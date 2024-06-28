import torch
import torch.nn.functional as F
import torch.nn as nn
from Attention import MultiHeadedAttention
from Embedding import SinusoidalEmbedding
from Residual import ResidualBlock

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

class MultiInputSequential(nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for module in self:
            if isinstance(module, ResidualBlock):
                input = module(input, *args, **kwargs)
            else:
                input = module(input)
        return input

def attention_layer(num_channels, num_heads):
    return MultiHeadedAttention(num_channels, num_heads)

# A LOT OF HELP FROM: https://github.com/filipbasara0/simple-diffusion/blob/main/simple_diffusion/
# Use a U-Net encoder-decoder architecture to predict the noise
class UNet(nn.Module):
    def __init__(
        self, 
        input_channels : int,
        output_channels : int,
        hidden_dims : list[int],
        time_embedding_dims : int
    ) -> None:
        
        super(UNet, self).__init__()

        self.time_embedding = SinusoidalEmbedding(time_embedding_dims)
        self.encoder_store = []

        self.encoder_store.append(
            self.encoder_layer(input_channels, hidden_dims[0], time_embedding_dims)
        )

        #Build a densely connected encoder with many skip connections
        for i in range(len(hidden_dims) - 1):
           self.encoder_store.append(
                self.encoder_layer(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    time_embedding_dims)
            )

        self.encoder = MultiInputSequential(
            *self.encoder_store
        )

        self.decoder_store = []

        hidden_dims.reverse() #Reverse the hidden state list

        for i in range(len(hidden_dims) - 1):
            self.decoder_store.append(
                self.decoder_layer(
                    hidden_dims[i] * 2,
                    hidden_dims[i + 1],
                    time_embedding_dims
                )
            )

        self.decoder = MultiInputSequential(*self.decoder_store)

        self.fl = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1] * 2,
                hidden_dims[-1],
                kernel_size = 3,
                stride = 2,
                padding = 1,
                output_padding= 1
            ),
            nn.BatchNorm2d(
                hidden_dims[-1]
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                hidden_dims[-1],
                output_channels,
                kernel_size = 3,
                padding = 1
            ),
            nn.Tanh()
        )
    
    def downsample(
        self,
        input_channels : int,
        output_channels: int,
    ):
        downsample = nn.Sequential(
            nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=1,
            stride = 2,
            ),
            nn.LeakyReLU(),
        )
        return downsample
    
    def upsample(
        self,
        input_channels : int,
        output_channels : int,
    ):
        upsample = MultiInputSequential(
            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size = 2,
                stride = 2
            ),
            nn.LeakyReLU()
        )

        return upsample   
    
    def encoder_layer(
        self,
        input_channels : int,
        output_channels : int,
        time_embed_size : int
        ):
            block = ResidualBlock(input_channels, output_channels, time_embed_size)
            block_two = ResidualBlock(output_channels, output_channels, time_embed_size)
            block_three = self.downsample(output_channels, output_channels)

            return MultiInputSequential(
                block,
                block_two,
                # attention_layer(output_channels, 8),
                block_three
            )
    

    def decoder_layer(
        self,
        input_channels : int,
        output_channels : int,
        time_embed_size : int
    ):
        block = ResidualBlock(input_channels, output_channels, time_embed_size)
        block_two = ResidualBlock(output_channels, output_channels, time_embed_size)
        block_three = self.upsample(output_channels, output_channels)

        #Use ConvTranspose2d to upsample back to the original image size
        return MultiInputSequential(
            block,
            block_two,
            # attention_layer(output_channels, 8),
            block_three
        )

    def encode(
        self, 
        input : torch.Tensor,
        time : torch.Tensor
    ):  
        skip_connections = []
        for layer in self.encoder_store:
            input = layer(input, time)
            skip_connections.append(input)  #Append all the inputs from the previous layers [128, 128, 256, 512, 512]

        return input, skip_connections
        
    def decode(
        self,
        input : torch.Tensor,
        skip_connections : list,
        time : torch.Tensor
    ):
        
        skip_connections = skip_connections[::-1]
        for i, layer in enumerate(self.decoder_store):
            # print("skip:", skip_connections[i].shape)

            input = torch.cat((input, skip_connections[i]), dim = 1)
            input = layer(input, time)
        a = torch.cat((input, skip_connections[-1]), dim = 1)
        a =  self.fl(a)

        return a

    def forward(
        self,
        input : torch.Tensor,
        time : torch.Tensor,
    ) -> torch.Tensor:
        
        a = self.time_embedding(time)
        z, skip_connections = self.encode(input, a)
        a = self.decode(z, skip_connections, a)

        return a
    
    #Compute the loss of the diffusion model
    def find_loss(
        self,   
        predicted_noise,
        actual_noise,
    ) -> int:
        loss = F.mse_loss(predicted_noise, actual_noise)
        return loss