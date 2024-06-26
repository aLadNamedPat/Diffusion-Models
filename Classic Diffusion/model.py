import torch
import torch.nn.functional as F
import torch.nn as nn
from Attention import AttentionLayer
from Embedding import SinusoidalEmbedding
from Residual import ResidualBlock

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
            self.encoder_layer(input_channels, hidden_dims[0])
        )

        #Build a densely connected encoder with many skip connections
        for i in range(len(hidden_dims) - 1):
           self.encoder_store.append(
                self.encoder_layer(hidden_dims[i],
                                    hidden_dims[i + 1])
            )

        self.encoder = nn.Sequential(
            *self.encoder_store
        )

        self.decoder_store = []

        hidden_dims.reverse() #Reverse the hidden state list

        for i in range(len(hidden_dims) - 1):
            self.decoder_store.append(
                self.decoder_layer(
                    hidden_dims[i] * 2,
                    hidden_dims[i + 1]
                )
            )

        self.decoder = nn.Sequential(*self.decoder_store)

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
        upsample = nn.Sequential(
            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size = 1,
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
            block_two = self.downsample(output_channels, output_channels)

            return nn.Sequential(
                block,
                block_two
            )
    

    def decoder_layer(
        self,
        input_channels : int,
        output_channels : int,
        time_embed_size : int
    ):
        block = ResidualBlock(input_channels, output_channels, time_embed_size)
        block_two = self.upsample(output_channels, output_channels)

        #Use ConvTranspose2d to upsample back to the original image size
        return nn.Sequential(
            block,
            block_two
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
            input = torch.cat((input, skip_connections[i]), dim = 1)
            input = layer(input, time)
        a = torch.cat((input, skip_connections[-1]), dim = 1)
        a =  self.fl(a)
        return a

    def forward(
        self,
        input : torch.Tensor,
        time : int
    ):
        a = self.time_embedding(time)
        z, skip_connections = self.encode(input, a)
        a = self.decode(z, skip_connections, time)
        return a
    
    #Compute the loss of the diffusion model
    def find_loss(
        self,   
        predicted_noise,
        actual_noise,
    ) -> int:
        loss = F.mse_loss(predicted_noise, actual_noise)
        return loss