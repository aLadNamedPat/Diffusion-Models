import torch
import torch.nn.functional as F
import torch.nn as nn

# A LOT OF HELP FROM: https://github.com/filipbasara0/simple-diffusion/blob/main/simple_diffusion/

# Embed the timestep of the action being performed in a sinusoidal fashion
# Recommend looking at this article for more info! 
# https://kazemnejad.com/blog/transformer_architecture_positional_encoding/

class SinusoidalEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dims : int = 100, 
    ) -> None:
        super(SinusoidalEmbedding, self).__init__()
        self.embedding_dims = embedding_dims            # Embedding dims is d
        self.k_max = embedding_dims // 2
        
    def forward(
        self,
        t : int
    ) -> torch.tensor:
        vals = torch.arange(0, self.k_max, dtype = torch.float32)
        w_ks = torch.exp(torch.log(10000) * -vals / self.k_max - 1)
        t = torch.tensor(t, dtype=torch.float32)

        sins = torch.sin(w_ks * t)
        cos = torch.cos(w_ks * t)

        pe = torch.zeros(self.embedding_dims)
        pe[0::2] = sins
        pe[1::2] = cos
        return pe
    
# Build a residual block based on the Wide ResNet architecture
class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels : int,
        output_channels : int,
        time_embed_size : torch.tensor
    ) -> None:
        # The goal of a residual block is to send residual data through while performing whatever needed function

        self.input_channels = input_channels
        self.output_channels = output_channels
        
        self.time_embedding = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(time_embed_size, output_channels),
            nn.LeakyReLU()
        )

        if input_channels != output_channels:
            self.residual = nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size = 1
            )
        else:
            self.residual = nn.Identity()

        self.block_one = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size = 3,
                padding = 2,
            ),
            nn.BatchNorm2d(
                output_channels
                ),
            nn.LeakyReLU()
        )

        self.block_two =  nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding = 2,
            ),
            nn.BatchNorm2d(
                output_channels,
            ),
            nn.LeakyReLU()
        )

    def forward(
        self,
        input : torch.tensor,
        time_embed : torch.tensor,
    ):
        res = self.residual(input)  # Convert the input channels to the output channels
        x = self.block_one(input) # Find the result of taking block one

        x += self.time_embedding(time_embed)[:, :, None, None] # Add the time embedding to the 

        x = self.block_two(x)

        return res + x


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