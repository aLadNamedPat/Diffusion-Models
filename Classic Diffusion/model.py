import torch
import torch.nn.functional as F
import torch.nn as nn

# Use a U-Net encoder-decoder architecture to predict the noise
class UNet(nn.Module):
    def __init__(
        self, 
        input_channels : int,
        output_channels : int,
        hidden_dims : list[int]
    ) -> None:
        
        super(UNet, self).__init__()


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
        
    def encoder_layer(
        self,
        input_channels : int,
        output_channels : int,
        ):

        return nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size= 3,
                    stride = 2,
                    padding = 1
                ), 
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU()
        )
    
    def decoder_layer(
        self,
        input_channels : int,
        output_channels : int,
    ):
        #Use ConvTranspose2d to upsample back to the original image size
        return nn.Sequential(
            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size = 3,
                stride = 2,
                padding = 1,
                output_padding= 1
            ),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU()
        )

    def encode(
        self, 
        input : torch.Tensor
    ):  
        skip_connections = []
        for layer in self.encoder_store:
            input = layer(input)
            skip_connections.append(input)  #Append all the inputs from the previous layers [128, 128, 256, 512, 512]

        return input, skip_connections
        
    def decode(
        self,
        input : torch.Tensor,
        skip_connections : list
    ):
        
        skip_connections = skip_connections[::-1]
        for i, layer in enumerate(self.decoder_store):
            input = torch.cat((input, skip_connections[i]), dim = 1)
            input = layer(input)
        a = torch.cat((input, skip_connections[-1]), dim = 1)
        a =  self.fl(a)
        return a

    def forward(
        self,
        input : torch.Tensor,
    ):
        z, skip_connections = self.encode(input)
        a = self.decode(z, skip_connections)
        return a
    
    #Compute the loss of the diffusion model
    def find_loss(
        self,   
        predicted_noise,
        actual_noise,
    ) -> int:
        loss = F.mse_loss(predicted_noise, actual_noise)
        return loss
    

#Encode the timestep of the denoising sequence to send as part of the U-Net

