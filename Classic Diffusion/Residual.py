import torch
import torch.nn as nn
# Build a residual block based on the Wide ResNet architecture
class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels : int,
        output_channels : int,
        time_embed_size : torch.tensor
    ) -> None:
        # The goal of a residual block is to send residual data through while performing whatever needed function
        super(ResidualBlock, self).__init__()
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
                padding = 1,
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
                padding = 1,
            ),
            nn.BatchNorm2d(
                output_channels,
            ),
            nn.LeakyReLU()
        )

    def forward(
        self,
        input : torch.Tensor,
        time_embed : torch.Tensor,
    ):
        res = self.residual(input)  # Convert the input channels to the output channels
        x = self.block_one(input) # Find the result of taking block one

        time_embed = self.time_embedding(time_embed)
        # print(time_embed.shape)
        time_embed = time_embed[:, :, None, None]

        x += time_embed # Add the time embedding to the output
        # print(time_embed.shape)
        # print(x.shape)
        # print(res.shape)
        x = self.block_two(x)

        return res + x