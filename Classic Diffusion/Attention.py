import torch
import torch.nn as nn
# Why use attention? Encoder long-distance dependencies as shown in this paper: https://arxiv.org/pdf/1805.08318
# https://discuss.pytorch.org/t/attention-in-image-classification/80147/2
#  
class AttentionLayer(nn.Module):
    def __init__(
        self,
        n_channels : int,
    ) -> None:
        
        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.q = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels // 8,
            kernel_size=1
        )
        
        self.k = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels // 8,
            kernel_size = 1
        )

        self.v = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels // 8,
            kernel_size = 1
        )

        self.l = nn.Conv2d(
            in_channels= n_channels,
            out_channels= n_channels,
            kernel_size= 1
        )
        self.softmax = nn.Softmax(dim = -1)

# The math is tricky here and definitely not intuitive so I'll try and break it down into a more understandable way
# What "attention" really is are dot products across each other to compute the "importance" of pixels relative to each other

# Let's say that we feed in data of shape (batch_size, num_channels, width, height)
# To compute the "attention", we want to multiply each point by each other point in a matrix multiplication manner
# So the first step is to resize, if we resize the input to size (batch_size, num_channels, width * height) then this essentailly compresses the pixel ranges linearly
# Let's permute this by rearranging the second and third term. You'll see why this is useful later on. This takes on the new form of (batch_size, width * height, num_channels)
# We'll call the above term the query
# Let's do the same for the keys, except we don't do any permutations. So we end up with (batch_size, num_channels, width * height)

# To visualize this, let's say width * height is a 2 * 2 image and that there are 3 channels. Each term is labeled as c_{pixel_identifier}{channel_number}
# This would look something like this in terms of queries (forget the batch_size term):
# [ [ c_11, c_12, c_13 ],
#   [ c_21, c_22, c_23 ],
#   [ c_31, c_32, c_33 ],
#   [ c_41, c_42, c_43 ]]

# In terms of values, this would look something like this (num_channels, width * height):
# [ [c_11, c_21, c_31, c_41],
#   [c_12, c_22, c_32, c_42],
#   [c_13, c_23, c_33, c_43]]

# If we take their matrix multiplication q * k, we end up getting
# [ [c_11 * c_11 + c_12 * c_12 + c_13 * c_13, c_11 * c_21 + c_12 * c_22 + c_13 * c_13, c_11 * c_31 + c_12 * c_32 + c_13 * c_33],
#   ...
# ]
# You get the point. What we've effectively done is taken the dot product of each pixel with each other pixel (in terms of channels)!
# This results in a matrix of [ [ s11, s12, s13, s14],
#                               [ s21, s22, s23, s24],
#                               [ s31, s32, s33, s34],
#                               [ s41, s42, s43, s44]]
# Now, we've effectively obtained the attention map (after a softmax along the last dimensions), we need to multiply again by the values to find the importance
# weight on itself (so far we've only looked at the relationship between two points, we now want to find how IMPACTFUL a weight is to another weight)
# This step is pretty straightforward since our new input size is (batch_size, w * h, w * h)    

    def forward(
        self,
        input : torch.Tensor,
    ) -> torch.Tensor:
        #input is of shape (batch_size, num_channels, width, height)
        batch_size, channels, w, h = input.size()

        query = self.q(input).view(batch_size, -1, w * h).permute(0, 2, 1)
        key = self.k(input).view(batch_size, -1, w * h)
        value = self.v(input).view(batch_size, -1, w * h)
        

        attention = self.softmax(torch.bmm(query * key))         # We use torch.bmm here for batch multiplication
        o = torch.bmm(value, attention.permute(0, 2, 1))        
        o = o.view(batch_size, channels, w, h)

        o = self.gamma * o + input
        return o, attention
