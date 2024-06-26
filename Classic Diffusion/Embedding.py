import torch
import torch.nn as nn
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
