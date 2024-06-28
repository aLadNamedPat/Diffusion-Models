import torch
import torch.nn as nn
# Embed the timestep of the action being performed in a sinusoidal fashion
# Recommend looking at this article for more info! 
# https://kazemnejad.com/blog/transformer_architecture_positional_encoding/

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        t : torch.Tensor
    ) -> torch.Tensor:
        vals = torch.arange(0, self.k_max, dtype = torch.float32).to(device)
        w_ks = torch.exp(torch.log(torch.tensor(10000.0)) * -vals / self.k_max).to(device)

        # print("vals:", vals)
        # print("w_ks:", w_ks)

        t = torch.tensor(t, dtype=torch.float32).to(device)

        sins = torch.sin(torch.ger(t, w_ks)).to(device)
        cos = torch.cos(torch.ger(t, w_ks)).to(device)

        pe = torch.zeros((t.shape[0], self.embedding_dims), dtype=torch.float32).to(device)
        pe[:, 0::2] = sins
        pe[:, 1::2] = cos

        return pe