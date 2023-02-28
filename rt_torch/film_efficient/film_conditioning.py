from torch import nn
import torch
from einops import rearrange, repeat, pack, unpack

class FiLM(nn.Module):
    def __init__(
        self,
        text_embedding_dim,
        num_channels,
    ):
        super().__init__()
        self.add = nn.Linear(text_embedding_dim, num_channels)
        self.mult = nn.Linear(text_embedding_dim, num_channels)
        nn.init.zeros_(self.add.weight)
        nn.init.zeros_(self.add.bias)
        nn.init.zeros_(self.mult.weight)
        nn.init.zeros_(self.mult.bias)

    def forward(self, conditions: torch.Tensor, feature_map: torch.Tensor):
        add = self.add(conditions)
        mult = self.mult(conditions)
        # import pdb; pdb.set_trace()
        add = add.view(*add.shape, 1, 1)
        mult = mult.view(*mult.shape, 1, 1)
        return (1 + mult) * feature_map + add