from rt_torch.utilizes import *
from torch import nn
from einops import repeat, reduce, rearrange
import math
import pdb
import torch
import torch.nn.functional as F
# token learner module


class TokenLearnerModuleV11(nn.Module):
    """TokenLearner module Version 1.1, using slightly different conv. layers.
    Instead of using 4 conv. layers with small channels to implement spatial
    attention, this version uses 2 grouped conv. layers with more channels. It
    also uses softmax instead of sigmoid. We confirmed that this version works
    better when having limited training data, such as training with ImageNet1K
    from scratch.
    Attributes:
      num_tokens: Number of tokens.
      dropout_rate: Dropout rate.
    """

    def __init__(self, in_channels, num_tokens, num_groups=1, dropout_rate=0.):
        """Applies learnable tokenization to the 2D inputs.
        Args:
          inputs: Inputs of shape `[bs, h, w, c]`.
        Returns:
          Output of shape `[bs, n_token, c]`.
        """
        super(TokenLearnerModuleV11, self).__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.num_groups = num_groups  # in_channels and out_channels must both be divisible by groups
        self.norm = nn.LayerNorm(self.in_channels)  # Operates on the last axis (c) of the input data.

        self.attention_maps = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, groups=self.num_groups, bias=False),
            nn.Conv2d(self.in_channels, self.num_tokens, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
        )
        self.feat_conv = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, groups=self.num_groups, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor):
        inputs = inputs.permute(0, 2, 3, 1) # inputs: [b c h w]
        feature_shape = inputs.shape  # Shape:  [bs, h, w, c]

        selected = inputs
        selected = self.norm(selected)
        selected = selected.permute(0, 3, 1, 2)  # Shape:  [bs, c, h, w]
        selected = self.attention_maps(selected)  # Shape: [bs, n_token, h, w].
        selected = selected.permute(0, 2, 3, 1)  # Shape: [bs, h, w, n_token].
        selected = selected.contiguous().view(feature_shape[0], feature_shape[1] * feature_shape[2],
                                 -1)  # Shape: [bs, h*w, n_token].
        selected = selected.permute(0, 2, 1)  # Shape: [bs, n_token, h*w].
        selected = F.softmax(selected, dim=-1)

        # Reshape the input to align it with the output of the conv block.
        feat = inputs
        feat = feat.permute(0, 3, 1, 2)   # Shape:  [bs, c, h, w]
        feat = self.feat_conv(feat)      # Shape: [bs, c, h, w].
        feat = feat.permute(0, 2, 3, 1)   # Shape: [bs, h, w, c].
        feat = feat.contiguous().view(feature_shape[0], feature_shape[1] * feature_shape[2], -1)  # Shape: [bs, h*w, c].

        # Produced the attended inputs.
        outputs = torch.einsum("...si,...id->...sd",  selected, feat)  # (B, n_token, c)
        outputs = self.dropout(outputs)

        return outputs


class TokenLearner(nn.Module):
    """
    https://arxiv.org/abs/2106.11297
    using the 1.1 version with the MLP (2 dense layers with gelu) for generating attention map
    """

    def __init__(
            self,
            *,
            dim,
            ff_mult=2,
            num_output_tokens=8,
            num_layers=2
    ):
        super().__init__()
        inner_dim = int(math.sqrt(dim * num_output_tokens))
        self.num_output_tokens = num_output_tokens
        self.layer_norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)
        self.attention = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, num_output_tokens),
            
        )

    def forward(self, x):
        # x: B C H W
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.layer_norm(x)
        x = rearrange(x, 'b h w c -> b (h w) c')
        attn = self.attention(x)    # attn: B h*w N
        attn = rearrange(attn, 'b l n -> b n l')
        attn = self.softmax(attn)
        x = rearrange(x, 'b l c -> b c l')
        x = x.unsqueeze(1)
        attn = attn.unsqueeze(2)
        # x B 1 C H*W 
        # attn: B N 1 h*w
        res = reduce(x * attn, 'b n c l -> b n c', 'mean')
        # import pdb; pdb.set_trace()
        return res


class TokenLearner_(nn.Module):
    """
    https://arxiv.org/abs/2106.11297
    using the 1.1 version with the MLP (2 dense layers with gelu) for generating attention map
    """

    def __init__(
            self,
            *,
            dim,
            ff_mult=2,
            num_output_tokens=8,
            num_layers=2
    ):
        super().__init__()
        inner_dim = dim * ff_mult * num_output_tokens

        self.num_output_tokens = num_output_tokens
        self.net = nn.Sequential(
            nn.Conv2d(dim * num_output_tokens, inner_dim, 1, groups=num_output_tokens),
            nn.GELU(),
            nn.Conv2d(inner_dim, num_output_tokens, 1, groups=num_output_tokens),
        )

    def forward(self, x):
        x, ps = pack_one(x, '* c h w')
        x = repeat(x, 'b c h w -> b (g c) h w', g=self.num_output_tokens)
        attn = self.net(x)

        attn = rearrange(attn, 'b g h w -> b 1 g h w')
        x = rearrange(x, 'b (g c) h w -> b c g h w', g=self.num_output_tokens)

        x = reduce(x * attn, 'b c g h w -> b c g', 'mean')
        x = unpack_one(x, ps, '* c n')
        return x