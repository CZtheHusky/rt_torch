from typing import Optional, Callable, Tuple
import torch
from torch import nn
from rt_torch.utilizes.utilize import *
from einops import rearrange, einsum
from einops.layers.torch import Rearrange, Reduce
from rt_torch.film_efficient.film_conditioning import FiLM
from rt_torch.utilizes.utilize import *


class ConditionalAttention(nn.Module):
    def __init__(
            self,
            feed_forward_size: int,
            key_dim: int,
            value_dim: int=None,
            output_dim: int=None,
            heads: int=8,
            dropout_att: float=0.1,
            conditioning: bool=False,
            text_embedding_dim=None
    ):
        super().__init__()
        self.heads = heads
        self.feed_forward_size = feed_forward_size
        self.output_dim = feed_forward_size if output_dim is None else output_dim
        self.value_dim = key_dim if value_dim is None else value_dim
        self.conditioning = conditioning
        self.scale = key_dim ** -0.5
        self.norm1 = nn.LayerNorm(feed_forward_size)
        self.attn_dropout = nn.Dropout(dropout_att)

        self.to_q = nn.Linear(feed_forward_size, self.heads * key_dim)
        self.to_k = nn.Linear(feed_forward_size, self.heads * key_dim)
        self.to_v = nn.Linear(feed_forward_size, self.heads * self.value_dim)
        self.to_out = nn.Linear(self.heads * self.value_dim, self.output_dim)
        if conditioning:
            raise NotImplementedError

    def forward(
            self,
            x,
            text_embeddings=None,
            attn_mask=None,
    ):
        # import pdb; pdb.set_trace()
        x = self.norm1(x)

        if text_embeddings is not None:
            # adaptive layer-norm
            raise NotImplementedError
        # q, k, v: b n inner_dim
        
        value = self.to_v(x)
        query = self.to_q(x)
        key = self.to_k(x)
        # q, k, v: b heads n dim_head
        query = rearrange(query, 'b n (h d) -> b h n d', h=self.heads)
        key = rearrange(key, 'b n (h d) -> b h n d', h=self.heads)
        value = rearrange(value, 'b n (h d) -> b h n d', h=self.heads)
        query = query * self.scale
        # import pdb; pdb.set_trace()
        sim = einsum(query, key, 'b h i d, b h j d -> b h i j')

        if exists(attn_mask):
            # import pdb; pdb.set_trace()
            sim = sim.masked_fill(attn_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        # attn: b h i j
        # v: b h j d
        
        out = einsum(attn, value, 'b h i j, b h j d -> b h i d')

        out = rearrange(out, 'b h n d -> b n (h d)')
        # import pdb; pdb.set_trace()
        return self.to_out(out)


class ConditionalTransformerLayer(nn.Module):
    def __init__(
            self,
            feed_forward_size,
            key_dim=4096,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
    ):
        super().__init__()
        self.attn = ConditionalAttention(feed_forward_size=feed_forward_size, key_dim=key_dim, heads=heads, dropout_att=attn_dropout)
        self.ffn = nn.Sequential(
            nn.Linear(feed_forward_size, feed_forward_size),
            nn.Dropout(ff_dropout),
        )
        self.ffn_norm = nn.LayerNorm(feed_forward_size)

    def forward(
            self,
            x,
            attn_mask=None
    ):

        x = self.attn(x, attn_mask=attn_mask) + x
        x = self.ffn_norm(x)
        x = self.ffn(x) + x
        return x
    
class TransformerBlocks(nn.Module):
    def __init__(self,
                num_layers: int = 1,
                key_dim: int = 4096,
                num_heads: int = 8,
                feed_forward_size: int = 512,
                vocab_size: int = 256,
                num_actions: int = 2,
                max_sequence_len: int = 256,
                input_embedding_size: int = 512,
                drop_out_rate: float = 0.1,
                ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            ConditionalTransformerLayer(feed_forward_size=feed_forward_size,
                                        key_dim=key_dim,
                                        heads=num_heads,
                                        attn_dropout=drop_out_rate,
                                        ff_dropout=drop_out_rate,
                                        ) for i in range(num_layers)
        ])
        self.token_embedding = nn.Linear(input_embedding_size, feed_forward_size)
        self.position_embedding = nn.Embedding(max_sequence_len, feed_forward_size)
        self.output_tokens = nn.Sequential(
            nn.Linear(feed_forward_size, num_actions * vocab_size),
            Rearrange('... (a b) -> ... a b', b=vocab_size),
            )

    def forward(self,
                inputs: torch.Tensor,
                attention_mask: torch.Tensor,
                gather_idx: torch.Tensor,
                ):
        bs = inputs.shape[0]
        seq_len = inputs.shape[1]
        token_embeddings = self.token_embedding(inputs)

        position_idx = torch.arange(seq_len, dtype=torch.long, device=inputs.device)
        position_idx = torch.tile(position_idx.unsqueeze(0), dims=(bs, 1))
        # import pdb; pdb.set_trace()
        position_embeddings = self.position_embedding(position_idx)
        x = token_embeddings + position_embeddings  # b (t n) d

        for layer in self.layers:
            x = layer(x, attention_mask)
        # import pdb; pdb.set_trace()
        if gather_idx is not None:
            gathered_x = x[:, gather_idx]
            return self.output_tokens(gathered_x)
        else:
            return self.output_tokens(x)

