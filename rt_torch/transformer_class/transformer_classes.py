from einops import rearrange, einsum
from einops.layers.torch import Rearrange
from torch import nn

from rt_torch.utilizes.utilize import *


class ConditionalAttention(nn.Module):
    def __init__(
            self,
            feed_forward_size: int,
            key_dim: int,
            value_dim: int = None,
            output_dim: int = None,
            heads: int = 8,
            dropout_att: float = 0.1,
            conditioning: bool = False,
            text_embedding_dim=None,
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
            sim = sim.masked_fill(attn_mask, torch.finfo(sim.dtype).min)

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
        self.attn = ConditionalAttention(feed_forward_size=feed_forward_size, key_dim=key_dim, heads=heads,
                                         dropout_att=attn_dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(feed_forward_size),
            nn.Linear(feed_forward_size, feed_forward_size),
            nn.Dropout(ff_dropout),
        )

    def forward(
            self,
            x,
            attn_mask=None
    ):
        x = self.attn(x, attn_mask=attn_mask) + x
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
                 max_sequence_len: int = 48,
                 input_embedding_size: int = 512,
                 drop_out_rate: float = 0.1,
                 return_last: bool = True,
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
        self.return_last = return_last

    def forward(self,
                inputs: torch.Tensor,
                attention_mask: torch.Tensor,
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
        if self.return_last:
            # import pdb; pdb.set_trace()
            # print(f"x {x.shape}")
            x = x[:, -1]  # b (t n) d
            # print(f"x after {x.shape}")
        return self.output_tokens(x)


class FusionTransformerBlocks(nn.Module):
    def __init__(
            self,
            nhead=8,
            dropout=0.1,
            d_model=512,
            dim_feedforward=None,
            batch_first=True,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model
        self.query_norm = nn.LayerNorm(d_model)
        self.key_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=batch_first)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, query, key, attn_mask=None):
        residual_query = query
        query = self.qeury_norm(query)
        key = self.key_norm(key)
        x, _ = self.attention(query, key, key, attn_mask=attn_mask)
        x = x + residual_query
        x = self.ffn_dropout(x) + x
        return x




class LanguageVisionFusion(nn.Module):
    def __init__(
            self,
            nhead=8,
            dropout=0.1,
            d_model=512,
            n_layer=2,
            dim_feedforward=2048,
    ) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [FusionTransformerBlocks(d_model=d_model,
                                     nhead=nhead,
                                     dim_feedforward=dim_feedforward,
                                     dropout=dropout,
                                     batch_first=True,
                                     )
             for _ in range(n_layer)]
        )

    def forward(self, lan_emb, vis_emb, attn_mask=None):
        for layer in self.transformer_blocks:
            lan_emb = layer(lan_emb, vis_emb, attn_mask=attn_mask)
        return lan_emb



def positional_encoding(x, d_model):
    """Adds positional encoding to a tensor.

    Args:
      x: Tensor with shape [batch_size, seq_length, d_model]
      d_model: Model dimensionality.

    Returns:
      Tensor with the same shape as x.
    """
    batch_size, seq_length = x.size(0), x.size(1)
    pos = torch.arange(seq_length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(10000.0) / d_model))
    pos_embedding = torch.zeros((seq_length, d_model))
    pos_embedding[:, 0::2] = torch.sin(pos * div_term)
    pos_embedding[:, 1::2] = torch.cos(pos * div_term)
    pos_embedding = pos_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
    x = x + pos_embedding.to(x.device)
    return x

def positional_encoding_2d(x, d_model):
    """Adds positional encoding to a 2D tensor.

    Args:
      x: Tensor with shape [batch_size, height, width, d_model]
      d_model: Model dimensionality.

    Returns:
      Tensor with the same shape as x.
    """
    batch_size, height, width = x.size(0), x.size(1), x.size(2)
    pos_x = torch.arange(height, dtype=torch.float32).unsqueeze(1).repeat(1, width)
    pos_y = torch.arange(width, dtype=torch.float32).unsqueeze(0).repeat(height, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(10000.0) / d_model))
    pos_x_embedding = torch.zeros((height, d_model))
    pos_x_embedding[:, 0::2] = torch.sin(pos_x * div_term)
    pos_x_embedding[:, 1::2] = torch.cos(pos_x * div_term)
    pos_y_embedding = torch.zeros((width, d_model))
    pos_y_embedding[:, 0::2] = torch.sin(pos_y * div_term)
    pos_y_embedding[:, 1::2] = torch.cos(pos_y * div_term)
    pos_x_embedding = pos_x_embedding.unsqueeze(1).repeat(1, width, 1)
    pos_y_embedding = pos_y_embedding.unsqueeze(0).repeat(height, 1, 1)
    pos_embedding = torch.cat([pos_x_embedding, pos_y_embedding], dim=2).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    x = x + pos_embedding.to(x.device)
    return x


if __name__ == "__main__":
    t_block = TransformerBlocks(num_layers=1, key_dim=512, num_heads=8, feed_forward_size=512, vocab_size=256)
    attn_mask = torch.ones((48, 48), dtype=torch.bool).triu(1)
    test_input = torch.randn((5, 48, 512))
    out = t_block(test_input, attn_mask)
    print(out.shape)
    print(out)
