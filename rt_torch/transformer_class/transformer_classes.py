from einops import rearrange, einsum
from einops.layers.torch import Rearrange
from torch import nn

from rt_torch.utilizes.utilize import *
import math

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
        self.act_idx = torch.arange(7, 48, 8, dtype=torch.long)

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
        x = x[:, self.act_idx]
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

    def forward(self, query, key=None, attn_mask=None):
        if key is None:
            key = query
        residual_query = query
        query = self.query_norm(query)
        key = self.key_norm(key)
        x, _ = self.attention(query, key, key, attn_mask=attn_mask)
        x = x + residual_query
        x = self.ffn_dropout(x) + x
        return x




class FusionTransformer(nn.Module):
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

    def forward(self, lan_emb, vis_emb=None, attn_mask=None):
        if vis_emb is None:
            for layer in self.transformer_blocks:
                lan_emb = layer(lan_emb, lan_emb, attn_mask=attn_mask)
        else:
            for layer in self.transformer_blocks:
                lan_emb = layer(lan_emb, vis_emb, attn_mask=attn_mask)
        return lan_emb
    
class LanguageVisionFusion(nn.Module):
    def __init__(
            self,
            nhead=8,
            dropout=0.1,
            d_model=512,
            n_layer=2,
            dim_feedforward=2048,
            image_embed_dim=512,
            text_embed_dim=768,
    ) -> None:
        super().__init__()
        self.transformer = FusionTransformer(
            nhead=nhead,
            dropout=dropout,
            d_model=d_model,
            n_layer=n_layer,
            dim_feedforward=dim_feedforward
        )
        self.scale = d_model ** 0.5
        self.image_token_embedding = nn.Linear(image_embed_dim, d_model)
        self.text_token_embedding = nn.Linear(text_embed_dim, d_model)
        self.image_dropout = nn.Dropout(p=dropout)
        self.lang_dropout = nn.Dropout(p=dropout)
        self.fused_norm = nn.LayerNorm(d_model)
    
    def forward(self, image_tokens, texts_embeddings):
        texts_embeddings = self.text_token_embedding(texts_embeddings)
        texts_embeddings = self.lang_dropout(texts_embeddings)
        height, width = image_tokens.shape[2], image_tokens.shape[3]
        image_tokens = rearrange(image_tokens, 'b c h w -> b (h w) c')
        vision_embed = self.image_token_embedding(image_tokens)
        vision_embed *= self.scale
        vision_embed = rearrange(vision_embed, 'b (h w) c -> b h w c', h=height, w=width)
        
        pe = positionalencoding2d(vision_embed.shape[-1], height=vision_embed.shape[1], width=vision_embed.shape[2]).to(vision_embed.device)
        vision_embed += pe.permute(1, 2, 0)
        vision_embed = rearrange(vision_embed, 'b h w c -> b (h w) c')
        vision_embed = self.image_dropout(vision_embed)
        # import pdb; pdb.set_trace()

        texts_embeddings = texts_embeddings.unsqueeze(1)
        fused_embed = self.transformer(texts_embeddings, vision_embed)
        fused_embed = fused_embed.squeeze(1)
        return self.fused_norm(fused_embed)


class FusionActionDecoder(nn.Module):
    def __init__(            
            self,
            token_dim,
            nhead=8,
            dropout=0.1,
            d_model=512,
            n_layer=2,
            dim_feedforward=2048,) -> None:
        super().__init__()
        self.transformer = FusionTransformer(
            nhead=nhead,
            dropout=dropout,
            d_model=d_model,
            n_layer=n_layer,
            dim_feedforward=dim_feedforward,
        )
        self.ffn = nn.Linear(token_dim, d_model)
        self.scale = d_model ** 0.5
        self.post_norm = nn.LayerNorm(d_model)
        self.dp = nn.Dropout(dropout)

    def forward(self, tokens, attn_mask=None):
        tokens = self.ffn(tokens)
        tokens *= self.scale
        # import pdb; pdb.set_trace()
        if len(tokens.shape) == 2:
            pe1 = positionalencoding1d(tokens.shape[-1], tokens.shape[0])
        elif len(tokens.shape) == 3:
            pe1 = positionalencoding1d(tokens.shape[-1], tokens.shape[1])
        else:
            import pdb; pdb.set_trace()
        tokens += pe1.to(tokens.device)
        tokens = self.dp(tokens)
        logits = self.transformer(tokens, attn_mask=attn_mask)
        # import pdb; pdb.set_trace()
        if len(logits.shape) == 2:
            logits = logits.mean(0)
            logits = self.post_norm(logits.unsqueeze(0))
        else:
            logits = logits.mean(1)
            logits = self.post_norm(logits)
        
        return logits
    

class FFNResidualBlock(nn.Module):
    def __init__(self, dim=1024) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
        )

    def forward(self, x):
        return x + self.layers(x)

class FFNResidual(nn.Module):
    def __init__(self, input_dim=512, dim=1024, num_blocks=2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            FFNResidualBlock(dim) for i in range(num_blocks)
        ])
        self.ffn = nn.Linear(input_dim, dim)
    
    def forward(self, x):
        x = self.ffn(x)
        for block in self.blocks:
            x = block(x)
        return x


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


if __name__ == "__main__":
    t_block = TransformerBlocks(num_layers=1, key_dim=512, num_heads=8, feed_forward_size=512, vocab_size=256)
    attn_mask = torch.ones((48, 48), dtype=torch.bool).triu(1)
    test_input = torch.randn((5, 48, 512))
    out = t_block(test_input, attn_mask)
    print(out.shape)
    print(out)
