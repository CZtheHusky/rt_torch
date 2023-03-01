from typing import List, Optional
from rt_torch.transformer_class.transformer_classes import TransformerBlocks
from rt_torch.tokenizers.action_tokenizer import ActionTokenizer
from rt_torch.tokenizers.image_tokenizer import ImageTokenizer
from rt_torch.tokenizers.text_tokenizer import TextTokenizer
from einops import repeat, reduce, rearrange
from torch import nn
import torch
# Robotic Transformer

class RT1_transformer(nn.Module):
    def __init__(
            self,
            num_actions=2,
            vocab_size=256,
            num_layers=8,
            heads=8,
            key_dim=4096,
            feed_forward_size=512,
            text_encoder='t5',
            seq_len=6,
            text_model_device='cpu',
            token_learner=True,
            learned_token_num=8,
            token_learner_dropout=0.1,
            transformer_dropout=0.1,
    ):
        super().__init__()
        self.action_tokenizer = ActionTokenizer
        self.image_tokenizer = ImageTokenizer(use_token_leraner=token_learner, 
                                              num_tokens=learned_token_num, 
                                              dropout_rate=token_learner_dropout)
        self.image_embed_dim = self.image_tokenizer._embedding_output_dim
        self.text_tokenizer = TextTokenizer(name=text_encoder, 
                                            device=text_model_device)
        self.num_learned_tokens = self.image_tokenizer._num_tokens
        self.transformer = TransformerBlocks(num_actions=num_actions, 
                                             num_layers=num_layers,
                                             key_dim=key_dim, 
                                             num_heads=heads, 
                                             feed_forward_size=feed_forward_size, 
                                             vocab_size=vocab_size, 
                                             input_embedding_size=self.image_embed_dim, 
                                             drop_out_rate=transformer_dropout,
                                             )
        self.attn_mask = torch.ones((seq_len * self.num_learned_tokens, seq_len * self.num_learned_tokens), dtype=torch.bool).triu(1)
        self.act_idx = torch.arange(self.num_learned_tokens - 1, seq_len * self.num_learned_tokens, self.num_learned_tokens)
        self.seq_len = seq_len
        self.memory_buffer = None

    def forward(
            self,
            video,
            texts: Optional[List[str]] = None,
            texts_embeddings=None,
    ):
        # depth = self.transformer_depth
        bs = video.shape[0]
        video = rearrange(video, 'b t c h w -> (b t) c h w')
        device = video.device
        if texts_embeddings is None:
            texts_embeddings = self.text_tokenizer(texts)
        else:
            texts_embeddings = rearrange(texts_embeddings, 'b t d -> (b t) d')
        image_tokens = self.image_tokenizer(video, texts_embeddings)
        # import pdb; pdb.set_trace()
        image_tokens = rearrange(image_tokens, '(b t) n c -> b (t n) c', b=bs)
        # attn_mask = repeat(attn_mask, 'i j -> (i r1) (j r2)', r1=self.num_learned_tokens, r2=self.num_learned_tokens)
        logits = self.transformer(image_tokens, attention_mask=self.attn_mask.to(device), gather_idx=self.act_idx)
        # import pdb; pdb.set_trace()
        return logits