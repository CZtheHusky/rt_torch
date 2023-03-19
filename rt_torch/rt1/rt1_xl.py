from typing import List, Optional
from rt_torch.transformer_class.transformer_xl import PositionalEmbedding, PositionwiseFF, RelPartialLearnableMultiHeadAttn, RelPartialLearnableDecoderLayer
from rt_torch.tokenizers.action_tokenizer import ActionTokenizer
from rt_torch.tokenizers.image_tokenizer import ImageTokenizer
from rt_torch.tokenizers.text_tokenizer import TextTokenizer
from einops import repeat, reduce, rearrange
from torch import nn
import torch
# Robotic Transformer

class RT1_transformerxl(nn.Module):
    def __init__(
            self,
            num_actions=2,
            vocab_size=256,
            num_layers=8,
            heads=8,
            key_dim=512,
            feed_forward_size=512,
            text_encoder='t5',
            frame_stack=6,
            text_model_device='cpu',
            token_learner=True,
            learned_token_num=8,
            token_learner_dropout=0.1,
            transformer_dropout=0.1,
            return_last=True,
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
#         if config.n_inner is None:
#             self.d_inner = 4 * self.d_model
#         else:
#             self.d_inner = config.n_inner

#         self.pre_lnorm = config.pre_lnorm

#         self.mem_len = config.mem_len if config.mem_len is not None else 0
#         self.same_length = config.same_length
#         self.clamp_len = self.n_position
#         self.untie_r = config.untie_r

#         ##### Build embedding ######
#         self.text_vocab_size = config.text_vocab_size
#         self.discrete_vocab_size = config.num_discrete_values
#         self.continuous_vocab_size = config.num_continuous_bin
#         self.discrete_overlap_with_text = config.overlap_with_text
#         total_vocab_size = (
#             self.text_vocab_size
#             + self.continuous_vocab_size
#             + (0 if self.discrete_overlap_with_text else self.discrete_vocab_size)
#         )

#         # include a additional token as rl seperator '|' in Gato.
#         self.total_vocab_size = total_vocab_size + 1
#         self.rl_separator_token_id = total_vocab_size
#         del total_vocab_size

        
#         self.word_embedding = nn.Embedding(self.total_vocab_size, self.n_embed)
#         # self.word_positional_embedding = nn.Embedding(self.n_position, self.n_embed)
#         self.pos_emb = PositionalEmbedding(
#             self.n_embed
#         )  # use relative positional encoding, fixed
#         if not self.untie_r:
#             self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
#             self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))

#         self.vision_encoder = VisionEmbedding(config)
#         self.ic_encoder = self.vision_encoder

#         # positional encoding for RL timesteps, 0 is used for unique action embedding
#         self.rl_local_timestep_embedding = nn.Embedding(512 + 1, self.n_embed)

#         self.drop = nn.Dropout(config.embd_pdrop)
#         self.h = nn.ModuleList(
#             [
#                 RelPartialLearnableDecoderLayer(
#                     self.n_head,
#                     self.d_model,
#                     self.d_head,
#                     self.d_inner,
#                     config.drop,
#                     dropatt=config.dropattn,
#                     activation=config.activation_fn,
#                     pre_lnorm=self.pre_lnorm,
#                     r_w_bias=None if self.untie_r else self.r_w_bias,
#                     r_r_bias=None if self.untie_r else self.r_r_bias,
#                     layer_norm_epsilon=config.layer_norm_epsilon,
#                 )
#                 for i in range(config.n_layer)
#             ]
#         )
#         # self.ln_f = nn.LayerNorm(self.n_embed, eps=config.layer_norm_epsilon)
#         self.share_input_output_embedding = config.share_input_output_embedding
#         if config.share_input_output_embedding:
#             self.lm_head = None
#         else:
#             self.lm_head = nn.Linear(config.n_embed, self.total_vocab_size, bias=False)

#         self.apply(self._init_weights)

        self.frame_stack = frame_stack
        self.memory_buffer = None


    def token_stack(self, tokens, split_idx, new_ep=False):
        start_idx = 0
        stacked_list = []
        for ep_idx, idx in enumerate(split_idx):
            current_split = tokens[start_idx:start_idx + idx]
            if ep_idx == 0:
                if new_ep:
                    current_split = torch.cat([tokens[-self.frame_stack + 1:], current_split])
            else:
                current_split = torch.cat([tokens[-self.frame_stack + 1:], current_split])
            start_idx += idx
            stacked_tokens = torch.stack([current_split[i - self.frame_stack:i] for i in range(self.frame_stack, len(current_split) + 1)])
            # import pdb; pdb.set_trace()
            stacked_list.append(stacked_tokens)
        return torch.cat(stacked_list)

    def forward(
            self,
            video,
            texts: Optional[List[str]] = None,
            texts_embeddings=None,
            new_ep=False,
    ):
        # depth = self.transformer_depth
        frames = self.frame_stack
        split_idx = None
        # import pdb; pdb.set_trace()
        if isinstance(video, list):
            device = video[0].device
            split_idx = []
            for vid in video:
                split_idx.append(len(vid))
            # import pdb; pdb.set_trace()
            video = torch.cat(video, dim=0)
            video = torch.cat([video, torch.zeros([self.frame_stack - 1, 3, 300, 300]).to(device)])
            if texts_embeddings is not None:
                texts_embeddings = torch.cat([texts_embeddings, torch.zeros([5, 768]).to(device)])
            else:
                texts.extend([""] * 5)
        else:
            device = video.device
        if texts_embeddings is None:
            texts_embeddings = self.text_tokenizer(texts)
        image_tokens = self.image_tokenizer(video, texts_embeddings)
        if split_idx is not None:
            image_tokens = self.token_stack(image_tokens, split_idx, new_ep)
            # import pdb; pdb.set_trace()
            image_tokens = rearrange(image_tokens, 'b f n c -> b (f n) c')
        attn_mask = torch.ones((frames * self.num_learned_tokens, frames * self.num_learned_tokens), dtype=torch.bool, device=device).triu(1)
        
        # attn_mask = repeat(attn_mask, 'i j -> (i r1) (j r2)', r1=self.num_learned_tokens, r2=self.num_learned_tokens)
        logits = self.transformer(image_tokens, attention_mask=attn_mask)
        # import pdb; pdb.set_trace()
        last_logits = logits[:, -self.num_learned_tokens:, :]
        last_pooled = reduce(last_logits, 'b n d -> b d', 'mean')
        return last_pooled