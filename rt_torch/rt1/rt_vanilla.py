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
        self.transformer = TransformerBlocks(num_actions=num_actions, 
                                             num_layers=num_layers,
                                             key_dim=key_dim, 
                                             num_heads=heads, 
                                             feed_forward_size=feed_forward_size, 
                                             vocab_size=vocab_size, 
                                             input_embedding_size=self.image_embed_dim, 
                                             drop_out_rate=transformer_dropout,
                                             return_last=return_last,
                                             )
        self.return_last = return_last
        self.seq_len = seq_len
        self.memory_buffer = None
        self.action_tokenizer = ActionTokenizer(num_action_bin=vocab_size, action_max=0.1, action_min=-0.1)
        self.criterion = nn.CrossEntropyLoss()


    def token_stack(self, tokens, split_idx, new_ep=False):
        start_idx = 0
        stacked_list = []
        for ep_idx, idx in enumerate(split_idx):
            current_split = tokens[start_idx:start_idx + idx]
            if ep_idx == 0:
                if new_ep:
                    current_split = torch.cat([tokens[-self.seq_len + 1:], current_split])
            else:
                current_split = torch.cat([tokens[-self.seq_len + 1:], current_split])
            start_idx += idx
            stacked_tokens = torch.stack([current_split[i - self.seq_len:i] for i in range(self.seq_len, len(current_split) + 1)])
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
        frames = self.seq_len
        split_idx = None
        # import pdb; pdb.set_trace()
        if isinstance(video, list):
            device = video[0].device
            split_idx = []
            for vid in video:
                split_idx.append(len(vid))
            # import pdb; pdb.set_trace()
            video = torch.cat(video, dim=0)
            video = torch.cat([video, torch.zeros([self.seq_len - 1, 3, 300, 300]).to(device)])
            if texts_embeddings is not None:
                texts_embeddings = torch.cat([texts_embeddings, torch.zeros([self.seq_len - 1, 768]).to(device)])
            else:
                texts.extend([""] * self.seq_len - 1)
        else:
            device = video.device
        if texts_embeddings is None:
            texts_embeddings = self.text_tokenizer(texts)
        image_tokens = self.image_tokenizer(video, texts_embeddings)
        # import pdb; pdb.set_trace()
        if split_idx is not None:
            image_tokens = self.token_stack(image_tokens, split_idx, new_ep)
            image_tokens = rearrange(image_tokens, 'b f n c -> b (f n) c')
        attn_mask = torch.ones((frames * self.num_learned_tokens, frames * self.num_learned_tokens), dtype=torch.bool, device=device).triu(1)
        
        # attn_mask = repeat(attn_mask, 'i j -> (i r1) (j r2)', r1=self.num_learned_tokens, r2=self.num_learned_tokens)
        logits = self.transformer(image_tokens, attention_mask=attn_mask)
        # import pdb; pdb.set_trace()
        if self.return_last:
            return logits
        else:
            return logits[:, -1]
    
    def cal_loss(self,
                 data,
                 device,
                 ):
        rgbs, instructions, actions, inst_embeddings, new_ep = data
        if isinstance(rgbs, list):
            rgbs = [rgb.to(device) for rgb in rgbs]
        else:
            rgbs.to(device)
        actions = actions.to(device)
        actions_discretes = self.action_tokenizer.discretize(actions)
        if inst_embeddings is not None:
            inst_embeddings = inst_embeddings.to(device)
        predicts = self.forward(video=rgbs, texts=instructions, texts_embeddings=inst_embeddings, new_ep=new_ep)
        predicts = predicts.permute(0, 2, 1)
        loss = self.criterion(predicts, actions_discretes)
        return loss