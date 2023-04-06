from typing import List, Optional
from rt_torch.transformer_class.transformer_classes import TransformerBlocks
from rt_torch.tokenizers.action_tokenizer import ActionTokenizer
from rt_torch.tokenizers.image_tokenizer import ImageTokenizer
from rt_torch.tokenizers.text_tokenizer import TextTokenizer
from einops import repeat, reduce, rearrange
from torch import nn
import torch
import os
import numpy as np
from torch.distributed import get_rank
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
            quantile_path=None,
    ):
        super().__init__()
        self.text_tokenizer = TextTokenizer(name=text_encoder,
                                            device=text_model_device)
        self.text_embed_dim = self.text_tokenizer.text_embed_dim
        self.image_tokenizer = ImageTokenizer(use_token_leraner=token_learner,
                                              num_tokens=learned_token_num,
                                              dropout_rate=token_learner_dropout,
                                              text_embedding_dim=self.text_embed_dim,
                                              conditioning=False)
        self.num_learned_tokens = self.image_tokenizer._num_tokens
        self.image_embed_dim = self.image_tokenizer._embedding_output_dim
        self.transformer = TransformerBlocks(num_actions=num_actions, 
                                             num_layers=num_layers,
                                             key_dim=key_dim, 
                                             num_heads=heads, 
                                             feed_forward_size=feed_forward_size, 
                                             vocab_size=vocab_size,
                                             max_sequence_len=seq_len * self.num_learned_tokens,
                                             input_embedding_size=self.image_embed_dim, 
                                             drop_out_rate=transformer_dropout,
                                             return_last=return_last,
                                             )
        self.return_last = return_last
        self.seq_len = seq_len
        self.memory_buffer = None
        self.criterion = nn.CrossEntropyLoss(reduction="mean")


    def forward(
            self,
            data,
    ):
        # bs sql c h w
        # bs sql d
        # bs d
        frames, texts_embeddings, actions = data
        batch_size = frames.shape[0]
        # rgb_size, embedding_size, action_size, split_idx_size = frames.shape, texts_embeddings.shape, actions.shape, split_idx.shape
        # print(f"rank: {get_rank()}, inference, rgb: {rgb_size}, embedding: {embedding_size}, action: {action_size}, split: {split_idx_size}")
        # print(f"inference, rgb: {rgb_size}, embedding: {embedding_size}, action: {action_size}, split: {split_idx_size}")
        device = frames.device
        # import pdb; pdb.set_trace()
        # print(f"rank: {get_rank()}, inference, frames: {frames.shape}, texts_embeddings: {texts_embeddings.shape}")
        frames = rearrange(frames, 'b l c h w -> (b l) c h w')
        texts_embeddings = rearrange(texts_embeddings, 'b l d -> (b l) d')
        image_tokens = self.image_tokenizer(frames, texts_embeddings)
        image_tokens = rearrange(image_tokens, '(b l) n d -> b (l n) d', b=batch_size)
        # print(f"rank: {get_rank()}, inference, image_tokens: {image_tokens.shape}")
        # import pdb; pdb.set_trace()
        # print(f"rank: {get_rank()}, inference, stacked_image_tokens: {image_tokens.shape}")
        attn_mask = torch.ones((self.seq_len * self.num_learned_tokens, self.seq_len * self.num_learned_tokens), dtype=torch.bool, device=device).triu(1)
        
        # attn_mask = repeat(attn_mask, 'i j -> (i r1) (j r2)', r1=self.num_learned_tokens, r2=self.num_learned_tokens)
        logits = self.transformer(image_tokens, attention_mask=attn_mask)
        if not self.return_last:
            logits = logits[:, -1]
        logits = logits.permute(0, 2, 1)
        # logits = logits.view(-1, logits.shape[-1])
        # actions_discretes = self.action_tokenizer.discretize(actions)
        # actions_discretes = actions_discretes.view(-1)
        # print(f"rank: {get_rank()}, inference: logits: {logits.shape}, actions_discretes: {actions_discretes.shape}")
        loss = self.criterion(logits, actions)
        # print(f"rank: {get_rank()}, inference: {float(loss.item())}")
        # import pdb; pdb.set_trace()
        return loss

    def inference(self, rgb, instruction, inst_buffer, device):
        # print(f"rgb: {rgb.shape}")
        # print(f"inst_buffer: {inst_buffer.shape}")
        texts_embeddings = self.text_tokenizer.embed_texts(instruction, device=device)
        texts_embeddings = texts_embeddings.to(inst_buffer.dtype)
        # print(f"texts_embeddings: {texts_embeddings.shape}")
        # print(f"inst_buffer {inst_buffer.device}, texts_embeddings {texts_embeddings.device}")
        all_embeddings = torch.cat([inst_buffer, texts_embeddings])
        image_tokens = self.image_tokenizer(rgb, all_embeddings)
        # print(f"image_tokens before: {image_tokens.shape}")
        image_tokens = rearrange(image_tokens, 'f n c -> (f n) c')
        image_tokens = image_tokens.view(1, *image_tokens.shape)
        # print(f"image_tokens after: {image_tokens.shape}")
        attn_mask = torch.ones((self.seq_len * self.num_learned_tokens, self.seq_len * self.num_learned_tokens), dtype=torch.bool, device=device).triu(1)
        # print(f"attn_mask: {attn_mask.shape}")
        logits = self.transformer(image_tokens, attention_mask=attn_mask)
        # print(f"logits: {logits.shape}")
        # action = self.action_tokenizer.discrete2Scalar(logits.squeeze(0))
        # print(f"action last: {action.shape}")
        out = logits.detach().cpu().squeeze(0)
        # print(f"action: {out}")
        return out, texts_embeddings

    def save_check_point(self, iteration, optimizer, save_path, logger, max_save_num, lr_scheduler=None):
        file_name = str(iteration)
        file_name = file_name.zfill(10)
        dict2save = {
            "model_state_dict": self.state_dict(),
            "iteration": iteration,
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if lr_scheduler is not None:
            dict2save["scheduler"] = lr_scheduler.state_dict()
        torch.save(dict2save,
            os.path.join(save_path, file_name + '.pt')
        )
        logger.info(f"check point saved")
        saved_list = os.listdir(save_path)
        if len(saved_list) > max_save_num:
            saved_list = sorted(saved_list)
            oldest = os.path.join(save_path, saved_list[0])
            logger.info(f"oldest check point removed, path: {oldest}")
            os.remove(oldest)
