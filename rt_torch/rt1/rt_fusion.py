import os

import torch
from einops import rearrange, repeat
from torch import nn
from einops.layers.torch import Rearrange
from rt_torch.tokenizers.action_tokenizer import ActionTokenizer
from rt_torch.tokenizers.image_tokenizer import ImageTokenizer
from rt_torch.tokenizers.text_tokenizer import TextTokenizer
from rt_torch.transformer_class.transformer_classes import LanguageVisionFusion, positionalencoding1d, positionalencoding2d, FFNResidual, FusionActionDecoder
import math


# Robotic Transformer

class RT1_fusion(nn.Module):
    def __init__(
            self,
            num_actions=2,
            vocab_size=256,
            fusion_layers=8,
            fusion_nhead=8,
            transformer_layers=2,
            transformer_nhead=8,
            key_dim=4096,
            feed_forward_size=512,
            text_encoder='t5',
            seq_len=4,
            text_model_device='cpu',
            token_learner=False,
            learned_token_num=8,
            dropout=0.1,
            ffn_res_dim=1024,
            ffn_res_blocks=2,
            quantile_path=None,
            d_model=512,
    ):
        super().__init__()
        self.text_tokenizer = TextTokenizer(name=text_encoder,
                                            device=text_model_device)
        self.text_embed_dim = self.text_tokenizer.text_embed_dim
        self.image_tokenizer = ImageTokenizer(use_token_leraner=token_learner,
                                              num_tokens=learned_token_num,
                                              dropout_rate=dropout,
                                              text_embedding_dim=self.text_embed_dim,
                                              conditioning=False)
        self.image_embed_dim = self.image_tokenizer._embedding_output_dim
        # self.num_learned_tokens = self.image_tokenizer._num_tokens
        self.d_model = d_model

        self.fusion = LanguageVisionFusion(
            nhead=fusion_nhead,
            dropout=dropout,
            d_model=d_model,
            n_layer=fusion_layers,
            dim_feedforward=d_model,
            image_embed_dim=self.image_embed_dim,
            text_embed_dim=self.text_embed_dim,
        )
        self.transformer = FusionActionDecoder(
            token_dim=d_model,
            nhead=transformer_nhead,
            dropout=dropout,
            d_model=d_model,
            n_layer=transformer_layers,
            dim_feedforward=d_model
        )
        self.ffn_residual = FFNResidual(
            input_dim=self.d_model,
            dim=ffn_res_dim,
            num_blocks=ffn_res_blocks,
        )
        self.action_proj = nn.Sequential(
            nn.Linear(ffn_res_dim, num_actions * vocab_size),
            Rearrange('... (a b) -> ... a b', b=vocab_size),
            )

        # self.position_embedding = nn.Embedding(seq_len, feed_forward_size)
        self.seq_len = seq_len
        self.memory_buffer = None
        
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(
            self,
            data,
    ):
        frames, texts_embeddings, actions, act_mask = data
        batch_size = frames.shape[0]
        # rgb_size, embedding_size, action_size, split_idx_size = frames.shape, texts_embeddings.shape, actions.shape, split_idx.shape
        # print(f"rank: {get_rank()}, inference, rgb: {rgb_size}, embedding: {embedding_size}, action: {action_size}, split: {split_idx_size}")
        # print(f"inference, rgb: {rgb_size}, embedding: {embedding_size}, action: {action_size}, split: {split_idx_size}")
        # import pdb; pdb.set_trace()
        # print(f"rank: {get_rank()}, inference, frames: {frames.shape}, texts_embeddings: {texts_embeddings.shape}")
        frames = rearrange(frames, 'b l c h w -> (b l) c h w')
        texts_embeddings = rearrange(texts_embeddings, 'b l d -> (b l) d')
        image_tokens = self.image_tokenizer(frames)
        fused_embed = self.fusion(image_tokens, texts_embeddings)
        fused_embed = rearrange(fused_embed, '(b l) d -> b l d', b=batch_size)
            # fused_embed = rearrange(image_tokens, 'b f n c -> b (f n) c')

        # print(f"rank: {get_rank()}, inference, image_tokens: {image_tokens.shape}")
        # import pdb; pdb.set_trace()

        # print(f"rank: {get_rank()}, inference, stacked_image_tokens: {image_tokens.shape}")
        fused_embed = self.transformer(fused_embed)
        fused_embed = self.ffn_residual(fused_embed)
        logits = self.action_proj(fused_embed)
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
        # import pdb; pdb.set_trace()
        texts_embeddings = self.text_tokenizer.embed_texts(instruction, device=device)
        texts_embeddings = texts_embeddings.to(inst_buffer.dtype)
        # print(f"texts_embeddings: {texts_embeddings.shape}")
        # print(f"inst_buffer {inst_buffer.device}, texts_embeddings {texts_embeddings.device}")
        all_embeddings = torch.cat([inst_buffer, texts_embeddings])
        image_tokens = self.image_tokenizer(rgb)
        fused_embed = self.fusion(image_tokens, all_embeddings)
        fused_embed = self.transformer(fused_embed)
        fused_embed = self.ffn_residual(fused_embed)
        logits = self.action_proj(fused_embed)
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
