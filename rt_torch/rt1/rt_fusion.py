import os

import torch
from einops import rearrange
from torch import nn

from rt_torch.tokenizers.action_tokenizer import ActionTokenizer
from rt_torch.tokenizers.image_tokenizer import ImageTokenizer
from rt_torch.tokenizers.text_tokenizer import TextTokenizer
from rt_torch.transformer_class.transformer_classes import LanguageVisionFusion, positional_encoding, positional_encoding_2d


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
            seq_len=6,
            text_model_device='cpu',
            token_learner=False,
            learned_token_num=8,
            token_learner_dropout=0.1,
            transformer_dropout=0.1,
            feature_dropout=0.1,
            return_last=True,
            d_model=512,
    ):
        super().__init__()
        self.action_tokenizer = ActionTokenizer()
        self.image_tokenizer = ImageTokenizer(use_token_leraner=token_learner,
                                              num_tokens=learned_token_num,
                                              dropout_rate=token_learner_dropout)
        self.image_embed_dim = self.image_tokenizer._embedding_output_dim
        self.text_tokenizer = TextTokenizer(name=text_encoder,
                                            device=text_model_device)
        self.num_learned_tokens = self.image_tokenizer._num_tokens
        self.d_model = d_model
        self.image_dropout = nn.Dropout(p=feature_dropout)
        self.lang_dropout = nn.Dropout(p=feature_dropout)
        self.fusion = LanguageVisionFusion(
            nheads=fusion_nhead,
            dropout=transformer_dropout,
            d_model=d_model,
            n_layer=fusion_layers,
            dim_feedforward=d_model
        )
        self.transformer = LanguageVisionFusion(
            nheads=transformer_nhead,
            dropout=transformer_dropout,
            d_model=d_model,
            n_layer=transformer_layers,
            dim_feedforward=d_model
        )
        self.image_token_embedding = nn.Linear(self.image_embed_dim, d_model)
        self.text_token_embedding = nn.Linear(self.text_tokenizer.text_embed_dim, d_model)
        self.position_embedding = nn.Embedding(seq_len, feed_forward_size)
        self.seq_len = seq_len
        self.memory_buffer = None
        self.action_tokenizer = ActionTokenizer(num_action_bin=vocab_size, action_max=0.1, action_min=-0.1)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def token_stack(self, tokens, split_idx):
        start_idx = 0
        stacked_list = []
        for ep_idx in range(len(split_idx)):
            idx = int(split_idx[ep_idx].item())
            # print(idx)
            # print(idx.dtype)
            if idx == -1:
                continue
            current_split = tokens[start_idx:start_idx + idx]
            if ep_idx > 0:
                current_split = torch.cat([tokens[-self.seq_len + 1:], current_split])
            start_idx += idx
            stacked_tokens = torch.stack(
                [current_split[i - self.seq_len:i] for i in range(self.seq_len, len(current_split) + 1)])
            # import pdb; pdb.set_trace()
            stacked_list.append(stacked_tokens)
        return torch.cat(stacked_list)

    def forward(
            self,
            data,
    ):
        frames, texts_embeddings, actions, split_idx = data
        # rgb_size, embedding_size, action_size, split_idx_size = frames.shape, texts_embeddings.shape, actions.shape, split_idx.shape
        # print(f"rank: {get_rank()}, inference, rgb: {rgb_size}, embedding: {embedding_size}, action: {action_size}, split: {split_idx_size}")
        # print(f"inference, rgb: {rgb_size}, embedding: {embedding_size}, action: {action_size}, split: {split_idx_size}")
        device = frames.device
        # import pdb; pdb.set_trace()
        frames = torch.cat([frames, torch.zeros([self.seq_len - 1, *frames.shape[1:]], dtype=frames.dtype).to(device)])
        # import pdb; pdb.set_trace()
        if not isinstance(texts_embeddings, list):
            texts_embeddings = torch.cat([texts_embeddings, torch.zeros([self.seq_len - 1, *texts_embeddings.shape[1:]],
                                                                        dtype=texts_embeddings.dtype).to(device)])
        else:
            texts_embeddings.extend([""] * self.seq_len - 1)
            texts_embeddings = self.text_tokenizer(texts_embeddings)
        # import pdb; pdb.set_trace()
        # print(f"rank: {get_rank()}, inference, frames: {frames.shape}, texts_embeddings: {texts_embeddings.shape}")
        texts_embeddings = self.text_token_embedding(texts_embeddings)
        texts_embeddings = self.lang_dropout(texts_embeddings)
        image_tokens = self.image_tokenizer(frames, texts_embeddings)

        height, width = image_tokens.shape[2], image_tokens.shape[3]
        image_tokens = rearrange(image_tokens, 'b c h w -> b (h w) c')
        vision_embed = self.image_token_embedding(image_tokens)
        vision_embed *= torch.sqrt(self.d_model)
        vision_embed = rearrange(vision_embed, 'b (h w) c -> b h w c', h=height, w=width)
        vision_embed = positional_encoding_2d(vision_embed, self.d_model)
        vision_embed = rearrange(vision_embed, 'b h w c -> b (h w) c')
        vision_embed = self.image_dropout(vision_embed)
        fused_embed = self.fusion(texts_embeddings, vision_embed)
        if split_idx is not None:
            fused_embed = self.token_stack(fused_embed, split_idx)
            fused_embed = rearrange(image_tokens, 'b f n c -> b (f n) c')

        # print(f"rank: {get_rank()}, inference, image_tokens: {image_tokens.shape}")
        # import pdb; pdb.set_trace()

        # print(f"rank: {get_rank()}, inference, stacked_image_tokens: {image_tokens.shape}")
        attn_mask = torch.ones((self.seq_len * self.num_learned_tokens, self.seq_len * self.num_learned_tokens),
                               dtype=torch.bool, device=device).triu(1)

        # attn_mask = repeat(attn_mask, 'i j -> (i r1) (j r2)', r1=self.num_learned_tokens, r2=self.num_learned_tokens)
        logits = self.transformer(image_tokens, attention_mask=attn_mask)
        if not self.return_last:
            logits = logits[:, -1]
        # logits = logits.permute(0, 2, 1)
        logits = logits.view(-1, logits.shape[-1])
        actions_discretes = self.action_tokenizer.discretize(actions)
        actions_discretes = actions_discretes.view(-1)
        # print(f"rank: {get_rank()}, inference: logits: {logits.shape}, actions_discretes: {actions_discretes.shape}")
        loss = self.criterion(logits, actions_discretes)
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
        attn_mask = torch.ones((self.seq_len * self.num_learned_tokens, self.seq_len * self.num_learned_tokens),
                               dtype=torch.bool, device=device).triu(1)
        # print(f"attn_mask: {attn_mask.shape}")
        logits = self.transformer(image_tokens, attention_mask=attn_mask)
        # print(f"logits: {logits.shape}")
        action = self.action_tokenizer.discrete2Scalar(logits.squeeze(0))
        # print(f"action last: {action.shape}")
        out = action.detach().cpu().numpy()
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
