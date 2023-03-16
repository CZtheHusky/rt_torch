import torch
from torch import nn
from torchvision.models import EfficientNet_B3_Weights

from rt_torch.film_efficient.film_efficientnet import film_efficientnet_b3
from rt_torch.tokenizers.token_learner import TokenLearnerModuleV11
from rt_torch.film_efficient.film_conditioning import FiLM


class ImageTokenizer(nn.Module):
    def __init__(self,
                 use_token_leraner: bool = True,
                 num_tokens: int = 8,
                 dropout_rate: float = 0.1,
                 eff_last_channels: int = 512,
                 text_embedding_dim: int = 768,
                 conditioning: bool = True,
                 ) -> None:
        super().__init__()
        self.film_efficient_net = film_efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1,
                                                       eff_last_channels=eff_last_channels,
                                                       text_embedding_dim=text_embedding_dim,
                                                       kwargs={"text_embedding_dim": text_embedding_dim, "conditioning": conditioning})
        self._embedding_output_dim = eff_last_channels
        self.conv_out = nn.Conv2d(self.film_efficient_net.embed_dim, eff_last_channels, kernel_size=1, stride=1,
                                  padding=0, bias=False)
        self._use_token_learner = use_token_leraner
        self.conditioning = conditioning
        if conditioning:
            self.film_layer = FiLM(text_embedding_dim=text_embedding_dim, num_channels=self._embedding_output_dim)
        if self._use_token_learner:
            self._num_tokens = num_tokens
            self.token_learner = TokenLearnerModuleV11(
                in_channels=self._embedding_output_dim,
                num_tokens=self._num_tokens,
                dropout_rate=dropout_rate,
            )

    @property
    def tokens_per_context_image(self) -> int:
        if self._use_token_learner:
            num_tokens = self._num_tokens
        else:
            num_tokens = 100
        return num_tokens

    def forward(self,
                images: torch.Tensor,
                text_embeddings: torch.Tensor=None,
                ):
        image_tokens = self.film_efficient_net(images, text_embeddings)
        image_tokens = self.conv_out(image_tokens)
        if self.conditioning:
            image_tokens = self.film_layer(text_embeddings, image_tokens)
        if self._use_token_learner:
            image_tokens = self.token_learner(image_tokens)
        return image_tokens  # b n c


if __name__ == "__main__":
    img_tk = ImageTokenizer(True, 8, 0.1)
    pass
