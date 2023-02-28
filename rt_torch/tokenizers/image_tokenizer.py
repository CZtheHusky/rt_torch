from rt_torch.film_efficient.film_efficientnet import film_efficientnet_b3
from torchvision.models import EfficientNet_B3_Weights
from torch import nn
from rt_torch.tokenizers.token_learner import TokenLearnerModuleV11
import torch



class ImageTokenizer(nn.Module):
    def __init__(self,
                 use_token_leraner: bool = True,
                 num_tokens: int = 8,
                 dropout_rate: float = 0.1,
                 ) -> None:
        super().__init__()
        self.film_efficient_net = film_efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self._embedding_output_dim = self.film_efficient_net.embed_dim
        self._use_token_learner = use_token_leraner
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
                text_embeddings: torch.Tensor,
                ):
        image_tokens = self.film_efficient_net(images, text_embeddings)
        if self._use_token_learner:
            image_tokens = self.token_learner(image_tokens)
        return image_tokens     # b n c
        



