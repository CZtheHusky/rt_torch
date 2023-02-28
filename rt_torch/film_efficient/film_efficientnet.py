from torchvision.models import EfficientNet_B3_Weights
from torchvision.models.efficientnet import MBConvConfig, FusedMBConvConfig, _MBConvConfig, WeightsEnum, Conv2dNormActivation, _ovewrite_named_param, _efficientnet_conf
import copy
import math
import warnings
from typing import Callable, Optional, List, Sequence, Union, Tuple, Any, Mapping
from torch import nn, Tensor
import torch
from collections import OrderedDict
from rt_torch.film_efficient.film_conditioning import FiLM



class film_EfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
        conditioning: bool = True,
        text_embedding_dim: int = 768,
        **kwargs: Any,
    ) -> None:
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")
        if "block" in kwargs:
            warnings.warn(
                "The parameter 'block' is deprecated since 0.13 and will be removed 0.15. "
                "Please pass this information on 'MBConvConfig.block' instead."
            )
            if kwargs["block"] is not None:
                for s in inverted_residual_setting:
                    if isinstance(s, MBConvConfig):
                        s.block = kwargs["block"]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conditioning = conditioning
        self.text_embedding_dim = text_embedding_dim
        self.features = nn.ModuleList([])
        if conditioning:
            self.conditioning_layers = nn.ModuleList([])
        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        # self.conv_in = Conv2dNormActivation(
        #         3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
        #     )
        self.features.append(Conv2dNormActivation(
                3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
            ))
        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage = nn.ModuleList([])
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                # if stage:
                if _:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                if conditioning:
                    self.conditioning_layers.append(FiLM(text_embedding_dim, block_cnf.input_channels))
                # self.features.append(block_cnf.block(block_cnf, sd_prob, norm_layer))

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            self.features.append(stage)

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        self.conv_out = Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        # self.features.append(Conv2dNormActivation(
        #         lastconv_input_channels,
        #         lastconv_output_channels,
        #         kernel_size=1,
        #         norm_layer=norm_layer,
        #         activation_layer=nn.SiLU,
        #     ))
        self.embed_dim = lastconv_output_channels

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)
        # import pdb; pdb.set_trace()
        for film_layer in self.conditioning_layers:
            nn.init.zeros_(film_layer.add.weight)
            nn.init.zeros_(film_layer.add.bias)
            nn.init.zeros_(film_layer.mult.weight)
            nn.init.zeros_(film_layer.mult.bias)

    def forward(
            self,
            x: Tensor,
            texts_embeddings: torch.Tensor = None,
    ) -> Tensor:
        x = self.features[0](x)
        stage_idx = 0
        for stages in self.features[1:]:
            for stage in stages:
                if self.conditioning:
                    x = self.conditioning_layers[stage_idx](texts_embeddings, x)
                x = stage(x)
                stage_idx += 1
        x = self.conv_out(x)

        # for stage, cond_fn in zip(self.stages, cond_fns):
        #     if exists(cond_fn):
        #         x = cond_fn(x)
        #
        #     x = stage(x)
        return x

def custom_load(model: nn.Module, pretrained_dict: OrderedDict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def _film_efficientnet(
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
    dropout: float,
    last_channel: Optional[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> film_EfficientNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = film_EfficientNet(inverted_residual_setting, dropout, last_channel=last_channel, **kwargs)
    # orig_state = weights.get_state_dict(progress=progress)
    # cur_state = model.state_dict()
    if weights is not None:
        # unexpected_keys = model.load_state_dict(weights.get_state_dict(progress=progress), strict=False)
        custom_load(model, weights.get_state_dict(progress=progress))
    return model

def film_efficientnet_b3(
    *, weights: Optional[EfficientNet_B3_Weights] = None, progress: bool = True, last_channel: int = 512, **kwargs: Any
) -> film_EfficientNet:
    """EfficientNet B3 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B3_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B3_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B3_Weights
        :members:
    """
    weights = EfficientNet_B3_Weights.verify(weights)

    inverted_residual_setting, _ = _efficientnet_conf("efficientnet_b3", width_mult=1.2, depth_mult=1.4)
    return _film_efficientnet(inverted_residual_setting, 0.3, last_channel, weights, progress, **kwargs)