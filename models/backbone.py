import torch
import torchvision
from torch import nn
from torch import Tensor
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import ResNet18_Weights
from typing import Optional, List

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask


    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)


    def decompose(self):
        return self.tensors, self.mask


    def __repr__(self):
        return str(self.tensors)


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d where the batch statistics and the affine parameters are fixed"""
  
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

  
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

  
    def forward(self, x):
        # move reshapes to the beginning to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

  
    def __init__(self, backbone: nn.Module, num_channels: int, return_interm_layers: bool):
        super().__init__()
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"} # only return the final output
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        """IntermediateLayerGetter
        Return the outputs of the specified layers
        Output format: dictionary
        e.g. For {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}, keys 
        are "0", "1", "2", "3" and values are outputs from layer 1, 2, 3, 4 of the backbone.
        
        """
        self.num_channels = num_channels

  
    def forward(self, tensor):
        xs = self.body(tensor)
        return xs


class Backbone(BackboneBase):
    """Visual encoder backbone (ResNet with frozen BatchNorm)"""
  
    def __init__(self, name: str,
                return_interm_layers: bool,
                dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            weights=ResNet18_Weights.DEFAULT, norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    """Visual encoder backbone + 2D positional encoding"""
  
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

  
    def forward(self, tensor_list: NestedTensor):
        """
        Returns:
            out: the output of the backbone (a list). If return_interm_layers==True, return
            the outputs of all the intermediate layers

            pos: the position embeddings of the outputs (a list)
        """
        xs = self[0](tensor_list)  # self[0]: backbone
        out: List[NestedTensor] = []
        pos = []
        for _, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.dtype))  # self[1]: positional encoding

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    backbone = Backbone(args.backbone, False, False)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
  
    return model