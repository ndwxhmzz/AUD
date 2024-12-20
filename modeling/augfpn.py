# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import torch
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import build_resnet_backbone

__all__ = ["build_resnet_augfpn_backbone", "AugFPN"]


class AugFPN(Backbone):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
            self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum",
            pool_ratios=(0.1, 0.2, 0.3)
    ):
        super(AugFPN, self).__init__()
        assert isinstance(bottom_up, Backbone)

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape() # resnet output is the input 
        in_strides = [input_shapes[f].stride for f in in_features]
        in_channels = [input_shapes[f].channels for f in in_features]
        aug_lateral_conv = in_channels[-1]

        _assert_strides_are_log2_contiguous(in_strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            ) # 1*1 conv C->M
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            ) # 3*3 M->P  keep width and height
            weight_init.c2_xavier_fill(lateral_conv) # init conv parameters
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(in_strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        # add lateral conv for features generated by rato-invariant scale adaptive pooling
        self.adaptive_pool_output_ratio = pool_ratios
        self.aug_lateral_conv = nn.ModuleList()
        self.aug_lateral_conv.extend([nn.Conv2d(aug_lateral_conv, out_channels, 1)
                                      for _ in range(len(self.adaptive_pool_output_ratio))])
        # the left part of ASF 
        self.aug_lateral_conv_attention = nn.Sequential(
            nn.Conv2d(out_channels * (len(self.adaptive_pool_output_ratio)), out_channels, 1), nn.ReLU(),
            nn.Conv2d(out_channels, len(self.adaptive_pool_output_ratio), 3, padding=1))

        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in in_strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = in_strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        bottom_up_features = self.bottom_up(x)
        x = [bottom_up_features[f] for f in self.in_features[::-1]]

        results = []
        # middle features M5
        prev_features_tmp = self.lateral_convs[0](x[0])
        raw_laternals = [prev_features_tmp.clone()]
        results.append(self.output_convs[0](prev_features_tmp))

        # Residual Feature Augmentation
        h, w = x[0].shape[-2:]
        # Ratio Invariant Adaptive Pooling x[0] is C5
        AdapPool_Features = [
            F.upsample(
                self.aug_lateral_conv[j](
                    F.adaptive_avg_pool2d(
                        x[0], output_size=(
                            max(1, int(h * self.adaptive_pool_output_ratio[j])),
                            max(1, int(w * self.adaptive_pool_output_ratio[j]))
                        )
                    )
                ), size=(h, w), mode='bilinear', align_corners=True
            ) for j in range(len(self.adaptive_pool_output_ratio))
        ]
        Concat_AdapPool_Features = torch.cat(AdapPool_Features, dim=1)
        fusion_weights = self.aug_lateral_conv_attention(Concat_AdapPool_Features)
        fusion_weights = F.sigmoid(fusion_weights)
        adap_pool_fusion = 0
        # part2 Residual Feature Augmentation 
        for i in range(len(self.adaptive_pool_output_ratio)):
            adap_pool_fusion += torch.unsqueeze(fusion_weights[:, i, :, :], dim=1) * AdapPool_Features[i]

        prev_features = prev_features_tmp + adap_pool_fusion
        # P5 top -> down
        for features, lateral_conv, output_conv in zip(
                x[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            lateral_features = lateral_conv(features)

            raw_laternals.insert(0, lateral_features.clone())
            # fusion
            ##### may be have mistake #####
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))

        assert len(self._out_features) == len(results) == len(raw_laternals)
        return dict(zip(self._out_features, results)), dict(zip(self._out_features, raw_laternals))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels, out_channels, in_feature="res5"):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_feature
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


@BACKBONE_REGISTRY.register()
def build_resnet_augfpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS #default channel is 256a
    backbone = AugFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=None,  # LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE, # default type is sum
    )
    return backbone
