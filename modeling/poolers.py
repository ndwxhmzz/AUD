# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import sys
from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import RoIPool
from typing import List, Optional
from detectron2.layers import ROIAlign, ROIAlignRotated, cat, nonzero_tuple, shapes_to_tensor
from detectron2.structures import Boxes
from detectron2.utils.tracing import assert_fx_safe, is_fx_tracing
import torch.nn.init as init

__all__ = ["ROIPooler"]


@torch.jit.script_if_tracing
def _create_zeros(
    batch_target: Optional[torch.Tensor],
    channels: int,
    height: int,
    width: int,
    like_tensor: torch.Tensor,
) -> torch.Tensor:
    batches = batch_target.shape[0] if batch_target is not None else 0
    sizes = (batches, channels, height, width)
    return torch.zeros(sizes, dtype=like_tensor.dtype, device=like_tensor.device)


def assign_boxes_to_levels(
    box_lists: List[Boxes],
    min_level: int,
    max_level: int,
    canonical_box_size: int,
    canonical_level: int,
):
    """
    Map each box in `box_lists` to a feature map level index and return the assignment
    vector.

    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]): A list of N Boxes or N RotatedBoxes,
            where N is the number of images in the batch.
        min_level (int): Smallest feature map level index. The input is considered index 0,
            the output of stage 1 is index 1, and so.
        max_level (int): Largest feature map level index.
        canonical_box_size (int): A canonical box size in pixels (sqrt(box area)).
        canonical_level (int): The feature map level index on which a canonically-sized box
            should be placed.

    Returns:
        A tensor of length M, where M is the total number of boxes aggregated over all
            N batch images. The memory layout corresponds to the concatenation of boxes
            from all images. Each element is the feature map index, as an offset from
            `self.min_level`, for the corresponding box (so value i means the box is at
            `self.min_level + i`).
    """
    box_sizes = torch.sqrt(cat([boxes.area() for boxes in box_lists]))
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-8)
    )
    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level

def convert_boxes_to_pooler_format(box_lists):
    """
    Convert all boxes in `box_lists` to the low-level format used by ROI pooling ops
    (see description under Returns).

    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]):
            A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.

    Returns:
        When input is list[Boxes]:
            A tensor of shape (M, 5), where M is the total number of boxes aggregated over all
            N batch images.
            The 5 columns are (batch index, x0, y0, x1, y1), where batch index
            is the index in [0, N) identifying which batch image the box with corners at
            (x0, y0, x1, y1) comes from.
        When input is list[RotatedBoxes]:
            A tensor of shape (M, 6), where M is the total number of boxes aggregated over all
            N batch images.
            The 6 columns are (batch index, x_ctr, y_ctr, width, height, angle_degrees),
            where batch index is the index in [0, N) identifying which batch image the
            rotated box (x_ctr, y_ctr, width, height, angle_degrees) comes from.
    """

    def fmt_box_list(box_tensor, batch_index):
        repeated_index = torch.full(
            (len(box_tensor), 1), batch_index, dtype=box_tensor.dtype, device=box_tensor.device
        )
        return cat((repeated_index, box_tensor), dim=1)

    pooler_fmt_boxes = cat(
        [fmt_box_list(box_list.tensor, i) for i, box_list in enumerate(box_lists)], dim=0
    )

    return pooler_fmt_boxes

class AFF(nn.Module):


    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        def init_weights(m):
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0.0)
        self.local_att.apply(init_weights)  
        self.global_att.apply(init_weights) 
    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
   
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = x * wei + residual * (1 - wei)
        return xo

class AFF_nBN(nn.Module):


    def __init__(self, channels=64, r=4):
        super(AFF_nBN, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

        self.sigmoid = nn.Sigmoid()
        def init_weights(m):
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0.0)
        self.local_att.apply(init_weights)  
        self.global_att.apply(init_weights) 
    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
   
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = x * wei + residual * (1 - wei)
        return xo
    
        roi_feats_list = []
        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            roi_feats_list.append(pooler(x_level, pooler_fmt_boxes))
        
        """
            multiple AFF
        """
      
        # aff = AFF(channels=num_channels).to(device)
        # feature_12 = aff(roi_feats_list[0], roi_feats_list[1])
        # feature_123 = aff(feature_12, roi_feats_list[2])
        # feature_1234 = aff(feature_123, roi_feats_list[3])
        # output = feature_1234
        # print(output.shape)
        # exit(0)
        # return output
    
        # # concat in channel dims [batchsize, NC, h, w]
        # concat_roi_feats = torch.cat(roi_feats_list, dim=1)
        # # [batchszie, N, h, w]
        # spatial_attention_map = self.spatial_attention_conv(concat_roi_feats)
 
        # for i in range(self.canonical_level):
        #     output += (F.sigmoid(spatial_attention_map[:, i, None, :, :]) * roi_feats_list[i])
        # print(output.shape)
        # exit(0)
        # return output

class ROIPooler_aug(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
            self,
            output_size,
            scales,
            sampling_ratio,
            pooler_type,
            canonical_box_size=224,
            canonical_level=4,
            out_channels=256,
    ):
        """
        Args:
            output_size (int, tuple[int] or list[int]): output size of the pooled region,
                e.g., 14 x 14. If tuple or list is given, the length must be 2.
            scales (list[float]): The scale for each low-level pooling op relative to
                the input image. For a feature map with stride s relative to the input
                image, scale is defined as a 1 / s. The stride must be power of 2.
                When there are multiple scales, they must form a pyramid, i.e. they must be
                a monotically decreasing geometric sequence with a factor of 1/2.
            sampling_ratio (int): The `sampling_ratio` parameter for the ROIAlign op.
            pooler_type (string): Name of the type of pooling operation that should be applied.
                For instance, "ROIPool" or "ROIAlignV2".
            canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
                is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
                pre-training).
            canonical_level (int): The feature map level index from which a canonically-sized box
                should be placed. The default is defined as level 4 (stride=16) in the FPN paper,
                i.e., a box of size 224x224 will be placed on the feature with stride=16.
                The box placement for all boxes will be determined from their sizes w.r.t
                canonical_box_size. For example, a box whose area is 4x that of a canonical box
                should be used to pool features from feature level ``canonical_level+1``.

                Note that the actual input feature maps given to this module may not have
                sufficiently many levels for the input boxes. If the boxes are too large or too
                small for the input feature maps, the closest level will be used.
        """
        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size

        self.aff = AFF(channels=out_channels)
        
        if pooler_type == "ROIAlign":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False
                )
                for scale in scales
            )
        elif pooler_type == "ROIAlignV2":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
                for scale in scales
            )
        elif pooler_type == "ROIPool":
            self.level_poolers = nn.ModuleList(
                RoIPool(output_size, spatial_scale=scale) for scale in scales
            )
        elif pooler_type == "ROIAlignRotated":
            self.level_poolers = nn.ModuleList(
                ROIAlignRotated(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
                for scale in scales
            )
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))

        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.
        min_level = -(math.log2(scales[0]))
        max_level = -(math.log2(scales[-1]))
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        ), "Featuremap stride is not power of 2!"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert (
            len(scales) == self.max_level - self.min_level + 1
        ), "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        assert 0 <= self.min_level and self.min_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size
        # input_size: concat_channel
        self.spatial_attention_conv = nn.Sequential(nn.Conv2d(out_channels * canonical_level, out_channels, 1),
                                                    nn.ReLU(),
                                                    nn.Conv2d(out_channels, canonical_level, 3, padding=1))

    def forward(self, x: List[torch.Tensor], box_lists):

        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Boxes] | list[RotatedBoxes]):  [514, 4]
                A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.

        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        num_level_assignments = len(self.level_poolers)
        
        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"
        assert (
                len(x) == num_level_assignments
        ), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )

        assert len(box_lists) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(box_lists)
        )

        # add batch index to every box_tensor, and flatten all box is  relatively to eatch orign img
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        # low level x: [level, NCHW]
        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)
        # assign each box to a level, [N], N is the number of boxes 
        level_assignments = assign_boxes_to_levels(
            box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )
       
        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros(
            (num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device
        )
        output = _create_zeros(pooler_fmt_boxes, num_channels, output_size, output_size, x[0])
        # aff.eval()
        for level, pooler in enumerate(self.level_poolers):
            # find boxes belong to this level
            inds = nonzero_tuple(level_assignments == level)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            # Use index_put_ instead of advance indexing, to avoid pytorch/issues/49852
            width, height = x[level].shape[2], x[level].shape[3]
            feature_1 = F.interpolate(x[(level + 1) % 4].clone(), size=(width, height), mode='bilinear')
            feature_2 = F.interpolate(x[(level + 2) % 4].clone(), size=(width, height), mode='bilinear')
            feature_3 = F.interpolate(x[(level + 3) % 4].clone(), size=(width, height), mode='bilinear')
            feature_12 = self.aff(x[level], feature_1)
            feature_123 = self.aff(feature_12, feature_2)
            feature_1234 = self.aff(feature_123, feature_3)
            output.index_put_((inds,), pooler(feature_1234, pooler_fmt_boxes_level))
        return output
    
    
    
        roi_feats_list = []
        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            roi_feats_list.append(pooler(x_level, pooler_fmt_boxes))
        
        """
            multiple AFF
        """
      
        # aff = AFF(channels=num_channels).to(device)
        # feature_12 = aff(roi_feats_list[0], roi_feats_list[1])
        # feature_123 = aff(feature_12, roi_feats_list[2])
        # feature_1234 = aff(feature_123, roi_feats_list[3])
        # output = feature_1234
        # print(output.shape)
        # exit(0)
        # return output
    
        # # concat in channel dims [batchsize, NC, h, w]
        # concat_roi_feats = torch.cat(roi_feats_list, dim=1)
        # # [batchszie, N, h, w]
        # spatial_attention_map = self.spatial_attention_conv(concat_roi_feats)
 
        # for i in range(self.canonical_level):
        #     output += (F.sigmoid(spatial_attention_map[:, i, None, :, :]) * roi_feats_list[i])
        # print(output.shape)
        # exit(0)
        # return output

class ROIPooler_aug_nBN(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
            self,
            output_size,
            scales,
            sampling_ratio,
            pooler_type,
            canonical_box_size=224,
            canonical_level=4,
            out_channels=256,
    ):
        """
        Args:
            output_size (int, tuple[int] or list[int]): output size of the pooled region,
                e.g., 14 x 14. If tuple or list is given, the length must be 2.
            scales (list[float]): The scale for each low-level pooling op relative to
                the input image. For a feature map with stride s relative to the input
                image, scale is defined as a 1 / s. The stride must be power of 2.
                When there are multiple scales, they must form a pyramid, i.e. they must be
                a monotically decreasing geometric sequence with a factor of 1/2.
            sampling_ratio (int): The `sampling_ratio` parameter for the ROIAlign op.
            pooler_type (string): Name of the type of pooling operation that should be applied.
                For instance, "ROIPool" or "ROIAlignV2".
            canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
                is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
                pre-training).
            canonical_level (int): The feature map level index from which a canonically-sized box
                should be placed. The default is defined as level 4 (stride=16) in the FPN paper,
                i.e., a box of size 224x224 will be placed on the feature with stride=16.
                The box placement for all boxes will be determined from their sizes w.r.t
                canonical_box_size. For example, a box whose area is 4x that of a canonical box
                should be used to pool features from feature level ``canonical_level+1``.

                Note that the actual input feature maps given to this module may not have
                sufficiently many levels for the input boxes. If the boxes are too large or too
                small for the input feature maps, the closest level will be used.
        """
        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size

        self.aff = AFF_nBN(channels=out_channels)
        
        if pooler_type == "ROIAlign":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False
                )
                for scale in scales
            )
        elif pooler_type == "ROIAlignV2":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
                for scale in scales
            )
        elif pooler_type == "ROIPool":
            self.level_poolers = nn.ModuleList(
                RoIPool(output_size, spatial_scale=scale) for scale in scales
            )
        elif pooler_type == "ROIAlignRotated":
            self.level_poolers = nn.ModuleList(
                ROIAlignRotated(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
                for scale in scales
            )
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))

        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.
        min_level = -(math.log2(scales[0]))
        max_level = -(math.log2(scales[-1]))
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        ), "Featuremap stride is not power of 2!"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert (
            len(scales) == self.max_level - self.min_level + 1
        ), "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        assert 0 <= self.min_level and self.min_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size
        # input_size: concat_channel
        self.spatial_attention_conv = nn.Sequential(nn.Conv2d(out_channels * canonical_level, out_channels, 1),
                                                    nn.ReLU(),
                                                    nn.Conv2d(out_channels, canonical_level, 3, padding=1))

    def forward(self, x: List[torch.Tensor], box_lists):

        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Boxes] | list[RotatedBoxes]):  [514, 4]
                A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.

        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        num_level_assignments = len(self.level_poolers)
        
        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"
        assert (
                len(x) == num_level_assignments
        ), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )

        assert len(box_lists) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(box_lists)
        )

        # add batch index to every box_tensor, and flatten all box is  relatively to eatch orign img
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        # low level x: [level, NCHW]
        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)
        # assign each box to a level, [N], N is the number of boxes 
        level_assignments = assign_boxes_to_levels(
            box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )
       
        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros(
            (num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device
        )
        output = _create_zeros(pooler_fmt_boxes, num_channels, output_size, output_size, x[0])
        # aff.eval()
        for level, pooler in enumerate(self.level_poolers):
            # find boxes belong to this level
            inds = nonzero_tuple(level_assignments == level)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            # Use index_put_ instead of advance indexing, to avoid pytorch/issues/49852
            width, height = x[level].shape[2], x[level].shape[3]
            feature_1 = F.interpolate(x[(level + 1) % 4].clone(), size=(width, height), mode='bilinear')
            feature_2 = F.interpolate(x[(level + 2) % 4].clone(), size=(width, height), mode='bilinear')
            feature_3 = F.interpolate(x[(level + 3) % 4].clone(), size=(width, height), mode='bilinear')
            feature_12 = self.aff(x[level], feature_1)
            feature_123 = self.aff(feature_12, feature_2)
            feature_1234 = self.aff(feature_123, feature_3)
            output.index_put_((inds,), pooler(feature_1234, pooler_fmt_boxes_level))
        return output
    
    
class ROIPooler_TwoFusion(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
            self,
            output_size,
            scales,
            sampling_ratio,
            pooler_type,
            canonical_box_size=224,
            canonical_level=4,
            out_channels=256,
    ):
        """
        Args:
            output_size (int, tuple[int] or list[int]): output size of the pooled region,
                e.g., 14 x 14. If tuple or list is given, the length must be 2.
            scales (list[float]): The scale for each low-level pooling op relative to
                the input image. For a feature map with stride s relative to the input
                image, scale is defined as a 1 / s. The stride must be power of 2.
                When there are multiple scales, they must form a pyramid, i.e. they must be
                a monotically decreasing geometric sequence with a factor of 1/2.
            sampling_ratio (int): The `sampling_ratio` parameter for the ROIAlign op.
            pooler_type (string): Name of the type of pooling operation that should be applied.
                For instance, "ROIPool" or "ROIAlignV2".
            canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
                is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
                pre-training).
            canonical_level (int): The feature map level index from which a canonically-sized box
                should be placed. The default is defined as level 4 (stride=16) in the FPN paper,
                i.e., a box of size 224x224 will be placed on the feature with stride=16.
                The box placement for all boxes will be determined from their sizes w.r.t
                canonical_box_size. For example, a box whose area is 4x that of a canonical box
                should be used to pool features from feature level ``canonical_level+1``.

                Note that the actual input feature maps given to this module may not have
                sufficiently many levels for the input boxes. If the boxes are too large or too
                small for the input feature maps, the closest level will be used.
        """
        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size

        self.aff = AFF(channels=out_channels)
        
        if pooler_type == "ROIAlign":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False
                )
                for scale in scales
            )
        elif pooler_type == "ROIAlignV2":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
                for scale in scales
            )
        elif pooler_type == "ROIPool":
            self.level_poolers = nn.ModuleList(
                RoIPool(output_size, spatial_scale=scale) for scale in scales
            )
        elif pooler_type == "ROIAlignRotated":
            self.level_poolers = nn.ModuleList(
                ROIAlignRotated(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
                for scale in scales
            )
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))

        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.
        min_level = -(math.log2(scales[0]))
        max_level = -(math.log2(scales[-1]))
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        ), "Featuremap stride is not power of 2!"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert (
            len(scales) == self.max_level - self.min_level + 1
        ), "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        assert 0 <= self.min_level and self.min_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size
        # input_size: concat_channel
        self.spatial_attention_conv = nn.Sequential(nn.Conv2d(out_channels * canonical_level, out_channels, 1),
                                                    nn.ReLU(),
                                                    nn.Conv2d(out_channels, canonical_level, 3, padding=1))

    def forward(self, x: List[torch.Tensor], box_lists):

        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Boxes] | list[RotatedBoxes]):  [514, 4]
                A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.

        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        num_level_assignments = len(self.level_poolers)
        
        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"
        assert (
                len(x) == num_level_assignments
        ), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )

        assert len(box_lists) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(box_lists)
        )

        # add batch index to every box_tensor, and flatten all box is  relatively to eatch orign img
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        # low level x: [level, NCHW]
        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)
        # assign each box to a level, [N], N is the number of boxes 
        level_assignments = assign_boxes_to_levels(
            box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )
       
        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros(
            (num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device
        )
        output = _create_zeros(pooler_fmt_boxes, num_channels, output_size, output_size, x[0])
        # aff.eval()
        for level, pooler in enumerate(self.level_poolers):
    
            # find boxes belong to this level
            inds = nonzero_tuple(level_assignments == level)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            # Use index_put_ instead of advance indexing, to avoid pytorch/issues/49852
            width, height = x[level].shape[2], x[level].shape[3]
            feature_1 = F.interpolate(x[(level + 1) % 4].clone(), size=(width, height), mode='bilinear')
            feature_12 = self.aff(x[level], feature_1)
            output.index_put_((inds,), pooler(feature_12, pooler_fmt_boxes_level))
        return output