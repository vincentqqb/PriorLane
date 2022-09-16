import math
from operator import or_

import torch
import torch.nn as nn
import torchvision.ops
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer, xavier_init
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from .transformer_util import TransformerLayerSequenceCustom
from mmcv.runner.base_module import BaseModule
from torch.nn.init import normal_
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from ..builder import NECKS
from ..orn import ORConv2d, RotationInvariantPooling
# from mmdet.models.utils.builder import TRANSFORMER

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)
        self.relu = nn.ReLU(inplace=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        # self.modulator_conv = nn.Conv2d(in_channels,
        #                                 1 * kernel_size * kernel_size,
        #                                 kernel_size=kernel_size,
        #                                 stride=stride,
        #                                 padding=self.padding,
        #                                 bias=True)

        # nn.init.constant_(self.modulator_conv.weight, 0.)
        # nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        # modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        # self.regular_conv.weight = self.regular_conv.weight.half() if x.dtype == torch.float16 else \
        #     self.regular_conv.weight
        x = torchvision.ops.deform_conv2d(input=x.float(),
                                          offset=offset.float(),
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=(self.padding, self.padding),
                                          # mask=modulator,
                                          stride=self.stride,
                                          )
        # return x
        return self.relu(x)

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            # nn.Conv2d(in_channels=512, out_channels=8, kernel_size=7),
            nn.Conv2d(in_channels=512, out_channels=8, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),  # in_features, out_features, bias = True
            nn.ReLU(True),
            nn.Linear(32, 3 * 2) # Affine
            # nn.Linear(32, 3 * 1) # Educ
        )
        self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0], dtype=torch.float))
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x, query):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)  
        theta = self.fc_loc(xs)  
        # yaw, d_x, d_y = torch.split(theta, 1, dim=1)
        # yaw_ang = math.pi * 2 * yaw
        # cos_y, _sin_y, sin_y, cos_y = torch.cos(yaw_ang), -torch.sin(yaw_ang), torch.sin(yaw_ang), torch.cos(yaw_ang)
        # theta = torch.cat((cos_y, _sin_y, d_x, sin_y, cos_y, d_y), 1)
        # theta = theta.view(-1,2,3)
        theta = theta.view(-1, 2, 3)  # 
        grid = F.affine_grid(theta=theta, size=x.size())  
        # x = F.grid_sample(x, grid)
        x = F.grid_sample(query, grid)
        return x



@TRANSFORMER_LAYER_SEQUENCE.register_module()
class FusionTransformerEncoder(TransformerLayerSequenceCustom):
    def __init__(self,
                 *args,
                 return_intermediate=None,
                 **kwargs):

        super(FusionTransformerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                *args,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            kwargs: key, value, query_pos, query_key_padding_mask, spatial_shapes, level_start_index

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        if hasattr(self, 'prompt_encoder_share'):
            kwargs['prompt_self_attn_layer'] = self.prompt_encoder_share

        output = query
        intermediate = []
        for lid, layer in enumerate(self.layers):
            # reference point has already been broadcasted across feature levels
            output = layer(
                output,
                *args,
                **kwargs)

        #     if self.return_intermediate[lid]:
        #         intermediate.append(output)

        # if self.return_intermediate is not None:
        #     return torch.stack(intermediate)

        return output


@NECKS.register_module()
class FusionTransformer(BaseModule):

    def __init__(self,
                 encoder=None,
                 map_embd=None,
                 patch_size=10,
                 init_cfg=None,
                 **kwargs,
                 ):
        super(FusionTransformer, self).__init__(init_cfg=init_cfg)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.init_layers(map_embd,patch_size)
        self.apply(self.init_weights)

    def init_layers(self, map_embd, patch_size=10):
        self.map_embd = None
        self.stn = None
        self.or_conv = None
        self.prompt_embd = None
        if "imp" in map_embd:
            self.prompt_embd = nn.Embedding(100, 512)
        self.resize = torchvision.transforms.Resize([200, 200])
        if "patch" in map_embd:
            self.map_embd = nn.Conv2d(3, 512, kernel_size=patch_size, stride=patch_size)
        elif "deform" in map_embd:
            self.map_embd = DeformableConv2d(3, 512, kernel_size=10, stride=10)
        else:
            print("NO map embd backbone")
        if "stn" in map_embd:
            self.stn = STN()

        if "orn" in map_embd:
            self.or_conv = ORConv2d(512, 512, kernel_size=3, padding=1, arf_config=(1, 8))
            self.or_pool = RotationInvariantPooling(256, 8)
            normal_init(self.or_conv, std=0.01)
        # self.norm = nn.LayerNorm(512)

    def init_weights(self, m):

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self,bb_feats,**kwargs):

        map_prompt = kwargs['map_prompt']
        if isinstance(map_prompt, list):
            map_prompt = map_prompt[0]
        feat_flatten = bb_feats[-1]
        bs, _, H, W = feat_flatten.shape
        feat_flatten = feat_flatten.flatten(2).permute(2,0,1)
        map_prompt = map_prompt.permute(0,3,1,2).contiguous().float()
        map_prompt = self.resize(map_prompt)
        if self.prompt_embd is None:
            query_embd = self.map_embd(map_prompt) 
            if self.or_conv is not None:
                or_feat = self.or_conv(query_embd)
                or_pool_feat = self.or_pool(or_feat)        
            if self.stn is not None:
                query = self.stn(or_pool_feat, query_embd)
            else:
                query = query_embd
            query = query.flatten(2).permute(2,0,1).contiguous() ######
        else:
            query = self.prompt_embd.weight.unsqueeze(1).repeat(1, bs, 1)

        query_feat = torch.cat([feat_flatten, query], dim=0) 
        inter_states = self.encoder(
            query=query_feat,
            key=None,
            value=None,
            img_end_index=H*W)
        inter_states = inter_states[:H*W,...]
        N, B, C = inter_states.shape
        bb_feats[-1] = inter_states.permute(1,2,0).reshape(B, C, H, W)
        return bb_feats
