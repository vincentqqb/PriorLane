import copy
import math
import warnings

import torch
import torch.nn as nn

from mmcv import ConfigDict
from mmcv.cnn import (Linear, build_activation_layer, build_norm_layer,
                      constant_init, xavier_init)
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)
from mmcv.runner.base_module import BaseModule
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import (ATTENTION, POSITIONAL_ENCODING,
                                      TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)
# from visualizer import get_local

class CustomAttention(nn.Module):
    def __init__(self, embd_dim, num_head):
        super(CustomAttention, self).__init__()
        self.embed_dims = embd_dim
        self.num_head = num_head
        self.self_attn = nn.MultiheadAttention(self.embed_dims, self.num_head, 0.1)
    # @get_local('attn_map')
    def forward(self, query, key, value, identity=None):
        if identity is None:
            identity = query
        query, attn_map = self.self_attn(query, key, value)

        return identity + torch.dropout(query, 0.1, train=True)

def build_positional_encoding(cfg, default_args=None):
    """Builder for Position Encoding."""
    return build_from_cfg(cfg, POSITIONAL_ENCODING, default_args)


def build_attention(cfg, default_args=None):
    """Builder for attention."""
    return build_from_cfg(cfg, ATTENTION, default_args)


def build_transformer_layer(cfg, default_args=None):
    """Builder for transformer layer."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER, default_args)


def build_transformer_layer_sequence(cfg, default_args=None):
    """Builder for transformer encoder and transformer decoder."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER_SEQUENCE, default_args)

class FFN(BaseModule):
    """Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        dropout (float, optional): Probability of an element to be
            zeroed. Default 0..
        add_residual (bool, optional): Whether to add the
            residual connection. Default: `True`.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 dropout=0.,
                 add_residual=True,
                 init_cfg=None):
        super(FFN, self).__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.dropout = dropout
        self.activate = build_activation_layer(act_cfg)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(dropout)))
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.add_residual = add_residual

    def forward(self, x, residual=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_residual:
            return self.dropout(out)
        if residual is None:
            residual = x
        return residual + self.dropout(out)

@TRANSFORMER_LAYER.register_module()
class FusionTransformerEncoderLayer(BaseModule):

    def __init__(self,
                 attn_cfgs=None,
                 feedforward_channels=None,
                 ffn_dropout=0.,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 init_cfg=None):

        super(FusionTransformerEncoderLayer, self).__init__(init_cfg)
        assert set(operation_order) & set(
            ['prompt_self_attn', 'self_attn', 'norm', 'ffn', 'cross_attn']) == \
            set(operation_order), f'The operation_order of' \
            f' {self.__class__.__name__} should ' \
            f'contains all five operation type ' \
            f"{['prompt_self_attn', 'self_attn', 'norm', 'ffn', 'cross_attn']}"
        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn') + operation_order.count('prompt_self_attn')
        if isinstance(attn_cfgs, ConfigDict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'
        self.init_cfg = init_cfg
        self.num_attn = num_attn
        self.feedforward_channels = feedforward_channels
        self.ffn_dropout = ffn_dropout
        self.operation_order = operation_order
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.ffn_num_fcs = ffn_num_fcs
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = nn.ModuleList()

        index = 0
        for operation in operation_order:
            if operation == "self_attn" or operation == "prompt_self_attn":
                attention = CustomAttention(attn_cfgs[index]['embed_dims'], attn_cfgs[index]['num_heads'])
                self.attentions.append(attention)
                index += 1
        self.embed_dims = self.attentions[0].embed_dims
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count('ffn')
        for _ in range(num_ffns):
            self.ffns.append(
                FFN(self.embed_dims, feedforward_channels, ffn_num_fcs,
                    act_cfg, ffn_dropout))

        self.norms = nn.ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(self,
                query,
                key,
                value,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                img_end_index=None,
                **kwargs):

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        inp_residual = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](query, temp_key, temp_value, inp_residual if self.pre_norm else None)
                attn_index += 1
                inp_residual = query
                
            elif layer == 'prompt_self_attn':
                object_queies_start_idx = img_end_index
                image_tokens = query[:object_queies_start_idx, ...]
                object_queies = query[object_queies_start_idx:, ...]
                inp_residual_propmt = inp_residual[object_queies_start_idx:, ...]
                # query_pos_propmt = query_pos[object_queies_start_idx:, ...]
                query_pos_propmt = None
                # query_key_padding_mask_propmt = query_key_padding_mask[:, object_queies_start_idx:]
                query_key_padding_mask_propmt = None

                temp_key = temp_value = object_queies  # only take object queries as input
                # temp_img_key = temp_img_value = image_tokens  # only take object queries as input

                if 'prompt_self_attn_layer' not in kwargs: #True
                    object_queies = self.attentions[attn_index](
                        object_queies,
                        temp_key,
                        temp_value,
                        inp_residual_propmt if self.pre_norm else None)

                    attn_index += 1
                else:
                    print("ERROR ERROR")
                query = torch.cat([image_tokens, object_queies], dim=0)
                inp_residual = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    inp_residual if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                inp_residual = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, inp_residual if self.pre_norm else None)
                ffn_index += 1

        return query


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class TransformerLayerSequenceCustom(BaseModule):
    def __init__(self, transformerlayers=None, num_layers=None, init_cfg=None):
        super(TransformerLayerSequenceCustom, self).__init__(init_cfg)
        if isinstance(transformerlayers, ConfigDict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        else:
            assert isinstance(transformerlayers, list) and \
                   len(transformerlayers) == num_layers
        self.num_layers = num_layers
        operation_order = transformerlayers[0]['operation_order']
        if 'shared' in transformerlayers[0]['attn_cfgs'][0]:
            prompt_encoder_share_cfg = copy.deepcopy(transformerlayers[0]['attn_cfgs'][0])
            assert prompt_encoder_share_cfg.pop('shared') is True
            self.prompt_encoder_share = build_attention(prompt_encoder_share_cfg)
        self.pre_norm = operation_order[0] == 'norm'
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(build_transformer_layer(transformerlayers[i]))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].operation_order[0] == 'norm'

    def forward(self,
                query,
                key,
                value,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                img_end_index=None,
                **kwargs):
        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                img_end_index=img_end_index,
                **kwargs)
        return query
