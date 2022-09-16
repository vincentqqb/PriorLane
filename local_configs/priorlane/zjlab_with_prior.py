_base_ = [
    '../_base_/datasets/zjlab_with_prior.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_15k_adamw_zjlab_with_prior.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(
        type='mit_b5',
        style='pytorch'),
    neck=dict(
        type='FusionTransformer',
        map_embd=["patch","orn","stn"],
        encoder=dict(
            type='FusionTransformerEncoder',
            num_layers=1,
            return_intermediate=(False, False),
            transformerlayers=dict(
                type='FusionTransformerEncoderLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention', 
                        embed_dims=512,
                        num_heads=8,
                        dropout=0.1,
                        # shared=True,
                    ),
                    dict(
                        type='MultiheadAttention', 
                        embed_dims=512,
                        num_heads=8,
                        dropout=0.1,
                        # shared=True,
                    ),
                    dict(
                        type='MultiheadAttention', 
                        embed_dims=512,
                        num_heads=8,
                        dropout=0.1,
                        # shared=True,
                    ),
                    dict(
                        type='MultiheadAttention', 
                        embed_dims=512,
                        num_heads=8,
                        dropout=0.1,
                        # shared=True,
                    ),
                    dict(
                        type='MultiheadAttention', 
                        embed_dims=512,
                        num_heads=8,
                        dropout=0.1,
                        # shared=True,
                    ),
                    dict(
                        type='MultiheadAttention', 
                        embed_dims=512,
                        num_heads=8,
                        dropout=0.1,
                        # shared=True,
                    ),
                    dict(
                        type='MultiheadAttention', 
                        embed_dims=512,
                        num_heads=8,
                        dropout=0.1,
                        # shared=True,
                    ),
                    dict(
                        type='MultiheadAttention', 
                        embed_dims=512,
                        num_heads=8,
                        dropout=0.1,
                        # shared=True,
                    ),
                ],
                feedforward_channels=1024,
                ffn_dropout=0.1,
                operation_order=('norm', 'prompt_self_attn', 'norm', 'ffn',
                        'norm', 'prompt_self_attn', 'norm', 'ffn',
                        'norm', 'prompt_self_attn', 'norm', 'ffn',
                        'norm', 'prompt_self_attn', 'norm', 'ffn',
                        'norm', 'self_attn', 'norm', 'ffn',
                        'norm', 'self_attn', 'norm', 'ffn',
                        'norm', 'self_attn', 'norm', 'ffn', 
                        'norm', 'self_attn', 'norm', 'ffn', 'norm')
            )
        ),
    ),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,class_weight=[0.1,1,1,1])),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)


