# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

model_cfgs = dict(
    stem_channel=16,
    cfg=[
        # k,  t,  c, s
        [3,   1,  16, 1], # 1/2          
        [3,   3,  24, 2], # 1/4                  
        [5,   2,  32, 2], # 1/8  
        # [5,   2,  32, 1], #       
        [3,   2,  56, 2], # 1/16   
        [3,   2,  56, 1], #    
        [3,   2,  56, 1], #    
        [5,   6,  104, 2], # 1/32  
        [5,   6,  104, 1], #                       
        [5,   6,  104, 1], #                       
    ],
    channels=[24, 32, 56, 104],
    out_channels=[None, 128, 128, 128],
    embed_out_indice=[1, 2, 5, 8],
    decode_out_indices=[1, 2, 3],
    key_dim=[18, 16, 18, 14], 
    num_heads=[6, 10, 4, 10],
    attn_ratios=[1.8, 2.0, 1.6, 1.8],
    mlp_ratios=[2.4, 2.0, 2.2, 1.8],
    c2t_stride=1,
)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='Topformer',
        cfgs=model_cfgs['cfg'], 
        stem_channel=model_cfgs['stem_channel'],
        channels=model_cfgs['channels'],
        out_channels=model_cfgs['out_channels'], 
        embed_out_indice=model_cfgs['embed_out_indice'],
        decode_out_indices=model_cfgs['decode_out_indices'],
        depths=[1, 2, 2, 2],
        key_dim=model_cfgs['key_dim'],
        num_heads=model_cfgs['num_heads'],
        attn_ratios=model_cfgs['attn_ratios'],
        mlp_ratios=model_cfgs['mlp_ratios'],
        c2t_stride=model_cfgs['c2t_stride'],
        drop_path_rate=0.1,
        norm_cfg=norm_cfg,
        init_cfg=dict(
            type='Pretrained', checkpoint='modelzoos/classification/topformer-T-224-66.2.pth')
    ),
    decode_head=dict(
        type='SimpleHead',
        in_channels=[128, 128, 128],
        in_index=[0, 1, 2],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        is_dw=True,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
