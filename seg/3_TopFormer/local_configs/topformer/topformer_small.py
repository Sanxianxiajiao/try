# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

cfgs_md2_middle = dict(
    stem_channel=16,
    cfg=[
        # k,  t,  c, s
        [5,   1,  16, 1], # 1/2          
        [5,   1,  16, 1],           
        [3,   3,  16, 2], # 1/4                  
        [3,   3,  16, 1],                   
        [5,   3,  48, 2], # 1/8  
        # [3,   2,  48, 1], #       
        [3,   2,  96, 2], # 1/16   
        [3,   2,  96, 1], #    
        [3,   2,  96, 1], #    
        [3,   6,  136, 2], # 1/32  
        [3,   6,  136, 1], #  
        [3,   6,  136, 1], #
        [3,   6,  136, 1], #                      
    ],
    channels=[16, 48, 96, 136],
    out_channels=[None, 192, 192, 192],
    embed_out_indice=[3, 4, 7, 11],
    decode_out_indices=[1, 2, 3],
    key_dim=[16, 14, 16, 14], 
    num_heads=[6, 4, 4, 8],
    attn_ratios=[1.6, 1.8, 1.8, 1.8],
    mlp_ratios=[2.2, 1.8, 2.0, 1.8],
    c2t_stride=1,
)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='Topformer',
        cfgs=cfgs_md2_middle['cfg'], 
        stem_channel=cfgs_md2_middle['stem_channel'],
        channels=cfgs_md2_middle['channels'],
        out_channels=cfgs_md2_middle['out_channels'], 
        embed_out_indice=cfgs_md2_middle['embed_out_indice'],
        decode_out_indices=cfgs_md2_middle['decode_out_indices'],
        depths=[2, 1, 2, 2],
        key_dim=cfgs_md2_middle['key_dim'],
        num_heads=cfgs_md2_middle['num_heads'],
        attn_ratios=cfgs_md2_middle['attn_ratios'],
        mlp_ratios=cfgs_md2_middle['mlp_ratios'],
        c2t_stride=cfgs_md2_middle['c2t_stride'],
        drop_path_rate=0.1,
        norm_cfg=norm_cfg,
        init_cfg=dict(
            type='Pretrained', checkpoint='modelzoos/classification/topformer-S-224-72.3.pth')
    ),
    decode_head=dict(
        type='SimpleHead',
        in_channels=[192, 192, 192],
        in_index=[0, 1, 2],
        channels=192,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))