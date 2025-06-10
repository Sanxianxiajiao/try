# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

model_cfgs = dict(
        stem_channel=16,
        cfg=[
            # k,  t,  c, s
            [5,   1,  16, 1], # 1/2        0.464K  17.461M
            [3,   4,  24, 2], # 1/4 1      3.44K   64.878M
            # [3,   3,  32, 1], #            4.44K   41.772M
            [3,   2,  72, 2], # 1/8 3      6.776K  29.146M
            [3,   2,  72, 1], #            13.16K  30.952M
            [3,   2,  128, 2], # 1/16 5     16.12K  18.369M
            [3,   2,  128, 1], #            41.68K  24.508M
            [3,   5,  176, 2], # 1/32 7     0.129M  36.385M
            [3,   5,  176, 1], #            0.335M  49.298M
            [3,   5,  176, 1], #            0.335M  49.298M
            [3,   5,  176, 1], #            0.335M  49.298M
        ],
        channels=[24, 72, 128, 176],
        out_channels=[None, 256, 256, 256],
        embed_out_indice=[1, 3, 5, 8], # TODO bug not fixed
        decode_out_indices=[1, 2, 3],
        key_dim=[18, 16, 16, 16],
        num_heads=[10, 8, 12, 4],
        attn_ratios=[1.6, 2.2, 1.8, 2.0],
        mlp_ratios=[2.2, 2.2, 1.6, 2.0],
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
        depths=[2, 1, 2, 2],
        key_dim=model_cfgs['key_dim'],
        num_heads=model_cfgs['num_heads'],
        attn_ratios=model_cfgs['attn_ratios'],
        mlp_ratios=model_cfgs['mlp_ratios'],
        c2t_stride=model_cfgs['c2t_stride'],
        drop_path_rate=0.1,
        norm_cfg=norm_cfg,
        init_cfg=dict(
            type='Pretrained', checkpoint='/media/disk1/leizhang/project/seg/TopFormer/base_pretrained_bug.pth.tar') # TODO
    ),
    decode_head=dict(
        type='SimpleHead',
        in_channels=[256, 256, 256],
        in_index=[0, 1, 2],
        channels=256,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))