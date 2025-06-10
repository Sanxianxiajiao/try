norm_cfg = dict(type='SyncBN', requires_grad=True)
model_cfgs = dict(
    stem_channel=16,
    cfg=[[5, 1, 16, 1], [3, 4, 24, 2], [3, 2, 72, 2], [3, 2, 72, 1],
         [3, 2, 128, 2], [3, 2, 128, 1], [3, 5, 176, 2], [3, 5, 176, 1],
         [3, 5, 176, 1], [3, 5, 176, 1]],
    channels=[24, 72, 128, 176],
    out_channels=[None, 256, 256, 256],
    embed_out_indice=[1, 3, 5, 9],
    decode_out_indices=[1, 2, 3],
    key_dim=[18, 16, 16, 16],
    num_heads=[10, 8, 12, 4],
    attn_ratios=[1.6, 2.2, 1.8, 2.0],
    mlp_ratios=[2.2, 2.2, 1.6, 2.0],
    c2t_stride=1)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='Topformer',
        cfgs=[[5, 1, 16, 1], [3, 4, 24, 2], [3, 2, 72, 2], [3, 2, 72, 1],
              [3, 2, 128, 2], [3, 2, 128, 1], [3, 5, 176, 2], [3, 5, 176, 1],
              [3, 5, 176, 1], [3, 5, 176, 1]],
        stem_channel=16,
        channels=[24, 72, 128, 176],
        out_channels=[None, 256, 256, 256],
        embed_out_indice=[1, 3, 5, 9],
        decode_out_indices=[1, 2, 3],
        depths=[2, 1, 2, 2],
        key_dim=[18, 16, 16, 16],
        num_heads=[10, 8, 12, 4],
        attn_ratios=[1.6, 2.2, 1.8, 2.0],
        mlp_ratios=[2.2, 2.2, 1.6, 2.0],
        c2t_stride=1,
        drop_path_rate=0.1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint=None)),
    decode_head=dict(
        type='SimpleHead',
        in_channels=[256, 256, 256],
        in_index=[0, 1, 2],
        channels=256,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
work_dir = 'debug'
gpu_ids = range(0, 1)
