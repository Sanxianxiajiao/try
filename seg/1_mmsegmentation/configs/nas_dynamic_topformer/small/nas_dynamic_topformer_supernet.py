norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='DynamicEncoderDecoder',
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    num_arch_training=4,
    sandwich_rule=True,
    distiller=dict(name='kd'), # pixel-wise kd loss
    # distiller=dict(name='cwd',tau=1.0, loss_weight=3.0), # 文中tau=4.0, loss_weight=3.0
    # distiller=None, # ce loss
    sync_bn=True, # True
    backbone=dict(
        type='NASDynamicTopFormer',
        supernet=dict(
            stem=dict(c=[16], act_func='relu6', s=2),
            embed_out_indice=[1, 2, 3, 4],
            c2t_stride=2,
            drop=0., # for mlp
            drop_path_rate=0.1, # for trans
            pretrained=None,
            mb1=dict(
                # c=16, d=1, k=3, t=1
                c=[16], d=[1, 2], k=[3, 5], t=[1], s=1,
                act_func='relu6', se=False
            ),
            mb2=dict(
                # c=24, d=2, k=3, t=4,3
                c=[16, 24, 32], d=[1, 2, 3], k=[3, 5], t=[3, 4, 5], s=2,
                act_func='relu6', se=False
            ),
            mb3=dict(
                # c=48, d=2, k=5, t=3
                c=[40, 48, 56], d=[1, 2, 3], k=[3, 5], t=[2, 3, 4, 5], s=2,
                act_func='relu6', se=False
            ),
            mb4=dict(
                # c=96, d=2, k=3, t=3
                c=[88, 96, 104, 112], d=[1, 2, 3], k=[3, 5], t=[2, 3, 4, 5], s=2,
                act_func='relu6', se=False
            ),
            mb5=dict(
                # c=128, d=3, k=5, t=6
                c=[120, 128, 136, 144], d=[2, 3, 4], k=[3, 5], t=[4, 5, 6, 7], s=2,
                act_func='relu6', se=False
            ),
            trans1=dict(
                num_heads=[4, 6, 8, 10], # head 6
                key_dim=[12, 14, 16, 18, 20], # 16
                attn_ratio=[1.6, 1.8, 2.0, 2.2, 2.4], # 2.0
                mlp_ratio=[1.6, 1.8, 2.0, 2.2, 2.4], # 2.0
                d=[1, 2], # 1
                act_func='relu6'
            ),
            trans2=dict(
                num_heads=[4, 6, 8, 10], # head 6
                key_dim=[12, 14, 16, 18, 20], # 16
                attn_ratio=[1.6, 1.8, 2.0, 2.2, 2.4], # 2.0
                mlp_ratio=[1.6, 1.8, 2.0, 2.2, 2.4], # 2.0
                d=[1, 2], # 1
                act_func='relu6'
            ),
            trans3=dict(
                num_heads=[4, 6, 8, 10], # head 6
                key_dim=[12, 14, 16, 18, 20], # 16
                attn_ratio=[1.6, 1.8, 2.0, 2.2, 2.4], # 2.0
                mlp_ratio=[1.6, 1.8, 2.0, 2.2, 2.4], # 2.0
                d=[1, 2], # 1
                act_func='relu6'
            ),
            trans4=dict(
                num_heads=[4, 6, 8, 10], # head 6
                key_dim=[12, 14, 16, 18, 20], # 16
                attn_ratio=[1.6, 1.8, 2.0, 2.2, 2.4], # 2.0
                mlp_ratio=[1.6, 1.8, 2.0, 2.2, 2.4], # 2.0
                d=[1, 2], # 1
                act_func='relu6'
            ),
            SIM=dict(
                act_func='relu6',
                # channels=[32, 64, 128, 160],
                decode_out_indices=[1, 2, 3],
                out_channels=[None, 192, 192, 192],
                injection=True,
                norm_cfg=norm_cfg # SyncBN
            ),
        ),
        fix_backbone=False,
        fix_trans=False
    ),
    decode_head=dict(
        type='SimpleHead',
        in_channels=[192, 192, 192],
        in_index=[0, 1, 2],
        channels=192,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg, # SyncBN
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
)