_base_ = [
    '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_160k.py',
    './nas_dynamic_topformer_supernet.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(backbone=dict(fix_backbone=False, fix_trans=False, supernet=dict(SIM=dict(norm_cfg=norm_cfg))),
             decode_head=dict(norm_cfg=norm_cfg),
             sync_bn=False)
constraint_flops=0.5
unfixed_ckpt='ckpt/nas_dynamic_topformer/tiny/ade20k/unfixed/448_latest.pth'
fix_backbone_ckpt='ckpt/nas_dynamic_topformer/tiny/ade20k/fix_backbone/448_latest.pth'
fix_trans_ckpt='ckpt/nas_dynamic_topformer/tiny/ade20k/fix_trans/448_latest.pth'
post_bn_calibration_batch_num=64
need_finetune=False

runner = dict(type='IterBasedRunner', max_iters=1500) # 采样子网后finetune
optimizer = dict(_delete_=True, type='AdamW', lr=0.00012, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (448, 448)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 448), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 448),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data_root = 'data/ade/ADEChallengeData2016'
data = dict(
    train=dict(pipeline=train_pipeline, data_root=data_root),
    val=dict(pipeline=test_pipeline, data_root=data_root),
    test=dict(pipeline=test_pipeline, data_root=data_root),
    samples_per_gpu=4)

log_config = dict(
    _delete_=True,
    interval=500, 
    hooks=[dict(type='TextLoggerHook', by_epoch=False),]
) 
find_unused_parameters = True
# evaluation = dict(interval=50, metric='mIoU', pre_eval=True) # for debug
evaluation = dict(interval=16000, metric='mIoU', pre_eval=True) # 32000
work_dir = 'output/nas_dynamic_topformer/tiny/unfixed'

