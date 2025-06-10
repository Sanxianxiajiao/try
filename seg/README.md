`rsync -avrP  PCL_hj_01:/userhome/lz/code/seg/mmsegmentation ./Downloads/exp/seg/`

reduce.py->last_pe.py

## base

```
2022-10-31 20:06:45,399 - mmseg - INFO - Best Architecture in evolution search: mIoU: 35.27 flops: 1.79 cfg:
 {'width': [16, 16, 24, 72, 128, 176],
  'depth': [1, 1, 2, 2, 4], 
  'kernel_size': [5, 3, 3, 3, 3], 
  'expand_ratio': [1, 4, 2, 2, 5], 
  'num_heads': [10, 8, 12, 4], 
  'key_dim': [18, 16, 16, 16], 
  'attn_ratio': [1.6, 2.2, 1.8, 2.0], 
  'mlp_ratio': [2.2, 2.2, 1.6, 2.0], 
  'transformer_depth': [2, 1, 2, 2]}
```



## small

```
2022-11-04 18:54:44,528 - mmseg - INFO - Best Architecture in evolution search: mIoU: 32.83 flops: 1.18 cfg: {
    'width': [16, 16, 16, 48, 96, 136],
    'depth': [2, 2, 1, 3, 4],
    'kernel_size': [5, 3, 5, 3, 3],
    'expand_ratio': [1, 3, 3, 2, 6],
    'num_heads': [6, 4, 4, 8],
    'key_dim': [16, 14, 16, 14],
    'attn_ratio': [1.6, 1.8, 1.8, 1.8],
    'mlp_ratio': [2.2, 1.8, 2.0, 1.8],
    'transformer_depth': [2, 1, 2, 2]}
```

## tiny

```
2022-11-04 19:48:56,430 - mmseg - INFO - Best Architecture in evolution search: mIoU: 29.47 flops: 0.58 cfg: {
    'width': [16, 16, 24, 32, 56, 104],
    'depth': [1, 1, 1, 3, 3],
    'kernel_size': [3, 3, 5, 3, 5],
    'expand_ratio': [1, 3, 2, 2, 6],
    'num_heads': [6, 10, 4, 10],
    'key_dim': [18, 16, 18, 14],
    'attn_ratio': [1.8, 2.0, 1.6, 1.8],
    'mlp_ratio': [2.4, 2.0, 2.2, 1.8],
    'transformer_depth': [1, 2, 2, 2]}
