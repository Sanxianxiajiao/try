# PC-DARTS

[PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search](https://openreview.net/forum?id=BJlS634tPr)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/yuhuixu1993/PC-DARTS">Official Repo</a>

<a href=" ">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Differentiable architecture search (DARTS) provided a fast solution in finding effective network architectures, but suffered from large memory and computing overheads in jointly training a super-network and searching for an optimal architecture. In this paper, we present a novel approach, namely, Partially-Connected DARTS, by sampling a small part of super-network to reduce the redundancy in exploring the network space, thereby performing a more efficient search without comprising the performance. In particular, we perform operation search in a subset of channels while bypassing the held out part in a shortcut. This strategy may suffer from an undesired inconsistency on selecting the edges of super-net caused by sampling different channels. We alleviate it using edge normalization, which adds a new set of edge-level parameters to reduce uncertainty in search. Thanks to the reduced memory cost, PC-DARTS can be trained with a larger batch size and, consequently, enjoys both faster speed and higher training stability. Experimental results demonstrate the effectiveness of the proposed method. Specifically, we achieve an error rate of 2.57% on CIFAR10 with merely 0.1 GPU-days for architecture search, and a state-of-the-art top-1 error rate of 24.2% on ImageNet (under the mobile setting) using 3.8 GPU-days for search. 

<!-- [IMAGE] -->

<div align=center>
<img src="" width="70%"/>
</div>



## Citation

```bibtex
@inproceedings{
xu2020pcdarts,
title={{\{}PC{\}}-{\{}DARTS{\}}: Partial Channel Connections for Memory-Efficient Architecture Search},
author={Yuhui Xu and Lingxi Xie and Xiaopeng Zhang and Xin Chen and Guo-Jun Qi and Qi Tian and Hongkai Xiong},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=BJlS634tPr}
}
```

## Results and models

### CIFAR-10

| Search space  | Random seed | Mem(GB) | Params(M) | Flops(G) | Genotype                                                     | Top-1 (%) | Top-5 (%) | Search-Cost(s) | Config                | Download                         |
| ------------- | ----------- | ------- | --------- | -------- | ------------------------------------------------------------ | --------- | --------- | -------------- | --------------------- | -------------------------------- |
| DARTS         | 1           | 20.4    |           |          | Genotype(normal=[[('skip_connect', 0), ('sep_conv_3x3', 1)], [('dil_conv_3x3', 0), ('dil_conv_5x5', 1)], [('sep_conv_3x3', 1), ('dil_conv_5x5', 0)], [('sep_conv_3x3', 3), ('sep_conv_3x3', 0)]], normal_concat=range(2, 6), reduce=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], [('sep_conv_5x5', 1), ('sep_conv_3x3', 0)], [('skip_connect', 0), ('max_pool_3x3', 1)], [('max_pool_3x3', 0), ('dil_conv_5x5', 3)]], reduce_concat=range(2, 6)) | 85.1      | 99.4      | 22890          | [config](https://.py) | [model](.pth) \|[log](.log.json) |
| NAS-Bench-201 | 1           |         |           |          |                                                              |           |           |                | [config](https://.py) | [model](.pth) \|[log](.log.json) |

