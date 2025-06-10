# PDARTS

[Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation](https://arxiv.org/abs/1904.12760)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/chenxin061/pdarts">Official Repo</a>

<a href=" ">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Recently, differentiable search methods have made major progress in reducing the computational costs of neural architecture search. However, these approaches often report lower accuracy in evaluating the searched architecture or transferring it to another dataset. This is arguably due to the large gap between the architecture depths in search and evaluation scenarios. In this paper, we present an efficient algorithm which allows the depth of searched architectures to grow gradually during the training procedure. This brings two issues, namely, heavier computational overheads and weaker search stability, which we solve using search space approximation and regularization, respectively. With a significantly reduced search time (∼7 hours on a single GPU), our approach achieves state-of-the-art performance on both the proxy dataset (CIFAR10 or CIFAR100) and the target dataset (ImageNet).

<!-- [IMAGE] -->

<div align=center>
<img src="https://raw.githubusercontent.com/chenxin061/pdarts/master/pipeline2.jpg" width="70%"/>
</div>



## Citation

```bibtex
@inproceedings{chen2019progressive,
  title={Progressive differentiable architecture search: Bridging the depth gap between search and evaluation},
  author={Chen, Xin and Xie, Lingxi and Wu, Jun and Tian, Qi},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={1294--1303},
  year={2019}
}
```

## Results and models

### CIFAR-10

| Search space  | Random seed | Mem(GB) | Params(M) | Flops(G) | Genotype                                                     | SuperNet Top-1 (%) | SuperNet Top-5 (%) | Search-Cost | Config                | Download                         |
| ------------- | ----------- | ------- | --------- | -------- | ------------------------------------------------------------ | --------- | --------- | ----------- | --------------------- | -------------------------------- |
| DARTS         | 1           | 4.79    |           |          | Genotype(normal=\[ \[('none', 1), ('skip_connect', 0)], \[('none', 2), ('none', 1)], \[('none', 3), ('none', 2)], \[('none', 4), ('none', 3)] ], normal_concat=range(2, 6), reduce=\[ \[('avg_pool_3x3', 1), ('sep_conv_5x5', 0)], \[('dil_conv_3x3', 2), ('dil_conv_3x3', 0)], \[('avg_pool_3x3', 1), ('skip_connect', 2)], \[('skip_connect', 3), ('avg_pool_3x3', 1)] ], reduce_concat=range(2, 6)) | 86.5      | 99.42     | 8825        | [config](https://.py) | [model](.pth) \|[log](.log.json) |
| DARTS         | 3           | 4.91    |           |          | Genotype(normal=\[ \[('none', 1), ('skip_connect', 0)], [('none', 1), ('none', 2)], [('none', 3), ('none', 2)], [('none', 4), ('none', 3)] ], normal_concat=range(2, 6), reduce=\[ \[('max_pool_3x3', 0), ('skip_connect', 1)], [('dil_conv_5x5', 2), ('sep_conv_3x3', 1)], [('skip_connect', 3), ('skip_connect', 2)], [('none', 4), ('skip_connect', 2)] ], reduce_concat=range(2, 6)) | 85.5      | 99.28     | 9549        | [config](https://.py) | [model](.pth) \|[log](.log.json) |
| DARTS         | 5           | 4.91    |           |          | Genotype(normal=\[ \[('none', 1), ('sep_conv_3x3', 0)], [('none', 2), ('none', 1)], [('none', 3), ('none', 2)], [('none', 4), ('none', 3)] ], normal_concat=range(2, 6), reduce=\[ \[('sep_conv_3x3', 0), ('skip_connect', 1)], [('skip_connect', 2), ('max_pool_3x3', 0)], [('none', 3), ('sep_conv_3x3', 0)], [('none', 4), ('max_pool_3x3', 0)] ], reduce_concat=range(2, 6)) | 86.7      | 99.3     | 9532        | [config](https://.py) | [model](.pth) \|[log](.log.json) |
| NAS-Bench-201 | 1           |         |           |          |                                                              |           |           |             | [config](https://.py) | [model](.pth) \|[log](.log.json) |

### CIFAR-100