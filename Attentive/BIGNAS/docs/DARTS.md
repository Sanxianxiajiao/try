# DARTS

[DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/quark0/darts">Official Repo</a>

<a href=" ">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

This paper addresses the scalability challenge of architecture search by formulating the task in a differentiable manner. Unlike conventional approaches of applying evolution or reinforcement learning over a discrete and non-differentiable search space, our method is based on the continuous relaxation of the architecture representation, allowing efficient search of the architecture using gradient descent. Extensive experiments on CIFAR-10, ImageNet, Penn Treebank and WikiText-2 show that our algorithm excels in discovering high-performance convolutional architectures for image classification and recurrent architectures for language modeling, while being orders of magnitude faster than state-of-the-art non-differentiable techniques. Our implementation has been made publicly available to facilitate further research on efficient architecture search algorithms.

<!-- [IMAGE] -->

<div align=center>
<img src="https://raw.githubusercontent.com/quark0/darts/master/img/darts.png" width="70%"/>
</div>


## Citation

```bibtex
@article{liu2018darts,
  title={DARTS: Differentiable Architecture Search},
  author={Liu, Hanxiao and Simonyan, Karen and Yang, Yiming},
  journal={arXiv preprint arXiv:1806.09055},
  year={2018}
}
```

## Results and models

### CIFAR-10

| Search space  | Random seed | Mem(GB) | Params(M) | Flops(G) | Genotype                                                     | SuperNet Top-1 (%) | SuperNet Top-5 (%) | Search cost(s) | Config                | Download                         |
| ------------- | ----------- | ------- | --------- | -------- | ------------------------------------------------------------ | --------- | --------- | -------------- | --------------------- | -------------------------------- |
| DARTS         | 1           | 8.6     |           |          | Genotype(normal=\[ \[\('sep_conv_3x3', 1), ('sep_conv_3x3', 0)], \[('sep_conv_3x3', 2), ('sep_conv_5x5', 1)], \[('sep_conv_3x3', 2), ('sep_conv_3x3', 3)], \[('sep_conv_3x3', 3), ('sep_conv_3x3', 2)] ], normal_concat=range(2, 6), reduce=\[ \[('max_pool_3x3', 0), ('max_pool_3x3', 1)], \[('max_pool_3x3', 0), ('dil_conv_3x3', 2)], \[('max_pool_3x3', 0), ('dil_conv_5x5', 2)], \[('sep_conv_3x3', 1), ('skip_connect', 3)] ], reduce_concat=range(2, 6)) | 89.6      | 99.5      | 98596          | [config](https://.py) | [model](.pth) \|[log](.log.json) |
| DARTS         | 3           | 8.76     |           |          | Genotype(normal=\[ \[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], \[('sep_conv_3x3', 1), ('sep_conv_3x3', 0)], \[('sep_conv_3x3', 1), ('sep_conv_3x3', 3)], \[('sep_conv_3x3', 1), ('sep_conv_5x5', 0)] ], normal_concat=range(2, 6), reduce=\[ \[('max_pool_3x3', 0), ('max_pool_3x3', 1)], \[('max_pool_3x3', 1), ('max_pool_3x3', 0)], \[('skip_connect', 3), ('skip_connect', 2)], \[('sep_conv_3x3', 3), ('sep_conv_3x3', 4)] ], reduce_concat=range(2, 6)) | 89.7      | 99.5      | 16891          | [config](https://.py) | [model](.pth) \|[log](.log.json) |
| NAS-Bench-201 | 1           | 0.95    |           |          | \|skip_connect~0\|+\|skip_connect~0\|nor_conv_3x3~1\|+\|skip_connect~0\|skip_connect~1\|skip_connect~2\| | 84.3      | 99.2      | 50372          | [config](https://.py) | [model](.pth) \|[log](.log.json) |
| NAS-Bench-201 | 3           | 0.95    |           |          | \|skip_connect~0\|+\|skip_connect~0\|nor_conv_3x3~1\|+\|skip_connect~0\|skip_connect~1\|skip_connect~2\| | 84.8      | 99.4      | 7115          | [config](https://.py) | [model](.pth) \|[log](.log.json) |
| NAS-Bench-201 | 5           | 0.95    |           |          | \|skip_connect~0\|+\|skip_connect~0\|none~1\|+\|skip_connect~0\|skip_connect~1\|skip_connect~2\| | 85.3      | 99.3      | 13500          | [config](https://.py) | [model](.pth) \|[log](.log.json) |

### CIFAR-100

