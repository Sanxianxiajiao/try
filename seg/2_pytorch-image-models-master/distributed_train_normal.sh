#!/bin/bash
NUM_PROC=$1
shift
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --nnodes=1 --node_rank=0 --master_port=1235 train.py "$@"

