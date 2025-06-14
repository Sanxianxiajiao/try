"""Configuration file (powered by YACS)."""

import os
import sys
import argparse
from yacs.config import CfgNode


# Global config object
_C = CfgNode(new_allowed=True)
cfg = _C


# -------------------------------------------------------- #
# Data Loader options
# -------------------------------------------------------- #
_C.LOADER = CfgNode(new_allowed=True)

_C.LOADER.DATASET = "cifar10"

# stay empty to use "./data/$dataset" as default
_C.LOADER.DATAPATH = ""

_C.LOADER.SPLIT = [0.8, 0.2]

# whether using val dataset (imagenet only)
_C.LOADER.USE_VAL = False

_C.LOADER.NUM_CLASSES = 10

_C.LOADER.NUM_WORKERS = 8

_C.LOADER.PIN_MEMORY = True

# batch size of training and validation
# type: int or list(different during validation)
# _C.LOADER.BATCH_SIZE = [256, 128]
_C.LOADER.BATCH_SIZE = 256

# augment type using by ImageNet only
# chosen from ['default', 'auto_augment_tf']
_C.LOADER.TRANSFORM = "default"


# ------------------------------------------------------------------------------------ #
# Search Space options
# ------------------------------------------------------------------------------------ #
_C.SPACE = CfgNode(new_allowed=True)

_C.SPACE.NAME = 'darts'

# first layer's channels, not channels for input image.
_C.SPACE.CHANNELS = 16

_C.SPACE.LAYERS = 8

_C.SPACE.NODES = 4

_C.SPACE.BASIC_OP = []



# ------------------------------------------------------------------------------------ #
# Optimizer options in network
# ------------------------------------------------------------------------------------ #
_C.OPTIM = CfgNode(new_allowed=True)

# Base learning rate
_C.OPTIM.BASE_LR = 0.1

_C.OPTIM.MIN_LR = 1.e-3


# Learning rate policy select from {'cos', 'exp', 'step'}
_C.OPTIM.LR_POLICY = "cos"
# Steps for 'steps' policy (in epochs)
_C.OPTIM.STEPS = [30, 60, 90]
# Learning rate multiplier for 'steps' policy
_C.OPTIM.LR_MULT = 0.1


# Momentum
_C.OPTIM.MOMENTUM = 0.9
# Momentum dampening
_C.OPTIM.DAMPENING = 0.0
# Nesterov momentum
_C.OPTIM.NESTEROV = False

# RMSProp alpha
_C.OPTIM.ALPHA = 0.9

_C.OPTIM.WEIGHT_DECAY = 5e-4

_C.OPTIM.GRAD_CLIP = 5.0

_C.OPTIM.MAX_EPOCH = 200
# Warm up epochs
_C.OPTIM.WARMUP_EPOCH = 0
# Start the warm up from OPTIM.WARMUP_LR to OPTIM.BASE_LR
_C.OPTIM.WARMUP_LR = 0.01
# Ending epochs
_C.OPTIM.FINAL_EPOCH = 0



# -------------------------------------------------------- #
# Searching options
# -------------------------------------------------------- #
_C.SEARCH = CfgNode(new_allowed=True)

_C.SEARCH.IM_SIZE = 224

# Multi-sized Crop
# NOTE: IM_SIZE in ImageNet will be covered if this one is setted. 
_C.SEARCH.MULTI_SIZES = []

# channels of input images, 3 for rgb
_C.SEARCH.INPUT_CHANNELS = 3

_C.SEARCH.LOSS_FUN = 'cross_entropy'
# label smoothing for cross entropy loss
_C.SEARCH.LABEL_SMOOTH = 0.

# resume and path of checkpoints
_C.SEARCH.AUTO_RESUME = True

_C.SEARCH.WEIGHTS = ""

_C.SEARCH.EVALUATION = ""


# ------------------------------------------------------------------------------------ #
# Options for model training
# ------------------------------------------------------------------------------------ #
_C.TRAIN = CfgNode(new_allowed=True)

_C.TRAIN.IM_SIZE = 32

# channels of input images, 3 for rgb
_C.TRAIN.INPUT_CHANNELS = 3

_C.TRAIN.DROP_PATH_PROB = 0.2

_C.TRAIN.LAYERS = 20

_C.TRAIN.CHANNELS = 36

_C.TRAIN.GENOTYPE = ""


# -------------------------------------------------------- #
# Model testing options
# -------------------------------------------------------- #
_C.TEST = CfgNode(new_allowed=True)

_C.TEST.IM_SIZE = 224

# using specific batchsize for testing
# using search.batch_size if this value keeps -1
_C.TEST.BATCH_SIZE = -1



# -------------------------------------------------------- #
# Benchmarks options
# -------------------------------------------------------- #
_C.BENCHMARK = CfgNode(new_allowed=True)

# Path to NAS-Bench-201 weights file
_C.BENCHMARK.NB201PATH = "./data/NAS-Bench-201-v1_1-096897.pth"

# path to NAS-Bench-301 folder
_C.BENCHMARK.NB301PATH = "./data/nb301models/"


# -------------------------------------------------------- #
# Misc options
# -------------------------------------------------------- #

_C.CUDNN_BENCH = True

_C.LOG_PERIOD = 10

_C.EVAL_PERIOD = 1

_C.SAVE_PERIOD = 1

_C.NUM_GPUS = 1

_C.OUT_DIR = "exp/"

_C.DETERMINSTIC = True

_C.RNG_SEED = 1

_C.MASTER_PORT = 23333


# -------------------------------------------------------- #

def dump_cfgfile(cfg_dest="config.yaml"):
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, cfg_dest)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfgfile(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def load_configs():
    """Load config from command line arguments and set any specified options.
       How to use: python xx.py --cfg path_to_your_config.cfg test1 0 test2 True
       opts will return a list with ['test1', '0', 'test2', 'True'], yacs will compile to corresponding values
    """
    parser = argparse.ArgumentParser(description="Config file options.")
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    _C.merge_from_file(args.cfg)
    _C.merge_from_list(args.opts)