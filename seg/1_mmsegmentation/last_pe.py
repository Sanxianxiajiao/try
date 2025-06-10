# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import pickle
import time
import warnings
from torch.distributed.elastic.multiprocessing.errors import record
import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes, get_root_logger

import random
import copy
import numpy as np
import torch.distributed as dist
from mmcv.cnn import get_model_complexity_info

# import math
# from zero_cost_pe.utils import pe


def parse_args():
    parser = argparse.ArgumentParser(description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument('--eval', type=str, nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--gpu-id', type=int, default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument('--constraint-flops', type=float, default=0) # in cfg file
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


@record
def main():
    args = parse_args()
    assert args.eval
    with open('./512-res-{}.pkl'.format(args.config.split('/')[-2]), 'rb') as f:
        spaces = pickle.load(f)
    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    assert cfg.constraint_flops != 0

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg.work_dir += '/search_evolution/'
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(f'The gpu-ids is reset from {cfg.gpu_ids} to '
                          f'{cfg.gpu_ids[0:1]} to avoid potential error in '
                          'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
    elif rank == 0:
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))     
    seed = init_random_seed(None, 'cuda') # 让不同gpu上采样的模型是相同的，否则
    set_random_seed(seed, deterministic=False)

    '''*************** val loader ***************'''
    # build the val_dataloader
    val_dataset = build_dataset(cfg.data.val) # 搜索时，使用训练集

    # The default loader config
    val_loader_cfg = dict(
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False)
    # The overall dataloader settings
    val_loader_cfg.update({
        k: v for k, v in cfg.data.items() if k not in \
        ['train', 'val', 'test', 'train_dataloader', 'val_dataloader','test_dataloader']})
    val_loader_cfg = {**val_loader_cfg, 'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('val_dataloader', {})}
    val_loader = build_dataloader(val_dataset, **val_loader_cfg)

    '''*************** train loader ***************'''
    # build the val_dataloader
    train_dataset = build_dataset(cfg.data.train)
    # The default loader config
    train_loader_cfg = dict(
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=seed,
        drop_last=True)
    # The overall dataloader settings
    train_loader_cfg.update({
        k: v for k, v in cfg.data.items() if k not in \
        ['train', 'val', 'test', 'train_dataloader', 'val_dataloader','test_dataloader']})
    train_loader_cfg = {**train_loader_cfg, **cfg.data.get('train_dataloader', {})}
    train_loader = build_dataloader(train_dataset, **train_loader_cfg)

    seed = init_random_seed(None, 'cuda') # 让不同gpu上采样的模型是相同的，否则
    set_random_seed(seed, deterministic=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    checkpoint = load_checkpoint(model, "{}-latest.pth".format(args.config.split('/')[-2]), map_location='cpu')
    # print("{}-latest.pth".format(args.config.split('/')[-2]))
    # print('"CLASSES" not found in meta, use dataset.CLASSES instead')
    # model.CLASSES = val_dataset.CLASSES
    # print('"PALETTE" not found in meta, use dataset.PALETTE instead')
    # model.PALETTE = val_dataset.PALETTE
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = val_dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = val_dataset.PALETTE
    
    eval_kwargs = {}
    cfg.device = get_device()
    rank, _ = get_dist_info()

    '''************* random search *************'''
    benchmarks = []
    # if rank == 0:

    for i, space in enumerate(spaces):
        subnet_cfg = space['subnet_cfg']
        flops = space['flops']
        model.set_active_subnet(
            subnet_cfg['width'], subnet_cfg['depth'], subnet_cfg['kernel_size'], subnet_cfg['expand_ratio'],
            subnet_cfg['num_heads'], subnet_cfg['key_dim'], subnet_cfg['attn_ratio'], subnet_cfg['mlp_ratio'], subnet_cfg['transformer_depth']
        )
        subnet = model.get_active_subnet()  
        subnet = build_dp_or_ddp(subnet, distributed, cfg)
        bn_calibration(subnet, train_loader, cfg.post_bn_calibration_batch_num)
        results = validate_subnet(subnet, distributed, val_loader, eval_kwargs)
        if args.eval and rank == 0:
            eval_kwargs.update(metric=args.eval)
            metric = val_dataset.evaluate(results, **eval_kwargs)
            mIoU = metric['mIoU']                     
            benchmarks.append({
                'subnet_cfg': subnet_cfg,
                'flops': flops,
                'mIoU': mIoU
            })
            logger.info('Initail population: {}th subnet, mIoU: {:.2f} flops: {} subnet_cfg:{}'.format(i, mIoU * 100, flops, subnet_cfg))   
    # supernet
    # benchmarks = []
    # for i in range(512):
    #     while True:
    #         subnet_cfg = model.sample_active_subnet()
    #         subnet = model.get_active_subnet()  
    #         # compute flops
    #         flops = compute_flops(subnet)
    #         if 0.9 * cfg.constraint_flops < flops < cfg.constraint_flops:
    #             break
    #         else:
    #             logger.info('subnet flops: {}, not satisfied'.format(flops))
    #     # eval
    #     subnet = build_dp_or_ddp(subnet, distributed, cfg)
    #     # bn_calibration(subnet, train_loader, cfg.post_bn_calibration_batch_num)
    #     results = validate_subnet(subnet, distributed, val_loader, eval_kwargs)
    #     if args.eval:
    #         eval_kwargs.update(metric=args.eval)
    #         metric = val_dataset.evaluate(results, **eval_kwargs)
    #         mIoU = metric['mIoU']                     
    #         benchmarks.append({
    #             'subnet_cfg': subnet_cfg,
    #             'flops': flops,
    #             'mIoU': mIoU
    #         })
    #         logger.info('Initail population: {}th subnet, mIoU: {:.2f} flops: {} subnet_cfg:{}'.format(i, mIoU * 100, flops, subnet_cfg))      

    # for evo_iter in range(20):
    #     benchmarks = list(filter(
    #         lambda x: x['flops'] < cfg.constraint_flops, 
    #         sorted(benchmarks, key=lambda d: d['mIoU'], reverse=True)))[0:128]
    #     logger.info('Choose best 128 arch from population...')

    #     for mutate_num in range(128):
    #         while True:
    #             subnet_cfg = model.sample_active_subnet()
    #             new_subnet_cfg = model.mutate_and_reset(subnet_cfg)
    #             subnet = model.get_active_subnet()  
    #             # compute flops
    #             flops = compute_flops(subnet)
    #             if 0.9 * cfg.constraint_flops < flops < cfg.constraint_flops:
    #                 break
    #             else:
    #                 logger.info('subnet flops: {}, not satisfied'.format(flops))
    #         # eval
    #         subnet = build_dp_or_ddp(subnet, distributed, cfg)
    #         bn_calibration(subnet, train_loader, cfg.post_bn_calibration_batch_num)
    #         results = validate_subnet(subnet, distributed, val_loader, eval_kwargs)
    #         if args.eval:
    #             eval_kwargs.update(metric=args.eval)
    #             metric = val_dataset.evaluate(results, **eval_kwargs)
    #             mIoU = metric['mIoU']                     
    #             benchmarks.append({
    #                 'subnet_cfg': new_subnet_cfg,
    #                 'flops': flops,
    #                 'mIoU': mIoU
    #             })
    #         logger.info('Evolution iter: {}, mutate num: {} mIoU: {:.2f} flops: {} subnet_cfg:{}'.format(evo_iter, mutate_num, mIoU * 100, flops, new_subnet_cfg))
        
    #     for cross_num in range(128):
    #         while True:
    #             subnet_cfg1 = model.sample_active_subnet()
    #             subnet_cfg2 = model.sample_active_subnet()
    #             new_subnet_cfg = model.crossover_and_reset(subnet_cfg1, subnet_cfg2)
    #             subnet = model.get_active_subnet()  
    #             # compute flops
    #             flops = compute_flops(subnet)
    #             if 0.9 * cfg.constraint_flops < flops < cfg.constraint_flops:
    #                 break
    #             else:
    #                 logger.info('subnet flops: {}, not satisfied'.format(flops))
    #         subnet = build_dp_or_ddp(subnet, distributed, cfg)
    #         bn_calibration(subnet, train_loader, cfg.post_bn_calibration_batch_num)
    #         results = validate_subnet(subnet, distributed, val_loader, eval_kwargs)
    #         if args.eval:
    #             eval_kwargs.update(metric=args.eval)
    #             metric = val_dataset.evaluate(results, **eval_kwargs)
    #             mIoU = metric['mIoU']                     
    #             benchmarks.append({
    #                 'subnet_cfg': new_subnet_cfg,
    #                 'flops': flops,
    #                 'mIoU': mIoU
    #             })
    #         logger.info('Evolution iter: {}, cross num: {} mIoU: {:.2f} flops: {} subnet_cfg:{}'.format(evo_iter, cross_num, mIoU * 100, flops, new_subnet_cfg))        
    if rank == 0:
        with open('./512-best-res-{}.pkl'.format(args.config.split('/')[-2]), 'wb') as f:
            pickle.dump(benchmarks, f)


        # 符合flops约束且精度从高到低的子网list
        benchmarks = list(filter(
            lambda x: x['flops'] < cfg.constraint_flops, 
            sorted(benchmarks, key=lambda d: d['mIoU'], reverse=True)))
        best_subnet_cfg, best_subnet_flops, best_subnet_mIoU = benchmarks[0]['subnet_cfg'], benchmarks[0]['flops'], benchmarks[0]['mIoU']
        logger.info("Best Architecture in evolution search: mIoU: {:.2f} flops: {} cfg: {}".format(
            best_subnet_mIoU * 100, best_subnet_flops, best_subnet_cfg))


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()

# 这个函数就是给各个随即模块设置随机种子，在之前的./tools/train.py中设置随机种子那里有调用
def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def build_dp_or_ddp(subnet, distributed, cfg):
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        subnet = revert_sync_batchnorm(subnet)
        subnet = build_dp(subnet, cfg.device, device_ids=cfg.gpu_ids)
    else:
        subnet = build_ddp(
            subnet, cfg.device, device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
    return subnet

def bn_calibration(model, train_loader, post_bn_calibration_batch_num):
    model.eval()
    with torch.no_grad():
        model.module.reset_running_stats_for_calibration()
        for batch_idx, x in enumerate(train_loader):
            img = x['img'].data[0].cuda()
            img_metas = x['img_metas'].data[0]
            gt_semantic_seg = x['gt_semantic_seg'].data[0].cuda()
            if batch_idx >= post_bn_calibration_batch_num:
                break
            model(img=img, img_metas=img_metas, return_loss=True, gt_semantic_seg=gt_semantic_seg)  #forward only     

def validate_subnet(subnet, distributed, val_loader, eval_kwargs):
    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    if not distributed:
        results = single_gpu_test(
            subnet, val_loader, False, False, False, 0.5,
            pre_eval=True, format_only=False,
            format_args=eval_kwargs)
    else:
        results = multi_gpu_test(
            subnet, val_loader, None, False, False, 
            pre_eval=True, format_only=False, 
            format_args=eval_kwargs)
    return results

def is_pareto(k, benckmarks: list):
    for x in benckmarks:
        if k['flops'] > x['flops'] and k['mIoU'] < x['flops']:
            return False
    return True    

def compute_flops(model): # 单卡
    tmp_model = copy.deepcopy(model).cuda()
    tmp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(tmp_model)
    tmp_model.forward = tmp_model.forward_dummy         
    tmp_model.eval()            
    flops, params = get_model_complexity_info(tmp_model, (3, 512, 512), print_per_layer_stat=False)
    flops = float(flops.split(' ')[0])
    return flops

# def get_zc_pe(network, loss_fn, train_loader, num_classes, device):
#     method_types = ["nwot", "l2_norm", "zen"]
#     scores = []
#     for method_type in method_types:
#         score = pe.find_measures(
#                         network,
#                         train_loader,
#                         ("random", 1, num_classes),
#                         device,
#                         loss_fn=loss_fn,
#                         measure_names=[method_type],
#                     )
#         if math.isnan(score):
#             score = -1e8
#         scores.append(score)
#     return scores
if __name__ == '__main__':

    main()