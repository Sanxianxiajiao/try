import os
import simplejson
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import xnas.logger.checkpoint as checkpoint

import xnas.core.config as config
import xnas.logger.meter as meter
import xnas.logger.logging as logging
from xnas.core.builder import *
from xnas.core.config import cfg
from xnas.datasets.loader import get_normal_dataloader
from xnas.logger.meter import TestMeter


# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)


def main(local_rank):
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=cfg.NUM_GPUS)
    setup_env()
    torch.cuda.set_device(local_rank)
    # load data
    [train_loader, valid_loader] = get_normal_dataloader()
    # load model
    supernet = space_builder().to(local_rank)
    supernet = DDP(supernet, device_ids=[local_rank], find_unused_parameters=True)
    checkpoint.load_checkpoint(cfg.SEARCH.WEIGHTS, supernet)
    logger.info('model loaded from {}'.format(cfg.SEARCH.WEIGHTS))
    # supernet.module.set_active_subnet(
    #     resolution=cfg.ATTENTIVENAS.ACTIVE_SUBNET.RESOLUTION,
    #     width = cfg.ATTENTIVENAS.ACTIVE_SUBNET.WIDTH,
    #     depth = cfg.ATTENTIVENAS.ACTIVE_SUBNET.DEPTH,
    #     kernel_size = cfg.ATTENTIVENAS.ACTIVE_SUBNET.KERNEL_SIZE,
    #     expand_ratio = cfg.ATTENTIVENAS.ACTIVE_SUBNET.EXPAND_RATIO,
    # )
    # supernet.module.sample_max_subnet()
    supernet.module.sample_min_subnet()
    subnet = supernet.module.get_active_subnet()
    subnet_flops = supernet.module.compute_active_subnet_flops()
    subnet_settings = supernet.module.get_active_subnet_settings()
    # house-keeping stuff: may using different values with supernet
    subnet.set_bn_param(momentum=cfg.ATTENTIVENAS.BN_MOMENTUM, eps=cfg.ATTENTIVENAS.BN_EPS)

    # Validate
    top1_err, top5_err = validate(subnet, train_loader, valid_loader, local_rank)

    logger.info("subnet_settings:{}".format(simplejson.dumps(subnet_settings)))
    logger.info("flops:{} top1_err:{} top5_err:{}".format(subnet_flops, top1_err, top5_err))


@torch.no_grad()
def validate(net, train_loader, valid_loader, rank):
    net.to(rank)
    # BN calibration
    net.eval()
    logger.info("Calibrating BN running statistics.")
    net.reset_running_stats_for_calibration()
    for cur_iter, (inputs, _) in enumerate(train_loader):
        if cur_iter >= cfg.ATTENTIVENAS.POST_BN_CALIBRATION_BATCH_NUM:
            break
        inputs = inputs.to(rank)
        net(inputs)      # forward only
    logger.info("Calibrated BN running statistics.")
    return test_epoch(net, valid_loader, rank)


def test_epoch(net, test_loader, rank):
    test_meter = TestMeter(len(test_loader))
    net.eval()
    test_meter.reset(True)
    test_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(rank, non_blocking=True), labels.to(rank, non_blocking=True)
        preds = net(inputs)

        # top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
        # loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()

        top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
        batch_size = inputs.size(0)
        wrong1, wrong5 = top1_err.item()*batch_size, top5_err.item()*batch_size #just in case the batch size is different on different nodes
        stats = torch.tensor([wrong1, wrong5, batch_size], device=rank)
        dist.barrier()  # synchronizes all processes
        dist.all_reduce(stats, op=torch.distributed.ReduceOp.SUM) 
        wrong1, wrong5, batch_size = stats.tolist()
        top1_err, top5_err = wrong1/batch_size, wrong5/batch_size
 
        test_meter.iter_toc()
        test_meter.update_stats(top1_err, top5_err, batch_size)
        test_meter.log_iter_stats(0, cur_iter)
        test_meter.iter_tic()
    # Log epoch stats
    epoch_stats = test_meter.get_epoch_stats(0)
    test_meter.log_epoch_stats(0, epoch_stats)
    # top1_err = test_meter.mb_top1_err.get_global_avg()
    # top5_err = test_meter.mb_top5_err.get_global_avg()
    top1_err, top5_err = epoch_stats['top1_err'], epoch_stats['top5_err']
    return top1_err, top5_err


if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(cfg.MASTER_PORT)
    mp.spawn(main, nprocs=cfg.NUM_GPUS, join=True)
