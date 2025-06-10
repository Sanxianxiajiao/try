"""BigNAS supernet training"""

import os
import random

import torch
import torch.nn as nn

import xnas.core.config as config
from xnas.datasets.loader import get_normal_dataloader
import xnas.logger.meter as meter
import xnas.logger.logging as logging
from xnas.core.config import cfg
from xnas.core.builder import *

# DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# BigNAS
from xnas.runner.trainer import Trainer
from xnas.runner.scheduler import adjust_learning_rate_per_batch
from xnas.spaces.OFA.utils import list_mean

# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)


def main(local_rank):
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=cfg.NUM_GPUS)
    setup_env()
    torch.cuda.set_device(local_rank)
    # Network
    net = space_builder().to(local_rank)
    # sync batchnorm
    if cfg.BIGNAS.SYNC_BN:
        net.apply(lambda m: setattr(m, 'need_sync', True))
    # Loss function
    criterion = criterion_builder()
    soft_criterion = criterion_builder('ce_soft')
    
    # Data loaders
    [train_loader, valid_loader] = get_normal_dataloader()
    
    # Optimizers
    net_params = [
        # parameters with weight decay
        {"params": net.get_parameters(['bn', 'bias'], mode="exclude"), "weight_decay": cfg.OPTIM.WEIGHT_DECAY},
        # parameters without weight decay
        {"params": net.get_parameters(['bn', 'bias'], mode="include") , "weight_decay": 0}, 
    ]
    optimizer = optimizer_builder("SGD", net_params)

    
    net = DDP(net, device_ids=[local_rank], find_unused_parameters=True)
    
    # Initialize Recorder
    bignas_trainer = BigNASTrainer(
        model=net,
        criterion=criterion,
        soft_criterion=soft_criterion,
        optimizer=optimizer,
        lr_scheduler=None,
        train_loader=train_loader,
        test_loader=valid_loader,
    )
    
    # Resume
    start_epoch = bignas_trainer.loading() if cfg.SEARCH.AUTO_RESUME else 0
    
    # Training
    logger.info("Start BigNAS training.")
    dist.barrier()
    bignas_trainer.start()
    for cur_epoch in range(start_epoch, cfg.OPTIM.WARMUP_EPOCH+cfg.OPTIM.MAX_EPOCH):
        bignas_trainer.train_epoch(cur_epoch, rank=local_rank)
        if local_rank == 0:
            if (cur_epoch+1) % cfg.EVAL_PERIOD == 0 or (cur_epoch+1) == cfg.OPTIM.MAX_EPOCH:
                bignas_trainer.validate(cur_epoch, local_rank)
    bignas_trainer.finish()
    dist.barrier()
    torch.cuda.empty_cache()


class BigNASTrainer(Trainer):
    """Trainer for BigNAS."""
    def __init__(self, model, criterion, soft_criterion, optimizer, lr_scheduler, train_loader, test_loader):
        super().__init__(model, criterion, optimizer, lr_scheduler, train_loader, test_loader)
        self.sandwich_sample_num = max(2, cfg.BIGNAS.SANDWICH_NUM)    # containing max & min
        self.soft_criterion = soft_criterion

    def train_epoch(self, cur_epoch, rank=0):
        self.model.train()

        cur_step = cur_epoch * len(self.train_loader)
        if self.lr_scheduler is not None:
            cur_lr = self.lr_scheduler.get_last_lr()[0]
            # Rule: constrant ending
            if cur_epoch >= cfg.OPTIM.WARMUP_EPOCH:
                cur_lr = max(cur_lr, 0.05 * cfg.OPTIM.BASE_LR)
            self.writer.add_scalar('train/lr', cur_lr, cur_step)

        self.train_meter.iter_tic()
        if cfg.NUM_GPUS > 1:
            self.train_loader.sampler.set_epoch(cur_epoch)  # DDP
        for cur_iter, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(rank, non_blocking=True), labels.to(rank, non_blocking=True)
            
            # Adjust lr per iter
            if self.lr_scheduler is None:
                cur_lr = adjust_learning_rate_per_batch(
                    epoch=cur_epoch,
                    n_iter=len(self.train_loader),
                    iter=cur_iter,
                    warmup=(cur_epoch < cfg.OPTIM.WARMUP_EPOCH),
                )
                # Rule: constrant ending
                if cur_epoch >= cfg.OPTIM.WARMUP_EPOCH:
                    cur_lr = max(cur_lr, 0.05 * cfg.OPTIM.BASE_LR)
                # set lr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = cur_lr
            else:
                cur_lr = self.lr_scheduler.get_last_lr()[0]
                # Rule: constrant ending
                if cur_epoch >= cfg.OPTIM.WARMUP_EPOCH:
                    cur_lr = max(cur_lr, 0.05 * cfg.OPTIM.BASE_LR)
            self.writer.add_scalar('train/lr', cur_lr, cur_step)

            top1_dict = {}
            top5_dict = {}
            loss_dict = {}

            ## Sandwich Rule ##
            # Step 1. Largest network sampling & regularization
            self.optimizer.zero_grad()
            self.model.module.sample_max_subnet()
            self.model.module.set_dropout_rate(cfg.TRAIN.DROP_PATH_PROB, cfg.BIGNAS.DROP_CONNECT)
            preds = self.model(inputs)
            loss = self.criterion(preds, labels)
            loss.backward()
            
            with torch.no_grad():
                soft_logits = preds.clone().detach()
            

            # calculating errors for max net
            top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
            loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()

            
            loss_dict.update({'max_net': loss})
            top1_dict.update({'max_net': top1_err})
            top5_dict.update({'max_net': top5_err})


            # Step 2. sample smaller networks
            self.model.module.set_dropout_rate(0, 0)
            for arch_id in range(1, self.sandwich_sample_num):
                if arch_id == self.sandwich_sample_num - 1:
                    self.model.module.sample_min_subnet()
                else:
                    subnet_seed = int("%d%.3d%.3d" % (cur_step, arch_id, 0))
                    random.seed(subnet_seed)
                    self.model.module.sample_active_subnet()
                preds = self.model(inputs)

                if self.soft_criterion is not None:
                    loss = self.soft_criterion(preds, soft_logits)
                else:
                    loss = self.criterion(preds, labels)
                loss.backward()
            self.optimizer.step()
            
            # calculating errors for min net
            top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
            loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()

            loss_dict.update({'min_net': loss})
            top1_dict.update({'min_net': top1_err})
            top5_dict.update({'min_net': top5_err})


            self.train_meter.iter_toc()
            self.train_meter.update_stats(top1_err, top5_err, loss, cur_lr, inputs.size(0) * cfg.NUM_GPUS)
            self.train_meter.log_iter_stats(cur_epoch, cur_iter)
            self.train_meter.iter_tic()
            self.writer.add_scalars('train/loss', loss_dict, cur_step)
            self.writer.add_scalars('train/top1_error', top1_dict, cur_step)
            self.writer.add_scalars('train/top5_error', top5_dict, cur_step)
            cur_step += 1
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(cur_epoch+cur_iter/len(self.train_loader))

        # Log epoch stats
        self.train_meter.log_epoch_stats(cur_epoch)
        self.train_meter.reset()
        # Saving checkpoint
        self.saving(cur_epoch)
    
    @torch.no_grad()
    def test_epoch(self, subnet, cur_epoch, rank=0):
        subnet.eval()
        self.test_meter.reset(True)
        self.test_meter.iter_tic()
        for cur_iter, (inputs, labels) in enumerate(self.test_loader):
            inputs, labels = inputs.to(rank, non_blocking=True), labels.to(rank, non_blocking=True)
            preds = subnet(inputs)
            top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
            top1_err, top5_err = top1_err.item(), top5_err.item()

            self.test_meter.iter_toc()
            self.test_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
            self.test_meter.log_iter_stats(cur_epoch, cur_iter)
            self.test_meter.iter_tic()
        # Log epoch stats
        epoch_stats = self.test_meter.get_epoch_stats(cur_epoch)
        self.test_meter.log_epoch_stats(cur_epoch, epoch_stats)
        # top1_err = test_meter.mb_top1_err.get_global_avg()
        # top5_err = test_meter.mb_top5_err.get_global_avg()
        top1_err, top5_err = epoch_stats['top1_err'], epoch_stats['top5_err']
        return top1_err, top5_err


    def validate(self, cur_epoch, rank, bn_calibration=True):
        subnets_to_be_evaluated = {
            'min_net': {},
            'max_net': {},
        }
        
        top1_list, top5_list = [], []
        with torch.no_grad():
            for net_id in subnets_to_be_evaluated:
                if net_id == 'min_net': 
                    self.model.module.sample_min_subnet()
                elif net_id == 'max_net':
                    self.model.module.sample_max_subnet()
                elif net_id.startswith('random_net'):
                    self.model.module.sample_active_subnet()
                else:
                    self.model.module.set_active_subnet(
                        subnets_to_be_evaluated[net_id]['resolution'],
                        subnets_to_be_evaluated[net_id]['width'],
                        subnets_to_be_evaluated[net_id]['depth'],
                        subnets_to_be_evaluated[net_id]['kernel_size'],
                        subnets_to_be_evaluated[net_id]['expand_ratio'],
                    )

                subnet = self.model.module.get_active_subnet()
                subnet.to(rank)
                logger.info("evaluating subnet {}".format(net_id))
                
                if bn_calibration:
                    subnet.eval()
                    logger.info("Calibrating BN running statistics.")
                    subnet.reset_running_stats_for_calibration()
                    for cur_iter, (inputs, _) in enumerate(self.train_loader):
                        if cur_iter >= cfg.BIGNAS.POST_BN_CALIBRATION_BATCH_NUM:
                            break
                        inputs = inputs.to(rank)
                        subnet(inputs)      # forward only
                
                top1_err, top5_err = self.test_epoch(subnet, cur_epoch, rank)
                top1_list.append(top1_err), top5_list.append(top5_err)

            top1_dict = { item[0]: item[1] for item in zip(subnets_to_be_evaluated, top1_list)}
            top5_dict = { item[0]: item[1] for item in zip(subnets_to_be_evaluated, top5_list)}
            top1_mean = list_mean(top1_list)
            top5_mean = list_mean(top5_list)
            top1_dict.update({'mean': top1_mean})
            top5_dict.update({'mean': top5_mean})
            # 分别可视化所有待评估子网的error
            self.writer.add_scalars(
                'val/top1_error', 
                top1_dict,
                cur_epoch
            )
            self.writer.add_scalars(
                'val/top5_error', 
                top5_dict,
                cur_epoch
            )
            # 输出待评估子网的error
            for net_id in top1_dict.keys():
                logger.info("{} | top1_err:{} top5_err:{}".format(net_id, top1_dict[net_id], top5_dict[net_id]))
            if self.best_err > top1_mean:
                self.best_err = top1_mean
                self.saving(cur_epoch, best=True)

if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(cfg.MASTER_PORT)
    
    mp.spawn(main, nprocs=cfg.NUM_GPUS, join=True)