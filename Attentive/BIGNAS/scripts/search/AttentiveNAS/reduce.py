"""BigNAS subnet searching: Coarse-to-fine Architecture Selection"""

import os
import pickle
import time
import warnings
import numpy as np
from itertools import product
from sklearn.ensemble import RandomForestClassifier

import torch

import xnas.core.config as config
import xnas.logger.meter as meter
import xnas.logger.logging as logging
from xnas.core.builder import *
from xnas.core.config import cfg
from xnas.datasets.loader import get_normal_dataloader
from xnas.logger.meter import TestMeter

from zero_cost_pe.utils import pe
import math
# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)


def get_all_subnets():
    # get all subnets
    all_subnets = []
    subnet_sets = cfg.BIGNAS.SEARCH_CFG_SETS
    stage_names = ['mb1', 'mb2', 'mb3', 'mb4', 'mb5', 'mb6', 'mb7']

    mb_stage_subnets = []
    for mbstage in stage_names:
        mb_block_cfg = getattr(subnet_sets, mbstage)
        mb_stage_subnets.append(list(product(
            mb_block_cfg.c,
            mb_block_cfg.d,
            mb_block_cfg.k,
            mb_block_cfg.t
        )))

    all_mb_stage_subnets = list(product(*mb_stage_subnets))

    resolutions = getattr(subnet_sets, 'resolutions')
    first_conv = getattr(subnet_sets, 'first_conv')
    last_conv = getattr(subnet_sets, 'last_conv')

    for res in resolutions:
        for fc in first_conv.c:
            for mb in all_mb_stage_subnets:
                np_mb_choice = np.array(mb)
                width = np_mb_choice[:, 0].tolist()  # c
                depth = np_mb_choice[:, 1].tolist()  # d
                kernel = np_mb_choice[:, 2].tolist() # k
                expand = np_mb_choice[:, 3].tolist() # t
                for lc in last_conv.c:
                    all_subnets.append({
                        'resolution': res,
                        'width': [fc] + width + [lc],
                        'depth': depth,
                        'kernel_size': kernel,
                        'expand_ratio': expand
                    })
    return all_subnets

def get_zc_pe(network, loss_fn, train_loader, num_classes, device):
    method_types = ["nwot", "l2_norm", "zen"]
    # for method_type in method_types:
    scores = []
    for method_type in method_types:
        scores.append(pe.find_measures(
                        network,
                        train_loader,
                        ("random", 1, num_classes),
                        device,
                        loss_fn=loss_fn,
                        measure_names=[method_type],
                    ))
    for i,score in enumerate(scores):
        if math.isnan(score):
            score = -1e8
            scores[i] = (score)
    return scores

def main():
    setup_env()
    supernet = space_builder().cuda()
    # supernet.load_weights_from_pretrained_models(cfg.SEARCH.WEIGHTS)

    [train_loader, valid_loader] = get_normal_dataloader()

    test_meter = TestMeter(len(valid_loader))
    
    suggest_samples = []
    print('flops:', cfg.BIGNAS.CONSTRAINT_FLOPS//1e6)
    print(os.path.join(cfg.OUT_DIR,'flops-{}.pkl'.format(
            cfg.BIGNAS.CONSTRAINT_FLOPS//1e6)
                     ))
    st = time.time()
    logger.info('start time: {}'.format(st))
    '''************* random sample *************'''
    # all_subnets = get_all_subnets()
    benchmarks = []
    ratio = 0.9
    scores = []
    features = []
    for i in range(200):
        while True:
            subnet_cfg = supernet.sample_active_subnet()
            subnet = supernet.get_active_subnet()  
            # compute flops
            # flops = compute_flops(subnet)
            flops = supernet.compute_active_subnet_flops()
            if flops*1e6 < cfg.BIGNAS.CONSTRAINT_FLOPS:
                break
            else:
                logger.info('subnet flops: {}, not satisfied'.format(flops))
        # eval
        # subnet = build_dp_or_ddp(subnet, distributed, cfg)
        # bn_calibration(subnet, train_loader, cfg.post_bn_calibration_batch_num)
        score = get_zc_pe(subnet, None,train_loader, cfg.LOADER.NUM_CLASSES,  "cuda")
        # top1_err, top5_err = validate(subnet, train_loader, valid_loader, test_meter)
        # flops = supernet.compute_active_subnet_flops()
        
        logger.info("[{}/{}] flops:{} score:{}".format(
            len(suggest_samples), 200, flops, score
        ))

        benchmarks.append({
            'subnet_cfg': subnet_cfg,
            'flops': flops,
            "score": score
            # 'top1_err': top1_err,
            # 'top5_err': top5_err
        })

        features.append(subnet_cfg_2_feature(subnet_cfg))
        # results = validate_subnet(subnet, distributed, val_loader, eval_kwargs)
        scores.append(score)


    scores = np.array(scores)
    thres = np.quantile(scores, q=ratio, axis=0, keepdims=True)
    first_mask = (scores >= thres).sum(axis=1) >= 2
    rf_model = RandomForestClassifier(n_estimators=30, random_state=0)
    rf_model.fit(features, first_mask)
    for i, mask in enumerate(first_mask):
        if mask:
            suggest_samples.append(benchmarks[i])
    
    second_space = []
    features = []
    # subnets = [] # TODO
    total_num = int(20*256/(1-ratio)) // 2 # search space size (about 20*256 subnet to eval) 1/10不到的评估
    for i in range(total_num):
        while True:
            subnet_cfg = supernet.sample_active_subnet()
            subnet = supernet.get_active_subnet()  
            # compute flops
            # flops = compute_flops(subnet)
            flops = supernet.compute_active_subnet_flops()
            # subnet_cfg = supernet.sample_active_subnet()
            if flops*1e6 < cfg.BIGNAS.CONSTRAINT_FLOPS:
                break
            else:
                logger.info('subnet flops: {}, not satisfied'.format(flops))
        second_space.append({
                'subnet_cfg': subnet_cfg,
                'flops': flops,
                # 'score': mIoU
            })
        # subnets.append(subnet)
        features.append(subnet_cfg_2_feature(subnet_cfg))
    idxs = np.argsort(rf_model.predict_proba(features)[:,1])[-int(2*(1 - ratio) * (total_num)):]
    for idx in idxs:
        # subnet = subnets[idx]
        subnet_cfg = second_space[idx]['subnet_cfg']
        flops = second_space[idx]['flops']
        supernet.set_active_subnet(
            subnet_cfg['resolution'], subnet_cfg['width'], subnet_cfg['depth'], subnet_cfg['kernel_size'], subnet_cfg['expand_ratio']
        )
        subnet = supernet.get_active_subnet()  
        # subnet = build_dp_or_ddp(subnet, distributed, cfg)
        # bn_calibration(subnet, train_loader, cfg.post_bn_calibration_batch_num)
        score = get_zc_pe(subnet, None,train_loader, cfg.LOADER.NUM_CLASSES,  "cuda")
        # features.append(subnet_cfg_2_feature(subnet_cfg))
        # results = validate_subnet(subnet, distributed, val_loader, eval_kwargs)subnet_cfg
        if (score >= thres).sum(axis=1) < 1.5:
            continue
        # scores.append(results)
        
             
        benchmarks.append({
            'subnet_cfg': subnet_cfg,
            'flops': flops,
            'score': score
        })
        logger.info("[{}/{}] flops:{} score:{}".format(
            len(suggest_samples), total_num, flops, score
        ))
        # logger.info('second PE: {}th subnet, score: {} flops: {} subnet_cfg:{}'.format(i, score, flops, subnet_cfg))   
        suggest_samples.append(benchmarks[-1])
    
    # with open('./flops-{}.pkl'.format(cfg.BIGNAS.CONSTRAINT_FLOPS//1e6), 'wb') as f:
    #     pickle.dump(suggest_samples, f)
    ed = time.time()
    logger.info('end time: {}'.format(ed))
    logger.info('cost time: {}'.format(ed-st))
    with open(
        os.path.join(cfg.OUT_DIR,'flops-{}.pkl'.format(
            cfg.BIGNAS.CONSTRAINT_FLOPS//1e6)
                     ), 'wb') as f:
        pickle.dump(suggest_samples, f)
    
            
    # # Phase 1. coarse search
    # for k,subnet_cfg in enumerate(all_subnets):
    #     supernet.set_active_subnet(
    #         subnet_cfg['resolution'],
    #         subnet_cfg['width'],
    #         subnet_cfg['depth'],
    #         subnet_cfg['kernel_size'],
    #         subnet_cfg['expand_ratio'],
    #     )
    #     subnet = supernet.get_active_subnet().cuda()
        
    #     # Validate
    #     top1_err, top5_err = validate(subnet, train_loader, valid_loader, test_meter)
    #     flops = supernet.compute_active_subnet_flops()

    #     logger.info("[{}/{}] flops:{} top1_err:{} top5_err:{}".format(
    #         k+1, len(all_subnets), flops, top1_err, top5_err
    #     ))

    #     benchmarks.append({
    #         'subnet_cfg': subnet_cfg,
    #         'flops': flops,
    #         'top1_err': top1_err,
    #         'top5_err': top5_err
    #     })

    # # Phase 2. fine-grained search
    # try:
    #     best_subnet_info = list(filter(
    #         lambda k: k['flops'] < cfg.BIGNAS.CONSTRAINT_FLOPS,
    #         sorted(benchmarks, key=lambda d: d['top1_err'])))[0]
    #     best_subnet_cfg = best_subnet_info['subnet_cfg']
    #     best_subnet_top1 = best_subnet_info['top1_err']
    # except IndexError:
    #     logger.info("Cannot find subnets under {} FLOPs".format(cfg.BIGNAS.CONSTRAINT_FLOPS))
    #     exit(1)
    
    # for mutate_epoch in range(cfg.BIGNAS.NUM_MUTATE):
    #     new_subnet_cfg = supernet.mutate_and_reset(best_subnet_cfg)
    #     prev_cfgs = [i['subnet_cfg'] for i in benchmarks]
    #     if new_subnet_cfg in prev_cfgs:
    #         continue
        
    #     subnet = supernet.get_active_subnet().cuda()
    #     # Validate
    #     top1_err, top5_err = validate(subnet, train_loader, valid_loader, test_meter)
    #     flops = supernet.compute_active_subnet_flops()
        
    #     logger.info("[{}/{}] flops:{} top1_err:{} top5_err:{}".format(
    #         mutate_epoch+1, cfg.BIGNAS.NUM_MUTATE, flops, top1_err, top5_err
    #     ))

    #     benchmarks.append({
    #         'subnet_cfg': subnet_cfg,
    #         'flops': flops,
    #         'top1_err': top1_err,
    #         'top5_err': top5_err
    #     })
        
    #     if flops < cfg.BIGNAS.CONSTRAINT_FLOPS and top1_err < best_subnet_top1:
    #         best_subnet_cfg = new_subnet_cfg
    #         best_subnet_top1 = top1_err
    
    # # Final best architecture
    # logger.info("="*20 + "\nMutate Finished.")
    # logger.info("Best Architecture:\n{}\n Best top1_err:{}".format(
    #     best_subnet_cfg, best_subnet_top1
    # ))


@torch.no_grad()
def validate(subnet, train_loader, valid_loader, test_meter):
    # BN calibration
    subnet.eval()
    logger.info("Calibrating BN running statistics.")
    subnet.reset_running_stats_for_calibration()
    for cur_iter, (inputs, _) in enumerate(train_loader):
        if cur_iter >= cfg.BIGNAS.POST_BN_CALIBRATION_BATCH_NUM:
            break
        inputs = inputs.cuda()
        subnet(inputs)      # forward only

    top1_err, top5_err = test_epoch(subnet, valid_loader, test_meter)
    return top1_err, top5_err


def test_epoch(subnet, test_loader, test_meter):
    subnet.eval()
    test_meter.reset(True)
    test_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        preds = subnet(inputs)
        top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
        top1_err, top5_err = top1_err.item(), top5_err.item()

        test_meter.iter_toc()
        test_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        test_meter.log_iter_stats(0, cur_iter)
        test_meter.iter_tic()
    top1_err = test_meter.mb_top1_err.get_win_avg()
    top5_err = test_meter.mb_top5_err.get_win_avg()
    # self.writer.add_scalar('val/top1_error', test_meter.mb_top1_err.get_win_avg(), cur_epoch)
    # self.writer.add_scalar('val/top5_error', test_meter.mb_top5_err.get_win_avg(), cur_epoch)
    # Log epoch stats
    test_meter.log_epoch_stats(0)
    # test_meter.reset()
    return top1_err, top5_err

def build_dp_or_ddp(subnet, distributed, cfg):
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        subnet = revert_sync_batchnorm(subnet)
        subnet = build_dp(subnet, cfg.device, device_ids=cfg.gpu_ids)
    else:
        raise NotImplementedError
        # subnet = build_ddp(
        #     subnet, cfg.device, device_ids=[int(os.environ['LOCAL_RANK'])],
        #     broadcast_buffers=False)
    return subnet

def subnet_cfg_2_feature(subnet_cfg):
    cfg_all = []
    for k in sorted(subnet_cfg.keys()):
        if isinstance(subnet_cfg[k], list):
            cfg_all.extend(subnet_cfg[k])
        else:
            cfg_all.append(subnet_cfg[k])
    return cfg_all

if __name__ == "__main__":
    main()
