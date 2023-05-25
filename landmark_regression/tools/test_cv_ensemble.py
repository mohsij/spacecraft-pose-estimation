# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate_cv
from utils.utils import create_logger

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model1, model2, model3, model4, model5, model6 = [None] * 6
    model_file_paths = [
        cfg.TEST.MODEL_FILE,cfg.TEST.MODEL_FILE2,cfg.TEST.MODEL_FILE3,
        cfg.TEST.MODEL_FILE4, cfg.TEST.MODEL_FILE5, cfg.TEST.MODEL_FILE6]
    
    cv_models = []
    
    for model_file_path in model_file_paths:
        if os.path.isfile(model_file_path):
            cv_models.append(eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
                cfg, is_train=False
            ))
            cv_models[-1].load_state_dict(torch.load(model_file_path))
            cv_models[-1] = torch.nn.DataParallel(cv_models[-1], device_ids=cfg.GPUS).cuda()

    # model1 = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    #     cfg, is_train=False
    # )
    # model1.load_state_dict(torch.load('output/PEdataset/hrnet_cms/sun_hpc_001/final_state.pth'), strict=False)
    # model1 = torch.nn.DataParallel(model1, device_ids=cfg.GPUS).cuda()
    
    # model2 = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    #     cfg, is_train=False
    # )
    # model2.load_state_dict(torch.load('output/PEdataset/hrnet_cms/sun_hpc_002/final_state.pth'), strict=False)
    # model2 = torch.nn.DataParallel(model2, device_ids=cfg.GPUS).cuda()

    # model3 = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    #     cfg, is_train=False
    # )
    # model3.load_state_dict(torch.load('output/PEdataset/hrnet_cms/sun_hpc_003/final_state.pth'), strict=False)
    # model3 = torch.nn.DataParallel(model3, device_ids=cfg.GPUS).cuda()

    # model4 = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    #     cfg, is_train=False
    # )
    # model4.load_state_dict(torch.load('output/PEdataset/hrnet_cms/sun_hpc_004/final_state.pth'), strict=False)
    # model4 = torch.nn.DataParallel(model4, device_ids=cfg.GPUS).cuda()

    # model5 = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    #     cfg, is_train=False
    # )
    # model5.load_state_dict(torch.load('output/PEdataset/hrnet_cms/sun_hpc_005/final_state.pth'), strict=False)
    # model5 = torch.nn.DataParallel(model5, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATA_DIR, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    validate_cv(cfg, valid_loader, valid_dataset, cv_models, criterion,
             final_output_dir, tb_log_dir, 'pred_real')


if __name__ == '__main__':
    main()
