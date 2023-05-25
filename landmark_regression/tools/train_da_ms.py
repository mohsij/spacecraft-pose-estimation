# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train_da, train_da_ms
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

from utils.transforms import RandomBloom
from utils.transforms import RandomHaze
from utils.transforms import RandomFlares
from utils.transforms import RandomStreaks
from utils.transforms import ToNumpy
from utils.transforms import RandomNoise

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

    # philly
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
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    # # # create discriminator
    # import torchvision
    # discriminator = torchvision.models.resnet34()
    # discriminator.conv1 = torch.nn.Conv2d(cfg.MODEL.NUM_JOINTS, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
    # discriminator.fc =  torch.nn.Linear(in_features=512, out_features=2, bias=True)
    discriminator = models.multi_scale_discriminator.resnet34_ms(num_classes=2, in_channels=cfg.MODEL.NUM_JOINTS*4)

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # dump_input = torch.rand(
    #     (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    # )
    # writer_dict['writer'].add_graph(model, (dump_input, ))

    # logger.info(get_model_summary(model, dump_input))

    model = torch.nn.DataParallel(model, device_ids=list(cfg.GPUS)).to(f'cuda:{cfg.GPUS[0]}')
    discriminator= torch.nn.DataParallel(discriminator, device_ids=cfg.GPUS).to(f'cuda:{cfg.GPUS[0]}')
    # discriminator.to('cuda:0')

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT)
    criterion2 = torch.nn.CrossEntropyLoss()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # pre_normalize = transforms.Normalize(
    #     mean=[0., 0., 0.], std=[1., 1., 1.]
    # )

    if cfg.DATASET.DATASET_ADVERSARIAL == 'lightbox':
        training_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.1),
            transforms.ToTensor(),
            RandomNoise(mean_min=0.03, mean_max=0.25, std_min=0.01, std_max=0.1), 
            normalize,
            transforms.RandomErasing(p=0.4, scale=(0.05, 0.2), ratio=(0.25, 4), value=(-0.485/0.229, -0.456/0.224, -0.406/0.225)),
        ])
    elif cfg.DATASET.DATASET_ADVERSARIAL == 'sunlamp':
        training_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.1),
            transforms.ToTensor(),
            RandomNoise(mean_min=0.01, mean_max=0.1, std_min=0.03, std_max=0.05), 
            normalize,
            transforms.RandomErasing(p=0.3, scale=(0.05, 0.2), ratio=(0.3, 3.3), value=(-0.485/0.229, -0.456/0.224, -0.406/0.225)),
            transforms.RandomErasing(p=0.6, scale=(0.05, 0.18), ratio=(0.5, 2.0), 
                value=((1-0.485)/0.229, (1-0.456)/0.224, (1-0.406)/0.225)),
        ])
    else:
        assert False, 'cfg.DATASET.DATASET_ADVERSARIAL value invalid.'

    numpy_transform = None
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATA_DIR, cfg.DATASET.TRAIN_SET, True,
        training_transforms,
        numpy_transform,
        cfg.MODEL.MULTI_SCALE_TARGET
    )
    train_dataset2 = eval('dataset.'+cfg.DATASET.DATASET_ADVERSARIAL)(
        cfg, cfg.DATASET.ROOT_ADVERSARIAL, cfg.DATA_DIR_ADVERSARIAL, cfg.DATASET.TRAIN_SET_ADVERSARIAL, True,
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=1, hue=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_dataset_syn = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATA_DIR, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET_ADVERSARIAL)(
        cfg, cfg.DATASET.ROOT_ADVERSARIAL, cfg.DATA_DIR_ADVERSARIAL, cfg.DATASET.TRAIN_SET_ADVERSARIAL, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    train_loader2 = torch.utils.data.DataLoader(
        train_dataset2,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU_ADVERSARIAL_SET*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader_syn = torch.utils.data.DataLoader(
        valid_dataset_syn,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    optimizer2 = get_optimizer(cfg, discriminator)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )
    lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer2, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):

        # train for one epoch
        if cfg.MODEL.MULTI_SCALE_TARGET:
            train_da_ms(cfg, train_loader, train_loader2, model, discriminator, criterion, criterion2, optimizer, optimizer2, 
                epoch, final_output_dir, tb_log_dir, writer_dict)
        else:
            train_da(cfg, train_loader, train_loader2, model, discriminator, criterion, criterion2, optimizer, optimizer2, 
                epoch, final_output_dir, tb_log_dir, writer_dict)

        lr_scheduler.step()
        lr_scheduler2.step()

        # evaluate on validation set
        perf_indicator = 0
        if epoch % 5 == 0:
            _ = validate(
                cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, pred_file_name='pred_val_real',
                writer_dict=writer_dict
            )

            perf_indicator = validate(
                cfg, valid_loader_syn, valid_dataset_syn, model, criterion,
                final_output_dir, tb_log_dir, pred_file_name='pred_val_syn',
                writer_dict=writer_dict
            )

        best_model = True

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'discriminator': discriminator.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
            'optimizer2': optimizer2.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
