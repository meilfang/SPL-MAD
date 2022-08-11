import numpy as np
import os
import csv
import copy
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict

from dataset import TrainDataset
from utils import  AvgrageMeter
import network

def run_training(log_file, args):
    # load data
    train_dataset = TrainDataset(csv_file=args.train_csv, input_shape=args.input_shape)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=32, pin_memory=True)
    print('Number of training images:', len(train_loader.dataset))

    # create folder for saving checkpoints
    checkpoint_save_dir = os.path.join('checkpoints', args.prefix)
    print('Checkpoint saving directory:', checkpoint_save_dir)
    os.makedirs(checkpoint_save_dir, exist_ok=True)

    # initial model, optimizer, and loss
    model = torch.nn.DataParallel(network.AEMAD(in_channels=3, features_root=args.features_root))
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    mse_criterion = torch.nn.MSELoss(reduction='none').cuda()
    scaler = GradScaler()

    num_step = 0
    for epoch in range(1, args.max_epoch+1):
        print('-------------- train ------------------------')
        model.train()
        loss_total = AvgrageMeter()
        progress_bar = tqdm(train_loader)
        for i, data in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))
            raw = data["images"].cuda()
            weights = torch.ones(raw.size(0)).cuda().type(torch.cuda.FloatTensor)
            lambda1_sp, lambda2_sp, lambda3_sp = 0, 0, 0
            easy_idx = []
            median_idx = []
            avg, std = 0, 0

            if epoch >= args.warmup_epoch:
                # SPL process
                model.eval()
                with torch.no_grad():
                    _, output_raw = model(raw)

                    scores = mse_criterion(output_raw, raw)
                    scores = torch.sum(torch.sum(torch.sum(scores, dim=3), dim=2), dim=1)

                    weights = torch.ones_like(scores)
                    avg, std = torch.mean(scores), torch.std(scores)
                    lambda_max = avg - std
                    lambda_spl = avg - (args.init_std_range - num_step * args.shrink_rate) * std
                    lambda_spl = max(min(lambda_spl, lambda1_sp), 1e-1) # equation 8
                    num_step += 1
                    # equation 7
                    easy_idx = torch.where(scores <= lambda_spl)[0]
                    hard_idx = torch.where(scores > lambda_spl)[0]
                    weights[hard_idx] = 1.0 - (lambda_spl / scores[hard_idx])
                    if len(easy_idx) > 0:
                        weights[easy_idx] = 0.0
                model.train()

            _, output_raw = model(raw)

            loss = mse_criterion(output_raw, raw)
            loss = loss.sum(dim=3).sum(dim=2).sum(dim=1)
            loss = loss @ weights / raw.size(0) / raw.size(1) / raw.size(2) / raw.size(3)
            loss_total.update(loss.data, raw.shape[0])

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            progress_bar.set_postfix(
                loss ='%.5f' % (loss_total.avg)
            )

        torch.save( model.state_dict(), os.path.join(checkpoint_save_dir, '{}.pth'.format(epoch)))
        tqdm.write('Epoch: %d, Train: loss_total= %.4f \n' % (epoch, loss_total.avg))
        log_file.write('Epoch: %d, Train: loss_total= %.4f \n' % (epoch, loss_total.avg))
        log_file.flush()

if __name__ == "__main__":


    torch.cuda.empty_cache()
    cudnn.benchmark = True

    if torch.cuda.is_available():
        print('GPU is available')
        torch.cuda.manual_seed(0)
    else:
        print('GPU is not available')
        torch.manual_seed(0)

    import argparse
    parser = argparse.ArgumentParser(description='SPL MAD')

    parser.add_argument("--train_csv", required=True, type=str, help="path of training data csv file")
    parser.add_argument("--prefix", default='training', type=str, help="prefix for log file")

    parser.add_argument("--input_shape", default=(224, 224), type=tuple, help="model input shape")
    parser.add_argument("--features_root", default=64, type=int, help="features root for network")
    parser.add_argument("--lr", default=0.0001, type=float, help="initial learning rate")
    parser.add_argument("--max_epoch", default=25, type=int, help="maximum epochs")
    parser.add_argument("--warmup_epoch", default=5, type=int, help="warm epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="train batch size")

    parser.add_argument("--shrink_rate", default=0.005, type=float, help="schrink rate")
    parser.add_argument("--init_std_range", default=4., type=float, help="init std range")

    args = parser.parse_args()

    logging_filename = os.path.join('logs',  '{}.txt'.format(args.prefix))
    if not os.path.isdir('logs'):
        os.makedirs('logs')
    log_file = open(logging_filename, 'a')
    log_file.write(f"feature root: {args.features_root}, lr: {args.lr}, shrink_rate: {args.shrink_rate}, init_std_range: {args.init_std_range} \n")
    log_file.flush()

    run_training(log_file=log_file, args=args)
