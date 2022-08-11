import numpy as np
import os
import csv
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from collections import defaultdict

from dataset import TestDataset
from utils import get_performance
import network

def run_test(args):
    test_dataset = TestDataset(csv_file=args.test_csv, input_shape=args.input_shape)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=64, pin_memory=True)

    print('Number of test images:', len(test_loader.dataset))
    model =  torch.nn.DataParallel(network.AEMAD(in_channels=3, features_root=args.features_root))
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    model.cuda()
    model.eval()

    mse_criterion = torch.nn.MSELoss(reduction='none').cuda()

    test_scores, gt_labels, test_scores_dict = [], [],[]

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            raw, labels, img_ids = data['images'].cuda(), data['labels'], data['img_path']
            _, output_raw = model(raw)

            scores = mse_criterion(output_raw, raw).cpu().data.numpy()
            scores = np.sum(np.sum(np.sum(scores, axis=3), axis=2), axis=1)
            test_scores.extend(scores)
            gt_labels.extend((1 - labels.data.numpy()))
            for j in range(labels.shape[0]):
                l = 'attack' if labels[j].detach().numpy() == 1 else 'bonafide'
                test_scores_dict.append({'img_path':img_ids[j], 'labels':l, 'prediction_score':float(scores[j])})

    eer, eer_th = get_performance(test_scores, gt_labels)
    print('Test EER:', eer*100)

    with open(args.output_path, mode='w') as csv_file:
        fieldnames = ['img_path', 'labels', 'prediction_score']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for d in test_scores_dict:
            writer.writerow(d)
        print('Prediction scores write done in', args.output_path)

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

    parser.add_argument("--test_csv", required=True, type=str, help="path of data directory including csv files")
    parser.add_argument("--model_path", required=True, type=str, help="model path")
    parser.add_argument("--output_path", default='test.csv', type=str, help="path for output prediction scores")

    parser.add_argument("--input_shape", default=(224, 224), type=tuple, help="model input shape")
    parser.add_argument("--features_root", default=64, type=int, help="feature root")
    parser.add_argument("--batch_size", default=32, type=int, help="test batch size")

    args = parser.parse_args()
    run_test(args=args)
