import numpy as np
import sklearn
#from sklearn import metrics
from sklearn.metrics import roc_curve, auc

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0/batch_size))
        return res

def get_eer_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1=tpr+fpr-1.0

    right_index = np.nanargmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    eer = fpr[right_index]

    return eer, best_th, right_index

def get_performance(prediction_scores, gt_labels, pos_label=1, verbose=True):

    data = [{'map_score': score, 'label': label} for score, label in zip(prediction_scores, gt_labels)]
    fpr, tpr, threshold = roc_curve(gt_labels, prediction_scores, pos_label=pos_label)

    eer, eer_th, _ = get_eer_threhold(fpr, tpr, threshold)
    #test_auc = auc(fpr, tpr)

    if verbose is True:
        print(f'EER is {eer}, threshold is {eer_th}')

    return eer, eer_th
