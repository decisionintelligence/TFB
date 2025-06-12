import copy

import numpy as np
import torch
import matplotlib.pyplot as plt
import time

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.lr * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.lr if epoch < 3 else args.lr * (0.9 ** ((epoch - 3) // 1))}
    
    # Sigmoid learning rate decay
    elif args.lradj == 'sigmoid':
        k = 0.5 # logistic growth rate
        s = 10  # decreasing curve smoothing rate
        w = 10  # warm-up coefficient
        lr_adjust = {epoch: args.lr / (1 + np.exp(-k * (epoch - w))) - args.lr / (1 + np.exp(-k/s * (epoch - w*s)))}
    
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.lr}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.lr if epoch < 10 else args.lr*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.lr if epoch < 15 else args.lr*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.lr if epoch < 25 else args.lr*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.lr if epoch < 5 else args.lr*0.1}  
 
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        print(
            f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
        )
        self.check_point = copy.deepcopy(model.state_dict())
        # self.check_point = model.state_dict()
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')