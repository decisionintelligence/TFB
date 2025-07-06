import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import copy


plt.switch_backend("agg")


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "type3":
        lr_adjust = {
            epoch: args.lr if epoch < 3 else args.lr * (0.9 ** ((epoch - 3) // 1))
        }
    elif args.lradj == "constant":
        lr_adjust = {epoch: args.lr}
    elif args.lradj == "3":
        lr_adjust = {epoch: args.lr if epoch < 10 else args.lr * 0.1}
    elif args.lradj == "4":
        lr_adjust = {epoch: args.lr if epoch < 15 else args.lr * 0.1}
    elif args.lradj == "5":
        lr_adjust = {epoch: args.lr if epoch < 25 else args.lr * 0.1}
    elif args.lradj == "6":
        lr_adjust = {epoch: args.lr if epoch < 5 else args.lr * 0.1}
    elif args.lradj == "TST":
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        if printout:
            print("Updating learning rate to {}".format(lr))
