import matplotlib.pyplot as plt

plt.switch_backend("agg")


def adjust_learning_rate(optimizer, epoch, args):
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

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))
