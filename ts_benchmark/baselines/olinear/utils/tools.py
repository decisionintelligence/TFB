import os
import shutil

# import functorch.dim
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch.nn.functional as F

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import time
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import random
import seaborn as sns
from einops import rearrange

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, scheduler=None, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    lr_adjust = {}
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == 'TST':
        assert scheduler is not None
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj in ['cosine', 'card']:
        # warmup-cosine
        min_lr = 0
        warmup_epochs = 0
        lr = (min_lr + (args.learning_rate - min_lr) * 0.5 *
              (1. + math.cos(math.pi * (epoch - warmup_epochs) / (args.train_epochs - warmup_epochs))))
        lr_adjust = {epoch: lr}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, save_every_epoch=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_every_epoch = save_every_epoch

    def __call__(self, val_loss, model, path, epoch=None):
        if np.isnan(val_loss):
            self.early_stop = True
            return
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, epoch)
            self.counter = 0
            self.early_stop = False

    def save_checkpoint(self, val_loss, model, path, epoch=None):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        file_path = os.path.join(path, 'checkpoint.pth')
        torch.save(model.state_dict(), file_path)

        # output checkpoint size
        file_size = os.path.getsize(file_path)
        file_size = convert_size(file_size)
        print(f"The size of checkpoint is {file_size}.")

        # delete txt files
        delete_txt_files_in_folder(path)
        file_path = os.path.join(path, f'Epoch_{epoch}.txt')
        # Create the file with the name "epoch_{i}.txt"
        with open(file_path, 'w') as file:
            file.write(f'Current Epoch: {epoch}')
        if self.save_every_epoch:
            if epoch:
                shutil.copy(os.path.join(path, 'checkpoint.pth'), os.path.join(path, f'checkpoint_epoch_{epoch:d}'
                                                                                     f'_val_loss_{val_loss:.5f}.pth'))
            else:
                shutil.copy(os.path.join(path, 'checkpoint.pth'), os.path.join(path, f'checkpoint_val_loss_'
                                                                                     f'{val_loss:.5f}.pth'))
        self.val_loss_min = val_loss


def convert_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f}PB"


def delete_txt_files_in_folder(path):
    # 遍历路径下的所有文件，并删除以 .txt 结尾的文件
    [os.remove(os.path.join(path, f)) for f in os.listdir(path) if f.endswith('.txt')]


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


def write_into_xls(excel_name, mat, columns=None):
    file_extension = os.path.splitext(excel_name)[1]

    if file_extension != ".xls" and file_extension != ".xlsx":
        raise ValueError('excel_name is not right in write_into_xls')

    folder_name = os.path.dirname(excel_name)
    if folder_name:
        os.makedirs(folder_name, exist_ok=True)

    if isinstance(mat, np.ndarray) and mat.ndim > 2:
        mat = mat.reshape(-1, mat.shape[-1])
        mat = mat[:1000]
    if columns is not None:
        dataframe = pd.DataFrame(mat, columns=columns)
    else:
        dataframe = pd.DataFrame(mat)
    # print(dataframe)
    # print(excel_name)
    dataframe.to_excel(excel_name, index=False)


def visual(true, preds=None, name='./pic/test.pdf', imp=False):
    """
    Results visualization
    """
    folder_name = os.path.dirname(name)
    if folder_name:
        os.makedirs(folder_name, exist_ok=True)
    label2 = 'Imputation' if imp else 'Prediction'

    if not isinstance(true, np.ndarray):
        true = true.numpy()
    if not isinstance(preds, np.ndarray):
        preds = preds.numpy()

    plt.figure()
    plt.plot(true, label='Ground Truth', linestyle='--', linewidth=2)
    if preds is not None:
        plt.plot(preds, label=label2, linewidth=2)
    plt.legend()
    plt.grid(linestyle=':', color='lightgray')
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def find_most_recently_modified_subfolder(base_dir, file_name='checkpoint.pth', contain_str=''):
    most_recent_time = 0
    most_recent_folder = None
    most_recent_subfolder = None

    if isinstance(contain_str, list):
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and
                   os.path.isfile(os.path.join(base_dir, d, file_name)) and all([cstr in d for cstr in contain_str])]
    else:
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and
                   os.path.isfile(os.path.join(base_dir, d, file_name)) and contain_str in d]

    # if not subdirs:
    #     raise ValueError('No such folder found!!! ')

    for subdir in subdirs:
        folder_path = os.path.join(base_dir, subdir)
        current_time = os.path.getmtime(folder_path)

        if current_time > most_recent_time:
            most_recent_time = current_time
            most_recent_folder = folder_path
            most_recent_subfolder = subdir

    return most_recent_folder, most_recent_subfolder


def compare_prefix_before_third_underscore(str1, str2, num=3):
    if str1 is None or str2 is None:
        return False
    prefix1 = ''.join(str1.split("_", num)[:num])
    prefix2 = ''.join(str2.split("_", num)[:num])

    are_prefixes_equal = prefix1 == prefix2

    return are_prefixes_equal


def compute_gradient_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        # elif param.requires_grad and param.grad is None:
        #     print('\t param.grad is None...')
    total_norm = total_norm ** 0.5
    return total_norm


def is_not_empty_or_nan(a):
    if isinstance(a, list):
        if not a:
            return False
        if any(isinstance(i, (float, np.float32, np.float64)) and np.isnan(i) for i in a):
            return False
    elif isinstance(a, torch.Tensor):
        if a.numel() == 0:
            return False
        if torch.isnan(a).any():
            return False
    else:
        if isinstance(a, (float, np.float32, np.float64)) and np.isnan(a):
            return False

    return True


def compute_uncert(mask, patch_len=16, temp_stride=8, temporal=True, channel_num=7, softmax=0, tau=1.0, tau2=0.5,
                   Patch_CI=True, eps=1e-5):
    # mask: [b,t,n]
    # return [b,token_num]
    mask = ~mask
    assert channel_num == mask.shape[-1]
    if temporal:
        # [b,t,n] --> [b,token_num,n,patch_len] --> [b,token_num,n]
        token_uncer_weight = mask.unfold(dimension=1, size=patch_len, step=temp_stride).sum(dim=-1)
        if Patch_CI:
            # [b*n, token_num]
            token_uncer_weight = token_uncer_weight.sum(dim=-1).repeat_interleave(repeats=channel_num, dim=0)
        else:
            # [b, token_num]
            token_uncer_weight = token_uncer_weight.sum(dim=-1)
    else:
        token_uncer_weight = mask.sum(dim=1)
    # float
    token_uncer_weight = token_uncer_weight.to(dtype=torch.float)

    tau = F.softplus(torch.tensor(tau)) if tau <= 0 else tau
    tau2 = F.softplus(torch.tensor(tau2)) if tau2 <= 0 else tau2

    if softmax > 0:
        # softmax
        token_uncer_weight = F.softmax(token_uncer_weight / tau, dim=-1)
    elif softmax == 0:
        # pow
        token_uncer_weight = F.normalize(token_uncer_weight.pow(tau2), p=1, dim=-1)
    else:
        # F.normalize
        token_uncer_weight = F.normalize(token_uncer_weight, p=1, dim=-1)

    return token_uncer_weight.clamp(min=eps)


def hier_half_token_weight(token_weight, ratio=2):
    if token_weight is None:
        return None
    # temp_token_weight_time: [b, token_num]
    B, N = token_weight.shape
    if N % ratio != 0:
        tmp = ratio - N % ratio
        token_weight = torch.cat([token_weight, token_weight[:, -tmp:]], dim=-1)
    token_weight = token_weight.reshape(B, -1, ratio).sum(dim=-1)
    return token_weight


def cosine_distance(tensor1, tensor2, keepdims=False):
    assert tensor1.shape == tensor2.shape, "Both tensors must have the same shape in cosine_distance"
    # F.cosine_similarity
    cosine_sim = F.cosine_similarity(tensor1, tensor2, dim=-1)
    # 1 - cosine_sim
    cosine_dist = 1 - cosine_sim

    if keepdims:
        return cosine_dist.unsqueeze(-1)
    else:
        return cosine_dist


def euclidean_distance(tensor1, tensor2, keepdims=False):
    assert tensor1.shape == tensor2.shape, "Both tensors must have the same shape in euclidean_distance"
    diff = tensor1 - tensor2
    squared_diff = diff ** 2
    euclidean_dist = torch.sqrt(squared_diff.sum(-1))
    if keepdims:
        return euclidean_dist.unsqueeze(-1)
    else:
        return euclidean_dist


def get_eval_feat(layer, tensor):
    # tensor: [b, l, n]
    # feat: [b, d_model]

    # [b,l,n] --> [n,b,d_model] --> [b,d_model]
    feat = layer(tensor.permute(2, 0, 1)).sum(dim=0)
    return feat


def undo_unfold(inp, length, stride, fft_flag=False):
    # [b,n,stride,period] --> [b,l,n]
    B, N, num, period = inp.shape
    if fft_flag:
        assert num == length // stride, (f'num:{num}, length:{length}, stride:{stride}. inp.shape: {inp.shape}. '
                                         f'Please check the inputs of undo_unfold().')
    else:
        assert num == (length - period) // stride + 1, 'Please check the inputs of undo_unfold().'

    if stride == period or fft_flag:
        reconstructed = inp.flatten(start_dim=2)
        return reconstructed.transpose(-1, -2)

    reconstructed = torch.zeros(B, N, length, device=inp.device)
    count_overlap = torch.zeros_like(reconstructed)

    for i in range(num):
        start = i * stride
        end = start + period
        reconstructed[:, :, start:end] += inp[:, :, i, :]
        count_overlap[:, :, start:end] += 1

    # average
    mask = count_overlap > 0
    reconstructed[mask] /= count_overlap[mask]

    return reconstructed.transpose(-1, -2)


def send_email(subject='Python Notification', body='Program complete!', to_email=r'mail@mail.com',
               from_email=r'mail@mail.com', password='xxxxxxxxxx', mail_host='xxxx.com',
               mail_port=465):
    # Create the message

    message = MIMEMultipart()
    message['From'] = from_email
    message['To'] = to_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain', 'utf-8'))  # utf-8 for compatibility

    try:
        # Connect to the SMTP server using SSL (port 465)
        with smtplib.SMTP_SSL(mail_host, mail_port) as server:
            # Login and send the email
            server.login(from_email, password)
            server.send_message(message)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")


def create_sub_diagonal_matrix(n, value=1, offset=0):
    if abs(offset) >= n:
        return None
    vec = torch.ones(n - abs(offset)) * value
    return torch.diag(vec, diagonal=offset)


def plot_mat(mat, str_cat='series_2D', str0='tmp', save_folder='./results'):
    if not isinstance(mat, np.ndarray):
        mat = mat.detach().cpu().numpy()
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    # fig, axs = plt.subplots(1, 1)
    # plt.imshow(mat, cmap='viridis', interpolation='nearest', vmin=0.0, vmax=1.0)  # viridis  hot
    # plt.colorbar()

    plt.figure(figsize=(8, 8))
    sns.heatmap(mat, annot=False, cmap='coolwarm', square=True, cbar=True)
    plt.xticks([])  # 去除x轴刻度
    plt.yticks([])  # 去除y轴刻度
    timestamp = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    plt.savefig(os.path.join(save_folder, f'{str_cat}_{str0}-{timestamp}.pdf'))
    plt.show()
    # save to excel
    excel_name = os.path.join(save_folder, f'{str_cat}_{str0}-{timestamp}.xlsx')
    write_into_xls(excel_name, mat)
    # save to npy
    np.save(os.path.join(save_folder, f'{str_cat}_{str0}-{timestamp}.npy'), mat)


def create_sin_pos_embed(max_len, d_model):
    pe = torch.zeros(max_len, d_model).float()

    position = torch.arange(0, max_len).float().unsqueeze(1)
    div_term = (torch.arange(0, d_model, 2).float()
                * -(math.log(10000.0) / d_model)).exp()

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)

    #  [1, max_len, d_model]
    return pe


def var2tuple2(x, num=2):
    num = int(num)
    if isinstance(x, tuple):
        if len(x) == num:
            return x
        elif len(x) > num:
            return x[:num]
        else:
            return x + (x[-1],) * (num - len(x))
    return (x,) * num


def create_swin_relative_index(window_size):
    # check
    window_size = var2tuple2(window_size)
    assert all(i > 0 for i in window_size)

    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    return relative_position_index


def get_relative_coords_table(window_size, h_times=torch.tensor(2.0), ws_scalar=torch.tensor(5.0),
                              ws_scalar2=None, pow_para=None, pow_mode=False):
    assert isinstance(window_size, tuple) and len(window_size) == 2 and ws_scalar > 0, f"ws_scalar:{ws_scalar}"
    # get relative_coords_table
    relative_coords_h = torch.arange(-(window_size[0] - 1), window_size[0], dtype=torch.float32)
    relative_coords_w = torch.arange(-(window_size[1] - 1), window_size[1], dtype=torch.float32)
    relative_coords_table = torch.stack(
        torch.meshgrid([relative_coords_h,
                        relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
    if torch.is_tensor(h_times):
        relative_coords_table = relative_coords_table.to(h_times.device)

    if window_size[0] > 1:
        relative_coords_table[:, :, :, 0] /= (window_size[0] - 1)
    if window_size[1] > 1:
        relative_coords_table[:, :, :, 1] /= (window_size[1] - 1)

    if pow_mode:
        relative_coords_table *= ws_scalar
        relative_coords_table = torch.sign(relative_coords_table) * torch.exp(
            torch.abs(relative_coords_table) - 1) / (torch.exp(ws_scalar) - 1)

        if torch.any(torch.isnan(relative_coords_table)):
            print('\t relative_coords_table is nan. please check...', ws_scalar)
            print("\t relative_coords_table.shape: ", relative_coords_table.shape)
    else:
        # if ws_scalar2 is None:
        relative_coords_table *= ws_scalar  # normalize to -ws_scalar, ws_scalar
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1) / torch.log2(ws_scalar)  # log2  log1p
        # else:
        #     tmp = relative_coords_table[:, :, :, 1].clone()
        #     tmp = tmp * ws_scalar
        #     tmp2 = relative_coords_table[:, :, :, 0].clone()
        #     tmp2 = tmp2 * ws_scalar2
        #
        #     relative_coords_table[:, :, :, 1] = torch.sign(tmp) * torch.log2(
        #         torch.abs(tmp) + 1.0) / torch.log2(ws_scalar)
        #     relative_coords_table[:, :, :, 0] = torch.sign(tmp2) * torch.log2(
        #         torch.abs(tmp2) + 1.0) / torch.log2(ws_scalar2)

    relative_coords_table[:, :, :, 0] *= h_times
    return relative_coords_table


def create_swin_relative_index_1d(window_size, period=None):
    window_size = to_2tuple(window_size)
    period = period or window_size[1]
    if torch.is_tensor(period):
        rel_pos = (torch.arange(window_size[1]).view(1, -1).to(period.device) +
                   torch.arange(window_size[0]).view(-1, 1).to(period.device) * period).view(1, -1)
    else:
        rel_pos = (torch.arange(window_size[1]).view(1, -1) +
                   torch.arange(window_size[0]).view(-1, 1) * period).view(1, -1)
    rel_pos = rel_pos - rel_pos.transpose(-1, -2)
    return rel_pos


def norm_rel_pos_1d(relative_position_index, ws_scalar):
    if not torch.is_tensor(ws_scalar):
        ws_scalar = torch.tensor(ws_scalar)
    # one way
    relative_position_index_norm = (relative_position_index / torch.max(relative_position_index).item()
                                    * ws_scalar)  # normalize to -8, 8
    relative_position_index_norm = torch.sign(relative_position_index_norm) * torch.log2(
        torch.abs(relative_position_index_norm) + 1.0) / torch.log2(ws_scalar)

    # more simple
    # relative_position_index_norm = (torch.sign(relative_position_index) * torch.log1p(
    #     torch.abs(relative_position_index) / torch.max(relative_position_index).item() * ws_scalar)
    #                                 / torch.log1p(ws_scalar))
    return relative_position_index_norm


def compute_weights(alpha, length, stages=None, multiple_flag=True):
    assert alpha <= 0
    if alpha == 0:
        weights = torch.ones(length)
        return weights
    stage_num = 1
    rem = 0
    if stages is not None:
        # assert (length + 1) % stages == 0 or length % stages == 0
        stage_num = (length + 1) // stages
        rem = length + 1 - stage_num * stages

    weights = torch.tensor([i ** alpha for i in range(length + 1, 0, -1)])
    # weights2 = torch.tensor([i ** (alpha / 2) for i in range(length + 1, 0, -1)])

    # iTransformer
    if multiple_flag and stages is not None:
        # on SDA now
        slices = list(range(stage_num - 1, length, stage_num))
        if rem > 0:
            slices = [a + i + 1 if i < rem else a + rem for i, a in enumerate(slices)]
        weights[slices] = torch.minimum(weights[slices] * 1.5, weights[-2])
        # weights[slices] = weights2[slices]

    # remove the first element
    weights = weights[:length]

    return weights


def roll_without_cycle(x, shifts, dims):
    if not isinstance(shifts, (tuple, list)):
        shifts = (shifts,)
    if not isinstance(dims, (tuple, list)):
        dims = (dims,)

    assert len(shifts) == len(dims), "shifts and dims must have the same length"

    shifted_x = torch.roll(x, shifts, dims)

    for shift, dim in zip(shifts, dims):
        zeros_slices = [slice(None)] * x.ndim
        if shift == 0:
            continue
        if shift > 0:
            zeros_slices[dim] = slice(0, shift)
        else:
            zeros_slices[dim] = slice(shift, None)

        shifted_x[tuple(zeros_slices)] = 0

    return shifted_x


def forward_fill(x, mask):
    b, l, n = x.size()
    # x = x.clone()
    mask = mask.clone()

    padding_positions = (mask == 1).nonzero(as_tuple=True)

    for batch_index, length_index, feature_index in zip(*padding_positions):
        # search backwards
        for prev_length_index in range(length_index - 1, -1, -1):
            if mask[batch_index, prev_length_index, feature_index] == 0:
                x[batch_index, length_index, feature_index] = x[batch_index, prev_length_index, feature_index]
                mask[batch_index, length_index, feature_index] = 0
                break

    padding_positions = (mask == 1).nonzero(as_tuple=True)

    for batch_index, length_index, feature_index in zip(*padding_positions):
        # search forwards
        for prev_length_index in range(length_index + 1, l, 1):
            if mask[batch_index, prev_length_index, feature_index] == 0:
                x[batch_index, length_index, feature_index] = x[batch_index, prev_length_index, feature_index]
                mask[batch_index, length_index, feature_index] = 0
                break

    return x, mask


def closest_divisor(a, b):
    # a=10,b=3; --> return 2
    if a % b == 0:
        return b

    left = b - 1
    right = b + 1

    while left > 0:
        if a % left == 0:
            return left
        left -= 1

    while right <= a:
        if a % right == 0:
            return right
        right += 1

    return None


def adapt_win(seq_len, period):
    H, W = math.ceil(seq_len / period), period
    scalar = 5
    max_hw = 7
    if H <= W:
        w = min(W // 2, max_hw)
        h = min(scalar ** 2 // w, H, w)
    else:
        h = min(H // 2, max_hw)
        w = min(scalar ** 2 // h, W, h)
    return h, w


def cross_correlation_fft(x, tau=None, circular_shift=False, first_row_shift=0):
    # input: b,l,h,w;  output: [h, 2tau+1]
    b, l, m, n = x.shape
    if m == 1:
        return x, [first_row_shift, ]

    fft_size = n if circular_shift else 2 * n - 1

    tau = tau or fft_size // 2
    tau = min(tau, fft_size // 2)

    if first_row_shift != 0:
        x[:, :, 0, :] = torch.roll(x[:, :, 0, :], shifts=first_row_shift, dims=-1)

    if not circular_shift:
        x = F.pad(x, (0, fft_size - n))

    x_fft = torch.fft.fft(x, dim=-1)

    first_row_fft = x_fft[:, :, 0, :].unsqueeze(2)

    cross_corr_fft = x_fft * torch.conj(first_row_fft)

    cross_corr = torch.fft.ifft(cross_corr_fft, dim=-1).real

    n_middle = (n - 1) // 2 if circular_shift else n - 1
    cross_corr = torch.roll(cross_corr, shifts=n_middle, dims=-1)

    cross_corr = cross_corr[:, :, :, max(n_middle - tau, 0):n_middle + tau + 1]  # / n

    # mean; [m, 2tau+1]
    cross_corr = cross_corr.flatten(0, 1).mean(0)

    # max delay
    delay_vec = min(tau, n_middle) - cross_corr.max(-1)[1]
    delay_vec[0] = first_row_shift

    return cross_corr, delay_vec


def cyclic_shift_per_row(x, vec):
    b, h, w, c = x.shape
    assert len(vec) == h, f"len(vec){len(vec)} should be equal to h{h}..."

    for i in range(h):
        shift_amount = vec[i].item() if torch.is_tensor(vec) else vec[i]
        x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=shift_amount, dims=1)

    return x


def find_period_multiple_k_ori(x, k=1):
    """
    Find the period of the signal x, where the period is a multiple of k.
    x is expected to be of shape (B,T,C) where C is the number of channels and T is the length of each channel.
    """
    B, T, C = x.shape

    len_fft = math.ceil(T / k) * k
    # Compute the FFT of the input
    X = torch.fft.rfft(x, n=len_fft, dim=1)
    frequency_list = abs(X).mean(0).mean(-1)
    frequency_list[0:2] = 0  # period cannot be 1
    top_fre = torch.argmax(frequency_list)
    top_fre = top_fre.detach().cpu().numpy()

    max_period = len_fft // int(top_fre)

    max_period = round(max_period / k) * k if max_period > k else max_period

    max_period = min(max(max_period, 2), T // 2)

    return int(max_period)


def find_period_multiple_k(x, k=1, harmonic=False):
    """
    Find the period of the signal x, where the period is a multiple of k.
    x is expected to be of shape (B,T,C) where C is the number of channels and T is the length of each channel.
    """
    B, T, C = x.shape

    len_fft = math.ceil(T / k) * k
    # Compute the FFT of the input
    X = torch.fft.rfft(x, n=len_fft, dim=1)
    frequency_list = abs(X).mean(0).mean(-1)
    frequency_list[0:2] = 0  # period cannot be 1

    if not harmonic:
        top_fre = torch.argmax(frequency_list)
        max_period = len_fft // int(top_fre)
        max_period = round(max_period / k) * k if max_period > k else max_period
        max_period = min(max(max_period, 2), T // 2)

        harm_period = 5
    else:
        # _, top_list = torch.topk(frequency_list, k=2)
        # top_list = top_list.detach().cpu().numpy()
        # top_list = list(top_list)
        #
        # top_fre, sub_fre = min(top_list), max(top_list)
        #
        # max_period = len_fft // int(top_fre)
        # max_period = round(max_period / k) * k if max_period > k else max_period
        # max_period = int(min(max(max_period, 2), T // 2))
        #
        # harm_period = len_fft // int(sub_fre)
        # harm_period = int(min(max(harm_period, 1), max_period))

        # another implementation
        top_fre = torch.argmax(frequency_list)
        max_period = len_fft // int(top_fre)
        # max_period = round(max_period / k) * k if max_period > k else max_period
        max_period = int(min(max(max_period, 2), T // 2))

        sub_fre_list = frequency_list[top_fre + 1:]
        sub_fre = torch.argmax(sub_fre_list) + top_fre + 1

        if frequency_list[sub_fre] > frequency_list[top_fre] * 0.95:
            harm_period = len_fft // int(sub_fre)
            harm_period = int(min(max(harm_period, 1), max_period))
        else:
            harm_period = 5

    return max_period, harm_period


def compute_harm_fre(x, period, win_size=5):
    """
        compute harmonic frequency under period
        x is expected to be of shape (B,L,N)
    """
    B, L, N = x.shape
    rem = L % period
    if rem != 0:
        x = F.pad(x, pad=[0, 0, 0, period - rem])
        L = x.shape[1]

    # [_, period]
    x = x.reshape(B, L // period, period, N).transpose(-1, -2).flatten(0, -2)

    x_fre = torch.fft.rfft(x, dim=-1).abs().mean(0)
    x_fre[0:period // win_size] = 0

    top_fre = torch.argmax(x_fre)

    harm_period = period // int(top_fre)

    return harm_period


def create_block_missing(input_size, mask_rate=0.1, block_length=(3, 10), device='cpu'):
    #  (batch, len, channels)
    batch, length, channels = input_size

    mask = torch.ones(input_size, dtype=torch.float32, device=device)

    total_elements = batch * length * channels
    total_mask_elements = int(total_elements * mask_rate)
    masked_elements = 0

    while masked_elements < total_mask_elements:
        b = random.randint(0, batch - 1)
        c = random.randint(0, channels - 1)
        block_len = random.randint(block_length[0], block_length[1])

        if masked_elements + block_len > total_mask_elements:
            block_len = total_mask_elements - masked_elements

        start = random.randint(0, length - block_len)

        mask[b, start:start + block_len, c] = 0

        masked_elements += block_len

    return mask


def apply_difference(data, n=1):
    """
    Apply differencing to the data.
    :param data: Input data [batch, length, channel]
    :param n: Order of differencing
    :return: Differenced data and the last original data point for each series
    """
    for i in range(n):
        data[..., 1:, :] = data[..., 1:, :] - data[..., :-1, :]
    return data


def moore_penrose_iter_pinv(x, iters=6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z
