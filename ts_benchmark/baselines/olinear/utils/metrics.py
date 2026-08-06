import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def calculate_r2_pearson_robust(x, y):
    # check shape
    # print(f'y.shape: {y.shape}')
    if x.shape != y.shape:
        raise ValueError("The shape of x and y must be the same!")

    assert x.ndim >= 3, 'Error in x.shape...'

    flattened_x = x.reshape(-1, x.shape[-2], x.shape[-1])
    flattened_y = y.reshape(-1, y.shape[-2], y.shape[-1])

    # there was a bug here!!!!
    mean_x = np.mean(flattened_x, axis=1, keepdims=True)
    mean_y = np.mean(flattened_y, axis=1, keepdims=True)

    # correlation coefficient
    numerator = np.sum((flattened_x - mean_x) * (flattened_y - mean_y), axis=1)
    denominator = np.sqrt(np.sum((flattened_x - mean_x) ** 2, axis=1) * np.sum((flattened_y - mean_y) ** 2, axis=1))
    # avoid potential zero-divide error
    denominator[denominator < 1e-5] = np.nan
    pearson_correlation = numerator / denominator

    # R2
    total_variance = np.sum((flattened_y - mean_y) ** 2, axis=1)
    residuals = flattened_y - flattened_x
    residual_sum_of_squares = np.sum(residuals ** 2, axis=1)
    # avoid potential zero-divide error
    total_variance[total_variance < 1e-5] = np.nan
    r2 = 1 - (residual_sum_of_squares / total_variance)

    # mean
    mean_pearson = np.nanmean(pearson_correlation)
    mean_r2 = np.nanmean(r2)

    return mean_r2, mean_pearson


def calculate_mase(y_pred, y_true, y_naive=None):
    """
    MASE（Mean Absolute Scaled Error）。
    y_true: (b, l, n)
    y_pred: (b, l, n)
    y_naive: (b, l, n)
    """

    assert y_pred.ndim >= 3, 'Error in y_pred.shape...'

    if y_naive is None:
        y_naive = y_true

    if y_true.shape != y_pred.shape or y_true.shape != y_naive.shape:
        raise ValueError("The input shape must be the same.")

    # reshape to [batch, pred_len, channel]
    y_true_flat = y_true.reshape(-1, y_true.shape[-2], y_true.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-2], y_pred.shape[-1])
    y_naive_flat = y_naive.reshape(-1, y_naive.shape[-2], y_naive.shape[-1])

    # MAE: [batch, channel]
    mae_model = np.mean(np.abs(y_true_flat - y_pred_flat), axis=1)

    # naive MAE [batch, channel]
    mae_naive = np.mean(np.abs(y_true_flat[:, 1:, :] - y_naive_flat[:, :-1, :]), axis=1)

    # avoid potential error
    mae_naive[mae_naive < 1e-5] = np.nan

    # MASE
    mase = np.nanmean(mae_model / mae_naive)

    return mase


def metric(pred, true):
    if np.any(np.isnan(true)):
        mask = ~np.isnan(true)
        pred = pred[mask]
        true = true[mask]

    mae = MAE(pred, true)
    mse = MSE(pred, true)

    # print(f'pred.shape[-2]: {pred.shape[-2]}')

    if pred.ndim > 1 and true.ndim > 1:
        r2, pear = calculate_r2_pearson_robust(pred, true)
        mase = calculate_mase(pred, true)
    else:
        r2, pear, mase = 0, 0, 0

    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe, r2, pear, mase
