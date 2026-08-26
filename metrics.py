import numpy as np


def RSE(pred, true):
    return np.sqrt(np.nansum((true - pred) ** 2)) / np.sqrt(np.nansum((true - np.nanmean(true)) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - np.nanmean(pred, 0))).sum(0)
    d = np.sqrt(np.nansum(((true - np.nanmean(true, 0)) ** 2 * (pred - np.nanmean(pred, 0)) ** 2), 0))
    d += 1e-12
    return np.nanmean(0.01*(u / d), -1)


def MAE(pred, true):
    return np.nanmean(np.abs(pred - true))


def MSE(pred, true):
    return np.nanmean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.nanmean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.nanmean(np.square((pred - true) / true))

def WAPE(pred, true):
    return np.nansum(np.abs(pred - true)) / np.nansum(np.abs(true))


def metric(pred, true):
    wape = WAPE(pred, true)
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return wape, mae, mse, rmse, mape, mspe, rse, corr