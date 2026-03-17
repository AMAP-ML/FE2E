import pandas as pd
import torch


# Adapted from: https://github.com/victoresque/pytorch-template/blob/master/utils/util.py
class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0.0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        # 确保value是数值类型
        value = float(value) if hasattr(value, '__float__') else value.item() if hasattr(value, 'item') else float(value)
        self._data.at[key, "total"] += value * n
        self._data.at[key, "counts"] += n
        self._data.at[key, "average"] = self._data.at[key, "total"] / self._data.at[key, "counts"]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

def pixel_mean(pred, gt, valid_mask):
    if valid_mask is not None:
        masked_pred = pred * valid_mask
        masked_gt = gt * valid_mask

        valid_pixel_count = torch.sum(valid_mask, dim=(0,1))

        pred_mean = torch.sum(masked_pred, dim=(0,1)) / valid_pixel_count
        gt_mean = torch.sum(masked_gt, dim=(0,1)) / valid_pixel_count
    else:
        pred_mean = torch.mean(pred, dim=(0,1))
        gt_mean = torch.mean(gt, dim=(0,1))

    mean_difference = torch.abs(pred_mean - gt_mean)
    return mean_difference

def pixel_var(pred, gt, valid_mask):
    if valid_mask is not None:
        masked_pred = pred * valid_mask
        masked_gt = gt * valid_mask

        valid_pixel_count = torch.sum(valid_mask, dim=(0,1))

        pred_mean = torch.sum(masked_pred, dim=(0,1)) / valid_pixel_count
        gt_mean = torch.sum(masked_gt, dim=(0,1)) / valid_pixel_count

        pred_var = torch.sum(valid_mask * (pred - pred_mean)**2, dim=(0,1)) / valid_pixel_count
        gt_var = torch.sum(valid_mask * (gt - gt_mean)**2, dim=(0,1)) / valid_pixel_count
    else:
        pred_var = torch.var(pred, dim=(0,1))
        gt_var = torch.var(gt, dim=(0,1))

    var_difference = torch.abs(pred_var - gt_var)

    return var_difference

def abs_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    return abs_relative_diff.mean()


def squared_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    square_relative_diff = (
        torch.pow(torch.abs(actual_output - actual_target), 2) / actual_target
    )
    if valid_mask is not None:
        square_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    square_relative_diff = torch.sum(square_relative_diff, (-1, -2)) / n
    return square_relative_diff.mean()


def rmse_linear(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()


def rmse_log(output, target, valid_mask=None):
    diff = torch.log(output) - torch.log(target)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()




# adapt from: https://github.com/imran3180/depth-map-prediction/blob/master/main.py
def threshold_percentage(output, target, threshold_val, valid_mask=None):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    bit_mat = (max_d1_d2 < threshold_val).to(output.dtype)
    if valid_mask is not None:
        bit_mat = bit_mat * valid_mask.to(output.dtype)
        n = valid_mask.sum((-1, -2))
    else:
        n = torch.tensor(output.shape[-1] * output.shape[-2], device=output.device)
    n = torch.clamp(n, min=1)
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n.to(count_mat.dtype)
    return threshold_mat.mean()


def delta1_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25, valid_mask)


def delta2_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**2, valid_mask)


def delta3_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**3, valid_mask)
