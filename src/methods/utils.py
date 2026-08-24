# timeit, cal_metrics, my_dataset
import time
import numpy as np
import torch
from functools import wraps
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)


def _amp_autocast(enabled=True):
    if hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _amp_grad_scaler(enabled=True):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _embed_amp_enabled(device):
    if device is None:
        return False
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return False
    try:
        from parse import args

        return not getattr(args, "no_amp", False)
    except Exception:
        return True


class my_dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        fn = self.data[index]
        label = self.labels[index]
        return fn, label

    def __len__(self):
        return len(self.data)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.4f} seconds\n\n")
        return result

    return timeit_wrapper


def metrics(label, pred_label, pred_posteriors):
    acc = accuracy_score(label, pred_label)

    fpr, tpr, thresholds = roc_curve(label, pred_posteriors)
    threshold_index = np.max((np.where(fpr <= 0.005)[0]))
    tpr1 = tpr[threshold_index]

    threshold_index = np.max((np.where(fpr <= 0.01)[0]))
    tpr2 = tpr[threshold_index]

    threshold_index = np.max((np.where(fpr <= 0.03)[0]))
    tpr3 = tpr[threshold_index]

    roc_auc = roc_auc_score(label, pred_posteriors)
    pr_auc = (
        average_precision_score(label, pred_posteriors)
        if len(set(label)) > 1
        else float("nan")
    )
    return acc, roc_auc, pr_auc, tpr1, tpr2, tpr3
