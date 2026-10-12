"""Open-set metrics (ADB protocol): unknown is class K."""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def evaluate(pred, score, y, K):
    pred, score, y = (np.asarray(a) for a in (pred, score, y))
    f1 = f1_score(y, pred, labels=np.arange(K + 1), average=None, zero_division=0)
    known = y < K
    thr = np.quantile(score[known], 0.05)                    # accept 95% of known test samples
    return {
        "acc": float((pred == y).mean()),
        "f1_all": float(f1.mean()),
        "f1_known": float(f1[:K].mean()),
        "f1_unknown": float(f1[K]),
        "auroc": float(roc_auc_score(known, score)),
        "fpr95": float((score[~known] >= thr).mean()),
    }
