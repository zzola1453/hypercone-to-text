import numpy as np
import pytest
import torch

from data import open_split
from methods import cone_predict, train_cone
from metrics import evaluate


def test_open_split_keeps_oos_unknown():
    labels = {"train": np.repeat(np.arange(10), 5), "test": np.repeat(np.arange(10), 2)}
    known, sp = open_split(labels, 10, oos=3, ratio=0.5, seed=0)
    assert 3 not in known and len(known) == 4          # round(0.5 * 9 candidates)
    _, ytr = sp["train"]
    assert ytr.max() < 4 and len(ytr) == 4 * 5
    idx, yte = sp["test"]
    assert len(idx) == 20
    assert (yte[labels["test"] == 3] == 4).all()       # oos -> unknown
    assert (yte < 4).sum() == 4 * 2


def test_evaluate_perfect():
    y = np.array([0, 1, 2, 2])                          # K = 2, class 2 = unknown
    r = evaluate(y, np.array([1.0, 1.0, 0.0, 0.0]), y, 2)
    assert r["acc"] == 1 and r["f1_all"] == 1 and r["auroc"] == 1 and r["fpr95"] == 0


def test_cone_tvmf_rejects_far_points():
    torch.manual_seed(0)
    K, din = 4, 32
    centers = torch.randn(K, din) * 3
    y = torch.arange(K).repeat_interleave(100)
    x = centers[y] + torch.randn(len(y), din) * 0.3
    net, theta = train_cone(x, y, K, epochs=15, seed=0, outliers="tvmf", d=16)
    pred, _ = cone_predict(net, theta, x, K)
    assert (pred == y).float().mean() > 0.9
    far = torch.randn(200, din) * 3                     # random points, mostly unlike any class
    pred_far, _ = cone_predict(net, theta, far, K)
    assert (pred_far == K).float().mean() > 0.5
