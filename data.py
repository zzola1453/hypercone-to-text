"""Intent datasets and the open-set split of ADB (Zhang et al., AAAI 2021).

A fraction of the classes is drawn as known. Train/val keep only known-class samples; in test every
other class (and CLINC's out-of-scope intent, which is never known) is relabelled as unknown = K.
"""
import numpy as np
from datasets import load_dataset


def load(name):
    """Returns ({split: (texts, labels)}, class names, oos label or None)."""
    if name == "clinc":
        ds = load_dataset("clinc/clinc_oos", "plus")
        names = ds["train"].features["intent"].names
        data = {s: (list(ds[s]["text"]), np.array(ds[s]["intent"])) for s in ("train", "validation", "test")}
        return data, names, names.index("oos")
    if name == "banking":
        # mteb/banking77 (9993 train / 3076 test) has no validation split: hold out 10% of train per class.
        ds = load_dataset("mteb/banking77")
        y = np.array(ds["train"]["label"])
        names = [None] * (y.max() + 1)
        for lab, txt in zip(ds["train"]["label"], ds["train"]["label_text"]):
            names[lab] = txt
        rng = np.random.RandomState(0)
        val = np.zeros(len(y), dtype=bool)
        for c in range(len(names)):
            idx = np.flatnonzero(y == c)
            val[rng.choice(idx, max(1, len(idx) // 10), replace=False)] = True
        texts = np.array(ds["train"]["text"], dtype=object)
        data = {"train": (list(texts[~val]), y[~val]), "validation": (list(texts[val]), y[val]),
                "test": (list(ds["test"]["text"]), np.array(ds["test"]["label"]))}
        return data, names, None
    raise ValueError(name)


def open_split(labels, num_classes, oos, ratio, seed):
    """labels: {split: int array}. Returns known class ids and {split: (index array, remapped labels)}."""
    cands = [c for c in range(num_classes) if c != oos]
    rng = np.random.RandomState(seed)
    known = np.sort(rng.choice(cands, int(round(ratio * len(cands))), replace=False))
    K = len(known)
    remap = np.full(num_classes, K)
    remap[known] = np.arange(K)
    out = {}
    for s, y in labels.items():
        r = remap[y]
        idx = np.arange(len(y)) if s == "test" else np.flatnonzero(r < K)
        out[s] = (idx, r[idx])
    return known, out
