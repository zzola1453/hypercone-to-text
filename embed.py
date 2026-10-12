"""Frozen sentence embeddings, cached per dataset and encoder."""
import os

import torch

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"


def embed(name, data, model_name=DEFAULT_MODEL, cache="its/cache"):
    """data: {split: (texts, labels)} -> {split: float tensor (N, D)} (raw, not normalised)."""
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"{name}_{model_name.split('/')[-1]}.pt")
    if os.path.exists(path):
        return torch.load(path)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    out = {s: torch.from_numpy(model.encode(t, batch_size=256, convert_to_numpy=True, show_progress_bar=False))
           for s, (t, _) in data.items()}
    torch.save(out, path)
    return out
