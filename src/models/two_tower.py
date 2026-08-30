"""Symmetric two-tower (siamese) encoder trained on MAL rec pairs.

Maps per-anime features (content emb + ALS co-watch emb + metadata) into a
space where cosine similarity = "MAL users would recommend these together".
Trained with in-batch sampled softmax, vote-weighted, both directions.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path(__file__).resolve().parent.parent.parent / "data"


class Tower(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 512, d_out: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def train_two_tower(ids: np.ndarray, X: np.ndarray,
                    pairs: pd.DataFrame,
                    epochs: int = 40, batch: int = 1024, lr: float = 1e-3,
                    temp: float = 0.05, d_hidden: int = 512, d_out: int = 256,
                    dropout: float = 0.1, seed: int = 42,
                    dev_eval_fn=None, log_fn=None,
                    device: str = "cuda") -> tuple[Tower, np.ndarray]:
    """pairs: columns src, dst, votes (mal ids). Returns (tower, item_emb)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    idx = {int(a): i for i, a in enumerate(ids)}
    p = pairs[pairs.src.isin(idx) & pairs.dst.isin(idx)]
    src = np.array([idx[s] for s in p.src], dtype=np.int64)
    dst = np.array([idx[d] for d in p.dst], dtype=np.int64)
    w = np.log1p(p.votes.to_numpy()).astype(np.float32)

    Xt = torch.tensor(X, device=device)
    tower = Tower(X.shape[1], d_hidden, d_out, dropout).to(device)
    opt = torch.optim.AdamW(tower.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = len(src)
    for ep in range(epochs):
        perm = np.random.permutation(n)
        tot, nb = 0.0, 0
        tower.train()
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            if len(b) < 16:
                continue
            za = tower(Xt[src[b]])
            zb = tower(Xt[dst[b]])
            logits = za @ zb.T / temp
            targets = torch.arange(len(b), device=device)
            bw = torch.tensor(w[b], device=device)
            la = (F.cross_entropy(logits, targets, reduction="none") * bw).mean()
            lb = (F.cross_entropy(logits.T, targets, reduction="none") * bw).mean()
            loss = (la + lb) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        if log_fn and (ep % 5 == 4 or ep == epochs - 1):
            dev_p5 = dev_eval_fn(encode_items(tower, Xt)) if dev_eval_fn else None
            log_fn(ep, tot / max(nb, 1), dev_p5)
    return tower, encode_items(tower, Xt)


@torch.no_grad()
def encode_items(tower: Tower, Xt: torch.Tensor, batch: int = 4096) -> np.ndarray:
    tower.eval()
    outs = [tower(Xt[i:i + batch]).cpu().numpy()
            for i in range(0, len(Xt), batch)]
    return np.concatenate(outs).astype(np.float32)
