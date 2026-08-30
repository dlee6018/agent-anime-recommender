"""Two-tower encoder trained on MAL rec pairs.

Maps per-anime features (content emb + ALS co-watch emb + metadata) into a
space where similarity = "MAL users would recommend these together".
In-batch sampled softmax (both directions), optional vote-proportional pair
sampling, optional shared hard negatives drawn from the popular candidate
pool, optional asymmetric query/candidate heads.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path(__file__).resolve().parent.parent.parent / "data"


class Tower(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 512, d_out: int = 256,
                 dropout: float = 0.1, asym: bool = False):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.head_q = nn.Linear(d_hidden, d_out)
        self.head_c = self.head_q if not asym else nn.Linear(d_hidden, d_out)

    def forward(self, x, side: str = "q"):
        h = self.trunk(x)
        z = self.head_q(h) if side == "q" else self.head_c(h)
        return F.normalize(z, dim=-1)


def train_two_tower(ids: np.ndarray, X: np.ndarray, pairs: pd.DataFrame,
                    epochs: int = 40, batch: int = 1024, lr: float = 1e-3,
                    temp: float = 0.05, d_hidden: int = 512, d_out: int = 256,
                    dropout: float = 0.1, seed: int = 42,
                    sample_by_votes: bool = False,
                    n_hard_neg: int = 0, hard_neg_pool: np.ndarray | None = None,
                    asym: bool = False,
                    dev_eval_fn=None, log_fn=None,
                    device: str = "cuda") -> tuple[Tower, np.ndarray]:
    """pairs: columns src, dst, votes (mal ids). hard_neg_pool: row indices
    into ids/X from which shared hard negatives are drawn each step.
    Returns (tower, candidate-side item_emb)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    idx = {int(a): i for i, a in enumerate(ids)}
    p = pairs[pairs.src.isin(idx) & pairs.dst.isin(idx)]
    src = np.array([idx[s] for s in p.src], dtype=np.int64)
    dst = np.array([idx[d] for d in p.dst], dtype=np.int64)
    w = np.log1p(p.votes.to_numpy()).astype(np.float32)

    # sparse per-src positive matrix, to mask false negatives among hard negs
    import scipy.sparse as _sp
    P = _sp.csr_matrix(
        (np.ones(len(src), dtype=np.int8),
         (src, dst)), shape=(len(ids), len(ids)))

    Xt = torch.tensor(X, device=device)
    tower = Tower(X.shape[1], d_hidden, d_out, dropout, asym=asym).to(device)
    opt = torch.optim.AdamW(tower.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = len(src)
    prob = w / w.sum() if sample_by_votes else None
    for ep in range(epochs):
        order = (np.random.choice(n, size=n, p=prob) if sample_by_votes
                 else np.random.permutation(n))
        tot, nb = 0.0, 0
        tower.train()
        for i in range(0, n, batch):
            b = order[i:i + batch]
            if len(b) < 16:
                continue
            za = tower(Xt[src[b]], "q")
            zb = tower(Xt[dst[b]], "c")
            logits = za @ zb.T / temp
            targets = torch.arange(len(b), device=device)
            if n_hard_neg and hard_neg_pool is not None:
                hn = np.random.choice(hard_neg_pool, size=n_hard_neg,
                                      replace=False)
                zn = tower(Xt[hn], "c")
                extra = za @ zn.T / temp
                fmask_np = (P[src[b]][:, hn].toarray() > 0) | \
                    (src[b][:, None] == hn[None, :])
                fmask = torch.tensor(fmask_np, device=device)
                extra = extra.masked_fill(fmask, -1e4)
                logits = torch.cat([logits, extra], dim=1)
            bw = torch.tensor(w[b], device=device)
            if sample_by_votes:
                bw = torch.ones_like(bw)
            la = (F.cross_entropy(logits, targets, reduction="none") * bw).mean()
            lb = (F.cross_entropy(logits[:, :len(b)].T, targets,
                                  reduction="none") * bw).mean()
            loss = (la + lb) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        if log_fn and (ep % 5 == 4 or ep == epochs - 1):
            dev_p5 = (dev_eval_fn(encode_items(tower, Xt, side="c"))
                      if dev_eval_fn else None)
            log_fn(ep, tot / max(nb, 1), dev_p5)
    return tower, encode_items(tower, Xt, side="c")


@torch.no_grad()
def encode_items(tower: Tower, Xt: torch.Tensor, side: str = "c",
                 batch: int = 4096) -> np.ndarray:
    tower.eval()
    outs = [tower(Xt[i:i + batch], side).cpu().numpy()
            for i in range(0, len(Xt), batch)]
    return np.concatenate(outs).astype(np.float32)


@torch.no_grad()
def encode_query(tower: Tower, Xt: torch.Tensor, rows: list[int]) -> np.ndarray:
    tower.eval()
    return tower(Xt[rows], "q").cpu().numpy().astype(np.float32)
