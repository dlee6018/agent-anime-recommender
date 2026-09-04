"""Gate: does cross-candidate attention beat pointwise GBDT scoring on the
SAME features and dataset? (survey: LiGR/NAR4Rec family — the one untried
mechanism; all our rankers score candidates independently.)

Uses the cached fold-honest dataset (rerank_dataset.npz groups). Train/val
split by group. Set model: feature MLP -> 3-layer transformer encoder over
the candidate set -> per-candidate score; listwise softmax CE on graded
labels. Baseline: LGBM LambdaRank on the identical split.
Report val P@5-in-group and NDCG@10 for both.
"""
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

rng = np.random.default_rng(0)
d = np.load("data/rerank_dataset.npz")
X, y, groups = d["X"].astype(np.float32), d["y"], list(d["groups"])
starts = np.concatenate([[0], np.cumsum(groups)])
G = len(groups)
perm = rng.permutation(G)
val_g = set(perm[: G // 5].tolist())
print(f"groups {G} (val {len(val_g)}), rows {len(X)}, feats {X.shape[1]}")

# normalize features (transformer needs it; LGBM doesn't care)
mu, sd = X.mean(0), X.std(0) + 1e-6
Xn = (X - mu) / sd


def group_slices(idxs):
    return [(starts[g], starts[g + 1]) for g in idxs]


tr_g = [g for g in range(G) if g not in val_g]
va_g = [g for g in range(G) if g in val_g]


def metrics(score_fn):
    p5s, ndcgs = [], []
    for a, b in group_slices(va_g):
        s = score_fn(a, b)
        order = np.argsort(-s)
        lab = y[a:b][order]
        rel = (lab >= 2).astype(float)  # top-10 members
        if rel.sum() == 0:
            continue
        p5s.append(rel[:5].mean())
        dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel[:10]))
        ideal = sum(1 / np.log2(i + 2)
                    for i in range(min(10, int(rel.sum()))))
        ndcgs.append(dcg / ideal)
    return np.mean(p5s), np.mean(ndcgs)


# ---- LGBM baseline on the split ----
tr_rows = np.concatenate([np.arange(a, b) for a, b in group_slices(tr_g)])
rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=400,
                    learning_rate=0.05, num_leaves=63, min_child_samples=20,
                    label_gain=[0, 1, 3, 7], random_state=0, verbose=-1)
rk.fit(X[tr_rows], y[tr_rows], group=[b - a for a, b in group_slices(tr_g)])
p5, nd = metrics(lambda a, b: rk.booster_.predict(X[a:b]))
print(f"LGBM       val P@5-in-group={p5:.3f} NDCG@10={nd:.3f}")

# ---- set transformer ----
dev = "cuda"


class SetRanker(nn.Module):
    def __init__(self, d_in, d_model=128, heads=4, layers=3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model))
        enc = nn.TransformerEncoderLayer(d_model, heads, d_model * 4,
                                         dropout=0.1, batch_first=True,
                                         norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):  # [B, n, d_in]
        h = self.tr(self.inp(x))
        return self.out(h).squeeze(-1)


model = SetRanker(X.shape[1]).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
gains = torch.tensor([0.0, 1.0, 3.0, 7.0], device=dev)

for ep in range(8):
    model.train()
    rng.shuffle(tr_g)
    tot = 0.0
    for g in tr_g:
        a, b = starts[g], starts[g + 1]
        x = torch.tensor(Xn[a:b], device=dev).unsqueeze(0)
        lab = torch.tensor(y[a:b], device=dev, dtype=torch.long)
        s = model(x).squeeze(0)
        tgt = gains[lab]
        if tgt.sum() == 0:
            continue
        loss = -(torch.log_softmax(s, dim=0) * (tgt / tgt.sum())).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    model.eval()
    with torch.no_grad():
        p5, nd = metrics(lambda a, b: model(
            torch.tensor(Xn[a:b], device=dev).unsqueeze(0)
        ).squeeze(0).cpu().numpy())
    print(f"ep {ep}: loss={tot/len(tr_g):.3f} "
          f"SetRanker val P@5={p5:.3f} NDCG@10={nd:.3f}", flush=True)
