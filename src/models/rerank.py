"""Shared reranker feature builder + inference-side reranking recommender."""
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..data import load_metadata, year_of
from ..franchise import same_franchise

DATA = Path(__file__).resolve().parent.parent.parent / "data"

FEATS = ["tt_cos", "content_cos", "als_cos", "cooc_lift", "cooc_logcnt",
         "genre_jac", "year_gap", "cand_logpop", "src_logpop", "type_match",
         "cand_season", "src_season", "studio_match", "cand_score",
         "transfer_in", "nbr_out", "cand_has_graph", "cand_age",
         "rev_edge", "colist", "rev_fam"]

SEASON_RE = re.compile(
    r"(?:(\d)(?:nd|rd|th) season|season (\d)|part (\d)|\b(ii|iii|iv)\b)", re.I)
ROMAN = {"ii": 2, "iii": 3, "iv": 4}


class FeatureBuilder:
    def __init__(self, ids: np.ndarray):
        self.ids = ids
        self.idx = {int(a): i for i, a in enumerate(ids)}
        self.meta = load_metadata()
        c = np.load(DATA / "content_emb.npz")
        self.CEMB = self._align(c["ids"], c["emb"].astype(np.float32))
        a = np.load(DATA / "als_emb.npz")
        self.AEMB = self._align(a["ids"], a["emb"].astype(np.float32))
        self.C = sp.load_npz(DATA / "cooc_csr.npz")
        cm = np.load(DATA / "cooc_meta.npz")
        self.cnt = cm["item_counts"]
        self.cooc_row_of = {int(x): i for i, x in enumerate(cm["ids"])}
        # cooc matrix may live on a different (older/smaller) id universe
        self.cooc_ids = cm["ids"]
        self.cooc2uni = np.array([self.idx.get(int(x), -1)
                                  for x in cm["ids"]], dtype=np.int64)
        self.out_lists: dict[int, list[tuple[int, float]]] = {}
        self.in_lists: dict[int, list[tuple[int, float]]] = {}

    def _align(self, src_ids, emb):
        out = np.zeros((len(self.ids), emb.shape[1]), dtype=np.float32)
        for x, e in zip(src_ids, emb):
            i = self.idx.get(int(x))
            if i is not None:
                out[i] = e
        return out

    def set_graph(self, pairs) -> None:
        """Visible rec graph for 2-hop features. Swap per fold during
        reranker training (fold srcs' edges hidden); use the widest legal
        graph at inference (full rec_pairs in production, eval-holdout file
        for milestone reads)."""
        self.out_lists: dict[int, list[tuple[int, float]]] = {}
        self.in_lists: dict[int, list[tuple[int, float]]] = {}
        self._fam_cache = {}  # family lookups depend on graph nodes
        for s, g in pairs.groupby("src"):
            v = np.log1p(g.votes.to_numpy().astype(np.float32))
            v /= v.sum() + 1e-9
            for d, wt in zip(g.dst, v):
                si, di = int(s), int(d)
                if si in self.idx and di in self.idx:
                    self.out_lists.setdefault(si, []).append((di, float(wt)))
                    self.in_lists.setdefault(di, []).append((si, float(wt)))

    def graph_feats(self, cand: int, tt_q: np.ndarray,
                    tt_emb: np.ndarray) -> tuple[float, float]:
        """(transfer_in, nbr_out) for one candidate vs the query embedding."""
        tin = 0.0
        for s, wt in self.in_lists.get(cand, ()):
            sim = float(tt_q @ tt_emb[self.idx[s]])
            if sim > 0:
                tin += (sim ** 3) * wt
        nbr = self.out_lists.get(cand, ())
        nout = 0.0
        if nbr:
            sims = sorted((float(tt_q @ tt_emb[self.idx[d]])
                           for d, _ in nbr), reverse=True)
            nout = float(np.mean(sims[:5]))
        return tin, nout

    def cooc_top(self, s_id: int, n: int, mask: np.ndarray) -> list[int]:
        crow = self.cooc_row_of.get(s_id)
        if crow is None:
            return []
        co = np.asarray(self.C[crow].todense()).ravel()
        s = co / (max(self.cnt[crow], 1) ** 0.65
                  * np.maximum(self.cnt, 1) ** 0.65)
        valid = (self.cooc2uni >= 0) & mask[np.maximum(self.cooc2uni, 0)]
        s[~valid] = -np.inf
        s[crow] = -np.inf
        n = min(n, len(s) - 1)
        top = np.argpartition(-s, n)[:n]
        top = top[np.argsort(-s[top])]
        return [int(self.cooc_ids[i]) for i in top if np.isfinite(s[i])]

    def content_top(self, s_id: int, n: int, mask: np.ndarray) -> list[int]:
        si = self.idx.get(s_id)
        if si is None:
            return []
        s = self.CEMB @ self.CEMB[si]
        s[~mask] = -np.inf
        s[si] = -np.inf
        top = np.argpartition(-s, n)[:n]
        return [int(self.ids[i]) for i in top[np.argsort(-s[top])]]

    def _family(self, aid: int) -> list[int]:
        """Franchise siblings of aid among graph nodes (cached)."""
        if not hasattr(self, "_fam_cache"):
            self._fam_cache: dict[int, list[int]] = {}
        if aid not in self._fam_cache:
            self._fam_cache[aid] = [
                s for s in self.in_lists
                if s != aid and same_franchise(s, aid)]
        return self._fam_cache[aid]

    def season_idx(self, aid: int) -> int:
        m = self.meta.get(aid)
        if not m:
            return 1
        for t in (m["name"], m["english"] or ""):
            mt = SEASON_RE.search(t)
            if mt:
                g = [g for g in mt.groups() if g]
                return ROMAN.get(g[0].lower(), None) or int(g[0])
        return 1

    def rows(self, s_id: int, cands: list[int], tt_q: np.ndarray,
             tt_emb: np.ndarray) -> np.ndarray:
        si = self.idx[s_id]
        crow = self.cooc_row_of.get(s_id)
        co = (np.asarray(self.C[crow].todense()).ravel()
              if crow is not None else None)
        ms = self.meta[s_id]
        sg, st = set(ms["genres"]), set(ms["studios"])
        sy = year_of(s_id) or 2005
        ssea = self.season_idx(s_id)
        # under src_only holdout the query's IN-edges are visible:
        # rev_edge = weight of (cand -> query); colist = query & cand
        # co-occurring in third-party lists. Both are 0 under symmetric.
        # (franchise-collapsed q_in was tried and reverted — exp 33: base
        # queries inherit noisy sequel in-edges, dev 0.709 -> 0.601)
        q_in = {s: w for s, w in self.in_lists.get(s_id, ())}
        out = []
        for c_id in cands:
            ci = self.idx[int(c_id)]
            mc = self.meta[int(c_id)]
            cg = set(mc["genres"])
            lift, con = 0.0, 0.0
            cj = self.cooc_row_of.get(int(c_id))
            if co is not None and cj is not None:
                con = float(co[cj])
                lift = con / (max(self.cnt[crow], 1) ** 0.65
                              * max(self.cnt[cj], 1) ** 0.65)
            tin, nout = self.graph_feats(int(c_id), tt_q, tt_emb)
            has_graph = float(int(c_id) in self.in_lists
                              or int(c_id) in self.out_lists)
            rev = q_in.get(int(c_id), 0.0)
            colist = 0.0
            if q_in:
                for s, w_sq in q_in.items():
                    for d, w_sc in self.out_lists.get(s, ()):
                        if d == int(c_id):
                            colist += w_sq * w_sc
                            break
            # sequel queries: candidate's list naming ANY franchise sibling
            # of the query is nearly as strong as naming the query itself
            rev_fam = rev
            for d, wt in self.out_lists.get(int(c_id), ()):
                if wt > rev_fam and same_franchise(d, s_id):
                    rev_fam = wt
            out.append([
                float(tt_q @ tt_emb[ci]),
                float(self.CEMB[si] @ self.CEMB[ci]),
                float(self.AEMB[si] @ self.AEMB[ci]),
                lift, np.log1p(con),
                len(sg & cg) / max(len(sg | cg), 1),
                abs((year_of(int(c_id)) or 2005) - sy),
                np.log1p(mc["popularity"] or 99999),
                np.log1p(ms["popularity"] or 99999),
                1.0 if mc["type"] == ms["type"] else 0.0,
                float(self.season_idx(int(c_id))), float(ssea),
                float(len(st & set(mc["studios"])) > 0),
                mc["score"] or 6.5,
                tin, nout,
                has_graph, float(2026 - (year_of(int(c_id)) or 2005)),
                rev, colist, rev_fam,
            ])
        return np.array(out, dtype=np.float32)


def make_rerank_recommender(ids: np.ndarray, tt_emb: np.ndarray,
                            booster, fb: FeatureBuilder,
                            maxrank_retrieve: int = 1500, top_cand: int = 80,
                            union_extra: int = 0):
    meta = load_metadata()
    pop = np.array([(meta.get(int(a), {}).get("popularity") or 99999)
                    for a in ids])
    rmask = pop <= maxrank_retrieve
    idx = fb.idx

    def recommend(query_ids: list[int], k: int) -> list[int]:
        qrows = [idx[q] for q in query_ids if q in idx]
        if not qrows:
            return []
        tt_q = tt_emb[qrows].mean(axis=0)
        tt_q /= np.linalg.norm(tt_q) + 1e-9
        sim = tt_emb @ tt_q
        sim[~rmask] = -np.inf
        for r in qrows:
            sim[r] = -np.inf
        top = np.argpartition(-sim, top_cand)[:top_cand]
        top = top[np.argsort(-sim[top])]
        cands = [int(ids[i]) for i in top]
        if union_extra:
            s_dom = min(query_ids, key=lambda q: meta.get(q, {})
                        .get("popularity") or 10**9)
            seen = set(cands)
            in_nbrs = [s for s, _ in sorted(fb.in_lists.get(s_dom, ()),
                                            key=lambda x: -x[1])[:union_extra]]
            for extra in (in_nbrs,
                          fb.cooc_top(s_dom, union_extra, rmask),
                          fb.content_top(s_dom, union_extra, rmask)):
                cands.extend(c for c in extra
                             if c not in seen and c in fb.idx)
                seen.update(extra)
        # featurize vs the most popular query anime (multi-query: mean would
        # need per-query rows; use the dominant query as src context)
        s_id = min(query_ids,
                   key=lambda q: meta.get(q, {}).get("popularity") or 10**9)
        if s_id not in idx:
            return cands[:k]
        Xr = fb.rows(s_id, cands, tt_q, tt_emb)
        scores = booster.predict(Xr)
        order = np.argsort(-scores)
        qset = set(query_ids)
        return [cands[i] for i in order if cands[i] not in qset][:k]

    return recommend
