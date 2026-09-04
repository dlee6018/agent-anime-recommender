"""Gate for the supervised-LLM-ranker branch (FIRST/RecRanker family).

QLoRA-SFT Qwen3-1.7B as a pointwise yes/no scorer: prompt carries query and
candidate docs (+ co-watch stat, per RecRanker's aux-info trick); label =
candidate in query's crowd top-10. Train on non-dev/eval srcs with hard
in-pool negatives; evaluate P@5 on dev bare pools via P("Yes").
LGBM bare baseline to beat: 0.504.

Phases: --phase data | train | eval
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import load_metadata, year_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True)
ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
ap.add_argument("--n_srcs", type=int, default=1500)
args = ap.parse_args()

DATA = ROOT / "data"
meta = load_metadata()


def desc(aid):
    m = meta[aid]
    tags = ", ".join((m["genres"] + m["themes"] + m["demographics"])[:8])
    syn = (m["synopsis"] or "").replace("\n", " ")
    if "no description available" in syn.lower()[:60]:
        syn = ""
    return (f"{m['name']} ({m['type']} {year_of(aid) or '?'}; {tags}). "
            f"{syn[:260]}")


def prompt(q, c, cos):
    return (f"Query anime: {desc(q)}\n"
            f"Candidate anime: {desc(c)}\n"
            f"Audience-overlap score: {cos:.2f}\n"
            f"Would MyAnimeList users commonly recommend the candidate to "
            f"fans of the query anime? Answer with one word, Yes or No.\n"
            f"Answer:")


def build_pools():
    import pandas as pd
    d = np.load(DATA / "tt_ens_emb.npz")
    ids, emb = d["ids"], d["emb"].astype(np.float32)
    idx = {int(a): i for i, a in enumerate(ids)}
    pop = np.array([(meta.get(int(a), {}).get("popularity") or 99999)
                    for a in ids])
    rmask = pop <= 8000
    pairs = pd.read_parquet(DATA / "rec_pairs.parquet")
    held = set()
    for f in ("eval_set.json", "dev_set.json"):
        held |= {int(q) for q in json.load(open(DATA / f))["queries"]}
    return pairs, held, ids, emb, idx, rmask


if args.phase == "data":
    import pandas as pd
    pairs, held, ids, emb, idx, rmask = build_pools()
    by_src = {int(s): g.sort_values("votes", ascending=False)
              for s, g in pairs.groupby("src")}
    srcs = [s for s in by_src if s not in held and s in idx
            and len(by_src[s]) >= 8][:args.n_srcs]
    rng = np.random.default_rng(0)
    rows = []
    for s in srcs:
        truth = [int(x) for x in by_src[s].dst.head(10) if int(x) in idx]
        tq = emb[idx[s]]
        sim = emb @ tq
        sim[~rmask] = -np.inf
        sim[idx[s]] = -np.inf
        pool = [int(ids[i]) for i in np.argpartition(-sim, 60)[:60]]
        negs = [c for c in pool if c not in set(truth)]
        rng.shuffle(negs)
        for c in truth[:5]:
            rows.append((s, c, float(emb[idx[s]] @ emb[idx[c]]), "Yes"))
        for c in negs[:10]:
            rows.append((s, c, float(emb[idx[s]] @ emb[idx[c]]), "No"))
    df = pd.DataFrame(rows, columns=["src", "cand", "cos", "label"])
    df.to_parquet(DATA / "llm_gate_train.parquet")
    print(f"gate training rows: {len(df)} "
          f"({(df.label == 'Yes').mean():.0%} positive)")

elif args.phase == "train":
    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments)

    df = pd.read_parquet(DATA / "llm_gate_train.parquet")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

    def gen():
        for r in df.itertuples():
            p = prompt(int(r.src), int(r.cand), r.cos)
            full = p + " " + r.label
            enc = tok(full, truncation=True, max_length=420)
            plen = len(tok(p)["input_ids"])
            labels = [-100] * plen + enc["input_ids"][plen:]
            labels = labels[:len(enc["input_ids"])]
            yield {"input_ids": enc["input_ids"], "labels": labels}

    ds = Dataset.from_generator(gen)

    def collate(feats):
        ml = max(len(f["input_ids"]) for f in feats)
        pad = tok.pad_token_id or tok.eos_token_id
        return {
            "input_ids": torch.tensor(
                [f["input_ids"] + [pad] * (ml - len(f["input_ids"]))
                 for f in feats]),
            "attention_mask": torch.tensor(
                [[1] * len(f["input_ids"]) + [0] * (ml - len(f["input_ids"]))
                 for f in feats]),
            "labels": torch.tensor(
                [f["labels"] + [-100] * (ml - len(f["labels"]))
                 for f in feats]),
        }

    Trainer(model=model, args=TrainingArguments(
        output_dir=str(ROOT / "models" / "llm_gate"), num_train_epochs=1,
        per_device_train_batch_size=8, gradient_accumulation_steps=2,
        learning_rate=1e-4, bf16=True, logging_steps=100,
        save_strategy="no", report_to=[]),
        train_dataset=ds, data_collator=collate).train()
    model.save_pretrained(str(ROOT / "models" / "llm_gate"))
    tok.save_pretrained(str(ROOT / "models" / "llm_gate"))
    print("gate model saved")

elif args.phase == "eval":
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, str(ROOT / "models" / "llm_gate"))
    model.eval()
    yes_id = tok(" Yes", add_special_tokens=False)["input_ids"][-1]
    no_id = tok(" No", add_special_tokens=False)["input_ids"][-1]

    pairs, held, ids, emb, idx, rmask = build_pools()
    dev = {int(q): [int(r) for r in v] for q, v in
           json.load(open(DATA / "dev_set.json"))["queries"].items()}
    ev_ids = {int(q) for q in json.load(open(DATA / "eval_set.json"))["queries"]}
    from src.franchise import franchise_filter, same_franchise

    p5s = []
    for qn, (q, truth) in enumerate(dev.items()):
        if q not in idx:
            continue
        tq = emb[idx[q]]
        sim = emb @ tq
        sim[~rmask] = -np.inf
        sim[idx[q]] = -np.inf
        pool = [int(ids[i]) for i in np.argpartition(-sim, 50)[:50]]
        keep = franchise_filter([q])
        pool = [c for c in pool if keep(c)]
        scores = []
        with torch.no_grad():
            for i in range(0, len(pool), 8):
                batch = pool[i:i + 8]
                enc = tok([prompt(q, c, float(emb[idx[q]] @ emb[idx[c]]))
                           for c in batch], return_tensors="pt",
                          padding=True, truncation=True,
                          max_length=420).to("cuda")
                out = model(**enc).logits[:, -1, :]
                scores += (out[:, yes_id] - out[:, no_id]).tolist()
        order = np.argsort(-np.array(scores))
        out5, t10 = [], set(truth[:10])
        for j in order:
            c = pool[j]
            if any(same_franchise(c, o) for o in out5):
                continue
            out5.append(c)
            if len(out5) == 5:
                break
        p5s.append(sum(c in t10 for c in out5) / 5)
        if qn % 30 == 29:
            print(f"[{qn + 1}] running P@5 = {np.mean(p5s):.3f}", flush=True)
    print(f"LLM-gate dev bare P@5 = {np.mean(p5s):.3f} "
          f"(LGBM baseline 0.504)", flush=True)
