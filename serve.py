#!/usr/bin/env python3
"""HTTP server for the recommender: models load once, queries are instant.

    python serve.py --port 8501 --model best
    curl 'localhost:8501/recommend?anime=Death+Note&k=5'
    curl 'localhost:8501/recommend?anime=Frieren&anime=Mushishi&k=10'
"""
import argparse
import json
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data import (load_metadata, nearest_servable,  # noqa: E402
                      resolve_title, titles)
from src.registry import get_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8501)
ap.add_argument("--model", default="rerank")  # pure ML: gauge the model, not the graph lookup
args = ap.parse_args()

print("loading models...", flush=True)
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import build_features
from src.models.product import make_heldout_recommender
from src.models.rerank import FeatureBuilder, make_rerank_recommender
from src.franchise import with_franchise_filter

_ids, _X, _ = build_features("content_emb_qwen.npz")
_emb = np.load(Path(__file__).parent / "data" / "tt_ens_emb.npz")["emb"].astype(np.float32)
_booster = lgb.Booster(model_file=str(Path(__file__).parent / "data" / "reranker_union.txt"))
_pairs = pd.read_parquet(Path(__file__).parent / "data" / "rec_pairs_fresh.parquet")
_fb = FeatureBuilder(_ids)  # shared; set_graph swapped under REC_LOCK
_al = Path(__file__).parent / "data" / "anilist_recs.json"
if _al.exists():
    _fb.set_anilist(json.load(open(_al)))

def _fb_factory():
    return _fb

_fb.set_graph(_pairs)
_full = with_franchise_filter(make_rerank_recommender(
    _ids, _emb, _booster, _fb, 8000, 250, union_extra=100))

# bare = no crowd-rec data about the query from ANY platform:
# no AniList loaded at all + per-query strict MAL-edge removal +
# a booster trained without AniList features (reranker_strict2, 21 feats)
_fb_bare = FeatureBuilder(_ids)  # never gets set_anilist
_booster_bare = lgb.Booster(
    model_file=str(Path(__file__).parent / "data" / "reranker_strict2.txt"))

MODES = {
    "bare": make_heldout_recommender(_pairs, _ids, _emb, _booster_bare,
                                     lambda: _fb_bare, mode="strict"),
    "strict": make_heldout_recommender(_pairs, _ids, _emb, _booster,
                                       _fb_factory, mode="strict"),
    "src_only": make_heldout_recommender(_pairs, _ids, _emb, _booster,
                                         _fb_factory, mode="src_only"),
    "full": lambda q, k: (_fb.set_graph(_pairs) or _full(q, k)),
}
REC_LOCK = threading.Lock()  # serializes set_graph swaps + LGBM predict

# cover images from the 2023 dump (display only)
import csv as _csv

IMG = {}
try:
    for _r in _csv.DictReader(open(Path(__file__).parent / "data" / "raw"
                                   / "anime_metadata_2023.csv")):
        IMG[int(_r["anime_id"])] = _r.get("Image URL") or ""
except Exception:
    pass

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ani-rec</title><style>
:root{--bg:#0f1115;--card:#181b22;--fg:#e8eaf0;--mut:#8a90a0;--acc:#7aa2ff}
*{box-sizing:border-box;margin:0}body{background:var(--bg);color:var(--fg);
font:16px/1.5 system-ui,sans-serif;max-width:720px;margin:0 auto;padding:24px}
h1{font-size:22px;margin-bottom:4px}p.sub{color:var(--mut);margin-bottom:20px}
form{display:flex;gap:8px;margin-bottom:8px}
input[type=text]{flex:1;padding:10px 14px;border-radius:10px;border:1px solid
#2a2f3a;background:var(--card);color:var(--fg);font-size:16px}
select,button{padding:10px 14px;border-radius:10px;border:1px solid #2a2f3a;
background:var(--card);color:var(--fg);font-size:15px}
button{background:var(--acc);color:#0b0d12;font-weight:600;cursor:pointer}
.hint{color:var(--mut);font-size:13px;margin-bottom:20px}
.card{display:flex;gap:14px;background:var(--card);border-radius:12px;
padding:12px;margin-bottom:10px;align-items:center}
.card img{width:56px;height:80px;object-fit:cover;border-radius:8px;
background:#242936}
.rank{color:var(--mut);font-size:14px;min-width:22px}
.card a{color:var(--fg);text-decoration:none;font-weight:600}
.card a:hover{color:var(--acc)}
#err{color:#ff8a8a;margin:12px 0}#spin{color:var(--mut)}
</style></head><body>
<h1>ani-rec</h1><p class="sub">anime you'll probably like, per the MAL crowd</p>
<form onsubmit="go();return false">
<input id="q" type="text" placeholder="e.g. Death Note, Frieren, jjk"
 autofocus><select id="k"><option>5</option><option selected>10</option>
<option>15</option></select><select id="m" title="how much the model may
peek at MAL's crowd rec graph for your query">
<option value="bare" selected>bare model (no crowd data)</option>
<option value="strict">no MAL edges (AniList ok)</option>
<option value="src_only">+MAL reverse edges</option>
<option value="full">crowd lookup</option></select>
<button>Recommend</button></form>
<p class="hint">Tip: comma-separate several titles to blend tastes.</p>
<div id="err"></div><div id="spin"></div><div id="out"></div>
<script>
async function go(){
 const q=document.getElementById('q').value.trim(); if(!q)return;
 const names=q.split(',').map(s=>s.trim()).filter(Boolean);
 const ps=names.map(n=>'anime='+encodeURIComponent(n)).join('&');
 const k=document.getElementById('k').value;
 document.getElementById('err').textContent='';
 document.getElementById('out').innerHTML='';
 document.getElementById('spin').textContent='thinking…';
 try{
  const m=document.getElementById('m').value;const r=await fetch('/recommend?'+ps+'&k='+k+'&mode='+m); const d=await r.json();
  document.getElementById('spin').textContent='';
  if(d.error){document.getElementById('err').textContent=
    'could not find: '+(d.unknown||[]).join(', ');return}
  let h='<p class="hint">for: '+d.query.map(x=>x.title).join(' + ')+'</p>';
  for(const s of (d.substituted||[])){h+='<p class="hint">'+s.asked+
    ' is outside the model\\'s catalogue — showing results for '+s.using+
    ' instead.</p>'}
  for(const rec of d.recommendations){
   h+='<div class="card"><span class="rank">'+rec.rank+'</span>'+
    (rec.image?'<img loading="lazy" src="'+rec.image+'">':'<img>')+
    '<a href="'+rec.url+'" target="_blank" rel="noopener">'+rec.title+
    '</a></div>'}
  document.getElementById('out').innerHTML=h;
 }catch(e){document.getElementById('spin').textContent='';
  document.getElementById('err').textContent='server error';}}
</script></body></html>"""
TT = titles()
META = load_metadata()
print("ready", flush=True)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path in ("/", "/index.html"):
            data = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if url.path != "/recommend":
            self.send_error(404)
            return
        qs = urllib.parse.parse_qs(url.query)
        names = qs.get("anime", [])
        k = min(int(qs.get("k", ["5"])[0]), 50)
        mode = qs.get("mode", ["bare"])[0]
        rec_fn = MODES.get(mode, MODES["bare"])
        ids, unknown, subbed = [], [], []
        for n in names:
            aid = resolve_title(n)
            if not aid:
                unknown.append(n)
                continue
            use, was_sub = nearest_servable(aid)
            if use is None:
                unknown.append(f"{TT.get(aid, n)} (outside model coverage)")
                continue
            if was_sub:
                subbed.append({"asked": TT.get(aid), "using": TT.get(use)})
            ids.append(use)
        if not ids:
            body = {"error": "no resolvable anime", "unknown": unknown}
            code = 400
        else:
            with REC_LOCK:
                recs = rec_fn(ids, k)
            body = {
                "mode": mode,
                "query": [{"mal_id": q, "title": TT.get(q)} for q in ids],
                "unknown": unknown,
                "substituted": subbed,
                "recommendations": [
                    {"rank": i + 1, "mal_id": r, "title": TT.get(r),
                     "url": f"https://myanimelist.net/anime/{r}",
                     "image": IMG.get(r, "")}
                    for i, r in enumerate(recs)],
            }
            code = 200
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *a):  # quiet
        pass


ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
