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

from src.data import load_metadata, resolve_title, titles  # noqa: E402
from src.registry import get_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8501)
ap.add_argument("--model", default="best")
args = ap.parse_args()

print("loading models...", flush=True)
REC = get_model(args.model)
REC_LOCK = threading.Lock()  # LightGBM predict across threads: unpinned safety
TT = titles()
META = load_metadata()
print("ready", flush=True)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path != "/recommend":
            self.send_error(404)
            return
        qs = urllib.parse.parse_qs(url.query)
        names = qs.get("anime", [])
        k = min(int(qs.get("k", ["5"])[0]), 50)
        ids, unknown = [], []
        for n in names:
            aid = resolve_title(n)
            (ids.append(aid) if aid else unknown.append(n))
        if not ids:
            body = {"error": "no resolvable anime", "unknown": unknown}
            code = 400
        else:
            with REC_LOCK:
                recs = REC(ids, k)
            body = {
                "query": [{"mal_id": q, "title": TT.get(q)} for q in ids],
                "unknown": unknown,
                "recommendations": [
                    {"rank": i + 1, "mal_id": r, "title": TT.get(r),
                     "url": f"https://myanimelist.net/anime/{r}"}
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
