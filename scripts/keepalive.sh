#!/bin/bash
# Keeps the recommender server AND the public tunnel alive; records the
# current public URL to data/public_url.txt whenever it changes.
cd /home/ubuntu/anime-rec
while true; do
  if ! curl -s --max-time 8 "localhost:8501/recommend?anime=monster&k=1&mode=full" >/dev/null 2>&1; then
    fuser -k 8501/tcp >/dev/null 2>&1
    nohup .venv/bin/python serve.py --port 8501 > serve.log 2>&1 &
    sleep 90
  fi
  if ! pgrep -f "cloudflared tunnel" >/dev/null 2>&1; then
    nohup /tmp/cloudflared tunnel --url http://localhost:8501 > tunnel.log 2>&1 &
    sleep 20
    grep -o "https://[a-z-]*\.trycloudflare\.com" tunnel.log | tail -1 > data/public_url.txt
  fi
  sleep 60
done
