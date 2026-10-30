# Setup Checklist

| # | Item | Owner | When needed | Status | Verify command |
|---|------|-------|-------------|--------|----------------|
| 1 | GitHub repo (`Anonymous-Ekansh/NLPSupportSystem`) | team | now | ✅ | `git remote -v` |
| 2 | Python ≥ 3.11 | local | now | ✅ | `python3 --version` |
| 3 | Node ≥ 20 + npm | local | web phase | ⬜ | `node --version && npm --version` |
| 4 | Embedding model (`BAAI/bge-small-en-v1.5`, ~67 MB ONNX) | local | training | ✅ | `python3 -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"` |
| 5 | Supabase project + pgvector + schema | cloud | Level 2 | ⬜ | `python3 scripts/check_env.py` (Supabase section) |
| 6 | `.env` with all keys (see `.env.example`) | local | API phase | ✅ | `test -f .env && grep SUPABASE_URL .env` |
| 7 | Vercel project (root = `web/`) | cloud | deploy | ⬜ | open `https://<project>.vercel.app` |
| 8 | Render service (start: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`, health: `/health`) | cloud | deploy | ⬜ | `curl $NLP_API_URL/health` |
| 9 | `NLP_API_URL` in Vercel env vars | cloud | deploy | ⬜ | Vercel dashboard → Settings → Env Vars |
| 10 | Uptime pinger (UptimeRobot / cron-job.org, 5 min, hits `/health`) | cloud | deploy | ⬜ | UptimeRobot dashboard shows UP |
| 11 | cloudflared fallback (local tunnel if Render cold-starts are unacceptable) | local | optional | ⬜ | `cloudflared tunnel --url http://localhost:8000` |
| 12 | Render Starter upgrade ($7/mo, removes 15-min sleep) | cloud | optional | ⬜ | Render dashboard → Service → Plan |
| 13 | LLM API key (optional, for Level 2+ generation) | cloud | Level 2 | ⬜ | `python3 -c "import os; print('set' if os.getenv('LLM_API_KEY') else 'not set')"` |

## How to use

```bash
# Verify everything automatically
python3 scripts/check_env.py

# Run the offline NLP pipeline (no cloud needed)
python3 scripts/run_all.py   # split → train → evaluate → examples

# Start API server locally
uvicorn api.main:app --reload --port 8000

# Test the API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "I want to cancel my subscription"}'
```

## Secrets policy
- `.env` is **gitignored** — never committed
- Supabase `service_role` key lives **only** on the Python service (Render env) and local `.env`
- Next.js `web/` **never** sees the service key; it calls Python API via `NLP_API_URL`
