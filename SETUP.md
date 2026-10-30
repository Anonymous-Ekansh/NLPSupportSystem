# Setup Checklist

| Item | Owner | Status | Verify Command |
|------|-------|--------|----------------|
| Python 3.11+ | local | ✅ | `python3 --version` |
| Core packages (pandas, numpy, sklearn, etc.) | local | ✅ | `python3 scripts/check_env.py` |
| fastembed | local | ✅ | `python3 -c "from fastembed import TextEmbedding"` |
| fastapi + pydantic | local | ⬜ | `python3 -c "import fastapi, pydantic"` |
| data/train.csv | local | ✅ | `wc -l data/train.csv` |
| data/validation.csv | local | ✅ | `wc -l data/validation.csv` |
| Supabase project | cloud | ⬜ | `python3 scripts/check_env.py` (SUPABASE_URL) |
| pgvector extension | cloud | ⬜ | SQL: `SELECT extversion FROM pg_extension WHERE extname='vector'` |
| .env file | local | ⬜ | `test -f .env && echo ok` |
| Render deploy (Python API) | cloud | ⬜ | `curl $RENDER_URL/health` |
| Vercel deploy (Next.js) | cloud | ⬜ | browser check |

## Phase 1 (offline pipeline) – no accounts needed
```bash
python3 scripts/check_env.py
python3 scripts/run_all.py
```

## Phase 2 – SETUP GATES (do these when prompted)
1. **Supabase**: create project at https://supabase.com/dashboard → copy URL + service_role key → `.env`
2. **Render**: push repo → connect at https://dashboard.render.com → set env vars
3. **Vercel**: `npx vercel` in `web/` → set env vars in dashboard
