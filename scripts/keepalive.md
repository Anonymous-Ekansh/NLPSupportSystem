# Service Keep-Alive & Uptime Guide

Render's free tier spins down web services after 15 minutes of inactivity. Cold starts can take 45–60 seconds.
Supabase free tier projects pause after 7 days of inactivity.

This guide provides three zero-cost methods to keep the API alive and ensure the Supabase database stays active.

---

## 1. Automated GitHub Actions Ping (Recommended)

Add `.github/workflows/keepalive.yml` to ping the API `/health` endpoint every 10 minutes. This prevents Render from sleeping.

```yaml
name: Render & Supabase Keepalive

on:
  schedule:
    # Runs every 10 minutes (UTC)
    - cron: '*/10 * * * *'
  workflow_dispatch: # Allows manual trigger from Actions tab

jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - name: Ping API /health
        run: |
          API_URL="${{ secrets.RENDER_API_URL || 'https://nlp-support-api.onrender.com' }}"
          echo "Pinging $API_URL/health..."
          STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health" --max-time 30)
          echo "Status: $STATUS"
          if [ "$STATUS" -ne 200 ]; then
            echo "Warning: API returned status $STATUS"
          fi

      - name: Verify Supabase Connectivity
        run: |
          API_URL="${{ secrets.RENDER_API_URL || 'https://nlp-support-api.onrender.com' }}"
          RESP=$(curl -s "$API_URL/health" --max-time 30)
          echo "Health response: $RESP"
```

---

## 2. Free Uptime Monitors (2-Minute Setup)

External uptime monitors ping your service continuously from multiple locations:

1. **Cron-Job.org** (100% Free, 1-minute intervals):
   - URL: `https://<your-render-service>.onrender.com/health`
   - Schedule: Every 10 minutes
   - Method: `GET`

2. **UptimeRobot** (Free up to 50 monitors, 5-minute intervals):
   - Monitor Type: `HTTP(s)`
   - URL: `https://<your-render-service>.onrender.com/health`
   - Monitoring Interval: `5 minutes`

---

## 3. Supabase Pause Prevention

Supabase free tier automatically pauses projects if they receive no database traffic for 7 consecutive days.

To ensure your Supabase instance remains active:
1. Pinging `/health` on the deployed API checks Supabase reachability (`SELECT 1 FROM examples LIMIT 1;`). When the API is pinged, it queries Supabase, keeping the project active automatically.
2. If using Supabase directly, you can also query via curl:
   ```bash
   curl -s -X GET "https://<PROJECT_REF>.supabase.co/rest/v1/examples?select=id&limit=1" \
     -H "apikey: <ANON_OR_SERVICE_KEY>" \
     -H "Authorization: Bearer <ANON_OR_SERVICE_KEY>"
   ```

---

## 4. Local Ping / Cron

If running on a local server or machine:

```bash
# Add to crontab via `crontab -e`:
*/10 * * * * curl -s https://<your-render-service>.onrender.com/health > /dev/null 2>&1
```
