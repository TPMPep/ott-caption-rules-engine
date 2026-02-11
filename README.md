# OTT Caption Rules Engine (Railway)

This is a small HTTP service that converts **word-level timestamps** (e.g., from AssemblyAI)
into **OTT-style captions** with configurable rules and a QC report.

## Endpoints

- `GET /health` → `{ "ok": true }`
- `POST /build-captions` → returns:
  - `cues[]` (start/end/text/speaker)
  - `srt` (string)
  - `vtt` (string)
  - `qc` (issues list)

## Deploy on Railway

1. Create a GitHub repo (e.g. `ott-caption-rules-engine`)
2. Add these files to the repo root
3. In Railway: New Project → GitHub Repository → select this repo
4. After deploy:
   - `https://YOUR-URL/health`
   - `https://YOUR-URL/docs` (Swagger UI)

## Example request

```json
{
  "words": [
    {"text":"Hello","start":1000,"end":1200,"speaker":"A"},
    {"text":"world!","start":1200,"end":1500,"speaker":"A"}
  ],
  "rules": {
    "maxCharsPerLine": 32,
    "maxLines": 2,
    "maxCPS": 17,
    "minDurationMs": 1000,
    "maxDurationMs": 7000
  }
}
```
