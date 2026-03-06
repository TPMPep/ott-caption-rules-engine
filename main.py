from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
from typing import Optional

from services.formatter import process_caption_job

app = FastAPI(title="OTT Caption Rules Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/generate")
async def generate(
    backbone_srt: UploadFile = File(...),
    timestamps_json: UploadFile = File(...),
    protected_phrases: Optional[str] = Form(None),   # JSON array string
    output_formats: Optional[str] = Form(None),      # JSON array string: ["srt","vtt","scc"]
):
    try:
        srt_text = (await backbone_srt.read()).decode("utf-8", errors="replace")
        ts_text = (await timestamps_json.read()).decode("utf-8", errors="replace")
        ts_data = json.loads(ts_text)

        prot = json.loads(protected_phrases) if protected_phrases else []
        formats = json.loads(output_formats) if output_formats else ["srt"]

        result = process_caption_job(
            backbone_srt_text=srt_text,
            timestamps=ts_data,
            protected_phrases=prot,
            output_formats=formats,
        )
        return result
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
