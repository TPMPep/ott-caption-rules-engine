import requests

ASSEMBLY_BASE = "https://api.assemblyai.com/v2"


def submit_transcription(api_key: str, media_url: str, speaker_labels: bool, language_detection: bool, webhook_url: str | None):
    headers = {
        "authorization": api_key,
        "content-type": "application/json"
    }

    payload = {
        "audio_url": media_url,
        "speaker_labels": bool(speaker_labels),
        "punctuate": True,
        "format_text": True,
        "disfluencies": False,
        "filter_profanity": False,
        "language_detection": bool(language_detection),
    }

    if webhook_url:
        payload["webhook_url"] = webhook_url

    r = requests.post(
        f"{ASSEMBLY_BASE}/transcript",
        headers=headers,
        json=payload,
        timeout=60
    )

    # 🔥 THIS IS THE IMPORTANT PART
    # Instead of raise_for_status(), we show the actual AssemblyAI error body.
    if not r.ok:
        raise RuntimeError(f"AssemblyAI {r.status_code}: {r.text}")

    data = r.json()
    return data["id"]


def fetch_transcript(api_key: str, transcript_id: str):
    headers = {
        "authorization": api_key
    }

    r = requests.get(
        f"{ASSEMBLY_BASE}/transcript/{transcript_id}",
        headers=headers,
        timeout=60
    )

    if not r.ok:
        raise RuntimeError(f"AssemblyAI fetch {r.status_code}: {r.text}")

    return r.json()
