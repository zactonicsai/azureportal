# Backend: AI App to Evaluate Long/Short Vowel Sounds
# Tools: FastAPI (Python), OpenAI Whisper API, Pydub, Librosa

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import openai
import tempfile
import os
import librosa
import numpy as np

app = FastAPI()

# Allow CORS for local HTML front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

# Expected phoneme dictionary for reference
# You can expand this dictionary
expected_phonemes = {
    "cake": "long",
    "cat": "short",
    "bike": "long",
    "bit": "short"
}

@app.post("/analyze")
async def analyze_audio(file: UploadFile, word: str = Form(...)):
    # Save temp file
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_audio.write(await file.read())
    temp_audio.close()

    # Transcribe using Whisper API
    try:
        transcript = openai.Audio.transcribe(
            model="whisper-1",
            file=open(temp_audio.name, "rb"),
            response_format="text"
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    # Analyze vowel length via duration of sound
    y, sr = librosa.load(temp_audio.name)
    duration = librosa.get_duration(y=y, sr=sr)

    # Simple heuristic based on duration
    classification = "long" if duration > 0.5 else "short"

    expected = expected_phonemes.get(word.lower(), "unknown")

    result = {
        "word": word,
        "transcript": transcript.strip(),
        "expected_sound": expected,
        "detected_sound": classification,
        "match": expected == classification if expected != "unknown" else False
    }
    os.unlink(temp_audio.name)
    return result

@app.get("/practice")
def get_practice_words():
    # Send words and recording links
    practice = [
        {"word": w, "record_url": f"/record?word={w}"} for w in expected_phonemes.keys()
    ]
    return {"practice_words": practice}
