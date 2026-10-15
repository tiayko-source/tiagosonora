import re
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import separator

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
ALLOWED = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".aif", ".webm"}
MAX_BYTES = 200 * 1024 * 1024
JOB_ID = re.compile(r"^[0-9a-f]{12}$")

app = FastAPI(title="Sonora")


def _job_or_404(job_id: str) -> separator.Job:
    job = separator.get_job(job_id) if JOB_ID.match(job_id) else None
    if job is None:
        raise HTTPException(404, "Música não encontrada.")
    return job


@app.get("/api/info")
def info():
    return {"device": separator.DEVICE, "device_name": separator.DEVICE_NAME,
            "models": separator.MODELS}


@app.post("/api/separate")
def separate(file: UploadFile = File(...), model: str = Form("4")):
    if model not in separator.MODELS:
        raise HTTPException(400, "Modelo inválido.")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, "Formato não suportado. Usa MP3, WAV, FLAC, OGG ou M4A.")

    name = Path(file.filename).stem[:120] or "Música"
    job, input_path = separator.create_job(name, model, suffix)
    with input_path.open("wb") as out:
        shutil.copyfileobj(file.file, out, length=1024 * 1024)
    if input_path.stat().st_size > MAX_BYTES:
        shutil.rmtree(job.dir, ignore_errors=True)
        raise HTTPException(413, "Ficheiro demasiado grande (máx. 200 MB).")

    separator.enqueue(job)
    return job.public()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return _job_or_404(job_id).public()


@app.get("/api/jobs/{job_id}/stems/{stem}.wav")
def stem_audio(job_id: str, stem: str):
    job = _job_or_404(job_id)
    if stem not in job.stems:
        raise HTTPException(404, "Faixa não encontrada.")
    return FileResponse(job.dir / "stems" / f"{stem}.wav", media_type="audio/wav")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
