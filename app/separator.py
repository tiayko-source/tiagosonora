"""Separação de stems com Demucs (Hybrid Transformer Demucs, Meta AI).

Corre num único thread de trabalho: os pedidos ficam numa fila e são
processados um de cada vez, para não rebentar a memória do CPU/GPU.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "jobs"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "4": "htdemucs",     # drums, bass, other, vocals
    "6": "htdemucs_6s",  # + guitar, piano
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"


@dataclass
class Job:
    id: str
    name: str
    model: str
    input_path: str
    status: str = "queued"          # queued | loading | separating | saving | done | error
    stage: str = "Na fila…"
    progress: float = 0.0
    stems: list[str] = field(default_factory=list)
    duration: float = 0.0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def dir(self) -> Path:
        return DATA_DIR / self.id

    def public(self) -> dict:
        d = asdict(self)
        d.pop("input_path")
        d["elapsed"] = (time.time() - self.started_at) if self.started_at and not self.finished_at else None
        d["queue_position"] = _queue_position(self.id)
        return d


_jobs: dict[str, Job] = {}
_queue: "queue.Queue[str]" = queue.Queue()
_models: dict[str, torch.nn.Module] = {}


def _queue_position(job_id: str) -> int:
    with _queue.mutex:
        pending = list(_queue.queue)
    return pending.index(job_id) + 1 if job_id in pending else 0


# ---------------------------------------------------------------- audio I/O

def _decode_soundfile(path: Path) -> tuple[np.ndarray, int]:
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return data.T, rate


def _decode_av(path: Path, rate: int) -> tuple[np.ndarray, int]:
    import av  # PyAV traz o ffmpeg embutido (m4a, aac, opus, …)

    chunks = []
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=rate)
        for frame in container.decode(stream):
            chunks.extend(f.to_ndarray() for f in resampler.resample(frame))
        chunks.extend(f.to_ndarray() for f in resampler.resample(None))
    if not chunks:
        raise ValueError("ficheiro sem áudio")
    return np.concatenate(chunks, axis=1).astype(np.float32), rate


def load_audio(path: Path, samplerate: int) -> torch.Tensor:
    """Devolve um tensor (2, n) float32 à taxa de amostragem do modelo."""
    try:
        data, rate = _decode_soundfile(path)
    except Exception:
        try:
            data, rate = _decode_av(path, samplerate)
        except ImportError:
            raise ValueError("Formato não suportado. Usa MP3, WAV, FLAC ou OGG.")
        except Exception:
            raise ValueError("Não foi possível ler este ficheiro de áudio.")

    wav = torch.from_numpy(np.ascontiguousarray(data))
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    if rate != samplerate:
        import julius
        wav = julius.resample_frac(wav, rate, samplerate)
    return wav


# ---------------------------------------------------------------- model

def get_model(key: str):
    name = MODELS[key]
    if name not in _models:
        from demucs.pretrained import get_model as demucs_get_model
        model = demucs_get_model(name)
        model.eval()
        _models[name] = model
    return _models[name]


class _ProgressShim:
    """Substitui o tqdm dentro do demucs.apply para reportarmos o progresso."""

    def __init__(self, n_models: int, callback):
        self.n_models = n_models
        self.callback = callback
        self.loop = -1

    def tqdm(self, iterable, **_):
        self.loop += 1
        items = list(iterable)
        total = max(1, len(items))
        for i, item in enumerate(items):
            yield item
            frac = (min(self.loop, self.n_models - 1) + (i + 1) / total) / self.n_models
            self.callback(frac)


def _separate(job: Job, model, wav: torch.Tensor, device: str) -> torch.Tensor:
    import demucs.apply as dapply

    n_models = len(model.models) if isinstance(model, dapply.BagOfModels) else 1

    def on_progress(frac: float):
        job.progress = 0.06 + 0.88 * frac

    original_tqdm = dapply.tqdm
    dapply.tqdm = _ProgressShim(n_models, on_progress)
    try:
        with torch.no_grad():
            return dapply.apply_model(
                model, wav[None], device=device, shifts=1, split=True,
                overlap=0.25, progress=True, num_workers=0,
            )[0]
    finally:
        dapply.tqdm = original_tqdm


def _process(job: Job):
    job.started_at = time.time()
    job.status, job.stage, job.progress = "loading", "A preparar o modelo de IA…", 0.01
    model = get_model(job.model)

    job.stage, job.progress = "A ler o áudio…", 0.04
    wav = load_audio(Path(job.input_path), model.samplerate)
    job.duration = wav.shape[1] / model.samplerate

    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std

    job.status, job.stage, job.progress = "separating", "A separar os instrumentos…", 0.06
    try:
        sources = _separate(job, model, wav, DEVICE)
    except RuntimeError as exc:
        if DEVICE == "cuda" and "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            job.stage = "Memória da GPU insuficiente — a continuar no CPU…"
            sources = _separate(job, model, wav, "cpu")
        else:
            raise
    sources = sources * std + mean

    job.status, job.stage, job.progress = "saving", "A guardar as faixas…", 0.95
    stems_dir = job.dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    for name, source in zip(model.sources, sources):
        audio = source.clamp(-1, 1).cpu().numpy().T
        sf.write(str(stems_dir / f"{name}.wav"), audio, model.samplerate, subtype="PCM_16")
    job.stems = list(model.sources)

    job.status, job.stage, job.progress = "done", "Pronto", 1.0
    job.finished_at = time.time()
    (job.dir / "meta.json").write_text(json.dumps(asdict(job)), encoding="utf-8")


def _worker():
    while True:
        job_id = _queue.get()
        job = _jobs.get(job_id)
        if job is None:
            continue
        try:
            _process(job)
        except Exception as exc:
            traceback.print_exc()
            job.status, job.stage = "error", "Erro"
            job.error = str(exc) if isinstance(exc, ValueError) else f"Falha na separação: {exc}"
            job.finished_at = time.time()
        finally:
            _queue.task_done()


threading.Thread(target=_worker, name="demucs-worker", daemon=True).start()


# ---------------------------------------------------------------- public API

def create_job(name: str, model: str, suffix: str) -> tuple[Job, Path]:
    job_id = uuid.uuid4().hex[:12]
    job_dir = DATA_DIR / job_id
    job_dir.mkdir(parents=True)
    input_path = job_dir / f"input{suffix}"
    job = Job(id=job_id, name=name, model=model, input_path=str(input_path))
    return job, input_path


def enqueue(job: Job):
    _jobs[job.id] = job
    _queue.put(job.id)


def get_job(job_id: str) -> Job | None:
    job = _jobs.get(job_id)
    if job is None:
        meta = DATA_DIR / job_id / "meta.json"
        if meta.is_file():
            job = Job(**json.loads(meta.read_text(encoding="utf-8")))
            _jobs[job_id] = job
    return job
