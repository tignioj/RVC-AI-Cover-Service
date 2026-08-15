"""LAN API for AstrBot AI-cover jobs.

The service deliberately keeps the GPU-heavy pipeline in the RVC workspace:

1. separate vocals and instrumental with the bundled PyMSS model;
2. remove reverb from the vocal stem;
3. convert the dry vocal with an RVC model and its matching index;
4. mix the converted vocal back with the instrumental.

Run with ``runtime\\python.exe tools\\ai_cover_service.py``. AstrBot only needs
HTTP access to this process; it does not need RVC or CUDA in its container.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import hmac
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from urllib.parse import quote, unquote

from fastapi import Body, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHT_ROOT = PROJECT_ROOT / "assets" / "weights"
INDEX_ROOT = PROJECT_ROOT / "logs"
PYMSS_WEIGHT_ROOT = PROJECT_ROOT / "assets" / "pymss_weights"
RMVPE_ROOT = PROJECT_ROOT / "assets" / "rmvpe"
JOB_ROOT = Path(
    os.getenv("AI_COVER_JOB_ROOT", str(PROJECT_ROOT / "TEMP" / "ai_cover"))
).resolve()
CACHE_ROOT = Path(
    os.getenv(
        "AI_COVER_CACHE_ROOT",
        str(PROJECT_ROOT / "TEMP" / "ai_cover_cache"),
    )
).resolve()
CACHE_LAYOUT_VERSION = "separation-v2"
CACHE_ENTRY_ROOT = CACHE_ROOT / CACHE_LAYOUT_VERSION
LEGACY_CACHE_LAYOUTS = ("separation-v1",)
FFMPEG = PROJECT_ROOT / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
FFPROBE = PROJECT_ROOT / ("ffprobe.exe" if os.name == "nt" else "ffprobe")

HOST = os.getenv("AI_COVER_HOST", "0.0.0.0")
PORT = int(os.getenv("AI_COVER_PORT", "18888"))
API_TOKEN = os.getenv("AI_COVER_API_TOKEN", "")
MAX_UPLOAD_BYTES = int(os.getenv("AI_COVER_MAX_UPLOAD_MB", "200")) * 1024 * 1024
MAX_AUDIO_SECONDS = float(os.getenv("AI_COVER_MAX_AUDIO_SECONDS", "900"))

# The RVC modules read these variables when imported.  Use absolute paths so
# the service also works when launched outside the repository directory.
os.environ.setdefault("RVC_CUDA_GRAPH", "0")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("weight_root", str(WEIGHT_ROOT))
os.environ.setdefault("weight_pymss_root", str(PYMSS_WEIGHT_ROOT))
os.environ.setdefault("index_root", str(INDEX_ROOT))
os.environ.setdefault("outside_index_root", str(PROJECT_ROOT / "assets" / "indices"))
os.environ.setdefault("rmvpe_root", str(RMVPE_ROOT))
os.environ.setdefault("TEMP", str(PROJECT_ROOT / "TEMP"))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=os.getenv("AI_COVER_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ai_cover_service")

app = FastAPI(title="RVC AI Cover Service", version="1.3.0")
job_lock = asyncio.Lock()


def _auth(token: str | None) -> None:
    """Require the configured shared token, while allowing tokenless LAN use."""
    if API_TOKEN and not hmac.compare_digest(token or "", API_TOKEN):
        raise HTTPException(status_code=401, detail="invalid API token")


def _model_files() -> list[Path]:
    return sorted(
        (path for path in WEIGHT_ROOT.glob("*.pth") if path.is_file()),
        key=lambda path: path.stem.casefold(),
    )


def _find_index(model: Path) -> Path | None:
    """Find the most suitable non-training index for an RVC model."""
    stem = model.stem.casefold()
    candidates: list[tuple[int, float, Path]] = []
    if not INDEX_ROOT.is_dir():
        return None
    for path in INDEX_ROOT.rglob("*.index"):
        name = path.stem.casefold()
        if "trained" in name:
            continue
        if name == stem:
            score = 0
        elif stem in name:
            score = 1
        else:
            continue
        candidates.append((score, -path.stat().st_mtime, path.resolve()))
    return min(candidates, default=(0, 0.0, None))[2]


def list_models() -> list[dict[str, object]]:
    return [
        {
            "name": path.stem,
            "file": path.name,
            "has_index": (index := _find_index(path)) is not None,
            "index": index.name if index else None,
        }
        for path in _model_files()
    ]


def _resolve_model(name: str) -> Path:
    requested = Path(str(name or "").strip()).stem.casefold()
    for path in _model_files():
        if requested in {path.stem.casefold(), path.name.casefold()}:
            return path
    raise ValueError(f"RVC model does not exist: {name}")


def _creation_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _run(command: list[str], *, label: str) -> None:
    completed = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=_creation_flags(),
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{label} failed: {detail[-2000:]}")


def _probe_audio(path: Path) -> float:
    completed = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        creationflags=_creation_flags(),
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"cannot decode the uploaded audio: {detail[-1000:]}")
    payload = json.loads(completed.stdout.decode("utf-8"))
    if not payload.get("streams"):
        raise ValueError("the uploaded file has no audio stream")
    duration = float(payload.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise ValueError("audio duration could not be determined")
    if duration > MAX_AUDIO_SECONDS:
        raise ValueError(f"audio is {duration:.1f}s; limit is {MAX_AUDIO_SECONDS:.0f}s")
    return duration


def _normalize_audio(source: Path, target: Path) -> None:
    _run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-c:a",
            "pcm_f32le",
            str(target),
        ],
        label="audio normalization",
    )


def _separate(
    source: Path,
    model_name: str,
    desired_dir: Path,
    secondary_dir: Path,
) -> tuple[Path, Path]:
    from tools.pymss_webui import MSSTBatchSeparator, resolve_model

    logger.info("PyMSS stage %s started", model_name)
    with MSSTBatchSeparator(
        resolve_model(model_name),
        "wav",
        str(desired_dir),
        str(secondary_dir),
    ) as separator:
        result = separator.separate_file(str(source))
    desired, secondary = (Path(item).resolve() for item in result["outputs"])
    logger.info("PyMSS stage %s completed", model_name)
    return desired, secondary


def _cache_entry(audio_hash: str) -> Path:
    return CACHE_ENTRY_ROOT / audio_hash


def _is_audio_hash(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _source_filename(value: str | None) -> str:
    decoded = unquote(str(value or ""))
    filename = decoded.replace("\\", "/").rsplit("/", 1)[-1].strip()
    return filename[:240] or "未知音频"


def _song_name(source_filename: str) -> str:
    name = Path(source_filename).stem.strip()
    return name[:200] or "未知歌曲"


def _utc_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _cache_metadata(audio_hash: str) -> dict[str, str] | None:
    manifest = _cache_entry(audio_hash) / "manifest.json"
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    required = {
        "layout": CACHE_LAYOUT_VERSION,
        "source_sha256": audio_hash,
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        return None
    if not all(
        isinstance(metadata.get(key), str) and metadata[key].strip()
        for key in ("song_name", "source_filename", "created_at")
    ):
        return None
    return metadata


def _cached_stems(audio_hash: str) -> tuple[Path, Path] | None:
    """Return a complete cached dry-vocal/instrumental pair, if available."""
    if not _is_audio_hash(audio_hash) or _cache_metadata(audio_hash) is None:
        return None
    entry = _cache_entry(audio_hash)
    vocal = entry / "vocals.wav"
    instrumental = entry / "instrumental.wav"
    try:
        if vocal.stat().st_size <= 0 or instrumental.stat().st_size <= 0:
            return None
    except OSError:
        return None
    return vocal, instrumental


def _restore_cached_stems(
    audio_hash: str,
    job_dir: Path,
) -> tuple[Path, Path] | None:
    cached = _cached_stems(audio_hash)
    if not cached:
        return None
    destination = job_dir / "cached_stems"
    destination.mkdir(parents=True, exist_ok=True)
    vocal = destination / "vocals.wav"
    instrumental = destination / "instrumental.wav"
    try:
        shutil.copyfile(cached[0], vocal)
        shutil.copyfile(cached[1], instrumental)
        os.utime(_cache_entry(audio_hash), None)
    except OSError:
        logger.warning(
            "Could not restore separation cache %s", audio_hash, exc_info=True
        )
        shutil.rmtree(destination, ignore_errors=True)
        return None
    logger.info("Separation cache hit: %s", audio_hash)
    return vocal, instrumental


def _store_cached_stems(
    audio_hash: str,
    vocal: Path,
    instrumental: Path,
    source_filename: str,
) -> None:
    """Publish a cache entry only after both stem copies are complete."""
    CACHE_ENTRY_ROOT.mkdir(parents=True, exist_ok=True)
    target = _cache_entry(audio_hash)
    temporary = CACHE_ENTRY_ROOT / f".tmp-{audio_hash}-{uuid.uuid4().hex}"
    normalized_filename = _source_filename(source_filename)
    try:
        temporary.mkdir()
        shutil.copyfile(vocal, temporary / "vocals.wav")
        shutil.copyfile(instrumental, temporary / "instrumental.wav")
        (temporary / "manifest.json").write_text(
            json.dumps(
                {
                    "layout": CACHE_LAYOUT_VERSION,
                    "source_sha256": audio_hash,
                    "song_name": _song_name(normalized_filename),
                    "source_filename": normalized_filename,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        temporary.replace(target)
        logger.info("Separation cache stored: %s", audio_hash)
    except OSError:
        logger.warning("Could not store separation cache %s", audio_hash, exc_info=True)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _cache_item(audio_hash: str) -> dict[str, object] | None:
    metadata = _cache_metadata(audio_hash)
    if metadata is None or _cached_stems(audio_hash) is None:
        return None
    entry = _cache_entry(audio_hash)
    try:
        total_bytes = sum(
            path.stat().st_size for path in entry.iterdir() if path.is_file()
        )
        last_accessed_at = _utc_timestamp(entry.stat().st_mtime)
    except OSError:
        return None
    source_filename = _source_filename(metadata["source_filename"])
    return {
        "id": audio_hash,
        "song_name": _song_name(source_filename),
        "source_filename": source_filename,
        "bytes": total_bytes,
        "created_at": metadata["created_at"],
        "last_accessed_at": last_accessed_at,
    }


def _cache_items(query: str = "") -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    if not CACHE_ENTRY_ROOT.is_dir():
        return items
    for entry in CACHE_ENTRY_ROOT.iterdir():
        if not entry.is_dir() or (item := _cache_item(entry.name)) is None:
            continue
        items.append(item)
    items.sort(key=lambda item: str(item["last_accessed_at"]), reverse=True)
    normalized_query = query.strip().casefold()
    if normalized_query:
        items = [
            item
            for item in items
            if normalized_query in str(item["song_name"]).casefold()
            or normalized_query in str(item["source_filename"]).casefold()
        ]
    return items


def _cache_stats() -> dict[str, int]:
    items = _cache_items()
    return {
        "entries": len(items),
        "bytes": sum(int(item["bytes"]) for item in items),
    }


def _cache_snapshot(query: str = "") -> dict[str, object]:
    all_items = _cache_items()
    normalized_query = query.strip().casefold()
    items = all_items
    if normalized_query:
        items = [
            item
            for item in all_items
            if normalized_query in str(item["song_name"]).casefold()
            or normalized_query in str(item["source_filename"]).casefold()
        ]
    return {
        "cache": {
            "entries": len(all_items),
            "bytes": sum(int(item["bytes"]) for item in all_items),
        },
        "items": items,
    }


def _delete_cache_entries(audio_hashes: list[str]) -> dict[str, int]:
    removed_entries = 0
    removed_bytes = 0
    for audio_hash in dict.fromkeys(audio_hashes):
        item = _cache_item(audio_hash)
        if item is None:
            continue
        shutil.rmtree(_cache_entry(audio_hash), ignore_errors=True)
        if not _cache_entry(audio_hash).exists():
            removed_entries += 1
            removed_bytes += int(item["bytes"])
    logger.info(
        "Separation cache entries removed: entries=%d bytes=%d",
        removed_entries,
        removed_bytes,
    )
    return {"entries": removed_entries, "bytes": removed_bytes}


def _discard_obsolete_cache() -> None:
    """Remove pre-metadata caches and incomplete v2 entries at service startup."""
    for layout in LEGACY_CACHE_LAYOUTS:
        legacy_root = CACHE_ROOT / layout
        if legacy_root.is_dir():
            shutil.rmtree(legacy_root, ignore_errors=True)
            logger.info("Obsolete separation cache removed: %s", legacy_root)
    if not CACHE_ENTRY_ROOT.is_dir():
        return
    for entry in CACHE_ENTRY_ROOT.iterdir():
        is_incomplete_entry = entry.is_dir() and _is_audio_hash(entry.name)
        if is_incomplete_entry and _cache_item(entry.name) is None:
            shutil.rmtree(entry, ignore_errors=True)
        elif entry.name.startswith(".tmp-"):
            shutil.rmtree(entry, ignore_errors=True)


def _clear_cache() -> dict[str, int]:
    """Delete only recognized cache entries below our versioned directory."""
    before = _cache_stats()
    if CACHE_ENTRY_ROOT.is_dir():
        for entry in CACHE_ENTRY_ROOT.iterdir():
            is_hash_entry = entry.is_dir() and _is_audio_hash(entry.name)
            if is_hash_entry or entry.name.startswith(".tmp-"):
                shutil.rmtree(entry, ignore_errors=True)
    logger.info(
        "Separation cache cleared: entries=%d bytes=%d",
        before["entries"],
        before["bytes"],
    )
    return before


def _convert_vocal(
    source: Path,
    target: Path,
    model: Path,
    transpose: int,
    index_rate: float,
    rms_mix_rate: float,
    protect: float,
) -> Path | None:
    import soundfile as sf
    import torch

    from configs.config import Config
    from infer.vc.modules import VC

    config = Config()
    vc = VC(config)
    try:
        vc.get_vc(model.name)
        original_index = _find_index(model)
        # The bundled Windows FAISS build opens paths through a narrow C API
        # and cannot read Chinese filenames. Use an ASCII job-local copy while
        # preserving the original index name in API metadata.
        runtime_index = target.parent / "rvc_feature.index"
        if original_index:
            shutil.copyfile(original_index, runtime_index)
        logger.info(
            "RVC stage started: model=%s index=%s transpose=%s",
            model.name,
            original_index.name if original_index else "none",
            transpose,
        )
        info, output = vc.vc_single(
            0,
            str(source),
            transpose,
            "rmvpe",
            str(runtime_index) if original_index else "",
            index_rate if original_index else 0.0,
            0,
            rms_mix_rate,
            protect,
        )
        if not output or output[0] is None or output[1] is None:
            raise RuntimeError(str(info))
        sample_rate, audio = output
        # RVC returns a PCM-scale int16 array. Writing those integer values into
        # a FLOAT WAV preserves values such as 32000.0 instead of normalizing
        # them to [-1, 1], which makes playback and the final mix clip heavily.
        # Match the WebUI/batch export path and store the result as PCM16.
        sf.write(str(target), audio, sample_rate, format="WAV", subtype="PCM_16")
        logger.info("RVC stage completed: %s", info.replace("\n", " | "))
        return original_index
    finally:
        del vc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _atempo_filters(tempo: float) -> list[str]:
    """Build an atempo chain compatible with FFmpeg's legacy 0.5-2 range."""
    filters: list[str] = []
    while tempo < 0.5:
        filters.append("atempo=0.5")
        tempo /= 0.5
    while tempo > 2.0:
        filters.append("atempo=2")
        tempo /= 2.0
    if not math.isclose(tempo, 1.0, rel_tol=0.0, abs_tol=1e-9):
        filters.append(f"atempo={tempo:.10f}")
    return filters


def _instrumental_filters(transpose: int, gain: float) -> str:
    """Pitch-shift the 44.1 kHz instrumental while preserving its duration."""
    filters: list[str] = []
    if transpose:
        pitch_ratio = 2 ** (transpose / 12)
        filters.extend(
            [
                f"asetrate={44100 * pitch_ratio:.10f}",
                "aresample=44100",
                *_atempo_filters(1 / pitch_ratio),
            ]
        )
    filters.append(f"volume={gain:.4f}")
    return ",".join(filters)


def _mix(
    instrumental: Path,
    vocal: Path,
    target: Path,
    instrumental_transpose: int,
    vocal_gain: float,
    instrumental_gain: float,
) -> None:
    mix_filter = (
        f"[0:a]{_instrumental_filters(instrumental_transpose, instrumental_gain)}[bg];"
        f"[1:a]volume={vocal_gain:.4f}[voice];"
        # The bundled FFmpeg predates amix's normalize option. Its legacy
        # behavior divides a two-input mix by two, so compensate afterwards.
        "[bg][voice]amix=inputs=2:duration=longest:dropout_transition=0,volume=2,"
        "alimiter=limit=0.95:attack=5:release=50[out]"
    )
    _run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(instrumental),
            "-i",
            str(vocal),
            "-filter_complex",
            mix_filter,
            "-map",
            "[out]",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "320k",
            "-id3v2_version",
            "3",
            str(target),
        ],
        label="final mix",
    )


def run_cover_pipeline(
    job_dir: Path,
    uploaded: Path,
    audio_hash: str,
    source_filename: str,
    model_name: str,
    transpose: int,
    instrumental_transpose: int,
    index_rate: float,
    rms_mix_rate: float,
    protect: float,
    vocal_gain: float,
    instrumental_gain: float,
) -> tuple[Path, dict[str, object]]:
    model = _resolve_model(model_name)
    cached = _restore_cached_stems(audio_hash, job_dir)
    cache_hit = cached is not None
    if cached:
        dry_vocal, instrumental = cached
    else:
        normalized = job_dir / "input.wav"
        _normalize_audio(uploaded, normalized)
        vocal, instrumental = _separate(
            normalized,
            "去伴奏",
            job_dir / "vocals",
            job_dir / "instrumental",
        )
        dry_vocal, _reverb = _separate(
            vocal,
            "去混响",
            job_dir / "dry_vocal",
            job_dir / "reverb",
        )
        _store_cached_stems(audio_hash, dry_vocal, instrumental, source_filename)

    converted = job_dir / "converted_vocal.wav"
    index = _convert_vocal(
        dry_vocal,
        converted,
        model,
        transpose,
        index_rate,
        rms_mix_rate,
        protect,
    )
    result = job_dir / "ai_cover.mp3"
    _mix(
        instrumental,
        converted,
        result,
        instrumental_transpose,
        vocal_gain,
        instrumental_gain,
    )
    return result, {
        "model": model.stem,
        "index": index.name if index else None,
        "transpose": transpose,
        "instrumental_transpose": instrumental_transpose,
        "cache_hit": cache_hit,
    }


async def _save_upload(upload: UploadFile, destination: Path) -> tuple[int, str]:
    total = 0
    digest = hashlib.sha256()
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise ValueError(
                    f"upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB"
                )
            output.write(chunk)
            digest.update(chunk)
    if total == 0:
        raise ValueError("uploaded file is empty")
    return total, digest.hexdigest()


@app.on_event("startup")
async def discard_obsolete_cache() -> None:
    await asyncio.to_thread(_discard_obsolete_cache)


@app.get("/health")
async def health(
    x_ai_cover_token: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    _auth(x_ai_cover_token)
    cache = await asyncio.to_thread(_cache_stats)
    return {
        "ok": True,
        "busy": job_lock.locked(),
        "models": len(_model_files()),
        "max_audio_seconds": MAX_AUDIO_SECONDS,
        "cache": cache,
    }


@app.get("/models")
async def models(
    x_ai_cover_token: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    _auth(x_ai_cover_token)
    return {"models": list_models()}


@app.get("/cache")
async def cache_status(
    query: Annotated[str, Query(max_length=200)] = "",
    x_ai_cover_token: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    _auth(x_ai_cover_token)
    return await asyncio.to_thread(_cache_snapshot, query)


@app.post("/cache/delete")
async def delete_cache_entries(
    ids: Annotated[list[str], Body(embed=True, min_length=1, max_length=500)],
    x_ai_cover_token: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    _auth(x_ai_cover_token)
    normalized_ids = list(dict.fromkeys(str(item).strip().lower() for item in ids))
    if any(not _is_audio_hash(item) for item in normalized_ids):
        raise HTTPException(status_code=422, detail="ids contains an invalid cache ID")
    async with job_lock:
        removed = await asyncio.to_thread(_delete_cache_entries, normalized_ids)
    return {"ok": True, "removed": removed, **_cache_snapshot()}


@app.delete("/cache")
async def clear_cache(
    x_ai_cover_token: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    _auth(x_ai_cover_token)
    async with job_lock:
        removed = await asyncio.to_thread(_clear_cache)
    return {"ok": True, "removed": removed, "cache": _cache_stats()}


@app.post("/cover")
async def cover(
    audio: Annotated[UploadFile, File()],
    model: Annotated[str, Form()],
    transpose: Annotated[int, Form()] = 0,
    instrumental_transpose: Annotated[int, Form()] = 0,
    index_rate: Annotated[float, Form()] = 0.75,
    rms_mix_rate: Annotated[float, Form()] = 0.25,
    protect: Annotated[float, Form()] = 0.25,
    vocal_gain: Annotated[float, Form()] = 1.0,
    instrumental_gain: Annotated[float, Form()] = 1.0,
    x_ai_cover_token: Annotated[str | None, Header()] = None,
) -> FileResponse:
    _auth(x_ai_cover_token)
    if not -24 <= transpose <= 24:
        raise HTTPException(
            status_code=422, detail="transpose must be between -24 and 24"
        )
    if not -24 <= instrumental_transpose <= 24:
        raise HTTPException(
            status_code=422,
            detail="instrumental_transpose must be between -24 and 24",
        )
    for name, value in {
        "index_rate": index_rate,
        "rms_mix_rate": rms_mix_rate,
        "protect": protect,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise HTTPException(
                status_code=422, detail=f"{name} must be between 0 and 1"
            )
    for name, value in {
        "vocal_gain": vocal_gain,
        "instrumental_gain": instrumental_gain,
    }.items():
        if not 0.0 <= value <= 3.0:
            raise HTTPException(
                status_code=422, detail=f"{name} must be between 0 and 3"
            )

    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    job_dir = JOB_ROOT / uuid.uuid4().hex
    job_dir.mkdir(parents=True)
    suffix = Path(audio.filename or "audio.bin").suffix.lower()
    if len(suffix) > 10 or not suffix.replace(".", "").isalnum():
        suffix = ".bin"
    uploaded = job_dir / f"upload{suffix}"

    try:
        _size, audio_hash = await _save_upload(audio, uploaded)
        duration = await asyncio.to_thread(_probe_audio, uploaded)
        logger.info(
            "Queued AI cover: file=%s duration=%.1fs model=%s "
            "vocal_transpose=%s instrumental_transpose=%s",
            audio.filename,
            duration,
            model,
            transpose,
            instrumental_transpose,
        )
        async with job_lock:
            result, metadata = await asyncio.to_thread(
                run_cover_pipeline,
                job_dir,
                uploaded,
                audio_hash,
                audio.filename or "未知音频",
                model,
                transpose,
                instrumental_transpose,
                index_rate,
                rms_mix_rate,
                protect,
                vocal_gain,
                instrumental_gain,
            )
        headers = {
            "X-AI-Cover-Model": quote(str(metadata["model"]), safe=""),
            "X-AI-Cover-Index": quote(str(metadata["index"] or ""), safe=""),
            "X-AI-Cover-Transpose": str(metadata["transpose"]),
            "X-AI-Cover-Instrumental-Transpose": str(
                metadata["instrumental_transpose"]
            ),
            "X-AI-Cover-Cache": "hit" if metadata["cache_hit"] else "miss",
        }
        return FileResponse(
            result,
            media_type="audio/mpeg",
            filename=f"ai_cover_{uuid.uuid4().hex[:8]}.mp3",
            headers=headers,
            background=BackgroundTask(shutil.rmtree, job_dir, True),
        )
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except (ValueError, FileNotFoundError) as error:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("AI cover job failed")
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"{type(error).__name__}: {error}",
        ) from error


if __name__ == "__main__":
    import uvicorn

    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ENTRY_ROOT.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
