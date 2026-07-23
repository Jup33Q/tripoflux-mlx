"""FastAPI server for the TripoFlux MLX pipeline."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .pipeline import PipelineConfig, TripoFluxPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="TripoFlux MLX", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store: {job_id: {"status": ..., "result": ...}}
_jobs: dict[str, dict] = {}

_pipeline: Optional[TripoFluxPipeline] = None


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    width: int = 1024
    height: int = 1024
    num_gaussians: int = 262144
    flux_quantize: Optional[int] = Field(None, description="4 or 8 for mflux quantization")


class GenerateResponse(BaseModel):
    job_id: str


class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    stage: str


def get_pipeline() -> TripoFluxPipeline:
    global _pipeline
    if _pipeline is None:
        config_path = os.environ.get(
            "TRIPOFLUX_CONFIG",
            str(Path(__file__).resolve().parent.parent / "configs" / "default.yaml"),
        )
        from .pipeline import load_pipeline_from_yaml

        _pipeline = load_pipeline_from_yaml(config_path)
    return _pipeline


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    index_file = frontend_dir / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/frontend/{file_path:path}")
async def frontend_static(file_path: str):
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    target = frontend_dir / file_path
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(target)


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "stage": "queued",
        "request": req.model_dump(),
        "result": None,
    }
    return GenerateResponse(job_id=job_id)


@app.get("/api/status/{job_id}", response_model=StatusResponse)
async def status(job_id: str) -> StatusResponse:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        stage=job["stage"],
    )


@app.get("/api/result/{job_id}/image")
async def result_image(job_id: str):
    job = _jobs.get(job_id)
    if job is None or job["result"] is None:
        raise HTTPException(status_code=404, detail="result not ready")
    img = job["result"].generated_image
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/result/{job_id}/rgba")
async def result_rgba(job_id: str):
    job = _jobs.get(job_id)
    if job is None or job["result"] is None:
        raise HTTPException(status_code=404, detail="result not ready")
    img = job["result"].rgba_image
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/result/{job_id}/splat")
async def result_splat(job_id: str):
    job = _jobs.get(job_id)
    if job is None or job["result"] is None:
        raise HTTPException(status_code=404, detail="result not ready")
    return StreamingResponse(
        io.BytesIO(job["result"].splat_bytes),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.splat"'},
    )


@app.get("/api/result/{job_id}/ply")
async def result_ply(job_id: str):
    job = _jobs.get(job_id)
    if job is None or job["result"] is None:
        raise HTTPException(status_code=404, detail="result not ready")
    return StreamingResponse(
        io.BytesIO(job["result"].ply_bytes),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.ply"'},
    )


@app.get("/api/result/{job_id}/spz")
async def result_spz(job_id: str):
    job = _jobs.get(job_id)
    if job is None or job["result"] is None:
        raise HTTPException(status_code=404, detail="result not ready")
    return StreamingResponse(
        io.BytesIO(job["result"].spz_bytes),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.spz"'},
    )


@app.get("/api/result/{job_id}/stage/{stage}")
async def result_stage(job_id: str, stage: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    img = None
    if stage == "flux":
        img = job.get("flux_image")
    elif stage == "rgba":
        img = job.get("rgba_image")
    if img is None:
        raise HTTPException(status_code=404, detail="stage image not ready")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/stream/{job_id}")
async def stream(job_id: str) -> StreamingResponse:
    async def event_generator() -> AsyncGenerator[str, None]:
        job = _jobs.get(job_id)
        if job is None:
            yield f"data: {json.dumps({'error': 'job not found'})}\n\n"
            return

        if job["status"] == "completed":
            yield f"data: {json.dumps({'status': 'completed', 'progress': 1.0, 'stage': 'done'})}\n\n"
            return

        req = job["request"]
        job["status"] = "running"
        job["progress"] = 0.0
        job["stage"] = "starting"
        job["log"] = []

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        _DONE = object()

        # Overall-progress span of each stage: (start, end)
        STAGE_SPAN = {"flux": (0.0, 0.45), "birefnet": (0.45, 0.55), "triposplat": (0.55, 1.0)}

        def emit(payload: dict) -> None:
            if "stage" in payload:
                job["stage"] = payload["stage"]
            if "progress" in payload:
                job["progress"] = payload["progress"]
            loop.call_soon_threadsafe(queue.put_nowait, payload)

        def on_progress(stage: str, frac: float) -> None:
            lo, hi = STAGE_SPAN.get(stage, (0.0, 1.0))
            overall = lo + (hi - lo) * frac
            emit({
                "status": "running",
                "stage": stage,
                "stage_progress": round(frac, 4),
                "progress": round(overall, 4),
            })

        def run_job() -> None:
            try:
                pipeline = get_pipeline()

                # Stage 1: FLUX image generation
                emit({"status": "running", "stage": "flux", "stage_progress": 0.0,
                      "progress": 0.0, "log": "Generating image with FLUX.2-klein-9B..."})
                image = pipeline.generate_image(
                    prompt=req["prompt"],
                    seed=req.get("seed"),
                    width=req.get("width"),
                    height=req.get("height"),
                    flux_quantize=req.get("flux_quantize"),
                    negative_prompt=req.get("negative_prompt"),
                    progress=on_progress,
                )
                job["flux_image"] = image
                emit({"status": "running", "stage": "flux", "stage_progress": 1.0,
                      "progress": STAGE_SPAN["flux"][1], "log": "Image generated",
                      "image_ready": "flux"})

                # Stage 2: BiRefNet background removal
                emit({"status": "running", "stage": "birefnet", "stage_progress": 0.0,
                      "progress": STAGE_SPAN["birefnet"][0],
                      "log": "Removing background with BiRefNet..."})
                rgba = pipeline.remove_background(image, progress=on_progress)
                job["rgba_image"] = rgba
                emit({"status": "running", "stage": "birefnet", "stage_progress": 1.0,
                      "progress": STAGE_SPAN["birefnet"][1], "log": "Background removed",
                      "image_ready": "rgba"})

                # Stage 3: TripoSplat generation
                emit({"status": "running", "stage": "triposplat", "stage_progress": 0.0,
                      "progress": STAGE_SPAN["triposplat"][0],
                      "log": "Generating 3D Gaussian Splat..."})
                ply, splat, spz, prepared = pipeline.generate_splat(
                    rgba,
                    num_gaussians=req.get("num_gaussians"),
                    seed=req.get("seed"),
                    progress=on_progress,
                )

                from .pipeline import PipelineResult
                result = PipelineResult(
                    prompt=req["prompt"],
                    generated_image=image,
                    rgba_image=rgba,
                    prepared_image=prepared,
                    ply_bytes=ply,
                    splat_bytes=splat,
                    spz_bytes=spz,
                    metadata={
                        "seed": req.get("seed") or pipeline.cfg.seed,
                        "width": req.get("width") or pipeline.cfg.image_width,
                        "height": req.get("height") or pipeline.cfg.image_height,
                        "num_gaussians": req.get("num_gaussians") or pipeline.cfg.num_gaussians,
                        "flux_quantize": req.get("flux_quantize") or pipeline.cfg.flux_quantize,
                        "negative_prompt": req.get("negative_prompt"),
                        "flux_backend": pipeline.cfg.flux_backend,
                        "birefnet_backend": pipeline.cfg.birefnet_backend,
                        "triposplat_backend": pipeline.cfg.triposplat_backend,
                    },
                )
                job["result"] = result
                job["status"] = "completed"
                job["progress"] = 1.0
                job["stage"] = "done"
                emit({"status": "completed", "stage": "done", "stage_progress": 1.0,
                      "progress": 1.0, "log": "All done!"})
            except Exception as exc:  # pragma: no cover
                logger.exception("job %s failed", job_id)
                job["status"] = "failed"
                job["error"] = str(exc)
                emit({"status": "failed", "error": str(exc), "log": f"Error: {exc}"})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        worker = threading.Thread(target=run_job, name=f"job-{job_id[:8]}", daemon=True)

        # Send immediate started event so the frontend knows the stream is alive.
        yield f"data: {json.dumps({'status': 'running', 'progress': 0.0, 'stage': 'starting', 'log': 'Job started'})}\n\n"
        worker.start()

        while True:
            item = await queue.get()
            if item is _DONE:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def main() -> None:
    host = os.environ.get("TRIPOFLUX_HOST", "127.0.0.1")
    port = int(os.environ.get("TRIPOFLUX_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
