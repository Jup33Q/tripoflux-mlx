#!/usr/bin/env python
"""End-to-end web test client for the TripoFlux MLX server.

Simulates exactly what the browser frontend does, over HTTP:

    1. GET  /                      – load the page (picks up any cookies the
                                     server sets; httpx keeps a cookie jar,
                                     so this client is session-faithful)
    2. POST /api/generate          – submit a job
    3. GET  /api/stream/{job_id}   – consume the SSE stream and render the
                                     same per-stage progress the UI shows
                                     (FLUX / BiRefNet / TripoSplat bars)
    4. GET  /api/result/{job_id}/* – download image / rgba / ply / spz and
                                     verify they are non-empty

Usage:
    python scripts/web_test.py --prompt "a red sports car" --seed 42
    python scripts/web_test.py --url http://127.0.0.1:8000 --output-dir /tmp/tf_out
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

STAGES = ("flux", "birefnet", "triposplat")
STAGE_LABELS = {
    "flux": "FLUX Image",
    "birefnet": "Background Removal",
    "triposplat": "TripoSplat 3D",
}


def render_bar(frac: float, width: int = 30) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def print_stage_line(stage: str, frac: float) -> None:
    label = STAGE_LABELS.get(stage, stage)
    line = f"  {label:<20} {render_bar(frac)} {frac * 100:5.1f}%"
    # \r + clear-to-EOL keeps each stage updating in place; use ANSI sparingly
    sys.stdout.write("\r\x1b[K" + line)
    sys.stdout.flush()


def end_stage_line(stage: str, frac: float, elapsed: float) -> None:
    label = STAGE_LABELS.get(stage, stage)
    sys.stdout.write(
        "\r\x1b[K" + f"  {label:<20} {render_bar(frac)} {frac * 100:5.1f}%  ({elapsed:.1f}s)\n"
    )
    sys.stdout.flush()


def stream_job(client: httpx.Client, base: str, job_id: str) -> dict:
    """Consume the SSE stream, rendering per-stage progress. Returns final event."""
    stage_start = time.monotonic()
    last_stage = None
    last_frac = 0.0
    final: dict = {}

    with client.stream("GET", f"{base}/api/stream/{job_id}", timeout=None) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            event = json.loads(line[len("data:"):].strip())

            if event.get("log"):
                # keep terminal-log parity with the frontend's log pane
                sys.stdout.write("\r\x1b[K" + f"  [log] {event['log']}\n")
                sys.stdout.flush()

            stage = event.get("stage")
            frac = event.get("stage_progress")
            if stage in STAGES and frac is not None:
                if stage != last_stage:
                    if last_stage is not None:
                        end_stage_line(last_stage, last_frac, time.monotonic() - stage_start)
                    stage_start = time.monotonic()
                    last_stage = stage
                last_frac = frac
                print_stage_line(stage, frac)

            if event.get("image_ready"):
                sys.stdout.write("\r\x1b[K" + f"  [preview] {event['image_ready']} image ready\n")
                sys.stdout.flush()

            status = event.get("status")
            if status in ("completed", "failed"):
                if last_stage is not None:
                    end_stage_line(last_stage, 1.0 if status == "completed" else last_frac,
                                   time.monotonic() - stage_start)
                final = event
                break

    return final


def verify_result(client: httpx.Client, base: str, job_id: str, kind: str,
                  out_dir: Path) -> int:
    resp = client.get(f"{base}/api/result/{job_id}/{kind}", timeout=120)
    resp.raise_for_status()
    data = resp.content
    assert len(data) > 0, f"{kind}: empty response"
    suffix = {"image": ".png", "rgba": ".png", "ply": ".ply", "spz": ".spz"}[kind]
    path = out_dir / f"{job_id[:8]}_{kind}{suffix}"
    path.write_bytes(data)
    return len(data)


def main() -> int:
    ap = argparse.ArgumentParser(description="TripoFlux web end-to-end test client")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--prompt", default="A red sports car, studio lighting, product photography")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--num-gaussians", type=int, default=262144)
    ap.add_argument("--flux-quantize", type=int, default=8, choices=[4, 8])
    ap.add_argument("--output-dir", default="/tmp/tripoflux_web_test")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    # trust_env=False: bypass any http_proxy/https_proxy env vars — this client
    # must talk to the local server directly, not through a system proxy.
    # httpx.Client keeps a cookie jar across requests, matching browser behavior.
    with httpx.Client(cookies=httpx.Cookies(), trust_env=False) as client:
        # 1. load the frontend page first, like a browser would
        page = client.get(f"{base}/", timeout=30)
        page.raise_for_status()
        assert "TripoFlux" in page.text, "frontend page did not load"
        print(f"[1/4] Frontend page OK ({len(page.text)} bytes, cookies: {dict(client.cookies) or 'none'})")

        # 2. submit the job
        payload = {
            "prompt": args.prompt,
            "seed": args.seed,
            "width": args.width,
            "height": args.height,
            "num_gaussians": args.num_gaussians,
            "flux_quantize": args.flux_quantize,
        }
        resp = client.post(f"{base}/api/generate", json=payload, timeout=30)
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
        print(f"[2/4] Job submitted: {job_id}")

        # 3. stream progress
        print("[3/4] Streaming progress:")
        final = stream_job(client, base, job_id)
        if final.get("status") != "completed":
            print(f"FAILED: {final.get('error', 'unknown error')}")
            return 1

        # 4. verify artifacts
        print("[4/4] Verifying artifacts:")
        ok = True
        for kind in ("image", "rgba", "ply", "spz"):
            try:
                size = verify_result(client, base, job_id, kind, out_dir)
                print(f"  {kind:<6} OK  {size / 1e6:8.2f} MB -> {out_dir}")
            except Exception as exc:
                print(f"  {kind:<6} FAIL: {exc}")
                ok = False

    total = time.monotonic() - t0
    print(f"\n{'WEB TEST PASSED' if ok else 'WEB TEST FAILED'} in {total:.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
