#!/usr/bin/env python3
"""Long-lived localhost server for persistent silent LTX-2.3 generation.

Purpose:
    Keep tokenizer, AV text encoder, AV transformer, VAE decoder, and MyPipeline
    loaded across separate generation requests. Gemma can be unloaded between
    requests to reduce idle memory.

Scope:
    V1 is localhost-only, LTX-2.3/V2 distilled-only, silent video-only, cfg=1,
    and one generation at a time. It reuses PersistentLTX23Runner so CLI and
    server generation behavior stay aligned.

Inspectability:
    Startup prints PID/model/load info, writes `server.pid` and
    `server_state.json`, and exposes `/health` plus `/status`. Kill with
    `kill "$(cat outputs/persistent_server_state/server.pid)"`.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_persistent import PersistentLTX23Runner, build_manifest  # noqa: E402


class GenerateRequest(BaseModel):
    prompt: str
    output_dir: str
    prefix: str = "persistent_ltx23"
    chunks: int = Field(default=1, ge=1)
    height: int = 800
    width: int = 448
    frames: int = 121
    steps: int = 8
    cfg: float = 1.0
    seed: int = 42
    seed_stride: int = 1
    generation_fps: float = 25.0
    output_fps: float = 24.0
    speed: float = 1.0


class PersistentServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.pid = os.getpid()
        self.started_at = time.time()
        self.state_dir = Path(args.state_dir)
        self.pid_path = self.state_dir / "server.pid"
        self.state_path = self.state_dir / "server_state.json"
        self.lock = threading.Lock()
        self.current_request: dict[str, Any] | None = None
        self.last_request: dict[str, Any] | None = None
        self.recent_requests: list[dict[str, Any]] = []
        self.request_count = 0
        self.last_activity_at = time.time()

        self._prepare_state_dir()
        compute_dtype = mx.float32 if args.fp32 else mx.float16
        self.runner = PersistentLTX23Runner(
            weights_path=args.weights,
            gemma_path=args.gemma_path,
            compute_dtype=compute_dtype,
            fast_mode=args.fast_mode,
            low_memory=args.low_memory,
            max_length=args.max_length,
            load_gemma=not args.unload_gemma_after_encode,
        )
        if args.unload_gemma_after_encode:
            self.runner.load_timings["gemma_loaded_at_startup"] = False
        else:
            self.runner.load_timings["gemma_loaded_at_startup"] = True
        self.last_activity_at = time.time()
        self._write_pid_file()
        atexit.register(self.cleanup)
        self.start_idle_watchdog()
        self.write_state()
        self.print_startup_banner()

    def _prepare_state_dir(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if not self.pid_path.exists():
            return
        try:
            existing_pid = int(self.pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            self.pid_path.unlink()
            return

        if existing_pid == self.pid:
            return
        try:
            os.kill(existing_pid, 0)
        except ProcessLookupError:
            self.pid_path.unlink()
            return
        except PermissionError as exc:
            raise RuntimeError(
                f"Cannot inspect existing server PID {existing_pid} from {self.pid_path}"
            ) from exc
        raise RuntimeError(
            f"Server already appears to be running with PID {existing_pid}. "
            f"Kill it or remove stale PID file: {self.pid_path}"
        )

    def _write_pid_file(self) -> None:
        self.pid_path.write_text(f"{self.pid}\n", encoding="utf-8")

    def cleanup(self) -> None:
        try:
            if self.pid_path.exists() and self.pid_path.read_text(encoding="utf-8").strip() == str(self.pid):
                self.pid_path.unlink()
        except OSError:
            pass

    def print_startup_banner(self) -> None:
        print("\nPersistent LTX 2.3 server loaded", flush=True)
        print(f"  PID: {self.pid}", flush=True)
        print(f"  URL: http://{self.args.host}:{self.args.port}", flush=True)
        print(f"  State dir: {self.state_dir}", flush=True)
        print(f"  Weights: {self.args.weights}", flush=True)
        print(f"  Gemma: {self.args.gemma_path}", flush=True)
        print(f"  Gemma loaded: {self.runner.gemma is not None}", flush=True)
        print(f"  Unload Gemma after encode: {self.args.unload_gemma_after_encode}", flush=True)
        print(f"  Idle timeout: {self.args.idle_timeout_minutes} minutes", flush=True)
        print(f"  Prompt cache: {len(self.runner.prompt_cache)} entries", flush=True)
        print("  Load timings:", flush=True)
        for name, value in self.runner.load_timings.items():
            if isinstance(value, bool):
                print(f"    {name}: {value}", flush=True)
            elif isinstance(value, (int, float)):
                print(f"    {name}: {value:.2f}s", flush=True)
            else:
                print(f"    {name}: {value}", flush=True)
        print(f"  Stop: kill \"$(cat {self.pid_path})\"", flush=True)
        print("", flush=True)

    def health(self) -> dict[str, Any]:
        return {
            "loaded": True,
            "pid": self.pid,
            "uptime_seconds": time.time() - self.started_at,
            "host": self.args.host,
            "port": self.args.port,
            "state_dir": str(self.state_dir),
            "weights": self.args.weights,
            "gemma_path": self.args.gemma_path,
            "load_timings": self.runner.load_timings,
            "prompt_cache_size": len(self.runner.prompt_cache),
            "gemma_loaded": self.runner.gemma is not None,
            "unload_gemma_after_encode": self.args.unload_gemma_after_encode,
            "idle_timeout_minutes": self.args.idle_timeout_minutes,
            "idle_seconds": time.time() - self.last_activity_at,
            "busy": self.lock.locked(),
            "current_request": self.current_request,
            "last_request": self.last_request,
        }

    def status(self) -> dict[str, Any]:
        data = self.health()
        data["recent_requests"] = self.recent_requests[-20:]
        return data

    def write_state(self) -> None:
        state = self.status()
        tmp_path = self.state_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp_path.replace(self.state_path)

    def generate(self, request: GenerateRequest) -> dict[str, Any]:
        self.validate_request(request)
        if not self.lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Generation already running")

        self.last_activity_at = time.time()
        self.request_count += 1
        request_id = f"req_{self.request_count:04d}"
        started_wall = time.time()
        started = time.perf_counter()
        summary = self.request_summary(request_id, request, "running", started_wall)
        self.current_request = summary
        self.write_state()
        print(f"[{request_id}] start {summary}", flush=True)

        try:
            output_dir = Path(request.output_dir).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)

            encode_started = time.perf_counter()
            self.runner.encode_prompt(request.prompt)
            prompt_encode_seconds = time.perf_counter() - encode_started
            if self.args.unload_gemma_after_encode:
                self.runner.unload_gemma()
                print(f"[{request_id}] Gemma unloaded after prompt encode", flush=True)

            chunks = []
            for chunk_index in range(request.chunks):
                seed = request.seed + chunk_index * request.seed_stride
                output_path = output_dir / f"{request.prefix}_chunk_{chunk_index:03d}_seed{seed}.mp4"
                print(f"[{request_id}] chunk {chunk_index + 1}/{request.chunks}: {output_path}", flush=True)
                chunks.append(
                    self.runner.generate_chunk(
                        prompt=request.prompt,
                        output_path=output_path,
                        height=request.height,
                        width=request.width,
                        frames=request.frames,
                        steps=request.steps,
                        seed=seed,
                        generation_fps=request.generation_fps,
                        output_fps=request.output_fps,
                        speed=request.speed,
                    )
                )

            total_seconds = time.perf_counter() - started
            manifest = build_manifest(
                args=self.manifest_args(request),
                load_timings=self.runner.load_timings,
                prompt_encode_seconds=prompt_encode_seconds,
                chunks=chunks,
                total_seconds=total_seconds,
            )
            manifest["server"] = {
                "request_id": request_id,
                "pid": self.pid,
                "uptime_seconds": time.time() - self.started_at,
                "state_dir": str(self.state_dir),
            }
            manifest_path = output_dir / "run_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            finished = self.request_summary(request_id, request, "completed", started_wall)
            finished["finished_at"] = time.time()
            finished["total_seconds"] = total_seconds
            finished["manifest_path"] = str(manifest_path)
            finished["output_paths"] = [chunk["output_path"] for chunk in chunks]
            self.last_request = finished
            self.recent_requests.append(finished)
            print(f"[{request_id}] completed in {total_seconds:.2f}s", flush=True)
            return manifest
        except Exception as exc:
            traceback.print_exc()
            failed = self.request_summary(request_id, request, "failed", started_wall)
            failed["finished_at"] = time.time()
            failed["error"] = str(exc)
            self.last_request = failed
            self.recent_requests.append(failed)
            raise HTTPException(status_code=500, detail={"error": str(exc), "request_id": request_id}) from exc
        finally:
            self.current_request = None
            self.last_activity_at = time.time()
            self.lock.release()
            self.write_state()

    def start_idle_watchdog(self) -> None:
        if self.args.idle_timeout_minutes <= 0:
            return
        threading.Thread(target=self._idle_watchdog, daemon=True).start()

    def _idle_watchdog(self) -> None:
        timeout_seconds = self.args.idle_timeout_minutes * 60
        while True:
            time.sleep(min(30, max(1, timeout_seconds / 4)))
            if self.lock.locked():
                continue
            idle_seconds = time.time() - self.last_activity_at
            if idle_seconds < timeout_seconds:
                continue
            print(
                f"Idle timeout reached after {idle_seconds:.0f}s; shutting down PID {self.pid}",
                flush=True,
            )
            os.kill(self.pid, signal.SIGINT)
            return

    def manifest_args(self, request: GenerateRequest) -> SimpleNamespace:
        return SimpleNamespace(
            prompt=request.prompt,
            weights=self.args.weights,
            gemma_path=self.args.gemma_path,
            height=request.height,
            width=request.width,
            frames=request.frames,
            steps=request.steps,
            cfg=request.cfg,
            seed=request.seed,
            seed_stride=request.seed_stride,
            generation_fps=request.generation_fps,
            output_fps=request.output_fps,
            speed=request.speed,
            fast_mode=self.args.fast_mode,
            low_memory=self.args.low_memory,
            fp32=self.args.fp32,
            chunks=request.chunks,
        )

    def request_summary(
        self,
        request_id: str,
        request: GenerateRequest,
        status: str,
        started_wall: float,
    ) -> dict[str, Any]:
        prompt_preview = request.prompt[:100]
        if len(request.prompt) > 100:
            prompt_preview += "..."
        return {
            "request_id": request_id,
            "status": status,
            "started_at": started_wall,
            "prompt_preview": prompt_preview,
            "output_dir": request.output_dir,
            "prefix": request.prefix,
            "chunks": request.chunks,
            "height": request.height,
            "width": request.width,
            "frames": request.frames,
            "steps": request.steps,
            "seed": request.seed,
        }

    def validate_request(self, request: GenerateRequest) -> None:
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="prompt must not be empty")
        if request.cfg != 1.0:
            raise HTTPException(status_code=400, detail="server V1 supports only cfg=1")
        if request.frames % 8 != 1:
            raise HTTPException(status_code=400, detail="frames must be 8*k + 1")
        if request.height % 32 != 0 or request.width % 32 != 0:
            raise HTTPException(status_code=400, detail="height and width must be divisible by 32")
        if request.generation_fps <= 0 or request.output_fps <= 0:
            raise HTTPException(status_code=400, detail="FPS values must be positive")
        if request.speed <= 0:
            raise HTTPException(status_code=400, detail="speed must be positive")
        output_dir = request.output_dir.strip()
        if not output_dir or output_dir in {".", "/"}:
            raise HTTPException(status_code=400, detail="output_dir must be explicit and not root/current dir")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-lived persistent silent LTX-2.3 server")
    parser.add_argument("--weights", required=True, help="LTX-2.3 distilled safetensors path")
    parser.add_argument("--gemma-path", required=True, help="Gemma 3 weights directory")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host; V1 allows localhost only")
    parser.add_argument("--port", type=int, default=8787, help="Bind port")
    parser.add_argument("--state-dir", default="outputs/persistent_server_state", help="PID/state directory")
    parser.add_argument("--max-length", type=int, default=1024, help="Tokenizer max length")
    parser.add_argument("--fast-mode", action="store_true", help="Skip intermediate transformer evals")
    parser.add_argument("--low-memory", action="store_true", help="Use more frequent transformer evals")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 compute instead of FP16")
    parser.add_argument(
        "--idle-timeout-minutes",
        type=float,
        default=30.0,
        help="Exit after this many idle minutes; set <=0 to disable",
    )
    parser.add_argument(
        "--unload-gemma-after-encode",
        action="store_true",
        help="Unload Gemma after caching prompt embeddings; reload for uncached prompts",
    )
    parser.add_argument(
        "--disable-shutdown-endpoint",
        action="store_true",
        help="Disable POST /shutdown; enabled by default on localhost",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("server V1 only binds to 127.0.0.1 or localhost")
    if args.port <= 0:
        raise ValueError("port must be positive")


def create_app(server: PersistentServer) -> FastAPI:
    app = FastAPI(title="Persistent LTX 2.3 Server")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return server.health()

    @app.get("/status")
    def status() -> dict[str, Any]:
        return server.status()

    @app.post("/generate")
    def generate(request: GenerateRequest) -> dict[str, Any]:
        return server.generate(request)

    @app.post("/shutdown")
    def shutdown() -> dict[str, Any]:
        if server.args.disable_shutdown_endpoint:
            raise HTTPException(status_code=403, detail="shutdown endpoint is disabled")

        def stop_process() -> None:
            time.sleep(0.2)
            os.kill(server.pid, signal.SIGINT)

        threading.Thread(target=stop_process, daemon=True).start()
        return {"status": "shutting_down", "pid": server.pid}

    return app


def main() -> None:
    args = parse_args()
    validate_args(args)
    server = PersistentServer(args)
    app = create_app(server)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
