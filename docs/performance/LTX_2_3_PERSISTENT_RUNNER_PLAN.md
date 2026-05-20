# LTX 2.3 Persistent Runner Plan

Goal: one Python process, load heavy models once, generate many clips/chunks sequentially.

Do **not** change pipeline internals yet. No tiling fixes, no fps experiments, no video-only transformer work. First make stable runner.

**What To Build**
Create:

`scripts/generate_persistent.py`

V1 scope:
- LTX 2.3 distilled only.
- Silent video only.
- `MyPipeline` only.
- `cfg=1` default.
- One prompt, many chunks/seeds.
- Sequential chunks, no batching yet.
- Save manifest with exact args/timings/output paths.

**Runner Structure**
`PersistentLTX23Runner`:

- Load tokenizer once.
- Load Gemma once.
- Load AV text encoder once.
- Load AV transformer once with `fast_mode`.
- Load VAE decoder once.
- Create `MyPipeline` once.
- Cache prompt embedding by prompt string.
- Loop chunks:
  - set seed
  - create `MyPipelineConfig`
  - run pipeline
  - save mp4
  - record timing
  - delete chunk arrays, keep models alive

**CLI Shape**
Recommended minimal CLI:

```bash
uv run python scripts/generate_persistent.py \
  "A golden retriever running through a sunny meadow" \
  --weights "/Users/vitorfrasson/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/76730e634e70a28f4e8d51f5e29c08e40e2d8e74/ltx-2.3-22b-distilled.safetensors" \
  --gemma-path "/Users/vitorfrasson/.cache/huggingface/hub/models--mlx-community--gemma-3-12b-it-bf16/snapshots/b8b9b412cb795bd6115fdce8c9a0ef0d1664db3a" \
  --output-dir "outputs/persistent_ltx23_v1" \
  --prefix "dog_meadow" \
  --chunks 3 \
  --height 800 \
  --width 448 \
  --frames 121 \
  --steps 8 \
  --cfg 1 \
  --seed 42 \
  --fast-mode
```

Outputs:
- `dog_meadow_chunk_000_seed42.mp4`
- `dog_meadow_chunk_001_seed43.mp4`
- `dog_meadow_chunk_002_seed44.mp4`
- `run_manifest.json`

**Manifest**
`run_manifest.json` should include:
- prompt
- model paths
- generation args
- output paths
- seed per chunk
- timing per chunk
- total wall time
- average chunk wall time
- estimated generated minutes per wall minute

**Key Simplicity Choices**
- Use `MyPipeline` unchanged.
- Pass dummy `negative_encoding = mx.zeros_like(positive_encoding)` because `cfg=1`, negative path unused.
- Reuse current V2 AV text encoder initially, even though it computes audio encoding. Prompt cache makes this acceptable for v1.
- Skip VAE encoder unless later adding image conditioning.
- Keep generation FPS same as current AV path first: `25.0`; add `--generation-fps` knob but default current behavior.

**Validation Plan**
After implementation:
1. Compile script with `uv run python -m py_compile scripts/generate_persistent.py`.
2. Smoke run tiny valid config: `--frames 9 --steps 1`.
3. Real benchmark: `448x800`, `121f`, `8 steps`, `2 chunks`.
4. Compare chunk 0 wall time vs old CLI run.
5. Confirm second chunk avoids model reload and is much faster than first total-per-run.

**Expected Win**
First chunk still pays load cost. Later chunks skip:
- Gemma load
- text encoder load
- transformer load
- VAE load

Main metric: chunk 2+ wall time, not total first-run wall.

## Next Fix: Verify `decode_tiled()` Dead First Loop

Source confirms bug:

- `tiling.py:295-348`: decodes every tile, `mx.eval()`, computes masks, then discards work.
- `tiling.py:351-407`: resets buffers and decodes same tiles again for real output.
- Our `448x800 121f` default tiling = `6` tiles, current code likely calls VAE decoder `12` times.

Plan:

1. Remove dead first loop only.
2. Keep real accumulation loop unchanged.
3. Run compile: `uv run python -m py_compile LTX_2_MLX/model/video_vae/tiling.py`
4. Run same persistent benchmark to compare chunk 2 vs `202.62s`.
5. Review output clips for visual regression.

Risk: low behavior risk, medium shared-code risk. It touches all tiled VAE users, but removed code has no output path. Speed win may be modest if transformer denoise dominates, but this is clearest high-value fix.

Status: implemented by removing the dead first tile loop only. Compile passed with `uv run python -m py_compile LTX_2_MLX/model/video_vae/tiling.py`.

Validation result:

- Same benchmark: `448x800`, `121f`, `8 steps`, `2 chunks`.
- Previous total: `550.36s`; fixed total: `491.50s` (`10.7%` faster).
- Previous chunk 2: `202.62s`; fixed chunk 2: `181.46s` (`10.4%` faster).
- Output dir: `outputs/persistent_ltx23_tiling_fix_448x800_121f_8step_v1`.

## Next Runner: Long-Lived Local Server

Goal: keep models loaded across separate user requests, not just across chunks inside one CLI run.

Create:

`scripts/serve_persistent.py`

Server scope:

- LTX 2.3 distilled only.
- Silent video only.
- `MyPipeline` only.
- `cfg=1` default.
- One loaded model set per process.
- One generation at a time guarded by a process-local lock.
- HTTP bound to `127.0.0.1` by default.
- Reuse `PersistentLTX23Runner` instead of duplicating loader/generation code.
- Keep tokenizer, AV text encoder, AV transformer, VAE decoder, `MyPipeline`, and prompt embedding cache alive until server exits.
- Optionally unload Gemma after prompt encoding to reduce idle memory.
- Exit automatically after idle timeout during inactive batch windows.

Startup CLI:

```bash
uv run python scripts/serve_persistent.py \
  --weights "/Users/vitorfrasson/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/76730e634e70a28f4e8d51f5e29c08e40e2d8e74/ltx-2.3-22b-distilled.safetensors" \
  --gemma-path "/Users/vitorfrasson/.cache/huggingface/hub/models--mlx-community--gemma-3-12b-it-bf16/snapshots/b8b9b412cb795bd6115fdce8c9a0ef0d1664db3a" \
  --fast-mode \
  --unload-gemma-after-encode \
  --idle-timeout-minutes 30 \
  --host 127.0.0.1 \
  --port 8787 \
  --state-dir outputs/persistent_server_state
```

Request shape:

```bash
curl -X POST "http://127.0.0.1:8787/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A golden retriever running through a sunny meadow",
    "output_dir": "outputs/daemon_dog_meadow_v1",
    "prefix": "dog_meadow",
    "chunks": 2,
    "height": 800,
    "width": 448,
    "frames": 121,
    "steps": 8,
    "seed": 42,
    "generation_fps": 25,
    "output_fps": 24
  }'
```

Endpoints:

- `GET /health`: returns loaded status, PID, uptime, model paths, load timings, cache size, current/last request summary.
- `GET /status`: returns whether a generation is running and recent request history.
- `POST /generate`: runs generation synchronously and returns manifest JSON plus output paths.
- `POST /shutdown`: local-only graceful shutdown, enabled by default. Use `--disable-shutdown-endpoint` only if needed.

Inspectability requirements:

- Print startup banner with PID, host, port, state dir, model paths, and load timings.
- Write `server.pid` in `--state-dir` after successful startup.
- Write `server_state.json` in `--state-dir` after startup and after each request.
- Write one request manifest per generation in that request's `output_dir`.
- Log each request start/end with prompt preview, dimensions, frames, steps, seeds, output dir, timing, and success/failure.
- Return structured JSON errors; do not hide tracebacks in terminal logs during development.
- Report `gemma_loaded`, `idle_seconds`, `idle_timeout_minutes`, and `prompt_cache_size` in health/status/state.

Kill/cleanup requirements:

- `Ctrl-C` in server terminal should exit cleanly.
- PID file kill path should be documented:

```bash
kill "$(cat outputs/persistent_server_state/server.pid)"
```

- Force kill path should be documented only as fallback:

```bash
kill -9 "$(cat outputs/persistent_server_state/server.pid)"
```

- Server should remove `server.pid` on clean exit when possible.
- If startup sees an existing `server.pid`, it should check whether that PID is alive and refuse to start unless stale.
- Idle timeout should self-stop the server when no generation is running.

Safety constraints:

- Bind to `127.0.0.1` by default, not `0.0.0.0`.
- No auth in V1 because localhost-only; if binding externally later, add auth first.
- Keep synchronous single-request execution first; no job queue until needed.
- Reject invalid `frames`, dimensions, `cfg != 1`, non-positive FPS/speed, and unsafe empty output dir.

Expected win:

- First server startup still pays load cost.
- Later `/generate` requests skip model loads.
- Repeated prompts skip prompt encoding through existing prompt cache.
- Different prompts pay prompt encode only, not model reload.
- With `--unload-gemma-after-encode`, different uncached prompts pay Gemma reload + prompt encode, but idle memory is lower.

Session-hot policy:

- Use server during active batch windows, not as an all-day daemon.
- Default `--idle-timeout-minutes 30` stops server after inactive batch windows.
- Use `--unload-gemma-after-encode` when idle memory matters more than fastest first uncached prompt.
- Keep repeated prompt embeddings cached so repeated prompts avoid Gemma even after unload.

Implementation status: `scripts/serve_persistent.py` added.

Validation result:

- Compile passed: `uv run python -m py_compile scripts/serve_persistent.py scripts/generate_persistent.py`.
- Help passed: `uv run python scripts/serve_persistent.py --help`.
- Smoke server loaded on `127.0.0.1:8788` with PID/state files in `outputs/persistent_server_smoke_state`.
- First tiny request: `384x576`, `9f`, `1 step`, seed `42`, total `83.87s`, prompt encode `79.07s`, chunk `4.76s`.
- Second same-prompt request: `384x576`, `9f`, `1 step`, seed `43`, total `2.50s`, prompt encode `0.000004s`, chunk `2.46s`.
- `GET /status` reported `prompt_cache_size=1`, recent request history, output paths, and server state.
- `POST /shutdown` stopped the smoke server and removed `server.pid`.

Session-hot strategy status:

- `--idle-timeout-minutes` added; default is `30`, set `<=0` to disable.
- `/shutdown` is enabled by default for localhost; `--disable-shutdown-endpoint` can disable it.
- `--unload-gemma-after-encode` added; Gemma loads lazily for uncached prompts and unloads after prompt encoding.
- `/health`, `/status`, and `server_state.json` report whether Gemma is currently loaded.

Session-hot validation:

- Compile passed after policy changes: `uv run python -m py_compile scripts/generate_persistent.py scripts/serve_persistent.py`.
- Startup with `--unload-gemma-after-encode` reported `gemma_loaded=false`, `gemma_seconds=0.0`, and `gemma_loaded_at_startup=false`.
- First uncached tiny request reloaded Gemma once, encoded prompt, unloaded Gemma, and completed successfully.
- Follow-up `/status` reported `prompt_cache_size=1` and `gemma_loaded=false`.
- Short idle timeout test with `--idle-timeout-minutes 0.2` self-stopped and removed `server.pid`.
- Cached same-prompt request with Gemma unloaded completed in `2.05s`; prompt encode was `0.000003s`, `gemma_reload_count` stayed `1`, and `gemma_loaded=false` after completion.
- Full session-hot benchmark with `--unload-gemma-after-encode --idle-timeout-minutes 30`, `448x800`, `121f`, `8 steps`, `2 chunks`: total `361.90s`, chunk 1 `152.73s`, chunk 2 `183.45s`, prompt encode `24.75s`, Gemma reload `20.20s`.
- After the full request, `/status` reported `prompt_cache_size=1` and `gemma_loaded=false`.
- Idle server memory after Gemma unload was `49G` by `top`, compared with previous all-hot idle process around `93G`.
- System-wide memory free after full request was `84%` by `memory_pressure`.
- Output dir: `outputs/persistent_server_session_hot_full_v1`.

Inspect commands:

```bash
curl "http://127.0.0.1:8787/health"
curl "http://127.0.0.1:8787/status"
cat outputs/persistent_server_state/server_state.json
cat outputs/persistent_server_state/server.pid
```

Kill commands:

```bash
curl -X POST "http://127.0.0.1:8787/shutdown"
kill "$(cat outputs/persistent_server_state/server.pid)"
kill -9 "$(cat outputs/persistent_server_state/server.pid)"
```
