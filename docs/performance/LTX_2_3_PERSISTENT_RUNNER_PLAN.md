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
