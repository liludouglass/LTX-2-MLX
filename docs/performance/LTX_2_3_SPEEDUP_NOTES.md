# LTX 2.3 Speed Notes

Memory not wall. GPU compute, MLX sync, model reload are wall.

## Findings

- LTX 2.3 always uses `Audio-Video Pipeline`, even no-audio.
- Default AV path has `use_internal_audio_branch=True`; denoise audio latents even if output audio off.
- Before patch, `--fast-mode` existed but AV loader ignored it.
- CLI reloads Gemma + transformer + VAE each short clip; wastes about `25-35s` per run.
- `128GB` helps only if we keep models loaded, use larger graphs, longer chunks, or batch independent work.

## Modes

Eval frequency logic:

```python
if fast_mode:
    _eval_frequency = 0
elif low_memory:
    _eval_frequency = 4
else:
    _eval_frequency = 8
```

- `--fast-mode`: no intermediate `mx.eval`, more memory, less sync, usually fastest.
- Default: `mx.eval` every `8` transformer blocks.
- `--low-memory`: `mx.eval` every `4` transformer blocks, lower peak memory, slower.
- If `--fast-mode` and `--low-memory` both set, `fast_mode` wins. Do not use both.

`mx.eval` outcomes:

- Materialize lazy MLX graph now.
- Free some intermediates sooner.
- Lower peak memory.
- More CPU/GPU sync.
- Usually same math/output.

## Precision

- Default is already FP16. Keep FP16 unless instability appears.
- FP32 uses more memory, likely slower, maybe tiny numerical change only.
- FP16 visible quality loss unlikely for inference, but exact seed pixels can change.
- FP8 not useful for speed here: loader dequantizes FP8 weights back to FP16. Use only for smaller download/disk, not generation speed.

## VAE Tiling

- Auto tiling triggers for bigger decode only.
- Can disable by making `MyPipelineConfig._get_tiling_config()` return `None`.
- Quality should stay basically same; possible tiny seam/numerical diffs.
- Untiled decode may be faster.
- Untiled decode may OOM or hit Metal watchdog on large outputs.
- At `384x576` with `49/97` frames, tiling likely does not trigger.

## Implemented

- Added notes: `docs/performance/LTX_2_3_SPEEDUP_NOTES.md`
- Added custom pipeline: `LTX_2_MLX/pipelines/my_pipeline.py`
- Patched route: `scripts/generate.py`
- Patched video-only transformer call: `LTX_2_MLX/model/transformer/model.py`
- Added `--pipeline my-pipeline`.
- Passed `--fast-mode` into AV transformer loading.
- `my-pipeline` disables internal audio branch for no-audio runs.
- Added persistent runner plan: `docs/performance/LTX_2_3_PERSISTENT_RUNNER_PLAN.md`.
- Added persistent runner script: `scripts/generate_persistent.py`.
- Added session-hot server script: `scripts/serve_persistent.py`.

## Benchmarks

Baseline command:

```bash
--pipeline one-stage --height 384 --width 576 --frames 49 --steps 12 --cfg 1
```

Results:

| Run | Wall time | Speedup | 1 min estimate |
| --- | ---: | ---: | ---: |
| `one-stage` | `101.55s` | baseline | `49.8 min` |
| `my-pipeline` | `91.13s` | `10.3%` | `44.6 min` |
| `my-pipeline --fast-mode` | `87.28s` | `14.1%` | `42.8 min` |
| `my-pipeline --fast-mode --frames 97` | `135.39s` | `32.6% per output second` | `33.5 min` |
| `my-pipeline --fast-mode --frames 97 --height 800 --width 448` | `219.51s` | `1.72x faster than 576x1024` | `54.3 min` |
| `my-pipeline --fast-mode --frames 97 --height 896 --width 512` | `258.07s` | `1.46x faster than 576x1024` | `63.9 min` |
| `my-pipeline --fast-mode --frames 97 --height 1024 --width 576` | `377.70s` | `1.88x slower per output second` | `93.5 min` |

Outputs:

```bash
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/acelogic_ltx_2_3_my_pipeline_384x576_49f_12step_cfg1_v1.mp4"
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/acelogic_ltx_2_3_my_pipeline_384x576_49f_12step_cfg1_v1_frame24.png"
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/acelogic_ltx_2_3_my_pipeline_fast_384x576_49f_12step_cfg1_v1.mp4"
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/acelogic_ltx_2_3_my_pipeline_fast_384x576_49f_12step_cfg1_v1_frame24.png"
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/acelogic_ltx_2_3_my_pipeline_fast_384x576_97f_12step_cfg1_v1.mp4"
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/acelogic_ltx_2_3_my_pipeline_fast_384x576_97f_12step_cfg1_v1_frame48.png"
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/acelogic_ltx_2_3_my_pipeline_fast_448x800_97f_12step_cfg1_v1.mp4"
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/acelogic_ltx_2_3_my_pipeline_fast_448x800_97f_12step_cfg1_v1_frame48.png"
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/acelogic_ltx_2_3_my_pipeline_fast_512x896_97f_12step_cfg1_v1.mp4"
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/acelogic_ltx_2_3_my_pipeline_fast_512x896_97f_12step_cfg1_v1_frame48.png"
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/acelogic_ltx_2_3_my_pipeline_fast_576x1024_97f_12step_cfg1_v1.mp4"
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/acelogic_ltx_2_3_my_pipeline_fast_576x1024_97f_12step_cfg1_v1_frame48.png"
```

Quality note:

- Disabling internal audio branch changes same-seed output.
- Middle frame still coherent dog/meadow, still motion-blurred.
- `--fast-mode` matched `my-pipeline` visually in this test.
- `97` frame run remained coherent, changed composition to dog plus people, still blurred.
- `448x800` and `512x896` vertical runs both triggered VAE tiling, but are much faster than `576x1024`.
- `576x1024` Shorts run triggered VAE tiling, took much longer, dog partly off-frame, and hallucinated visible text.

## LTX 2.0 vs 2.3 A/B

Same prompt/settings:

- Prompt: `A golden retriever running through a sunny meadow`
- Resolution: `448x800`
- Frames: `97`
- Steps: `12`
- CFG: `1`
- Seed: `42`

Results:

| Model | Pipeline | Wall time | 1 min estimate | Visual note |
| --- | --- | ---: | ---: | --- |
| LTX 2.0 distilled | `text-to-video --fast-mode` | `152.94s` | `37.8 min` | Prompt-literal meadow/dog, but dog shape morphs badly by later frames. |
| LTX 2.3 distilled | `my-pipeline --fast-mode` | `219.51s` | `54.3 min` | More photoreal/coherent motion, but adds human runner not in prompt. |

Current read: LTX 2.0 is faster; LTX 2.3 is better for coherence/photorealism. Use 2.3 for quality path, 2.0 only for cheap prompt/shot exploration.

Outputs:

```bash
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/ltx_2_0_distilled_fast_448x800_97f_12step_cfg1_v1.mp4"
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/ltx_2_0_distilled_fast_448x800_97f_12step_cfg1_v1_frame48.png"
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/ltx_2_0_distilled_fast_448x800_97f_12step_cfg1_v1_contact_v1.jpg"
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/acelogic_ltx_2_3_my_pipeline_fast_448x800_97f_12step_cfg1_v1.mp4"
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/acelogic_ltx_2_3_my_pipeline_fast_448x800_97f_12step_cfg1_v1_frame48.png"
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/acelogic_ltx_2_3_my_pipeline_fast_448x800_97f_12step_cfg1_v1_contact_v1.jpg"
```

## Current Defaults

Use `--frames 97` for speed tests.

- `49` frames = `2.04s` video.
- `97` frames = `4.04s` video.
- Same load cost, more output per run.
- Still likely below tiling threshold at `384x576`.
- Do not change global CLI default yet; use this in benchmark commands.

Use `448x800` for vertical tests now.

- CLI shape: `--pipeline my-pipeline --fast-mode --height 800 --width 448 --frames 97 --steps 12 --cfg 1`.
- Best current portrait speed/quality tradeoff.
- Wall: `219.51s`; one-minute estimate: `54.3 min`.
- `512x896` costs `258.07s`; keep as quality check only.
- `576x1024` costs `377.70s`; avoid unless detail beats wall time.
- Tiled VAE still triggers.

## Persistent Runner

Goal: keep model objects alive across chunks.

Detailed V1 plan: [LTX 2.3 Persistent Runner Plan](LTX_2_3_PERSISTENT_RUNNER_PLAN.md).

Current CLI loop:

- Load Gemma.
- Encode prompt.
- Clear Gemma.
- Load transformer.
- Load VAE.
- Generate one chunk.
- Exit.
- Repeat all loads next chunk.

Persistent runner loop:

- Start one Python process.
- Load tokenizer/Gemma/text connector once.
- Load transformer once.
- Load VAE once.
- Cache prompt embeddings when prompt repeats.
- Generate chunk 1, chunk 2, chunk 3 in loop.
- Save each chunk.

Validation runs:

| Run | Settings | Total | Chunk 1 | Chunk 2 | Note |
| --- | --- | ---: | ---: | ---: | --- |
| Smoke | `384x576`, `9f`, `1 step`, `1 chunk` | `134.76s` | `5.05s` | n/a | Validated imports, loads, prompt cache, MP4 save, manifest. |
| Benchmark | `448x800`, `121f`, `8 steps`, `2 chunks` | `550.36s` | `223.85s` | `202.62s` | Chunk 2 skips model loads and prompt encode. |
| Tiling fix benchmark | `448x800`, `121f`, `8 steps`, `2 chunks` | `491.50s` | `181.97s` | `181.46s` | Dead first tiled decode loop removed. |
| Session-hot server | `448x800`, `121f`, `8 steps`, `2 chunks`, Gemma unload | `361.90s` | `152.73s` | `183.45s` | First full request after lazy startup; Gemma unloaded after encode. |

Persistent benchmark details:

- Load subtotal: tokenizer `1.58s`, Gemma `10.19s`, text encoder `2.84s`, transformer `34.11s`, VAE `1.07s`.
- Prompt encode: `73.45s` once.
- Chunk 2 steady-state: `202.62s` for `121f` at `24fps` output = about `40.2 min` wall per generated minute.
- 2-chunk total including loads/encode: `54.58 min` wall per generated minute.

After tiled decode fix:

- Total: `491.50s`, down from `550.36s` (`10.7%` faster).
- Chunk 2 steady-state: `181.46s`, down from `202.62s` (`10.4%` faster).
- Wall per generated minute including loads/encode: `48.74 min`, down from `54.58 min`.
- Output dir: `outputs/persistent_ltx23_tiling_fix_448x800_121f_8step_v1`.

Session-hot server result:

- Command used `--unload-gemma-after-encode --idle-timeout-minutes 30`.
- Startup skipped Gemma load: `gemma_seconds=0.0`, `gemma_loaded_at_startup=false`.
- Full request total: `361.90s`; prompt encode `24.75s`; Gemma reload `20.20s`.
- Chunks: `152.73s` and `183.45s`.
- `/status` after request: `prompt_cache_size=1`, `gemma_loaded=false`.
- Idle server memory after Gemma unload: `49G` process memory by `top`; previous all-hot idle process was about `93G`.
- System memory free after request: `84%` by `memory_pressure`.
- Output dir: `outputs/persistent_server_session_hot_full_v1`.

Persistent outputs:

```bash
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/persistent_ltx23_smoke_v1/dog_meadow_smoke_chunk_000_seed42.mp4"
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/persistent_ltx23_benchmark_448x800_121f_8step_v1/dog_meadow_chunk_000_seed42.mp4"
mpv "/Users/vitorfrasson/code/cockpit-backend-runtime-library-cli/tests/models/outputs/persistent_ltx23_benchmark_448x800_121f_8step_v1/dog_meadow_chunk_001_seed43.mp4"
mpv "/Users/vitorfrasson/code/LTX-2-MLX/outputs/persistent_server_session_hot_full_v1/dog_meadow_session_hot_chunk_000_seed42.mp4"
mpv "/Users/vitorfrasson/code/LTX-2-MLX/outputs/persistent_server_session_hot_full_v1/dog_meadow_session_hot_chunk_001_seed43.mp4"
```

Implementation path:

- Add `scripts/generate_persistent.py`.
- Import same loaders and `MyPipeline`.
- Accept `--chunks`, `--frames 97`, `--steps`, `--cfg`, `--prefix`.
- Keep all model objects in memory between chunks.
- Best result: persistent runner + `97` frame chunks.

## Stage Capacity

Treat persistent runner as staged capacity problem, not plain `for` loop.

Measure max safe work per stage:

- Text/token stage: how many prompts/chunks can tokenize + encode at once. Cache repeated prompt embeddings.
- Transformer stage: how many videos or independent chunks fit in one denoise batch for resolution, frames, steps, CFG.
- VAE stage: how many generated latents decode at once, tiled and untiled.
- Save stage: split batched frames back into separate outputs.

Scheduling rule:

- If video chunk `N+1` depends on chunk `N`, same video is sequential. Cannot batch same chain to speed that one video.
- If chunks are independent or stitched without dependency, batch them. Faster aggregate, worse continuity risk.
- If same video can be batched safely, use memory to make that video faster.
- If same video cannot be batched, use memory to run more independent videos/requests in parallel.

Goal for `128GB`: max safe batch size per stage. If transformer bottleneck, batch transformer only. If VAE bottleneck, denoise several requests then decode smaller batches.

## OneStage 2.3 Pipeline Analysis

Analysis target: `OneStageCFGConfig` and `OneStagePipeline` in `LTX_2_MLX/pipelines/one_stage.py`.

Current best base: `OneStagePipeline`, but for our silent 2.3 use case it carries unused AV/audio/general-purpose baggage.

Main waste:

- Negative prompt encoded even when `cfg=1`, then never used.
- Audio text encoding computed even when silent.
- VAE encoder loaded even when no image conditioning. This matters less after persistent runner because it becomes one-time startup cost, so do not prioritize before `scripts/generate_persistent.py` exists.
- AV transformer loads audio/cross-modal weights even when audio branch disabled.
- Tiled VAE decoder appears to decode tiles twice. Verify before changing: current code looks like an abandoned first loop in `decode_tiled()` that discards output, but confirm no intentional warm-up/cache behavior by benchmark and visual/pixel parity.
- Many CLI knobs are no-op for 2.3 AV path: `cross_attn_scale`, `stg_scale`, `ge_gamma`, `sampler`, `tiled_vae`, temporal upscaler.

Highest-value changes:

| Priority | Change | Why |
| --- | --- | --- |
| 0 | Add persistent runner first | Removes repeated Gemma/transformer/VAE load cost and changes priority of loader-only cleanup. |
| 1 | Verify and likely fix `decode_tiled()` double-decode | Direct speed win for `448x800+` if first loop is dead work. |
| 2 | Skip negative prompt encode when `cfg_scale == 1` | Distilled path does no CFG, so null embedding unused. |
| 3 | Skip audio text encoding for silent | V2 encoder computes audio projection/connector needlessly. |
| 4 | Hardcode silent video-only denoise path | Remove AV/audio/Heun/STG branches from hot path. |
| 5 | Test configurable generation fps, including `12` | Current CLI forces `25.0` into AV config even when output is `24`. |
| 6 | Expose real tiling control | Current auto-tiling always triggers at `448x800/121f`; `--tiled-vae` no-op. |
| 7 | Try V2 video-only transformer load | Big memory/load win if weights map cleanly. |
| 8 | Pass latent into scheduler | Quality/schedule correctness test; speed neutral. |

Config review:

| Field | Current | Keep/Change |
| --- | --- | --- |
| `height`, `width` | Validates divisible by `32` | Keep. |
| `num_frames` | Requires `8k+1` | Keep; error text outdated, says up to `121`. |
| `seed` | Sets MLX seed | Keep. |
| `fps` | Default `24`, CLI forces `25` in AV path | Change silent pipeline to configurable generation fps; test `12`, `16`, `24`, `25`. |
| `num_inference_steps` | Default `30`, CLI uses `8` | Keep `8` distilled baseline. |
| `cfg_scale` | Distilled forced to `1.0` | Keep for distilled; dev profile separate. |
| `audio_cfg_scale` | Irrelevant when silent | Remove from silent config. |
| `rescale_scale` | Used only as boolean switch, not actual `0.7` strength | Remove for distilled; fix if dev tuning. |
| `tiling_config` | `None` means auto, no force-off | Add explicit `tiling_mode: auto/off/on`. |
| `dtype` | CLI passes `float16` | Keep. |
| `audio_enabled` | False | Remove/hardcode false. |
| `use_internal_audio_branch` | OneStage default true; MyPipeline false | Hardcode false. |
| Audio shape fields | Unused silent | Remove. |

Pipeline review:

| Area | Current | Worth Changing |
| --- | --- | --- |
| `__init__` | Always needs `video_encoder`; creates audio patchifier | Make `video_encoder` optional after persistent runner; no audio patchifier in silent path. |
| `X0Model` wrap | Needed, clean | Keep. |
| `is_av_model` | Enables internal audio branch unless disabled | Remove from silent path. |
| `_create_video_tools` | Builds positions/masks | Keep, but cache by shape in persistent runner. |
| `_create_audio_tools`, `_decode_audio` | Silent unused | Remove. |
| Cross-attn scaling | Implemented in pipeline | Wire into CLI for 2.3 or test in custom. |
| `_denoise_loop_cfg` | Good core path | Keep only Euler/no-audio version. |
| CFG branch | Skips neg pass when scale `1.0` | Good; also skip neg encoding before pipeline. |
| STG | Extra model pass | Remove from default; experimental only. |
| GE | Math-only but unproven | Leave off. |
| Heun | 2x model evals | Remove from speed path. |
| AV denoise loops | Expensive/general | Remove from silent path. |
| `scheduler.execute()` | Does not pass latent | Test `latent=video_state.latent`. |
| `post_process_latent()` | Runs even with no conditioning | Skip when no images. |
| `clear_conditioning()` | Runs even with no conditioning | Skip when no images. |
| Temporal upscaler | Present but CLI does not pass | Later test `121f -> 241f` cheaper path. |
| VAE decode | Auto tiled above threshold | Verify/fix tiled decode first, then benchmark full vs tiled. |

Tiling finding:

- `448x800` triggers auto tiling: `121f` latent is `16x25x14 = 5600`, above `4000` threshold.
- `448x800` triggers auto tiling: `241f` latent is `31x25x14 = 10850`, above `4000` threshold.
- `decode_tiled()` currently has a first tile loop that decodes each tile, discards result, then a second tile loop that decodes again and yields output.
- This looks accidental because comments say the first approach was abandoned: `we'll use a different approach` and `Since MLX doesn't have efficient scatter operations...`.
- Before removing first loop, confirm with one benchmark and compare output. If output unchanged and wall drops, remove dead loop.

FPS notes:

- There are two FPS meanings: generation fps in `OneStageCFGConfig.fps`, and saved playback fps in CLI `--fps` / `output_fps`.
- Generation fps affects temporal coordinates fed to transformer: `positions[:, 0] / fps`. It changes model's sense of clip duration and motion.
- Saved playback fps affects MP4 duration only.
- Current AV route sets `av_config.fps=25.0` unconditionally, then saves with `--fps` default `24`. Silent path should expose generation fps directly.
- `fps=12` is valid as an experiment. No divisibility constraint found; field is float.
- `121f` at generation `24fps` represents about `5.04s`; at generation `12fps` represents about `10.08s` to model.
- If generated at `12fps` and saved at `12fps`, output plays longer and may need fewer frames per target second.
- If generated at `12fps` and saved at `24fps`, model sees longer action span but playback compresses it, likely faster perceived motion.
- Test matrix: `generation_fps=12/output_fps=12`, `generation_fps=12/output_fps=24`, `generation_fps=24/output_fps=24`, `generation_fps=25/output_fps=24`.
- Use same prompt/seed/resolution/frames before judging; fps changes same-seed output.

Recommended custom pipeline:

- Build `SilentOneStage23Pipeline`.
- Inputs: positive video encoding only, no negative unless `cfg > 1`.
- Transformer: maybe V2 video-only model, no audio modules.
- Config: `height`, `width`, `frames`, `steps`, `seed`, `generation_fps`, `dtype`, `tiling_mode`, optional `cross_attn_scale`.
- Denoise: Euler only, video only, no STG/GE/Heun/audio.
- Decode: fixed tiled decode or full decode, selectable.
- Loader: skip VAE encoder unless image conditioning exists; lower priority after persistent runner.
- Encoder: V2 video-only text output, no audio connector, no negative prompt when CFG disabled.

Benchmark plan:

- Baseline: current `my-pipeline`, `448x800`, `121f`, `8step`.
- Test 1: persistent runner baseline with unchanged pipeline.
- Test 2: verified/fixed tiled decode.
- Test 3: full VAE decode at `448x800/121f`.
- Test 4: skip negative/audio text encode.
- Test 5: configurable generation fps, including `12`.
- Test 6: V2 video-only transformer.
- Test 7: `fps=24` vs `25` and scheduler with latent token count.

Best likely first win after persistent runner: verify/fix tiled decode + skip unused encoders.

## Next Order

1. Add persistent runner.
2. Benchmark `448x800 --steps 8` inside persistent runner if quality holds.
3. Verify/fix tiled VAE double-decode before changing tiling policy.
4. Test generation fps options, including `12`.
5. Consider batch support for independent chunks/requests.
