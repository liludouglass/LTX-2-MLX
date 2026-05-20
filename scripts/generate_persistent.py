#!/usr/bin/env python3
"""Persistent silent LTX-2.3 video runner.

Purpose:
    Keep Gemma, AV text encoder, AV transformer, VAE decoder, and MyPipeline
    alive in one Python process while generating several silent video chunks.

Scope:
    V1 is intentionally narrow: LTX-2.3 distilled, silent video, MyPipeline,
    sequential chunks, and CFG fixed at 1.0. No tiling changes, batching,
    image conditioning, or video-only transformer surgery here.

Outputs:
    Writes one MP4 per chunk plus `run_manifest.json` with exact args, paths,
    seeds, and timings. Later chunks reuse loaded models and cached prompt
    embeddings; chunk wall time is the main speed metric.

Example:
    uv run python scripts/generate_persistent.py \
      "A golden retriever running through a sunny meadow" \
      --weights /path/to/ltx-2.3-22b-distilled.safetensors \
      --gemma-path /path/to/gemma-3-12b-it-bf16 \
      --output-dir outputs/persistent_ltx23_v1 \
      --prefix dog_meadow --chunks 3 --height 800 --width 448 \
      --frames 121 --steps 8 --cfg 1 --seed 42 --fast-mode
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from LTX_2_MLX.model.text_encoder.encoder import (  # noqa: E402
    create_av_text_encoder,
    create_av_text_encoder_v2_from_checkpoint,
    load_av_text_encoder_v2_weights,
    load_av_text_encoder_weights,
)
from LTX_2_MLX.model.text_encoder.gemma3 import (  # noqa: E402
    Gemma3Config,
    Gemma3Model,
    load_gemma3_weights,
)
from LTX_2_MLX.model.video_vae.simple_decoder import (  # noqa: E402
    SimpleVideoDecoder,
    load_vae_decoder_weights,
)
from LTX_2_MLX.pipelines.my_pipeline import MyPipeline, MyPipelineConfig  # noqa: E402

from generate import (  # noqa: E402
    get_vae_config,
    is_v2_model,
    load_av_transformer,
    load_tokenizer,
)


class PersistentLTX23Runner:
    def __init__(
        self,
        weights_path: str,
        gemma_path: str,
        compute_dtype: mx.Dtype,
        fast_mode: bool,
        low_memory: bool,
        max_length: int,
    ):
        self.weights_path = weights_path
        self.gemma_path = gemma_path
        self.compute_dtype = compute_dtype
        self.max_length = max_length
        self.prompt_cache: dict[str, dict[str, mx.array]] = {}
        self.load_timings: dict[str, float] = {}

        if not is_v2_model(weights_path):
            raise ValueError("Persistent runner V1 supports LTX-2.3/V2 checkpoints only")

        started = time.perf_counter()
        self.tokenizer = self._load_tokenizer()
        self.load_timings["tokenizer_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        self.gemma = Gemma3Model(Gemma3Config())
        load_gemma3_weights(self.gemma, gemma_path, use_fp16=False)
        self.load_timings["gemma_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        self.text_encoder = self._load_text_encoder()
        self.load_timings["text_encoder_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        self.transformer = load_av_transformer(
            weights_path,
            num_layers=48,
            compute_dtype=compute_dtype,
            use_fp8=False,
            low_memory=low_memory,
            fast_mode=fast_mode,
            caption_channels=None,
            cross_attention_adaln=True,
            apply_gated_attention=True,
        )
        self.load_timings["transformer_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        self.video_decoder = self._load_video_decoder()
        self.load_timings["vae_decoder_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        self.pipeline = MyPipeline(
            transformer=self.transformer,
            video_encoder=None,
            video_decoder=self.video_decoder,
        )
        self.load_timings["pipeline_seconds"] = time.perf_counter() - started

    def _load_tokenizer(self):
        tokenizer = load_tokenizer(self.gemma_path)
        if tokenizer is None:
            raise RuntimeError("Failed to load Gemma tokenizer")
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_text_encoder(self):
        if is_v2_model(self.weights_path):
            text_encoder = create_av_text_encoder_v2_from_checkpoint(self.weights_path)
            load_av_text_encoder_v2_weights(text_encoder, self.weights_path)
            return text_encoder
        text_encoder = create_av_text_encoder()
        load_av_text_encoder_weights(text_encoder, self.weights_path)
        return text_encoder

    def _load_video_decoder(self) -> SimpleVideoDecoder:
        vae_config = get_vae_config(self.weights_path)
        video_decoder = SimpleVideoDecoder(
            decoder_blocks=vae_config.get("decoder_blocks"),
            base_channels=vae_config.get("decoder_base_channels", 128),
            timestep_conditioning=vae_config.get("timestep_conditioning", True),
            compute_dtype=self.compute_dtype,
        )
        load_vae_decoder_weights(video_decoder, self.weights_path)
        return video_decoder

    def encode_prompt(self, prompt: str) -> dict[str, mx.array]:
        cached = self.prompt_cache.get(prompt)
        if cached is not None:
            return cached

        encoding = self.tokenizer(
            prompt,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = mx.array(encoding["input_ids"])
        attention_mask = mx.array(encoding["attention_mask"])

        last_hidden, all_hidden_states = self.gemma(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        mx.eval(last_hidden)
        if all_hidden_states is None:
            raise RuntimeError("Gemma did not return hidden states")

        real_token_count = int(attention_mask.sum())
        seq_len = all_hidden_states[0].shape[1]
        if real_token_count < seq_len:
            all_hidden_states = [h[:, -real_token_count:, :] for h in all_hidden_states]
            attention_mask = attention_mask[:, -real_token_count:]

        av_output = self.text_encoder.encode_from_hidden_states(
            hidden_states=all_hidden_states,
            attention_mask=attention_mask,
            padding_side="left",
        )
        mx.eval(av_output.video_encoding)
        mx.eval(av_output.audio_encoding)

        result = {
            "video": av_output.video_encoding,
            "audio": av_output.audio_encoding,
            "mask": av_output.attention_mask,
        }
        self.prompt_cache[prompt] = result

        del all_hidden_states, last_hidden
        gc.collect()
        _clear_mlx_cache()
        return result

    def generate_chunk(
        self,
        prompt: str,
        output_path: Path,
        height: int,
        width: int,
        frames: int,
        steps: int,
        seed: int,
        generation_fps: float,
        output_fps: float,
        speed: float,
    ) -> dict[str, Any]:
        prompt_encoding = self.encode_prompt(prompt)
        negative_encoding = mx.zeros_like(prompt_encoding["video"])
        config = MyPipelineConfig(
            height=height,
            width=width,
            num_frames=frames,
            seed=seed,
            fps=generation_fps,
            num_inference_steps=steps,
            cfg_scale=1.0,
            audio_cfg_scale=1.0,
            rescale_scale=0.0,
            dtype=self.compute_dtype,
            audio_enabled=False,
        )

        def progress_callback(step: int, total: int):
            print(f"\r  Denoising seed {seed}: {step}/{total}", end="", flush=True)

        started = time.perf_counter()
        video, _audio = self.pipeline(
            positive_encoding=prompt_encoding["video"],
            negative_encoding=negative_encoding,
            config=config,
            images=[],
            callback=progress_callback,
        )
        print()
        denoise_decode_seconds = time.perf_counter() - started

        started = time.perf_counter()
        video_np = _video_to_numpy(video)
        save_video_exact_fps(video_np, output_path, fps=output_fps, speed=speed)
        save_seconds = time.perf_counter() - started

        del video, video_np, negative_encoding
        gc.collect()
        _clear_mlx_cache()

        return {
            "seed": seed,
            "output_path": str(output_path),
            "denoise_decode_seconds": denoise_decode_seconds,
            "save_seconds": save_seconds,
            "chunk_seconds": denoise_decode_seconds + save_seconds,
        }


def _clear_mlx_cache() -> None:
    if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()
    elif hasattr(mx, "clear_cache"):
        mx.clear_cache()


def _video_to_numpy(video: mx.array) -> np.ndarray:
    video_np = np.array(video)
    video_np = np.squeeze(video_np)
    if video_np.ndim == 4 and video_np.shape[0] == 3:
        video_np = np.transpose(video_np, (1, 2, 3, 0))
    if video_np.dtype != np.uint8:
        video_np = np.clip((video_np + 1) / 2 * 255.0, 0, 255).astype(np.uint8)
    if video_np.ndim != 4 or video_np.shape[-1] != 3:
        raise ValueError(f"Unexpected video shape after conversion: {video_np.shape}")
    return video_np


def save_video_exact_fps(
    video_np: np.ndarray,
    output_path: Path,
    fps: float,
    speed: float,
) -> None:
    import subprocess

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for index, frame in enumerate(video_np):
            Image.fromarray(frame).save(tmp_path / f"frame_{index:04d}.png")

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_path / "frame_%04d.png"),
        ]
        if speed != 1.0:
            cmd.extend(["-vf", f"setpts={1.0 / speed}*PTS"])
        cmd.extend([
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-loglevel",
            "error",
            str(output_path),
        ])
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")


def build_manifest(
    args: argparse.Namespace,
    load_timings: dict[str, float],
    prompt_encode_seconds: float,
    chunks: list[dict[str, Any]],
    total_seconds: float,
) -> dict[str, Any]:
    generated_seconds = args.chunks * args.frames / args.output_fps / args.speed
    avg_chunk_seconds = sum(chunk["chunk_seconds"] for chunk in chunks) / len(chunks)
    return {
        "prompt": args.prompt,
        "weights": args.weights,
        "gemma_path": args.gemma_path,
        "generation": {
            "height": args.height,
            "width": args.width,
            "frames": args.frames,
            "steps": args.steps,
            "cfg": args.cfg,
            "seed": args.seed,
            "seed_stride": args.seed_stride,
            "generation_fps": args.generation_fps,
            "output_fps": args.output_fps,
            "speed": args.speed,
            "fast_mode": args.fast_mode,
            "low_memory": args.low_memory,
            "dtype": "float32" if args.fp32 else "float16",
        },
        "timings": {
            "load": load_timings,
            "prompt_encode_seconds": prompt_encode_seconds,
            "chunks": chunks,
            "average_chunk_seconds": avg_chunk_seconds,
            "total_seconds": total_seconds,
            "generated_seconds": generated_seconds,
            "generated_minutes_per_wall_minute": generated_seconds / total_seconds,
            "wall_minutes_per_generated_minute": total_seconds / generated_seconds,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent silent LTX-2.3 runner")
    parser.add_argument("prompt", type=str, help="Text prompt for all chunks")
    parser.add_argument("--weights", required=True, help="LTX-2.3 distilled safetensors path")
    parser.add_argument("--gemma-path", required=True, help="Gemma 3 weights directory")
    parser.add_argument("--output-dir", required=True, help="Directory for MP4 chunks and manifest")
    parser.add_argument("--prefix", default="persistent_ltx23", help="Output filename prefix")
    parser.add_argument("--chunks", type=int, default=1, help="Number of chunks to generate")
    parser.add_argument("--height", type=int, default=800, help="Video height")
    parser.add_argument("--width", type=int, default=448, help="Video width")
    parser.add_argument("--frames", type=int, default=121, help="Frames per chunk; must be 8k+1")
    parser.add_argument("--steps", type=int, default=8, help="Denoising steps")
    parser.add_argument("--cfg", type=float, default=1.0, help="V1 supports only 1.0")
    parser.add_argument("--seed", type=int, default=42, help="Seed for first chunk")
    parser.add_argument("--seed-stride", type=int, default=1, help="Seed increment per chunk")
    parser.add_argument("--generation-fps", type=float, default=25.0, help="FPS used for model temporal positions")
    parser.add_argument("--output-fps", type=float, default=24.0, help="Exact MP4 frame rate")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--max-length", type=int, default=1024, help="Tokenizer max length")
    parser.add_argument("--fast-mode", action="store_true", help="Skip intermediate transformer evals")
    parser.add_argument("--low-memory", action="store_true", help="Use more frequent transformer evals")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 compute instead of FP16")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.cfg != 1.0:
        raise ValueError("Persistent runner V1 supports only --cfg 1")
    if args.frames % 8 != 1:
        raise ValueError("--frames must be 8*k + 1")
    if args.height % 32 != 0 or args.width % 32 != 0:
        raise ValueError("--height and --width must be divisible by 32")
    if args.chunks < 1:
        raise ValueError("--chunks must be at least 1")
    if args.output_fps <= 0 or args.generation_fps <= 0:
        raise ValueError("FPS values must be positive")
    if args.speed <= 0:
        raise ValueError("--speed must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compute_dtype = mx.float32 if args.fp32 else mx.float16

    total_started = time.perf_counter()
    runner = PersistentLTX23Runner(
        weights_path=args.weights,
        gemma_path=args.gemma_path,
        compute_dtype=compute_dtype,
        fast_mode=args.fast_mode,
        low_memory=args.low_memory,
        max_length=args.max_length,
    )

    encode_started = time.perf_counter()
    runner.encode_prompt(args.prompt)
    prompt_encode_seconds = time.perf_counter() - encode_started

    chunks = []
    for chunk_index in range(args.chunks):
        seed = args.seed + chunk_index * args.seed_stride
        output_path = output_dir / f"{args.prefix}_chunk_{chunk_index:03d}_seed{seed}.mp4"
        print(f"\nChunk {chunk_index + 1}/{args.chunks}: {output_path}")
        chunks.append(
            runner.generate_chunk(
                prompt=args.prompt,
                output_path=output_path,
                height=args.height,
                width=args.width,
                frames=args.frames,
                steps=args.steps,
                seed=seed,
                generation_fps=args.generation_fps,
                output_fps=args.output_fps,
                speed=args.speed,
            )
        )

    total_seconds = time.perf_counter() - total_started
    manifest = build_manifest(
        args=args,
        load_timings=runner.load_timings,
        prompt_encode_seconds=prompt_encode_seconds,
        chunks=chunks,
        total_seconds=total_seconds,
    )
    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    print(f"\nManifest: {manifest_path}")
    print(f"Total: {total_seconds:.2f}s")
    print(f"Average chunk: {manifest['timings']['average_chunk_seconds']:.2f}s")


if __name__ == "__main__":
    main()
