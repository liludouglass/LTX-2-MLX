"""Custom one-stage pipeline for faster silent video generation.

Starts from OneStagePipeline and disables the internal audio branch by default.
This keeps no-audio runs from denoising audio latents while preserving the
standard one-stage behavior otherwise.
"""

from dataclasses import dataclass

from .one_stage import OneStageCFGConfig, OneStagePipeline


@dataclass
class MyPipelineConfig(OneStageCFGConfig):
    """One-stage config tuned for silent video throughput."""

    use_internal_audio_branch: bool = False


class MyPipeline(OneStagePipeline):
    """OneStagePipeline with MyPipelineConfig defaults."""
