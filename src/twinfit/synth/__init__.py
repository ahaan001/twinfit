"""Synthetic ground-truth clips rendered with MuJoCo. See generate.py for the pipeline."""

from twinfit.synth.generate import SUITES, generate_clip, generate_suite, suite_dir
from twinfit.synth.scenes import SCENE_NAMES, build_scene

__all__ = ["SCENE_NAMES", "SUITES", "build_scene", "generate_clip", "generate_suite", "suite_dir"]
