"""Maps each stage to its real implementation and its oracle stub."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from twinfit.contracts.stage import StageName
from twinfit.stubs import oracle

REAL_IMPLS: dict[StageName, str] = {
    "perceive": "twinfit.perception.stage:run",
    "kinematics": "twinfit.identify.kinematics_stage:run",
    "dynamics": "twinfit.identify.dynamics_stage:run",
    "certify": "twinfit.verify.stage:run",
    "report": "twinfit.report.stage:run",
}
IMPL_CHOICES = ("auto", "real", "stub")


def real_available(stage: StageName) -> bool:
    """True when the real module imports. Only a ModuleNotFoundError for that exact module path
    counts as 'absent'; any other import error propagates."""
    module_path = REAL_IMPLS[stage].split(":")[0]
    try:
        importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        if e.name == module_path:
            return False
        raise
    return True


def load_real(stage: StageName) -> Callable:
    module_path, attr = REAL_IMPLS[stage].split(":")
    return getattr(importlib.import_module(module_path), attr)


def resolve(stage: StageName, impl: str = "auto") -> tuple[Callable, str]:
    """Returns (callable, "real" | "stub") for `impl` in {auto, real, stub}."""
    if impl not in IMPL_CHOICES:
        raise ValueError(f"impl must be one of {IMPL_CHOICES}, got {impl!r}")
    if impl == "stub":
        return oracle.STUBS[stage], "stub"
    if impl == "real":
        return load_real(stage), "real"
    if real_available(stage):
        return load_real(stage), "real"
    return oracle.STUBS[stage], "stub"
