"""Fixed-step MuJoCo replay.

Milestones 2 and 3 must both use `simulate` and `windowed_replay_rms`, so that fitting and
certification measure exactly the same quantity with the same integrator and step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

DEFAULT_TIMESTEP = 2e-4
DEFAULT_INTEGRATOR = "RK4"
_INTEGRATORS = {
    "EULER": mujoco.mjtIntegrator.mjINT_EULER,
    "RK4": mujoco.mjtIntegrator.mjINT_RK4,
    "IMPLICIT": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "IMPLICITFAST": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}


def load_model(model_xml: str | Path) -> mujoco.MjModel:
    """Accepts a path to an MJCF file or an MJCF string (anything containing '<')."""
    text = str(model_xml)
    if "<" in text:
        return mujoco.MjModel.from_xml_string(text)
    return mujoco.MjModel.from_xml_path(text)


def joint_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]


def configure(model: mujoco.MjModel, timestep: float, integrator: str) -> None:
    model.opt.timestep = timestep
    model.opt.integrator = _INTEGRATORS[integrator.upper()]


def step_to(model: mujoco.MjModel, data: mujoco.MjData, t_target: float) -> None:
    """Advance with full steps of model.opt.timestep, then one shorter step to land exactly on
    t_target. The step length is restored afterwards."""
    dt = model.opt.timestep
    while data.time + dt <= t_target + 1e-12:
        mujoco.mj_step(model, data)
    rem = t_target - data.time
    if rem > 1e-12:
        model.opt.timestep = rem
        mujoco.mj_step(model, data)
        model.opt.timestep = dt
    data.time = t_target


def simulate(
    model_xml: str | Path,
    q0: np.ndarray,
    qd0: np.ndarray,
    t_out: np.ndarray,
    timestep: float = DEFAULT_TIMESTEP,
    integrator: str = DEFAULT_INTEGRATOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Passive simulation from state (q0, qd0) at time t_out[0]. Returns q and qd at t_out, each
    of shape (len(t_out), nq), where q[0] = q0. Joint order is the model's joint order."""
    model = load_model(model_xml)
    configure(model, timestep, integrator)
    data = mujoco.MjData(model)
    t_out = np.asarray(t_out, dtype=float)
    data.qpos[:] = np.asarray(q0, dtype=float)
    data.qvel[:] = np.asarray(qd0, dtype=float)
    data.time = float(t_out[0])
    mujoco.mj_forward(model, data)
    q = np.empty((len(t_out), model.nq))
    qd = np.empty((len(t_out), model.nv))
    q[0] = data.qpos
    qd[0] = data.qvel
    for i in range(1, len(t_out)):
        step_to(model, data, float(t_out[i]))
        q[i] = data.qpos
        qd[i] = data.qvel
    return q, qd


@dataclass
class ReplayWindow:
    t_start: float
    t_end: float
    frame_start: int
    frame_end: int
    rms: np.ndarray
    max_abs: np.ndarray


@dataclass
class ReplayResult:
    joint_names: list[str]
    rms: np.ndarray
    windows: list[ReplayWindow] = field(default_factory=list)
    n_skipped: int = 0

    def as_dict(self) -> dict[str, float]:
        return {name: float(v) for name, v in zip(self.joint_names, self.rms, strict=True)}


def windowed_replay_rms(
    model_xml: str | Path,
    t: np.ndarray,
    q_obs: np.ndarray,
    qd_obs: np.ndarray,
    valid: np.ndarray,
    segments_s: list[list[float]],
    window_s: float = 0.5,
    timestep: float = DEFAULT_TIMESTEP,
    integrator: str = DEFAULT_INTEGRATOR,
) -> ReplayResult:
    """Windowed replay error, the shared fitting and certification metric.

    Inside each segment [t0, t1], consecutive windows of length `window_s` tile the segment from
    the first frame with t >= t0. A window covers the frames with t_start <= t < t_start + window_s
    and must end at or before t1. Each window restarts the simulation from the observed (q, qd) at
    its first frame and compares the simulated q with q_obs at every later frame of the window.
    Windows containing any invalid frame are skipped. The per-joint RMS is taken over all compared
    frames of all windows; `windows` holds the per-window details.

    q_obs and qd_obs are (N, nq) in the model's joint order; valid is (N,) or (N, nq) bool.
    """
    model = load_model(model_xml)
    configure(model, timestep, integrator)
    data = mujoco.MjData(model)
    t = np.asarray(t, dtype=float)
    q_obs = np.asarray(q_obs, dtype=float).reshape(len(t), -1)
    qd_obs = np.asarray(qd_obs, dtype=float).reshape(len(t), -1)
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim == 2:
        valid = valid.all(axis=1)
    names = joint_names(model)
    if q_obs.shape[1] != model.nq:
        raise ValueError(f"q_obs has {q_obs.shape[1]} columns, model has nq={model.nq}")

    windows: list[ReplayWindow] = []
    sq_sum = np.zeros(model.nq)
    count = 0
    n_skipped = 0
    for t0, t1 in segments_s:
        start = int(np.searchsorted(t, t0 - 1e-9))
        while start < len(t):
            t_start = t[start]
            t_end = t_start + window_s
            if t_end > t1 + 1e-9:
                break
            end = int(np.searchsorted(t, t_end - 1e-9))  # exclusive: frames with t < t_end
            if end - start < 2:
                break
            if not valid[start:end].all():
                n_skipped += 1
                start = end
                continue
            data.qpos[:] = q_obs[start]
            data.qvel[:] = qd_obs[start]
            data.time = t_start
            mujoco.mj_forward(model, data)
            err = np.zeros((end - start - 1, model.nq))
            for k, i in enumerate(range(start + 1, end)):
                step_to(model, data, float(t[i]))
                err[k] = data.qpos - q_obs[i]
            sq_sum += (err**2).sum(axis=0)
            count += len(err)
            windows.append(
                ReplayWindow(
                    t_start=float(t_start),
                    t_end=float(t[end - 1]),
                    frame_start=start,
                    frame_end=end,
                    rms=np.sqrt((err**2).mean(axis=0)),
                    max_abs=np.abs(err).max(axis=0),
                )
            )
            start = end
    rms = np.sqrt(sq_sum / count) if count else np.full(model.nq, np.nan)
    return ReplayResult(joint_names=names, rms=rms, windows=windows, n_skipped=n_skipped)
