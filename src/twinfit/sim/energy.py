"""Total mechanical energy of a MuJoCo state.

MuJoCo's energy flag computes `data.energy = [potential, kinetic]`. Verified on the installed
MuJoCo 3.17: the potential term already includes gravity and the joint spring energy
0.5 k (q - springref)^2, so nothing is added here (see tests/setup/test_sim.py).
"""

from __future__ import annotations

import mujoco
import numpy as np


def enable_energy(model: mujoco.MjModel) -> None:
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY


def total_energy(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Potential (gravity plus springs) plus kinetic energy of the current state."""
    enable_energy(model)
    mujoco.mj_energyPos(model, data)
    mujoco.mj_energyVel(model, data)
    return float(data.energy[0] + data.energy[1])


def energies_at(model: mujoco.MjModel, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """Total energy for each row of (q, qd), shape (N,)."""
    data = mujoco.MjData(model)
    q = np.asarray(q, float).reshape(-1, model.nq)
    qd = np.asarray(qd, float).reshape(-1, model.nv)
    out = np.empty(len(q))
    for i in range(len(q)):
        data.qpos[:] = q[i]
        data.qvel[:] = qd[i]
        mujoco.mj_forward(model, data)
        out[i] = total_energy(model, data)
    return out
