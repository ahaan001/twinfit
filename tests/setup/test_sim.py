import mujoco
import numpy as np
import pytest

from twinfit.sim import energy, replay

PENDULUM = """
<mujoco>
  <compiler angle="radian"/>
  <option gravity="0 9.81 0" timestep="0.0002" integrator="RK4"/>
  <worldbody>
    <body name="p1" pos="0 0 0">
      <joint name="j0" type="hinge" axis="0 0 -1" damping="{damping}" frictionloss="{friction}"
             stiffness="{stiffness}" springref="{springref}"/>
      <inertial pos="0 0.13 0" mass="0.06" fullinertia="4.5e-4 4.5e-6 4.5e-4 0 0 0"/>
      <geom type="box" pos="0 0.13 0" size="0.015 0.15 0.002" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def pendulum(damping=0.0, friction=0.0, stiffness=0.0, springref=0.0):
    return PENDULUM.format(
        damping=damping, friction=friction, stiffness=stiffness, springref=springref
    )


def test_undamped_pendulum_conserves_energy():
    xml = pendulum()
    t = np.linspace(0.0, 5.0, 301)
    q, qd = replay.simulate(xml, [1.0], [0.0], t)
    model = replay.load_model(xml)
    E = energy.energies_at(model, q, qd)
    assert np.abs(E - E[0]).max() / abs(E[0]) < 1e-3
    assert abs(q).max() <= 1.0 + 1e-6


def test_energy_includes_spring_potential():
    """Independent computation: potential = m g h + 0.5 k (q - ref)^2 against MuJoCo's value."""
    xml = pendulum(stiffness=0.6, springref=0.2)
    model = replay.load_model(xml)
    data = mujoco.MjData(model)
    data.qpos[0] = 0.7
    data.qvel[0] = 1.5
    mujoco.mj_forward(model, data)
    E = energy.total_energy(model, data)
    com = data.xipos[1]
    g = model.opt.gravity
    U_grav = -0.06 * np.dot(g, com)
    U_spring = 0.5 * 0.6 * (0.7 - 0.2) ** 2
    I_about_pivot = 4.5e-4 + 0.06 * 0.13**2
    T_kin = 0.5 * I_about_pivot * 1.5**2
    assert np.isclose(E, U_grav + U_spring + T_kin, rtol=1e-9)


def test_simulate_matches_plain_mj_step():
    xml = pendulum(damping=2e-4, friction=5e-4)
    dt = 2e-4
    n_frames = 40
    t = np.arange(n_frames) * dt * 25  # every output aligns with 25 steps
    q, qd = replay.simulate(xml, [0.8], [0.3], t, timestep=dt)
    model = replay.load_model(xml)
    model.opt.timestep = dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    data = mujoco.MjData(model)
    data.qpos[0] = 0.8
    data.qvel[0] = 0.3
    mujoco.mj_forward(model, data)
    ref = [data.qpos[0]]
    for _ in range(n_frames - 1):
        for _ in range(25):
            mujoco.mj_step(model, data)
        ref.append(data.qpos[0])
    np.testing.assert_allclose(q[:, 0], ref, atol=1e-12)


def test_simulate_lands_on_unaligned_times():
    xml = pendulum()
    t = np.array([0.0, 0.01234, 0.0333, 0.1])
    q, qd = replay.simulate(xml, [0.5], [0.0], t)
    q2, _ = replay.simulate(xml, [0.5], [0.0], np.linspace(0, 0.1, 501))
    assert abs(q[-1, 0] - q2[-1, 0]) < 1e-7


def test_windowed_replay_rms_is_zero_on_own_trajectory_and_skips_invalid():
    xml = pendulum(damping=2e-4, friction=5e-4)
    t = np.arange(0.0, 2.0, 1 / 60)
    q, qd = replay.simulate(xml, [1.0], [0.0], t)
    valid = np.ones(len(t), bool)
    res = replay.windowed_replay_rms(xml, t, q, qd, valid, [[0.0, 2.0]], window_s=0.5)
    assert res.joint_names == ["j0"]
    assert res.rms[0] < 1e-9
    assert len(res.windows) == 4 and res.n_skipped == 0
    valid[40] = False
    res = replay.windowed_replay_rms(xml, t, q, qd, valid, [[0.0, 2.0]], window_s=0.5)
    assert len(res.windows) == 3 and res.n_skipped == 1
    assert all(not (w.frame_start <= 40 < w.frame_end) for w in res.windows)
    wrong = replay.windowed_replay_rms(xml, t, q + 0.01, qd, np.ones(len(t), bool), [[0.0, 2.0]])
    assert wrong.rms[0] > 1e-4
    assert wrong.as_dict()["j0"] == pytest.approx(wrong.rms[0])


def test_windowed_replay_restarts_catch_wrong_model():
    """Replay with a different damping must produce a nonzero windowed RMS."""
    xml_true = pendulum(damping=2e-3)
    xml_wrong = pendulum(damping=2e-2)
    t = np.arange(0.0, 3.0, 1 / 60)
    q, qd = replay.simulate(xml_true, [1.2], [0.0], t)
    good = replay.windowed_replay_rms(xml_true, t, q, qd, np.ones(len(t), bool), [[0.5, 3.0]])
    bad = replay.windowed_replay_rms(xml_wrong, t, q, qd, np.ones(len(t), bool), [[0.5, 3.0]])
    assert good.rms[0] < 1e-9
    assert bad.rms[0] > 1e-3
    assert len(good.windows) == 5
