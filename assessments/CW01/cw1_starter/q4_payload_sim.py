"""COMP0037 2026/27 Coursework 1 — Q4 Starter Code: MedBot Payload Trolley MJCF.

A new MJCF rigid-body model representing a wheeled medical delivery trolley
carrying a top-heavy specimen mast with variable payload mass (0.2–1.5 kg),
mast hinge damping (log-uniform [1e-3, 1.0]), and floor friction ([0.2, 1.5]).
"""
from __future__ import annotations

import sys

TROLLEY_XML = """
<mujoco model="medbot_payload_trolley">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" friction="0.8 0.005 0.0001"/>
    <body name="trolley" pos="0 0 1.0">
      <joint name="trolley_slide" type="slide" axis="1 0 0" damping="0.15"/>
      <geom name="chassis" type="box" size="0.22 0.14 0.06" mass="2.5"/>
      <body name="mast" pos="0 0 0.06">
        <joint name="mast_hinge" type="hinge" axis="0 1 0" damping="0.05"/>
        <geom name="mast_rod" type="capsule" fromto="0 0 0 0 0 0.70" size="0.025" mass="0.3"/>
        <body name="payload" pos="0 0 0.72">
          <geom name="payload_box" type="box" size="0.07 0.07 0.07" mass="0.6"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="drive" joint="trolley_slide" gear="15" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


def trolley_model():
    """Return a compiled mujoco.MjModel for the MedBot payload trolley."""
    if sys.platform.startswith("linux"):
        import ctypes, glob
        for p in sys.path:
            for c in glob.glob(f"{p}/mujoco/libmujoco.so*"):
                try:
                    ctypes.CDLL(c, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass
    import mujoco
    return mujoco.MjModel.from_xml_string(TROLLEY_XML)


def sample_trolley_randomization(rng, nworld: int, log_uniform_damping: bool = True):
    """TODO(CW1-Q4c): Sample per-world (mast_damping, floor_friction, payload_mass).
    - mast_damping in [1e-3, 1.0] (log-uniform if log_uniform_damping else linear-uniform)
    - floor_friction in [0.2, 1.5] (uniform)
    - payload_mass in [0.2, 1.5] kg (uniform)
    """
    raise NotImplementedError("TODO(CW1-Q4c): implement 3-parameter trolley domain randomization")


if __name__ == "__main__":
    m = trolley_model()
    print(f"MedBot payload trolley MJCF OK: nq={m.nq}, nv={m.nv}, nu={m.nu}, nbody={m.nbody}")
    for i in range(m.nbody):
        print(f"  body[{i}]: mass={m.body_mass[i]:.3f} kg")

