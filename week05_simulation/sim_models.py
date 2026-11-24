"""Embedded MJCF models shared by the week 05 simulation labs.

Keeping the models in a single place lets every script load the same physics
so the CPU and GPU numbers you measure are directly comparable.

Nothing to implement here - this file is complete. Read it to see how a
MuJoCo model is described (bodies, joints, actuators) before moving on.
"""

MJCF_CARTPOLE = """
<mujoco model="cartpole">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="cart" pos="0 0 0.1">
      <joint name="slider" type="slide" axis="1 0 0" range="-2.0 2.0" damping="0.1"/>
      <geom name="cart_geom" type="box" size="0.15 0.1 0.05" mass="1.0"/>
      <body name="pole" pos="0 0 0.05">
        <joint name="hinge" type="hinge" axis="0 1 0" range="-3.0 3.0" damping="0.01"/>
        <geom name="pole_geom" type="capsule" size="0.05" fromto="0 0 0 0 0 0.6" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="slide_motor" joint="slider" ctrlrange="-10 10" gear="1"/>
  </actuator>
</mujoco>
"""

N_DOFS = 2
N_ACTUATORS = 1


def cartpole_model():
    """Return a mujoco.MjModel compiled from the embedded cartpole MJCF.

    Works with both the plain ``mujoco`` package (CPU) and, after
    ``mjw.put_model``, with MuJoCo Warp.
    """
    import sys
    if sys.platform.startswith("linux"):
        import ctypes, glob
        for p in sys.path:
            for c in glob.glob(f"{p}/mujoco/libmujoco.so*"):
                try:
                    ctypes.CDLL(c, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass

    import mujoco

    return mujoco.MjModel.from_xml_string(MJCF_CARTPOLE)


if __name__ == "__main__":
    m = cartpole_model()
    print(f"loaded cartpole: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody}")
    assert m.nq == N_DOFS and m.nv == N_DOFS and m.nu == N_ACTUATORS
    print("sim_models self-test OK")
