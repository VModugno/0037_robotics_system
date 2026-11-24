"""COMP0037 2026/27 Coursework 2 — Custom Dual-Table Mobile Manipulation Environment.

A wheeled mobile manipulator (`PandaOmron`: Omron LD-60 holonomic base + 7-DoF
Franka Emika Panda arm + parallel-jaw gripper) operating between two side-by-side
hospital pharmacy workstations:
  - Table A (Pickup Bench):   center (x = 0.0, y = -0.55, z = 0.80 m)
  - Table B (Drop-off Bench): center (x = 0.0, y = +0.55, z = 0.80 m)

Task:
  1. Drive the mobile base along the service aisle to Table A.
  2. Pick up the red specimen cube from Table A.
  3. Transport the cube by driving the mobile base to Table B.
  4. Place the cube onto the green target pad on Table B and release.

Action space (11-D, BASIC composite controller on PandaOmron):
  a[0:6]  = right arm OSC_POSE delta (dx, dy, dz, droll, dpitch, dyaw) in base frame
  a[6]    = torso joint position command (keep 0.0)
  a[7:10] = mobile base holonomic velocity (vx, vy, yaw_rate)
  a[10]   = right parallel-jaw gripper (-1.0 = open, +1.0 = close)
"""
from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET

import numpy as np

# Ensure Windows Robosuite temp directory exists before importing robosuite
if os.name == "nt":
    os.makedirs("C:\\tmp", exist_ok=True)

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

# Canonical geometry constants shared across all CW2 starter & solution scripts
TABLE_FULL_SIZE = (0.58, 0.54, 0.05)
TABLE_A_POS = np.array([0.0, -0.55, 0.80], dtype=np.float64)
TABLE_B_POS = np.array([0.0, +0.55, 0.80], dtype=np.float64)
TARGET_PAD_POS = np.array([0.0, +0.55, 0.802], dtype=np.float64)
TARGET_PAD_RADIUS = 0.085
BASE_DOCK_X = -0.56
STATION_A_BASE = np.array([BASE_DOCK_X, -0.55], dtype=np.float64)
STATION_B_BASE = np.array([BASE_DOCK_X, +0.55], dtype=np.float64)

# Safety corridor bounds (used by Q4 Whole-Body CBF Shield)
AISLE_X_MAX = -0.44  # Keep base chassis strictly clear of table front edge (x = -0.29)
AISLE_X_MIN = -1.10
AISLE_Y_MIN = -0.95
AISLE_Y_MAX = +0.95
EE_Z_MIN = 0.815     # Prevent slamming EE into tabletops (z = 0.80)


class DualTableArena(TableArena):
    """Side-by-side dual-table pharmacy arena (Table A pickup + Table B drop-off)."""

    def __init__(
        self,
        table_full_size=TABLE_FULL_SIZE,
        table_friction=(1.0, 0.005, 0.0001),
        table_a_offset=tuple(TABLE_A_POS),
        table_b_offset=tuple(TABLE_B_POS),
    ):
        self.table_a_offset = np.asarray(table_a_offset, dtype=np.float64)
        self.table_b_offset = np.asarray(table_b_offset, dtype=np.float64)
        super().__init__(
            table_full_size=table_full_size,
            table_friction=table_friction,
            table_offset=table_a_offset,
            has_legs=True,
        )
        self._add_second_table_and_markers()
        self._configure_cameras()

    def _add_second_table_and_markers(self) -> None:
        # Clone Table A body to create Table B at table_b_offset
        table_b = copy.deepcopy(self.table_body)
        table_b.set("name", "table_b")
        center_b = self.bottom_pos + np.array([0.0, 0.0, -self.table_half_size[2]]) + self.table_b_offset
        table_b.set("pos", array_to_string(center_b))
        for elem in table_b.iter():
            name = elem.get("name")
            if name and name != "table_b":
                elem.set("name", name.replace("table", "table_b", 1))
        self.worldbody.append(table_b)

        # Visual green target drop-off pad on Table B (non-colliding)
        pad_b = ET.Element(
            "geom",
            attrib={
                "name": "target_pad_b_visual",
                "type": "cylinder",
                "size": f"{TARGET_PAD_RADIUS:.4f} 0.002",
                "pos": array_to_string(np.array([0.0, 0.0, self.table_half_size[2] + 0.002])),
                "rgba": "0.14 0.78 0.32 0.90",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            },
        )
        table_b.append(pad_b)

        # Subtle blue staging ring on Table A (non-colliding)
        pad_a = ET.Element(
            "geom",
            attrib={
                "name": "pickup_pad_a_visual",
                "type": "cylinder",
                "size": "0.095 0.0015",
                "pos": array_to_string(np.array([0.0, 0.0, self.table_half_size[2] + 0.0015])),
                "rgba": "0.20 0.50 0.92 0.45",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            },
        )
        self.table_body.append(pad_a)

    def _configure_cameras(self) -> None:
        # Wide elevated third-person camera framing Table A, Table B, and PandaOmron
        cam = self.worldbody.find("./camera[@name='agentview']")
        if cam is not None:
            cam.set("pos", "1.65 0.0 1.95")
            cam.set("quat", "0.653 0.271 0.271 0.653")  # looks from +x downward toward origin
        front = self.worldbody.find("./camera[@name='frontview']")
        if front is not None:
            front.set("pos", "1.85 0.0 1.45")
            front.set("quat", "0.561 0.430 0.430 0.561")


class MobilePickPlaceEnv(ManipulationEnv):
    """Dual-Table Mobile Pick-and-Place environment for PandaOmron (CW2)."""

    def __init__(
        self,
        robots="PandaOmron",
        controller_configs=None,
        use_camera_obs=True,
        use_object_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names="agentview",
        camera_heights=128,
        camera_widths=128,
        control_freq=20,
        horizon=300,
        base_y_init_range=(-0.25, 0.05),
        cube_xy_noise=0.04,
        seed=None,
        **kwargs,
    ):
        self.table_full_size = TABLE_FULL_SIZE
        self.table_friction = (1.0, 5e-3, 1e-4)
        self.table_a_offset = TABLE_A_POS.copy()
        self.table_b_offset = TABLE_B_POS.copy()
        self.base_y_init_range = base_y_init_range
        self.cube_xy_noise = float(cube_xy_noise)
        self.use_object_obs = use_object_obs
        self.placement_initializer = None

        if controller_configs is None:
            controller_configs = load_composite_controller_config(robot="PandaOmron")
            controller_configs["type"] = "BASIC"

        super().__init__(
            robots=robots,
            controller_configs=controller_configs,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=False,
            hard_reset=True,
            lite_physics=True,
            seed=seed,
            **kwargs,
        )

    def _load_model(self):
        super()._load_model()
        # Spawn PandaOmron in the service aisle facing +x
        self.robots[0].robot_model.set_base_xpos([BASE_DOCK_X, -0.15, 0.0])

        mujoco_arena = DualTableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_a_offset=tuple(self.table_a_offset),
            table_b_offset=tuple(self.table_b_offset),
        )
        mujoco_arena.set_origin([0, 0, 0])

        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )
        self.cube = BoxObject(
            name="cube",
            size_min=[0.022, 0.022, 0.022],
            size_max=[0.022, 0.022, 0.022],
            rgba=[0.90, 0.12, 0.12, 1.0],
            material=redwood,
            rng=self.rng,
        )

        self.placement_initializer = UniformRandomSampler(
            name="CubeTableASampler",
            mujoco_objects=self.cube,
            x_range=[-self.cube_xy_noise, self.cube_xy_noise],
            y_range=[-self.cube_xy_noise, self.cube_xy_noise],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_a_offset,
            z_offset=0.01,
            rng=self.rng,
        )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def _setup_references(self):
        super()._setup_references()
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

            @sensor(modality=modality)
            def target_pad_pos(obs_cache):
                return TARGET_PAD_POS.copy()

            sensors = [cube_pos, cube_quat, target_pad_pos]
            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])
            sensors += [
                self._get_obj_eef_sensor(full_pf, "cube_pos", f"{arm_pf}gripper_to_cube_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            for s in sensors:
                observables[s.__name__] = Observable(
                    name=s.__name__, sensor=s, sampling_rate=self.control_freq
                )
        return observables

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                )
            # Randomize initial mobile base y-position in the service aisle
            y0 = float(self.rng.uniform(self.base_y_init_range[0], self.base_y_init_range[1]))
            base_qpos_idx = self.robots[0]._ref_joint_pos_indexes
            # Set mobile base y-slider joint if present
            for jname in self.robots[0].robot_model.base.joints:
                if "y" in jname.lower():
                    self.sim.data.set_joint_qpos(jname, y0 - (-0.15))
            self.sim.forward()

    def is_grasping_cube(self) -> bool:
        return bool(self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube))

    def _check_success(self) -> bool:
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        xy_dist = float(np.linalg.norm(cube_pos[:2] - TARGET_PAD_POS[:2]))
        on_table_b = (TABLE_B_POS[2] + 0.005) <= float(cube_pos[2]) <= (TABLE_B_POS[2] + 0.065)
        released = not self.is_grasping_cube()
        return (xy_dist < 0.12) and on_table_b and released

    def reward(self, action=None) -> float:
        if self._check_success():
            return 10.0
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        d_target = float(np.linalg.norm(cube_pos[:2] - TARGET_PAD_POS[:2]))
        return float(1.0 - np.tanh(1.5 * d_target))


def make_mobile_pick_place_env(
    use_camera_obs: bool = True,
    camera_names: str | list[str] = "agentview",
    camera_size: int = 128,
    horizon: int = 300,
    seed: int | None = 0,
    **kwargs,
) -> MobilePickPlaceEnv:
    """Factory helper used across all CW2 starter and solution scripts."""
    return MobilePickPlaceEnv(
        use_camera_obs=use_camera_obs,
        has_offscreen_renderer=use_camera_obs,
        camera_names=camera_names,
        camera_heights=camera_size,
        camera_widths=camera_size,
        horizon=horizon,
        seed=seed,
        **kwargs,
    )


def extract_proprio(obs: dict) -> np.ndarray:
    """Compact 16-D proprioception + task vector shared by BC, AIRL, and VLA."""
    base_xy = obs["robot0_base_pos"][:2]
    eef_pos = obs["robot0_eef_pos"]
    grip = obs["robot0_gripper_qpos"][:2]
    cube = obs["cube_pos"]
    rel_cube = cube - eef_pos
    rel_pad = TARGET_PAD_POS - cube
    return np.concatenate([base_xy, eef_pos, grip, cube, rel_cube, rel_pad]).astype(np.float32)


if __name__ == "__main__":
    env = make_mobile_pick_place_env(use_camera_obs=True, camera_size=128, seed=0)
    obs = env.reset()
    print(f"MobilePickPlaceEnv ready: action_dim={env.action_dim}")
    print(f"  base_pos={obs['robot0_base_pos'].round(3)}, eef_pos={obs['robot0_eef_pos'].round(3)}")
    print(f"  cube_pos={obs['cube_pos'].round(3)}, target_pad={obs['target_pad_pos'].round(3)}")
    print(f"  proprio_dim={extract_proprio(obs).shape[0]}, agentview={obs['agentview_image'].shape}")
    env.close()
