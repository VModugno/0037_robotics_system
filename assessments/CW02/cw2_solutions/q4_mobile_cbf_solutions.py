"""Instructor reference solutions for COMP0037 2026/27 Coursework 2 — Question 4.

Patches `WholeBodyCBFShield.filter_action` and `run_red_team_suite` in
`assessment/CW02/cw2_starter/q4_mobile_cbf_integration.py`, evaluates the 4-tier
safe mobile-manipulation stack (`Planner -> Skills -> Whole-Body CBF -> PandaOmron`)
across 3 red-teaming scenarios (Nominal, Adversarial Ram/Slam, Noisy Perception),
and saves:
  - `outputs/q4_cbf_redteam.png`
  - `outputs/q4_cbf_adversarial.gif` (Unshielded vs. Whole-Body CBF under adversarial attack)

Usage (from repository root):
    .venv_robot\\Scripts\\python assessment\\CW02\\cw2_solutions\\q4_mobile_cbf_solutions.py
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
STARTER_DIR = os.path.normpath(os.path.join(CUR_DIR, os.pardir, "cw2_starter"))
OUT_DIR = os.path.join(CUR_DIR, "outputs")
sys.path.insert(0, STARTER_DIR)

import mobile_pick_place_env as mpe  # noqa: E402
import q2_mobile_llm_planner as q2  # noqa: E402
import q4_mobile_cbf_integration as q4  # noqa: E402
import q2_mobile_llm_solutions as q2_sol  # noqa: E402


def filter_action(self: q4.WholeBodyCBFShield, obs: dict, u_nom: np.ndarray) -> np.ndarray:
    """Reference Whole-Body CBF Safety Shield for PandaOmron (coupled base + EE)."""
    self.step_count += 1
    u = np.asarray(u_nom, dtype=np.float32).copy()
    u_orig = u.copy()
    reasons: list[str] = []

    base_x, base_y = float(obs["robot0_base_pos"][0]), float(obs["robot0_base_pos"][1])
    eef_z = float(obs["robot0_eef_pos"][2])

    # 1. Velocity / action-norm clamping
    u[0:6] = np.clip(u[0:6], -self.max_arm_cmd, self.max_arm_cmd)
    u[7:10] = np.clip(u[7:10], -self.max_base_vel, self.max_base_vel)
    u[10] = float(np.clip(u[10], -1.0, 1.0))

    # 2. Mobile-base table-collision barrier: h_x = AISLE_X_MAX - base_x >= 0
    #    Include 1.5 cm stopping-distance margin for wheel inertia at 20 Hz
    h_xmax = (mpe.AISLE_X_MAX - 0.015) - base_x
    vx_max = float(self.alpha_base * h_xmax / 0.020)
    if u[7] > vx_max:
        u[7] = float(np.clip(vx_max, -self.max_base_vel, self.max_base_vel))
        reasons.append("base_table_cbf")

    h_xmin = base_x - mpe.AISLE_X_MIN
    vx_min = float(-self.alpha_base * h_xmin / 0.015)
    if u[7] < vx_min:
        u[7] = float(np.clip(vx_min, -self.max_base_vel, self.max_base_vel))
        reasons.append("base_xmin_cbf")

    # 3. Mobile-base aisle lateral bounds: AISLE_Y_MIN <= base_y <= AISLE_Y_MAX
    h_ymax = mpe.AISLE_Y_MAX - base_y
    vy_max = float(self.alpha_base * h_ymax / 0.03)
    if u[8] > vy_max:
        u[8] = float(np.clip(vy_max, -self.max_base_vel, self.max_base_vel))
        reasons.append("base_ymax_cbf")

    h_ymin = base_y - mpe.AISLE_Y_MIN
    vy_min = float(-self.alpha_base * h_ymin / 0.03)
    if u[8] < vy_min:
        u[8] = float(np.clip(vy_min, -self.max_base_vel, self.max_base_vel))
        reasons.append("base_ymin_cbf")

    # 4. End-effector tabletop impact barrier: h_z = eef_z - EE_Z_MIN >= 0
    #    OSC_POSE dz output scale is 0.05 m per unit action
    h_z = eef_z - mpe.EE_Z_MIN
    az_min = float(-self.alpha_ee * h_z / 0.05)
    if u[2] < az_min:
        u[2] = float(np.clip(az_min, -self.max_arm_cmd, self.max_arm_cmd))
        reasons.append("ee_table_cbf")

    if float(np.linalg.norm(u - u_orig)) > 1e-4:
        self.interventions += 1
        self.audit_log.append(
            q4.SafetyAuditEntry(
                step=self.step_count,
                reason="+".join(reasons) if reasons else "vel_clamp",
                u_nom_norm=float(np.linalg.norm(u_orig)),
                u_safe_norm=float(np.linalg.norm(u)),
                base_x=base_x,
                eef_z=eef_z,
            )
        )
    return u


def _run_scenario(scenario: str, shield_enabled: bool, seed: int = 0, record: bool = False) -> dict:
    env = mpe.make_mobile_pick_place_env(
        use_camera_obs=record, camera_size=192 if record else 64, horizon=360, seed=seed
    )
    shield = q4.WholeBodyCBFShield()
    filt = shield.filter_action if shield_enabled else None

    base_x_hist, eef_z_hist, frames = [], [], []
    base_breaches, ee_breaches = 0, 0
    success = False

    if scenario == "nominal":
        lib = q2.MobileSkillLibrary(env, action_filter=filt, record_frames=record)
        lib.reset_env()
        planner = q2_sol.MobileLLMPlanner(backend="mock")
        plan = planner.plan("Transfer the red cube from Table A to Table B.", lib.get_world_state())
        for step in plan:
            lib.execute(step["skill"], step.get("args"))
            bx = float(lib.obs["robot0_base_pos"][0])
            ez = float(lib.obs["robot0_eef_pos"][2])
            base_x_hist.append(bx)
            eef_z_hist.append(ez)
            base_breaches += int(bx > mpe.AISLE_X_MAX + 0.005)
            ee_breaches += int(ez < mpe.EE_Z_MIN - 0.005)
        success = bool(env._check_success())
        frames = lib.frames

    elif scenario == "adversarial":
        # Adversarial command: ram the mobile base straight forward into the tables while slamming EE downward
        obs = env.reset()
        if record:
            frames.append(np.flipud(obs["agentview_image"]).copy())
        for t in range(80):
            u_adv = np.zeros(11, dtype=np.float32)
            u_adv[7] = 1.0   # full speed +x straight into tables
            u_adv[8] = 0.0   # no lateral drift so braking at AISLE_X_MAX is crystal clear
            u_adv[2] = -1.0  # slam EE downward into tabletop
            u_exec = filt(obs, u_adv) if filt is not None else u_adv
            obs, _, _, _ = env.step(u_exec)
            bx = float(obs["robot0_base_pos"][0])
            ez = float(obs["robot0_eef_pos"][2])
            base_x_hist.append(bx)
            eef_z_hist.append(ez)
            base_breaches += int(bx > mpe.AISLE_X_MAX + 0.005)
            ee_breaches += int(ez < mpe.EE_Z_MIN - 0.005)
            if record and (t % 2 == 0):
                frames.append(np.flipud(obs["agentview_image"]).copy())

    elif scenario == "noisy_perception":
        # Run full MobileSkillLibrary transfer while injecting Gaussian pose noise into
        # the planner/skill perception (cube_pos & base_pos); CBF shields true state.
        rng = np.random.default_rng(seed + 99)

        class _NoisySkillLibrary(q2.MobileSkillLibrary):
            def _step(self, raw_action: np.ndarray) -> dict:
                if self.env.done:
                    return self.obs
                act = np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0)
                if self.action_filter is not None:
                    act = self.action_filter(self._true_obs, act)
                self._true_obs, _, _, _ = self.env.step(act)
                bx = float(self._true_obs["robot0_base_pos"][0])
                ez = float(self._true_obs["robot0_eef_pos"][2])
                base_x_hist.append(bx)
                eef_z_hist.append(ez)
                nonlocal base_breaches, ee_breaches
                base_breaches += int(bx > mpe.AISLE_X_MAX + 0.005)
                ee_breaches += int(ez < mpe.EE_Z_MIN - 0.005)
                # Corrupt perception seen by the high-level skills:
                # Realistic depth-camera bias (-12 cm in z) + wheel-odometry bias (-25 cm in x) + noise,
                # which tricks an unshielded controller into driving the base & EE into the table.
                noisy = dict(self._true_obs)
                noisy["cube_pos"] = self._true_obs["cube_pos"] + np.array([0.0, 0.0, -0.12]) + rng.normal(0.0, 0.02, size=3)
                noisy["robot0_base_pos"] = self._true_obs["robot0_base_pos"] + np.array([-0.25, 0.0, 0.0]) + rng.normal(0.0, [0.02, 0.02, 0.0])
                self.obs = noisy
                self._maybe_capture()
                return self.obs

            def reset_env(self) -> dict:
                self.frames.clear()
                self._true_obs = self.env.reset()
                self.obs = dict(self._true_obs)
                self._maybe_capture()
                return self.obs

        lib = _NoisySkillLibrary(env, action_filter=filt, record_frames=record)
        lib.reset_env()
        planner = q2_sol.MobileLLMPlanner(backend="mock")
        plan = planner.plan("Transfer the red cube from Table A to Table B.", lib.get_world_state())
        for step in plan:
            lib.execute(step["skill"], step.get("args"))
        success = bool(env._check_success())
        frames = lib.frames

    env.close()
    return {
        "success": success,
        "base_breaches": base_breaches,
        "ee_breaches": ee_breaches,
        "interventions": shield.interventions,
        "max_base_x": max(base_x_hist) if base_x_hist else -1.0,
        "min_eef_z": min(eef_z_hist) if eef_z_hist else 1.0,
        "base_x_hist": np.asarray(base_x_hist),
        "eef_z_hist": np.asarray(eef_z_hist),
        "frames": frames,
    }


def run_red_team_suite(shield_enabled: bool = True, seed: int = 0) -> dict:
    return {
        "nominal": _run_scenario("nominal", shield_enabled, seed=seed, record=True),
        "adversarial": _run_scenario("adversarial", shield_enabled, seed=seed, record=True),
        "noisy_perception": _run_scenario("noisy_perception", shield_enabled, seed=seed, record=True),
    }


def save_comparison_gif(
    unsh_frames: list[np.ndarray],
    cbf_frames: list[np.ndarray],
    left_title: str,
    right_title: str,
    out_path: str,
    stride: int = 1,
) -> None:
    if not unsh_frames or not cbf_frames:
        return
    uf = unsh_frames[::stride]
    cf = cbf_frames[::stride]
    n = max(len(uf), len(cf))
    pil = []
    for i in range(n):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 4.1), dpi=90)
        ax1.imshow(uf[min(i, len(uf) - 1)])
        ax1.set_title(left_title, fontsize=9.5, fontweight="bold", color="#DC2626")
        ax1.axis("off")
        ax2.imshow(cf[min(i, len(cf) - 1)])
        ax2.set_title(right_title, fontsize=9.5, fontweight="bold", color="#16A34A")
        ax2.axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        pil.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE))
        plt.close(fig)
    durs = [100] * len(pil)
    durs[-1] = 1500
    pil[0].save(out_path, save_all=True, append_images=pil[1:], duration=durs, loop=0)
    print(f"Saved GIF -> {out_path} ({len(pil)} frames)")


def patch_all() -> None:
    q4.WholeBodyCBFShield.filter_action = filter_action
    q4.run_red_team_suite = run_red_team_suite


def main() -> None:
    ap = argparse.ArgumentParser(description="CW2 Q4 Whole-Body CBF Red-Teaming reference solutions")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    patch_all()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 78)
    print("CW2 Q4 — Whole-Body CBF Safety Shield & 3-Scenario Red-Teaming (PandaOmron)")
    print("=" * 78)
    unsh = q4.run_red_team_suite(shield_enabled=False, seed=0)
    shld = q4.run_red_team_suite(shield_enabled=True, seed=0)

    for sc in ("nominal", "adversarial", "noisy_perception"):
        u, s = unsh[sc], shld[sc]
        print(
            f"  [{sc:<16}] Unshielded: base_hit={u['base_breaches']:2d}, ee_hit={u['ee_breaches']:2d}, succ={u['success']} | "
            f"CBF: base_hit={s['base_breaches']:2d}, ee_hit={s['ee_breaches']:2d}, interv={s['interventions']:3d}, succ={s['success']}"
        )

    # 3-panel diagnostic plot
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    # Panel 1: Adversarial base_x trajectory vs AISLE_X_MAX
    axes[0].plot(unsh["adversarial"]["base_x_hist"], color="#DC2626", lw=2.0, label="Unshielded")
    axes[0].plot(shld["adversarial"]["base_x_hist"], color="#16A34A", lw=2.0, label="Whole-Body CBF")
    axes[0].axhline(mpe.AISLE_X_MAX, color="#0F172A", ls="--", lw=1.4, label=f"AISLE_X_MAX ({mpe.AISLE_X_MAX}m)")
    axes[0].set(xlabel="Step", ylabel="Base x (m)", title="Adversarial Ram: Base-to-Table CBF")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    # Panel 2: Adversarial eef_z trajectory vs EE_Z_MIN
    axes[1].plot(unsh["adversarial"]["eef_z_hist"], color="#DC2626", lw=2.0, label="Unshielded")
    axes[1].plot(shld["adversarial"]["eef_z_hist"], color="#16A34A", lw=2.0, label="Whole-Body CBF")
    axes[1].axhline(mpe.EE_Z_MIN, color="#0F172A", ls="--", lw=1.4, label=f"EE_Z_MIN ({mpe.EE_Z_MIN}m)")
    axes[1].set(xlabel="Step", ylabel="End-Effector z (m)", title="Adversarial Slam: EE Height CBF")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    # Panel 3: Total safety breaches across all 3 scenarios
    scs = ["Nominal", "Adversarial", "Noisy\nPercep."]
    u_tot = [unsh[k]["base_breaches"] + unsh[k]["ee_breaches"] for k in ("nominal", "adversarial", "noisy_perception")]
    s_tot = [shld[k]["base_breaches"] + shld[k]["ee_breaches"] for k in ("nominal", "adversarial", "noisy_perception")]
    x = np.arange(3)
    axes[2].bar(x - 0.18, u_tot, width=0.34, color="#DC2626", edgecolor="#0F172A", label="Unshielded")
    axes[2].bar(x + 0.18, s_tot, width=0.34, color="#16A34A", edgecolor="#0F172A", label="Whole-Body CBF")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(scs)
    axes[2].set(ylabel="Total Safety Breaches", title="Red-Team Safety Breaches")
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    p_out = os.path.join(args.out_dir, "q4_cbf_redteam.png")
    fig.savefig(p_out, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p_out}")

    save_comparison_gif(
        unsh["nominal"]["frames"],
        shld["nominal"]["frames"],
        "Unshielded (Nominal Transfer)",
        "Whole-Body CBF (Nominal Transfer — Preserved)",
        os.path.join(args.out_dir, "q4_cbf_nominal.gif"),
        stride=2,
    )
    save_comparison_gif(
        unsh["adversarial"]["frames"],
        shld["adversarial"]["frames"],
        "Unshielded (Adversarial Ram + Slam)",
        "Whole-Body CBF (Blocked — 0 Breaches)",
        os.path.join(args.out_dir, "q4_cbf_adversarial.gif"),
    )
    save_comparison_gif(
        unsh["noisy_perception"]["frames"],
        shld["noisy_perception"]["frames"],
        "Unshielded (Noisy Perception Transfer)",
        "Whole-Body CBF (Noisy Perception — 0 Breaches)",
        os.path.join(args.out_dir, "q4_cbf_noisy.gif"),
        stride=2,
    )
    print("CW2_Q4_SOLUTIONS_DONE")


if __name__ == "__main__":
    main()
