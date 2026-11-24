"""Generate vector PDF figures (with live MuJoCo render) for COMP0037 2026/27 Coursework 2.

Produces:
  - figures/fig1_mobile_arena.pdf
  - figures/fig3_wholebody_cbf.pdf

Run with `.venv_robot` from the repository root:
    .venv_robot\\Scripts\\python assessment\\CW02\\generate_cw2_figures.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

CW2_DIR = os.path.dirname(os.path.abspath(__file__))
STARTER_DIR = os.path.join(CW2_DIR, "cw2_starter")
OUT_DIR = os.path.join(CW2_DIR, "figures")
sys.path.insert(0, STARTER_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

import mobile_pick_place_env as mpe  # noqa: E402


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "figure.dpi": 200,
})


def make_fig1_mobile_arena():
    # Capture real high-res MuJoCo agentview render of PandaOmron + Table A + Table B
    env = mpe.make_mobile_pick_place_env(
        use_camera_obs=True, camera_names="agentview", camera_size=384, seed=0
    )
    obs = env.reset()
    render_img = np.flipud(obs["agentview_image"]).copy()
    env.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.1), gridspec_kw={"width_ratios": [1.0, 1.12]})

    # Panel (a): Real MuJoCo render
    ax1.imshow(render_img)
    ax1.set_title("(a) MuJoCo Render: PandaOmron Dual-Table Arena", pad=7)
    ax1.axis("off")

    # Panel (b): Top-down metric schematic
    ax2.add_patch(patches.Rectangle((-1.10, -0.95), 0.66, 1.90, facecolor="#ECFDF5", edgecolor="#10B981", ls="--", lw=1.2, alpha=0.65, label="Safe service aisle"))
    ax2.axvline(mpe.AISLE_X_MAX, color="#DC2626", lw=1.8, ls="-.", label=f"Base CBF wall ($x={mpe.AISLE_X_MAX}$ m)")

    # Table A (y = -0.55) and Table B (y = +0.55)
    tx, ty = mpe.TABLE_FULL_SIZE[0], mpe.TABLE_FULL_SIZE[1]
    ax2.add_patch(patches.Rectangle((-tx / 2, -0.55 - ty / 2), tx, ty, facecolor="#E2E8F0", edgecolor="#334155", lw=1.6))
    ax2.text(0.0, -0.74, "Table A (Pickup)\n(0.00, -0.55)", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#1E293B")
    ax2.add_patch(patches.Rectangle((-0.03, -0.58), 0.06, 0.06, facecolor="#DC2626", edgecolor="#7F1D1D", lw=1.2, zorder=4))
    ax2.text(0.05, -0.55, "Cube", ha="left", va="center", fontsize=7.8, fontweight="bold", color="#991B1B")

    ax2.add_patch(patches.Rectangle((-tx / 2, +0.55 - ty / 2), tx, ty, facecolor="#E2E8F0", edgecolor="#334155", lw=1.6))
    ax2.text(0.0, +0.74, "Table B (Drop-off)\n(0.00, +0.55)", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#1E293B")
    ax2.add_patch(patches.Circle((0.0, 0.55), mpe.TARGET_PAD_RADIUS, facecolor="#86EFAC", edgecolor="#15803D", lw=1.4, zorder=4))
    ax2.text(0.11, +0.55, "Target\npad", ha="left", va="center", fontsize=7.8, fontweight="bold", color="#166534")

    # Station A & B docking markers + PandaOmron base
    ax2.scatter([mpe.BASE_DOCK_X, mpe.BASE_DOCK_X], [-0.55, 0.55], s=55, color="#1D4ED8", zorder=5)
    ax2.text(mpe.BASE_DOCK_X - 0.06, -0.55, "Station A\n(-0.56, -0.55)", ha="right", va="center", fontsize=7.6, color="#1E3A8A", fontweight="bold")
    ax2.text(mpe.BASE_DOCK_X - 0.06, +0.55, "Station B\n(-0.56, +0.55)", ha="right", va="center", fontsize=7.6, color="#1E3A8A", fontweight="bold")

    # Mobile base footprint at initial spawn
    ax2.add_patch(patches.FancyBboxPatch((-0.68, -0.25), 0.24, 0.20, boxstyle="round,pad=0.02", facecolor="#3B82F6", edgecolor="#1E3A8A", lw=1.4, zorder=5))
    ax2.text(-0.56, -0.15, "Panda\nOmron", ha="center", va="center", color="white", fontsize=7.0, fontweight="bold", zorder=6)

    # 4-stage transfer arrows
    ax2.annotate("", xy=(-0.56, -0.49), xytext=(-0.56, -0.26), arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.8))
    ax2.annotate("", xy=(-0.56, +0.49), xytext=(-0.56, -0.49), arrowprops=dict(arrowstyle="->", color="#7C3AED", lw=2.0, ls="--", connectionstyle="arc3,rad=-0.22"))
    ax2.text(-0.88, 0.08, "3. Drive B\n(arm stowed)", ha="center", va="center", fontsize=7.6, color="#6D28D9", fontweight="bold")

    ax2.set_xlim(-1.18, 0.42)
    ax2.set_ylim(-1.02, 1.02)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title("(b) Top-Down Metric Layout & Service Aisle", pad=7)
    ax2.grid(alpha=0.25)
    ax2.legend(loc="lower right", fontsize=7.2, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_mobile_arena.pdf"), bbox_inches="tight")
    plt.close(fig)


def _draw_box(ax, x, y, w, h, text, fc, ec="#0F172A", tc="#0F172A", fs=8.0):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=fc, edgecolor=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=tc)


def make_fig2_vla_skill_architecture():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.6))
    for ax in (ax1, ax2):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")

    # Panel (a): Hierarchical LLM Planner + Typed Skills + Recovery (Q2)
    ax1.set_title("(a) Typed Skill API & Precondition Validation (Q2)", pad=5)
    _draw_box(ax1, 0.5, 4.7, 9.0, 0.9, "Natural-Language Instruction + Symbolic World State\n(station, arm_stowed, holding_cube)", "#DBEAFE", "#1D4ED8")
    _draw_box(ax1, 0.5, 3.3, 4.1, 0.95, "Structured JSON\nLLM / Mock Planner", "#EDE9FE", "#6D28D9")
    _draw_box(ax1, 5.4, 3.3, 4.1, 0.95, "Schema & Symbolic\nPrecondition Validator", "#FEF3C7", "#B45309")
    _draw_box(ax1, 0.5, 1.7, 9.0, 1.0, "MobileSkillLibrary (Closed-Loop P-Controllers)\ndrive_to_station(A/B) | stow_arm() | pick_cube() | verify_grasp() | place_cube()", "#DCFCE7", "#15803D")
    _draw_box(ax1, 2.0, 0.25, 6.0, 0.9, "PandaOmron (11-D Base + Arm + Gripper)\nClosed-loop failure detection -> Replan", "#F1F5F9", "#334155")

    ax1.annotate("", xy=(2.55, 4.28), xytext=(2.55, 4.68), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax1.annotate("", xy=(5.38, 3.78), xytext=(4.62, 3.78), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax1.annotate("", xy=(7.45, 2.74), xytext=(7.45, 3.28), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax1.annotate("", xy=(5.0, 1.18), xytext=(5.0, 1.68), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax1.annotate("", xy=(0.45, 3.8), xytext=(1.95, 0.7), arrowprops=dict(arrowstyle="->", lw=1.4, ls="--", color="#DC2626", connectionstyle="arc3,rad=-0.28"))
    ax1.text(0.25, 2.4, "Retry", rotation=90, fontsize=7.5, color="#DC2626", fontweight="bold")

    # Panel (b): Track A VLA + Conditional Diffusion Chunk-MPC (Q3)
    ax2.set_title("(b) Track A VLA: nanoVLM + Diffusion Chunk-MPC (Q3)", pad=5)
    _draw_box(ax2, 0.3, 4.55, 2.6, 1.0, "RGB Image\n(64x64 agentview)", "#DBEAFE", "#1D4ED8")
    _draw_box(ax2, 3.4, 4.55, 2.8, 1.0, "TinyViT\n(64 patch tokens)", "#E0E7FF", "#4338CA")
    _draw_box(ax2, 6.7, 4.55, 3.0, 1.0, "2x2 Pixel-Shuffle\n(16 prefix tokens)", "#EDE9FE", "#6D28D9")
    _draw_box(ax2, 5.2, 2.85, 4.5, 1.05, "NanoVLM (Prefix-Causal)\nFrozen Visual Embedding", "#FCE7F3", "#BE185D")
    _draw_box(ax2, 0.3, 2.85, 4.2, 1.05, "16-D Proprioception\n(base_xy, eef, grip, cube)", "#FEF3C7", "#B45309")
    _draw_box(ax2, 1.0, 1.35, 8.0, 0.95, "Conditional DDPM Diffusion Action Head\nPredicts 4-step x 11-D = 44-D Action Chunk", "#DCFCE7", "#15803D")
    _draw_box(ax2, 1.8, 0.15, 6.4, 0.75, "Receding-Horizon Chunk-MPC (execute k in {1, 2, 4} steps)", "#F1F5F9", "#334155")

    ax2.annotate("", xy=(3.38, 5.05), xytext=(2.92, 5.05), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax2.annotate("", xy=(6.68, 5.05), xytext=(6.22, 5.05), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax2.annotate("", xy=(7.45, 3.92), xytext=(8.20, 4.53), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax2.annotate("", xy=(7.0, 2.32), xytext=(7.45, 2.83), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax2.annotate("", xy=(2.8, 2.32), xytext=(2.4, 2.83), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax2.annotate("", xy=(5.0, 0.92), xytext=(5.0, 1.33), arrowprops=dict(arrowstyle="->", lw=1.4))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_vla_skill_architecture.pdf"), bbox_inches="tight")
    plt.close(fig)


def make_fig3_wholebody_cbf():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.6))

    # Panel (a): Top-down Base-to-Table & Aisle CBF
    ax1.add_patch(patches.Rectangle((-1.10, -0.95), 0.66, 1.90, facecolor="#ECFDF5", edgecolor="#10B981", ls="--", lw=1.1, alpha=0.65))
    ax1.add_patch(patches.Rectangle((mpe.AISLE_X_MAX, -0.95), 0.74, 1.90, facecolor="#FEE2E2", edgecolor="#DC2626", hatch="///", alpha=0.55, label="Forbidden table zone"))
    ax1.axvline(mpe.AISLE_X_MAX, color="#B91C1C", lw=2.0)
    ax1.add_patch(patches.Rectangle((-0.29, -0.82), 0.58, 0.54, facecolor="#CBD5E1", edgecolor="#334155", lw=1.4))
    ax1.add_patch(patches.Rectangle((-0.29, +0.28), 0.58, 0.54, facecolor="#CBD5E1", edgecolor="#334155", lw=1.4))
    ax1.text(0.0, -0.55, "Table A", ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax1.text(0.0, +0.55, "Table B", ha="center", va="center", fontsize=8.5, fontweight="bold")

    ax1.add_patch(patches.FancyBboxPatch((-0.62, -0.08), 0.16, 0.16, boxstyle="round,pad=0.02", facecolor="#2563EB", edgecolor="#1E3A8A", lw=1.4, zorder=5))
    ax1.annotate("", xy=(-0.22, 0.12), xytext=(-0.46, 0.0), arrowprops=dict(arrowstyle="->", color="#DC2626", lw=2.2))
    ax1.text(-0.26, 0.16, r"$u_{\mathrm{nom}}$ (ram)", color="#B91C1C", fontsize=8.2, fontweight="bold")
    ax1.annotate("", xy=(-0.46, 0.32), xytext=(-0.46, 0.0), arrowprops=dict(arrowstyle="->", color="#16A34A", lw=2.2))
    ax1.text(-0.88, 0.25, r"$u_{\mathrm{safe}}$ (CBF)", color="#15803D", fontsize=8.2, fontweight="bold")
    ax1.set_xlim(-1.15, 0.35)
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title(r"(a) Base CBF: $h_x = x_{\max} - x_{\mathrm{base}} \geq 0$", pad=6)
    ax1.grid(alpha=0.25)

    # Panel (b): Side-view EE Tabletop Impact CBF
    ax2.add_patch(patches.Rectangle((-0.30, 0.75), 0.60, 0.05, facecolor="#CBD5E1", edgecolor="#334155", lw=1.5, label="Tabletop ($z=0.80$ m)"))
    ax2.add_patch(patches.Rectangle((-0.45, 0.65), 0.90, mpe.EE_Z_MIN - 0.65, facecolor="#FEE2E2", edgecolor="#DC2626", hatch="///", alpha=0.50))
    ax2.axhline(mpe.EE_Z_MIN, color="#B91C1C", lw=2.0, ls="-.", label=f"EE CBF floor ($z={mpe.EE_Z_MIN}$ m)")
    ax2.scatter([0.0], [0.89], s=95, color="#F59E0B", edgecolor="#0F172A", zorder=6, label="End-effector")
    ax2.annotate("", xy=(0.0, 0.72), xytext=(0.0, 0.89), arrowprops=dict(arrowstyle="->", color="#DC2626", lw=2.2))
    ax2.text(0.04, 0.76, r"$u_{\mathrm{nom},z}$ (slam)", color="#B91C1C", fontsize=8.2, fontweight="bold")
    ax2.annotate("", xy=(0.0, 0.835), xytext=(0.0, 0.89), arrowprops=dict(arrowstyle="->", color="#16A34A", lw=2.4))
    ax2.text(-0.38, 0.85, r"$u_{\mathrm{safe},z}$ (clamped)", color="#15803D", fontsize=8.2, fontweight="bold")
    ax2.set_xlim(-0.45, 0.45)
    ax2.set_ylim(0.68, 1.12)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("z (m)")
    ax2.set_title(r"(b) EE Height CBF: $h_z = z_{\mathrm{ee}} - z_{\min} \geq 0$", pad=6)
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper right", fontsize=7.5, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_wholebody_cbf.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_fig1_mobile_arena()
    make_fig3_wholebody_cbf()
    print(f"Generated MedBot CW2 figures in {OUT_DIR}")

