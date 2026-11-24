"""Generate publication-grade vector figures for COMP0037 2026/27 Coursework 1 (MedBot)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
})


def make_fig1_hospital_ward():
    # Exact layout matching cw1_starter/q2_hospital_ward.py
    layout = [
        "...........G",
        "..##..II..#.",
        "..##..II..#.",
        "..D...II..D.",
        "......##....",
        ".##...##..#.",
        ".##..BBBB.#.",
        "S...........",
    ]
    rows, cols = len(layout), len(layout[0])
    fig, ax = plt.subplots(figsize=(8.6, 4.4))

    colors = {
        ".": "#F8FAFC",
        "#": "#334155",
        "D": "#F59E0B",
        "I": "#93C5FD",
        "B": "#DC2626",
        "S": "#2563EB",
        "G": "#16A34A",
    }
    labels = {
        "#": ("#", "white"),
        "D": ("D", "#1E293B"),
        "I": ("ICU", "#1E3A8A"),
        "B": ("BIO", "white"),
        "S": ("S", "white"),
        "G": ("G", "white"),
    }

    for r, row in enumerate(layout):
        for c, ch in enumerate(row):
            y = rows - 1 - r
            rect = patches.Rectangle((c, y), 1, 1, facecolor=colors[ch], edgecolor="#94A3B8", linewidth=0.9)
            ax.add_patch(rect)
            if ch in labels:
                txt, fg = labels[ch]
                fs = 7.2 if len(txt) > 1 else 9.5
                ax.text(c + 0.5, y + 0.5, txt, ha="center", va="center", color=fg, fontweight="bold", fontsize=fs)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_xticklabels(range(cols), fontsize=8)
    ax.set_yticks(np.arange(rows) + 0.5)
    ax.set_yticklabels(range(rows - 1, -1, -1), fontsize=8)
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    ax.tick_params(length=0)
    ax.set_title("HospitalWardEnv (8×12) — MedBot Autonomous Pharmacy Delivery Map", pad=8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("#334155")

    legend_handles = [
        patches.Patch(facecolor=colors["S"], edgecolor="#334155", label="Pharmacy Start (S)"),
        patches.Patch(facecolor=colors["G"], edgecolor="#334155", label="Ward Goal (G, +10)"),
        patches.Patch(facecolor=colors["#"], edgecolor="#334155", label="Wall (#)"),
        patches.Patch(facecolor=colors["D"], edgecolor="#334155", label="Auto Door (D, -0.5)"),
        patches.Patch(facecolor=colors["I"], edgecolor="#334155", label="Sterile ICU (I, -5.0)"),
        patches.Patch(facecolor=colors["B"], edgecolor="#334155", label="Biohazard Spill (B, -50 + reset)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=8.2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(os.path.join(OUT_DIR, "fig1_hospital_ward.pdf"), bbox_inches="tight")
    plt.close(fig)


def make_fig2_slip_model():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.1))

    def draw_case(ax, title, nominal, slips, arrow_vec):
        for r in range(3):
            for c in range(3):
                fc = "#FFFFFF"
                txt, fg = "", "black"
                if (r, c) == (1, 1):
                    fc, txt, fg = "#1E293B", "MedBot", "white"
                elif (r, c) == nominal:
                    fc, txt, fg = "#64748B", r"$1 - p_{\mathrm{slip}}$", "white"
                elif (r, c) in slips:
                    fc, txt, fg = "#CBD5E1", r"$p_{\mathrm{slip}}/2$", "#1E293B"
                rect = patches.Rectangle((c, 2 - r), 1, 1, facecolor=fc, edgecolor="#334155", linewidth=1.0)
                ax.add_patch(rect)
                if txt:
                    ax.text(c + 0.5, 2 - r + 0.5, txt, ha="center", va="center", color=fg, fontsize=8.3, fontweight="bold")
        dx, dy = arrow_vec
        ax.annotate(
            "", xy=(1.5 + dx * 0.72, 1.5 + dy * 0.72), xytext=(1.5, 1.5),
            arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.28", color="#E63946", lw=2.0),
        )
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, pad=6)

    draw_case(ax1, "Command: UP", nominal=(0, 1), slips=[(1, 0), (1, 2)], arrow_vec=(0, 1))
    draw_case(ax2, "Command: RIGHT", nominal=(1, 2), slips=[(0, 1), (2, 1)], arrow_vec=(1, 0))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_slip_model.pdf"), bbox_inches="tight")
    plt.close(fig)


def make_fig3_sterile_arm_trolley():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.7), gridspec_kw={"width_ratios": [1.05, 1.0]})

    # Left panel: SterileArmEnv with joint limits AND circular centrifuge obstacle
    L1, L2 = 0.55, 0.45
    th1, th2 = np.deg2rad(-25), np.deg2rad(20)
    p0 = np.array([0.0, 0.0])
    p1 = np.array([L1 * np.cos(th1), L1 * np.sin(th1)])
    p2 = p1 + np.array([L2 * np.cos(th1 + th2), L2 * np.sin(th1 + th2)])

    wedge = patches.Wedge((0, 0), L1 + L2, -92, 92, facecolor="#ECFDF5", edgecolor="#34D399", ls="--", lw=1.0, alpha=0.55, label="Safe joint workspace")
    ax1.add_patch(wedge)
    forb = patches.Wedge((0, 0), L1 + L2, 92, 268, facecolor="#FEF2F2", edgecolor="#F87171", ls=":", lw=1.0, alpha=0.65, label="Joint-limit forbidden zone")
    ax1.add_patch(forb)

    # Centrifuge keep-out obstacle at (0.45, 0.50), r=0.16
    obs = patches.Circle((0.45, 0.50), 0.16, facecolor="#FCA5A5", edgecolor="#B91C1C", lw=1.6, hatch="///", alpha=0.85, zorder=3, label="Centrifuge keep-out (r=0.16m)")
    ax1.add_patch(obs)

    # Arm links
    ax1.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#1E3A8A", lw=5.5, solid_capstyle="round", zorder=4)
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#2563EB", lw=4.5, solid_capstyle="round", zorder=4)
    ax1.scatter([p0[0], p1[0]], [p0[1], p1[1]], s=65, color="#0F172A", zorder=5)
    ax1.scatter([p2[0]], [p2[1]], s=75, color="#F59E0B", edgecolor="#0F172A", zorder=6, label="End-effector (x, y)")

    # Sample vial target
    tgt = np.array([0.15, 0.78])
    circ = patches.Circle(tgt, 0.06, facecolor="#BBF7D0", edgecolor="#15803D", lw=1.4, zorder=3)
    ax1.add_patch(circ)
    ax1.scatter([tgt[0]], [tgt[1]], marker="x", s=55, color="#15803D", lw=2.0, zorder=5, label="Sample vial target")

    ax1.set_xlim(-1.12, 1.12)
    ax1.set_ylim(-1.05, 1.12)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("(a) SterileArmEnv: Dual-Barrier CBF Cabinet", pad=7)
    ax1.grid(alpha=0.25)
    ax1.legend(loc="lower left", fontsize=6.9, framealpha=0.92)

    # Right panel: Mechanical schematic of medbot_payload_trolley (Q4 MJCF model)
    ax2.axhline(0.0, color="#334155", lw=2.2, zorder=2)
    # Floor hatch
    floor = patches.Rectangle((-0.65, -0.10), 1.30, 0.10, facecolor="#E2E8F0", edgecolor="none", hatch="///", zorder=1)
    ax2.add_patch(floor)

    # Wheels
    for wx in (-0.15, 0.15):
        wheel = patches.Circle((wx, 0.055), 0.055, facecolor="#1E293B", edgecolor="#0F172A", lw=1.2, zorder=4)
        ax2.add_patch(wheel)
        hub = patches.Circle((wx, 0.055), 0.02, facecolor="#94A3B8", zorder=5)
        ax2.add_patch(hub)

    # Trolley chassis (2.5 kg)
    chassis = patches.FancyBboxPatch((-0.23, 0.09), 0.46, 0.13, boxstyle="round,pad=0.015",
                                     facecolor="#3B82F6", edgecolor="#1E3A8A", lw=1.5, zorder=3)
    ax2.add_patch(chassis)
    ax2.text(0.0, 0.155, "Chassis (2.5 kg)", ha="center", va="center", color="white", fontsize=8.0, fontweight="bold", zorder=5)

    # Slide actuator force arrow
    ax2.annotate("", xy=(-0.25, 0.155), xytext=(-0.54, 0.155),
                 arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.25", color="#DC2626", lw=2.0))
    ax2.text(-0.56, 0.21, "Drive $u$\n(slide joint)", ha="left", va="bottom", color="#B91C1C", fontsize=7.8, fontweight="bold")

    # Vertical reference dashed line
    hinge = np.array([0.0, 0.235])
    ax2.plot([0.0, 0.0], [0.235, 1.02], color="#94A3B8", ls="--", lw=1.1, zorder=2)

    # Tilted mast (0.70 m, angle theta = 22 deg)
    th = np.deg2rad(22)
    L_mast = 0.68
    top = hinge + np.array([L_mast * np.sin(th), L_mast * np.cos(th)])
    ax2.plot([hinge[0], top[0]], [hinge[1], top[1]], color="#475569", lw=5.0, solid_capstyle="round", zorder=4)
    ax2.scatter([hinge[0]], [hinge[1]], s=75, color="#0F172A", zorder=6)

    # Angle arc for theta
    arc = patches.Arc((hinge[0], hinge[1]), 0.34, 0.34, angle=0, theta1=90 - 22, theta2=90, color="#DC2626", lw=1.5)
    ax2.add_patch(arc)
    ax2.text(0.065, 0.43, r"$\theta$", color="#DC2626", fontsize=10, fontweight="bold")

    # Top specimen payload box (rotated)
    box_s = 0.14
    t_rot = matplotlib.transforms.Affine2D().rotate_around(top[0], top[1], -th) + ax2.transData
    payload = patches.Rectangle((top[0] - box_s / 2, top[1] - box_s / 2), box_s, box_s,
                                facecolor="#F59E0B", edgecolor="#92400E", lw=1.6, zorder=5, transform=t_rot)
    ax2.add_patch(payload)
    ax2.text(top[0], top[1], r"$m_p$", ha="center", va="center", color="#1E293B", fontsize=8.5, fontweight="bold", zorder=6)

    # Randomized parameter callouts
    ax2.annotate(r"Payload mass $m_p \in [0.2, 1.5]\,$kg",
                 xy=(top[0] + 0.06, top[1] + 0.02), xytext=(0.18, 1.04),
                 arrowprops=dict(arrowstyle="->", color="#92400E", lw=1.1),
                 fontsize=7.6, color="#92400E", fontweight="bold", ha="center")
    ax2.annotate(r"Mast damping $d \in [10^{-3}, 1.0]$" + "\n(log-uniform)",
                 xy=(hinge[0] + 0.03, hinge[1] + 0.01), xytext=(0.24, 0.36),
                 arrowprops=dict(arrowstyle="->", color="#1E3A8A", lw=1.1),
                 fontsize=7.5, color="#1E3A8A", fontweight="bold")
    ax2.text(0.0, -0.055, r"Floor friction $f \in [0.2, 1.5]$ (uniform)",
             ha="center", va="center", fontsize=7.8, color="#334155", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#CBD5E1", alpha=0.9))

    ax2.set_xlim(-0.62, 0.66)
    ax2.set_ylim(-0.12, 1.14)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("(b) MedBot Payload Trolley MJCF (3 DR Params)", pad=7)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_sterile_arm_trolley.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_fig1_hospital_ward()
    make_fig2_slip_model()
    make_fig3_sterile_arm_trolley()
    print("Generated all 3 MedBot CW1 figures in", OUT_DIR)
