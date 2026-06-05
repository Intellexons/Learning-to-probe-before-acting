"""Render a looping GIF of the branched-rollout credit-assignment scheme.

Animated counterpart of paper Figure 1(c). With ``G_1 = 2`` first-turn
variants and ``G_2 = 3`` continuations per variant, the tree builds up
top-down, then per-prefix and root Monte Carlo baselines fade in, and
finally the per-turn advantage equation appears at the bottom.

Output: ``docs/assets/branched_rollout.gif``.

Run:
    uv run python scripts/render_branched_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Palette: matches the paper's TikZ Figure 1(c).
PROMPT_FC = "#e8e8e8"
T1_FC = "#cfd8e8"
T2_FC = "#f8d6b3"
TOOL_FC = "#f0f0f0"
EDGE = "#444444"
BASELINE = "#b32525"
TEXT = "#222222"

G1 = 2
G2 = 3
N_LEAVES = G1 * G2

S_XY = (0.5, 0.86)
Y1_XY = [(0.27, 0.71), (0.73, 0.71)]
M_XY = [(0.27, 0.57), (0.73, 0.57)]
LEAF_X_BY_PREFIX = [
    [0.15, 0.27, 0.39],
    [0.61, 0.73, 0.85],
]
LEAF_Y = 0.42
REWARD_Y = 0.34
V1_TOP, V1_BOT = 0.27, 0.255
V1_LABEL_Y = 0.225
V0_TOP, V0_BOT = 0.19, 0.175
V0_LABEL_Y = 0.145
EQ_Y = 0.06
TITLE_Y = 0.965

STAGES = {
    "prompt": (0, 14),
    "t1": (14, 32),
    "tool": (32, 48),
    "t2": (48, 78),
    "rewards": (78, 96),
    "v1": (96, 120),
    "v0": (120, 140),
    "adv": (140, 168),
    "hold": (168, 200),
}
TOTAL_FRAMES = 200
NODE_R = 0.022


def _alpha(stage: str, frame: int) -> float:
    start, end = STAGES[stage]
    if frame < start:
        return 0.0
    if frame >= end:
        return 1.0
    return (frame - start) / (end - start)


def _draw_circle(ax, xy, alpha, fc):
    if alpha <= 0:
        return
    ax.add_patch(
        mpatches.Circle(xy, NODE_R, facecolor=fc, edgecolor=EDGE,
                        linewidth=0.9, alpha=alpha, zorder=3)
    )


def _draw_edge(ax, xy_from, xy_to, alpha):
    if alpha <= 0:
        return
    ax.plot(
        [xy_from[0], xy_to[0]],
        [xy_from[1], xy_to[1]],
        color=EDGE, alpha=alpha * 0.65, linewidth=0.8, zorder=1,
    )


def _draw_bracket(ax, x_left, x_right, y_top, y_bot, alpha, label, label_y):
    if alpha <= 0:
        return
    ax.plot(
        [x_left, x_left, x_right, x_right],
        [y_top, y_bot, y_bot, y_top],
        color=BASELINE, alpha=alpha, linewidth=1.3, zorder=2,
    )
    ax.text(
        (x_left + x_right) / 2, label_y, label,
        ha="center", va="top",
        fontsize=10, color=BASELINE, alpha=alpha,
    )


def make_frame(ax, frame: int) -> None:
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        0.5, TITLE_Y,
        "branched rollout & per-turn credit assignment",
        ha="center", va="top",
        fontsize=10, color=TEXT, weight="bold",
    )

    # Stage 1: prompt s.
    a_p = _alpha("prompt", frame)
    _draw_circle(ax, S_XY, a_p, PROMPT_FC)
    if a_p > 0:
        ax.text(*S_XY, "$s$", ha="center", va="center",
                fontsize=10, alpha=a_p, color=TEXT, zorder=4)

    # Stage 2: turn-1 fan-out.
    a_t1 = _alpha("t1", frame)
    for i, xy in enumerate(Y1_XY):
        _draw_edge(ax, S_XY, xy, a_t1)
        _draw_circle(ax, xy, a_t1, T1_FC)
        if a_t1 > 0:
            ax.text(xy[0], xy[1], f"$y^{{(1)}}_{i + 1}$",
                    ha="center", va="center", fontsize=7,
                    alpha=a_t1, color=TEXT, zorder=4)

    # Stage 3: tool messages.
    a_m = _alpha("tool", frame)
    for i, xy in enumerate(M_XY):
        _draw_edge(ax, Y1_XY[i], xy, a_m)
        if a_m > 0:
            ax.add_patch(mpatches.Rectangle(
                (xy[0] - 0.032, xy[1] - 0.020),
                0.064, 0.040,
                facecolor=TOOL_FC, edgecolor=EDGE,
                linewidth=0.8, alpha=a_m, zorder=3,
            ))
            ax.text(*xy, f"$m_{i + 1}$",
                    ha="center", va="center", fontsize=8,
                    alpha=a_m, color=TEXT, zorder=4)

    # Stage 4: turn-2 fan-out.
    a_t2 = _alpha("t2", frame)
    for i in range(G1):
        for j in range(G2):
            leaf = (LEAF_X_BY_PREFIX[i][j], LEAF_Y)
            _draw_edge(ax, M_XY[i], leaf, a_t2)
            _draw_circle(ax, leaf, a_t2, T2_FC)

    # Stage 5: leaf rewards.
    a_r = _alpha("rewards", frame)
    if a_r > 0:
        for i in range(G1):
            for j in range(G2):
                x = LEAF_X_BY_PREFIX[i][j]
                ax.text(x, REWARD_Y, f"$R_{{{i + 1},{j + 1}}}$",
                        ha="center", va="center", fontsize=8,
                        color=TEXT, alpha=a_r)

    # Stage 6: per-prefix MC baselines V_1^(i).
    a_v1 = _alpha("v1", frame)
    if a_v1 > 0:
        for i in range(G1):
            xs = LEAF_X_BY_PREFIX[i]
            x_left = min(xs) - 0.045
            x_right = max(xs) + 0.045
            _draw_bracket(
                ax, x_left, x_right, V1_TOP, V1_BOT,
                a_v1, f"$\\hat V_1^{{({i + 1})}}$", V1_LABEL_Y,
            )

    # Stage 7: root MC baseline V_0.
    a_v0 = _alpha("v0", frame)
    if a_v0 > 0:
        all_xs = [x for prefix in LEAF_X_BY_PREFIX for x in prefix]
        x_left = min(all_xs) - 0.055
        x_right = max(all_xs) + 0.055
        _draw_bracket(
            ax, x_left, x_right, V0_TOP, V0_BOT,
            a_v0 * 0.85, "$\\hat V_0$", V0_LABEL_Y,
        )

    # Stage 8: advantage equation.
    a_eq = _alpha("adv", frame)
    if a_eq > 0:
        ax.text(
            0.5, EQ_Y,
            r"$A^{(1)}_i = \hat V_1^{(i)} - \hat V_0$"
            r"$\quad\quad$"
            r"$A^{(2)}_{i,j} = R_{i,j} - \hat V_1^{(i)}$",
            ha="center", va="center", fontsize=10,
            color=TEXT, alpha=a_eq,
        )


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "docs" / "assets" / "branched_rollout.gif"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=110)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def update(frame):
        make_frame(ax, frame)
        return []

    anim = FuncAnimation(
        fig, update, frames=TOTAL_FRAMES,
        interval=42, blit=False,
    )
    writer = PillowWriter(fps=24)
    anim.save(out, writer=writer)
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
