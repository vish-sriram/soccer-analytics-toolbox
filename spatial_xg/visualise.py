"""
Plot freeze frame snapshots for the top 20% of shots by spatial P(goal).
Each subplot shows a half-pitch with all visible players at the moment of the shot.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

# ── Pitch drawing (attacking half only, x: 60–120) ────────────────────────────

def draw_half_pitch(ax):
    ax.set_facecolor("#3a7d44")

    # Pitch outline (right half)
    ax.add_patch(patches.Rectangle((60, 0), 60, 80, linewidth=1.5,
                                    edgecolor="white", facecolor="none"))
    # Centre line
    ax.add_line(Line2D([60, 60], [0, 80], color="white", linewidth=1.5))
    # Centre circle (right half only)
    centre = patches.Arc((60, 40), 20, 20, angle=0,
                          theta1=-90, theta2=90, color="white", linewidth=1.5)
    ax.add_patch(centre)
    # Penalty area
    ax.add_patch(patches.Rectangle((102, 18), 18, 44, linewidth=1.5,
                                    edgecolor="white", facecolor="none"))
    # 6-yard box
    ax.add_patch(patches.Rectangle((114, 30), 6, 20, linewidth=1.5,
                                    edgecolor="white", facecolor="none"))
    # Goal
    ax.add_patch(patches.Rectangle((120, 36), 2, 8, linewidth=2,
                                    edgecolor="white", facecolor="#555"))
    # Penalty spot
    ax.plot(108, 40, "o", color="white", markersize=3)
    # Penalty arc
    arc = patches.Arc((108, 40), 20, 20, angle=0,
                       theta1=37, theta2=143, color="white", linewidth=1.5)
    ax.add_patch(arc)

    ax.set_xlim(58, 122)
    ax.set_ylim(-2, 82)
    ax.set_aspect("equal")
    ax.axis("off")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open("data/graphs.pkl", "rb") as f:
        payload = pickle.load(f)
    graphs   = payload["graphs"]
    metadata = payload["metadata"]

    results = pd.read_parquet("data/outputs/spatial_xg_values.parquet")

    threshold = results["p_goal"].quantile(0.80)
    top = results[results["p_goal"] >= threshold].sort_values("p_goal", ascending=False)
    print(f"Top 20% threshold: P(goal) >= {threshold:.3f}")
    print(f"Shots to plot: {len(top)}")

    legend_elements = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#FFD700",
               markersize=12, label="Shooter"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4fc3f7",
               markersize=9, label="Teammate"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ef5350",
               markersize=9, label="Opponent"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#ef5350",
               markersize=9, label="Goalkeeper"),
    ]

    N_COLS, N_ROWS = 4, 5          # 20 shots per page
    PER_PAGE = N_COLS * N_ROWS
    top_list = list(top.iterrows())
    n_pages  = int(np.ceil(len(top_list) / PER_PAGE))

    for page in range(n_pages):
        chunk = top_list[page * PER_PAGE : (page + 1) * PER_PAGE]

        fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(16, 20))
        fig.patch.set_facecolor("#1a1a2e")
        axes = axes.flatten()

        for plot_i, (_, row) in enumerate(chunk):
            ax = axes[plot_i]
            draw_half_pitch(ax)

            idx    = row.name
            g      = graphs[idx]
            meta   = metadata[idx]
            x_feat = g.x.numpy()

            px = x_feat[:, 0] * PITCH_LENGTH
            py = x_feat[:, 1] * PITCH_WIDTH
            is_gk       = x_feat[:, 4].astype(bool)
            is_teammate = x_feat[:, 5].astype(bool)
            is_shooter  = x_feat[:, 6].astype(bool)

            tm_mask  = is_teammate & ~is_shooter & ~is_gk
            opp_mask = ~is_teammate & ~is_gk

            ax.scatter(px[tm_mask],  py[tm_mask],  s=100, color="#4fc3f7",
                       edgecolors="white", linewidths=0.7, zorder=4)
            ax.scatter(px[is_shooter], py[is_shooter], s=200, color="#FFD700",
                       edgecolors="white", linewidths=0.9, marker="*", zorder=5)
            ax.scatter(px[opp_mask], py[opp_mask], s=100, color="#ef5350",
                       edgecolors="white", linewidths=0.7, zorder=4)
            ax.scatter(px[is_gk],    py[is_gk],    s=130, color="#ef5350",
                       edgecolors="white", linewidths=0.7, marker="s", zorder=4)

            if is_shooter.any():
                sx, sy = px[is_shooter][0], py[is_shooter][0]
                ax.annotate("", xy=(120, 40), xytext=(sx, sy),
                            arrowprops=dict(arrowstyle="-|>", color="#FFD700",
                                            lw=1.2, mutation_scale=10), zorder=3)

            is_goal     = bool(row["is_goal"])
            outcome_str = "GOAL" if is_goal else row.get("shot_outcome", "")
            title = (f"{meta.get('player', '?')}\n"
                     f"{meta.get('team', '')} | {meta.get('minute', '?')}'\n"
                     f"P(goal)={row['p_goal']:.2f}  [{outcome_str}]")
            ax.set_title(title, fontsize=7, pad=3,
                         color="#FFD700" if is_goal else "white",
                         fontweight="bold" if is_goal else "normal")

        for j in range(plot_i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.legend(handles=legend_elements, loc="lower center", ncol=4,
                   framealpha=0.15, fontsize=9, labelcolor="white",
                   facecolor="#222")
        fig.suptitle(
            f"Top 20% Shots by Spatial P(Goal) — Leverkusen 2023/24  "
            f"(page {page+1}/{n_pages})",
            fontsize=13, color="white", fontweight="bold"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        out = f"data/outputs/top_shots_p{page+1:02d}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
