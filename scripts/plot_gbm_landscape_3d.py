"""
GBM 3D Waddington Epigenetic Landscape
=======================================
Two-basin surface: GBM cancer attractor (deep well, MES + NPC sub-basins)
vs. Normal Brain attractor (astrocyte + neuron sub-basins).
TCIP reversion trajectory shown as a 3D ribbon path.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

OUT_PATH = "outputs/gbm_landscape_3d.png"
os.makedirs("outputs", exist_ok=True)

# ── Potential function ────────────────────────────────────────────────────────
def U(x, y):
    """
    Waddington potential with GBM (left) and normal brain (right) attractors.
    Sub-basins: MES + NPC for cancer; astrocyte + neuron for normal.
    """
    # ── Cancer well (left): deep, two sub-basins ──────────────────────────
    cancer_main  = -2.2  * np.exp(-((x + 1.6)**2 + (y - 0.1)**2) / 0.75)
    mes_sub      = -0.55 * np.exp(-((x + 2.0)**2 + (y + 0.65)**2) / 0.22)
    npc_sub      = -0.50 * np.exp(-((x + 1.25)**2 + (y - 0.75)**2) / 0.20)

    # ── Normal brain well (right): slightly shallower, two sub-basins ─────
    normal_main  = -1.75 * np.exp(-((x - 1.6)**2 + (y - 0.1)**2) / 0.65)
    astro_sub    = -0.40 * np.exp(-((x - 2.0)**2 + (y + 0.55)**2) / 0.20)
    neuron_sub   = -0.38 * np.exp(-((x - 1.25)**2 + (y - 0.70)**2) / 0.18)

    # ── Barrier ridge ─────────────────────────────────────────────────────
    barrier      =  0.55 * np.exp(-x**2 / 0.18) * np.exp(-y**2 / 2.5)

    # ── Bowl background ───────────────────────────────────────────────────
    background   =  0.09 * (x**2 + y**2) + 0.05 * y**2

    return (cancer_main + mes_sub + npc_sub
            + normal_main + astro_sub + neuron_sub
            + barrier + background)


# ── Grid ──────────────────────────────────────────────────────────────────────
res = 320
x = np.linspace(-3.2, 3.2, res)
y = np.linspace(-2.2, 2.2, res)
X, Y = np.meshgrid(x, y)
Z = U(X, Y)

# Clip top for cleaner view
Z_clipped = np.clip(Z, -2.5, 0.8)

# ── Colormap: deep blue-purple (high energy) → teal → dark gold (low energy) ─
colors_land = [
    (0.00, "#0D0D1A"),
    (0.15, "#0A1628"),
    (0.30, "#0E2F4A"),
    (0.48, "#1A5C6E"),
    (0.60, "#2E8B78"),
    (0.72, "#3DAA5C"),
    (0.83, "#7DC962"),
    (0.92, "#C8E07A"),
    (1.00, "#FFF0A0"),
]
cmap = LinearSegmentedColormap.from_list("waddington3d",
                                          [(v, c) for v, c in colors_land])

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 13), facecolor="#05060D")

ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
ax.set_facecolor("#05060D")
ax.patch.set_alpha(0)

# Grid pane colors
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor("#111122")
ax.yaxis.pane.set_edgecolor("#111122")
ax.zaxis.pane.set_edgecolor("#111122")
ax.grid(False)

# ── Surface ───────────────────────────────────────────────────────────────────
norm_z = (Z_clipped - Z_clipped.min()) / (Z_clipped.max() - Z_clipped.min())
surf = ax.plot_surface(
    X, Y, Z_clipped,
    facecolors=cmap(norm_z),
    rstride=2, cstride=2,
    linewidth=0,
    antialiased=True,
    alpha=0.93,
    shade=True,
)

# ── Contour lines projected on the surface (topographic) ─────────────────────
contour_levels = np.linspace(-2.3, 0.6, 18)
ax.contour(X, Y, Z_clipped, levels=contour_levels, zdir="z",
           offset=None, colors="#FFFFFF", alpha=0.07, linewidths=0.5)

# ── Basin markers: bottom of each well ────────────────────────────────────────
def basin_marker(ax, x0, y0, color, size=80, zorder=10):
    z0 = float(np.clip(U(np.array([[x0]]), np.array([[y0]])).ravel()[0], -2.5, 0.8))
    ax.scatter([x0], [y0], [z0 + 0.04], c=[color], s=size,
               edgecolors="white", linewidths=1.2, zorder=zorder, depthshade=False)
    return z0

z_cancer  = basin_marker(ax, -1.6,  0.0,  "#FF4040", size=180)
z_mes     = basin_marker(ax, -2.0, -0.65, "#FF8C42", size=100)
z_npc     = basin_marker(ax, -1.25, 0.75, "#FFC857", size=100)
z_normal  = basin_marker(ax,  1.6,  0.0,  "#00E676", size=180)
z_astro   = basin_marker(ax,  2.0, -0.55, "#69F0AE", size=100)
z_neuron  = basin_marker(ax,  1.25, 0.70, "#B9F6CA", size=100)

# ── 3D Text annotations ───────────────────────────────────────────────────────
def label3d(ax, x, y, z, text, color, fs=9, zorder=12, bold=False):
    ax.text(x, y, z, text, color=color, fontsize=fs, zorder=zorder,
            fontweight="bold" if bold else "normal",
            fontfamily="monospace",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="#05060D")])

# Cancer basin labels
label3d(ax, -1.6,  0.12, z_cancer + 0.28, "GBM\nCANCER\nATTRACTOR",
        "#FF4040", fs=10, bold=True)
label3d(ax, -2.1, -0.65, z_mes + 0.22, "MES\nstate\n(TWIST1·VIM)",
        "#FF8C42", fs=7.5)
label3d(ax, -1.25, 0.80, z_npc + 0.22, "NPC\nstate\n(SOX2·MYC)",
        "#FFC857", fs=7.5)

# Normal basin labels
label3d(ax,  1.6,  0.12, z_normal + 0.28, "NORMAL\nBRAIN\nATTRACTOR",
        "#00E676", fs=10, bold=True)
label3d(ax,  2.0, -0.55, z_astro + 0.22, "Astrocyte\n(GFAP·S100B)",
        "#69F0AE", fs=7.5)
label3d(ax,  1.25, 0.75, z_neuron + 0.22, "Neuron\n(NEUROD1·RBFOX1)",
        "#B9F6CA", fs=7.5)

# Barrier annotation
ax.text(0.0, -0.1, 0.72,
        "← epigenetic barrier →\n   (EZH2·BRD4 lock)",
        color="#CCCCFF", fontsize=8, ha="center",
        path_effects=[pe.withStroke(linewidth=2, foreground="#05060D")])

# ── TCIP reversion trajectory ─────────────────────────────────────────────────
t = np.linspace(0, 1, 220)

# Path in x-y plane: sigmoid from cancer (-1.6) to normal (1.6), slight arc in y
traj_x = -1.6 + 3.2 * (1 / (1 + np.exp(-8 * (t - 0.5))))
traj_y = 0.0 + 0.35 * np.sin(np.pi * t)

# Z follows the surface + a small lift so it's visible above the mesh
traj_z = np.array([float(U(np.array([[tx]]), np.array([[ty]])).ravel()[0])
                   for tx, ty in zip(traj_x, traj_y)])
traj_z = np.clip(traj_z, -2.5, 0.8) + 0.12  # lift off surface

ax.plot(traj_x, traj_y, traj_z,
        color="#A78BFA", linewidth=3.0, zorder=20, alpha=0.95)

# Arrow at end of trajectory
ax.quiver(traj_x[-8], traj_y[-8], traj_z[-8],
          traj_x[-1] - traj_x[-8],
          traj_y[-1] - traj_y[-8],
          traj_z[-1] - traj_z[-8],
          color="#A78BFA", linewidth=2.5, arrow_length_ratio=0.5, zorder=21)

# ── TCIP intervention diamonds on trajectory ──────────────────────────────────
interventions = [
    (0.12, "REPRESS\nSOX2·MYC",    "#FF6B35"),
    (0.30, "REPRESS\nTWIST1·EZH2", "#FF6B35"),
    (0.52, "REPRESS\nBRD4",        "#FF6B35"),
    (0.75, "ACTIVATE\nNEUROD1·\nRBFOX1·GFAP", "#FFD700"),
]

for pos, label, col in interventions:
    idx = int(pos * len(t))
    ix, iy, iz = traj_x[idx], traj_y[idx], traj_z[idx]
    ax.scatter([ix], [iy], [iz + 0.05], c=[col], s=140,
               marker="D", edgecolors="white", linewidths=1.0,
               zorder=22, depthshade=False)
    label3d(ax, ix + 0.12, iy + 0.18, iz + 0.35, label, col, fs=7.5)

# ── Projected contour shadow on bottom ────────────────────────────────────────
ax.contourf(X, Y, Z_clipped,
            levels=contour_levels,
            zdir="z", offset=-2.55,
            cmap=cmap, alpha=0.18)

# ── Colorbar ──────────────────────────────────────────────────────────────────
m = cm.ScalarMappable(cmap=cmap)
m.set_array(Z_clipped)
cbar = fig.colorbar(m, ax=ax, shrink=0.45, aspect=18, pad=0.02,
                    orientation="vertical")
cbar.ax.set_ylabel("Epigenetic Potential  U(x,y)", color="#AAAACC",
                    fontsize=9, rotation=270, labelpad=18)
cbar.ax.yaxis.set_tick_params(color="#AAAACC", labelcolor="#AAAACC", labelsize=7)
cbar.set_ticks([-2.4, -1.8, -1.2, -0.6, 0.0, 0.6])
cbar.set_ticklabels(["deep stable\nbasin", "", "", "", "", "energy\nbarrier"])
cbar.outline.set_edgecolor("#333355")

# ── Axis formatting ───────────────────────────────────────────────────────────
ax.set_xlabel("GRN Identity Axis  (cancer ← → normal)", color="#8888AA",
              fontsize=9, labelpad=10)
ax.set_ylabel("Cell-state Variance", color="#8888AA", fontsize=9, labelpad=10)
ax.set_zlabel("Epigenetic Potential", color="#8888AA", fontsize=9, labelpad=10)
ax.tick_params(colors="#444466", labelsize=7, pad=2)

ax.set_xlim(-3.2, 3.2)
ax.set_ylim(-2.2, 2.2)
ax.set_zlim(-2.55, 1.2)

# Remove tick labels for cleaner look
ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

# ── Viewing angle ─────────────────────────────────────────────────────────────
ax.view_init(elev=28, azim=-55)

# ── Title & subtitle ─────────────────────────────────────────────────────────
fig.text(0.44, 0.97,
         "ORACLE  ·  GBM 3D Epigenetic Landscape",
         ha="center", va="top", color="#E8E8FF",
         fontsize=17, fontweight="bold", fontfamily="monospace")
fig.text(0.44, 0.93,
         "Waddington two-basin topology  ·  MES + NPC sub-states  ·  "
         "TCIP reversion trajectory  (purple)",
         ha="center", va="top", color="#8888AA", fontsize=9.5, fontstyle="italic")

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    (dict(marker="o", color="#FF4040", markersize=9, linestyle="None"), "Cancer attractor"),
    (dict(marker="o", color="#FF8C42", markersize=7, linestyle="None"), "MES sub-basin (TWIST1/VIM)"),
    (dict(marker="o", color="#FFC857", markersize=7, linestyle="None"), "NPC sub-basin (SOX2/MYC)"),
    (dict(marker="o", color="#00E676", markersize=9, linestyle="None"), "Normal brain attractor"),
    (dict(marker="o", color="#69F0AE", markersize=7, linestyle="None"), "Astrocyte sub-basin (GFAP)"),
    (dict(marker="o", color="#B9F6CA", markersize=7, linestyle="None"), "Neuron sub-basin (NEUROD1/RBFOX1)"),
    (dict(color="#A78BFA", linewidth=2.5),                              "TCIP reversion trajectory"),
    (dict(marker="D", color="#FF6B35", markersize=7, linestyle="None"), "REPRESS intervention (HDAC2/PRMT5/EZH2)"),
    (dict(marker="D", color="#FFD700", markersize=7, linestyle="None"), "ACTIVATE intervention (p300/BRD4)"),
]
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], **kw) for kw, _ in legend_items]
labels  = [lbl for _, lbl in legend_items]
leg = ax.legend(handles, labels, loc="upper left",
                fontsize=7.8, facecolor="#0D0F1A",
                edgecolor="#333355", labelcolor="#CCCCDD",
                framealpha=0.92, ncol=1,
                bbox_to_anchor=(0.0, 0.88))

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight",
            facecolor="#05060D", edgecolor="none")
print(f"Saved → {OUT_PATH}")
plt.close()
