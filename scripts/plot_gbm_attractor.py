"""
GBM Cancer Attractor Map — ORACLE Visualization
================================================
Renders three panels:
  1. GBM Cancer Attractor (stem/MES+NPC state)
  2. Normal Brain Attractor (astrocyte/neuron state)
  3. Reversion trajectory with TCIP intervention arrows
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

OUT_PATH = "outputs/gbm_attractor_map.png"
os.makedirs("outputs", exist_ok=True)

# ── Color palette ────────────────────────────────────────────────────────────
C_ON    = "#E84855"   # crimson   – gene ON  in cancer attractor
C_OFF   = "#3D8EB9"   # steel-blue– gene OFF in cancer attractor
C_NRM_ON  = "#2ECC71" # emerald   – gene ON  in normal attractor
C_NRM_OFF = "#95A5A6" # grey      – gene OFF in normal attractor
C_BG    = "#0F1117"   # near-black background
C_PANEL = "#1A1D25"   # panel bg
C_TEXT  = "#E8E8E8"
C_ACCEL = "#FFD700"   # gold  – ACTIVATE arrow
C_REPR  = "#FF6B35"   # orange-red– REPRESS arrow
C_TRAJ  = "#AAAAFF"   # lavender – reversion trajectory
C_NORM_BASIN = "#1E3A2F"   # dark green – normal basin
C_CANCER_BASIN = "#3A1E1E" # dark red   – cancer basin

# ── Gene sets ─────────────────────────────────────────────────────────────────
CANCER_ON = [
    "SOX2", "NES", "MYC", "TWIST1", "EZH2", "BRD4",
    "CDK4", "EGFR", "STAT3", "VIM", "CDH2", "ALDH1A3",
    "CHI3L1", "NOTCH1",
]
CANCER_OFF = [
    "NEUROD1", "RBFOX1", "GFAP", "TUBB3", "MAP2",
    "CDKN2A", "PTEN",
]
NORMAL_ON = [
    "GFAP", "NEUROD1", "RBFOX1", "TUBB3", "MAP2",
    "S100B", "ALDH1L1",
]
NORMAL_OFF = [
    "SOX2", "NES", "MYC", "TWIST1", "EZH2", "BRD4",
]

ACTIVATE_GENES = ["NEUROD1", "RBFOX1", "GFAP"]
REPRESS_GENES  = ["SOX2", "MYC", "TWIST1", "EZH2", "BRD4"]

# ── TCIP labels for arrows ────────────────────────────────────────────────────
TCIP_LABELS = {
    "SOX2":    "TCIP-S (PRMT5↓\nH4R3me2s)",
    "MYC":     "TCIP-M (PRMT5↓\nH4R3me2s)",
    "TWIST1":  "TCIP-T (HDAC2↓\n−H3K27ac)",
    "EZH2":    "TCIP-E2 (PRMT5↓\nH4R3me2s)",
    "BRD4":    "TCIP-B (EZH2↓\n+H3K27me3)",
    "NEUROD1": "TCIP-N (p300↑\n+H3K27ac)",
    "RBFOX1":  "TCIP-R (p300↑\n+H3K27ac)",
    "GFAP":    "TCIP-G (BRD4↑\n+H3K27ac)",
}

# ── Figure setup ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 15), facecolor=C_BG)
fig.text(0.5, 0.97, "ORACLE  ·  GBM Cancer Attractor Map",
         ha="center", va="top", color=C_TEXT, fontsize=18, fontweight="bold",
         fontfamily="monospace")
fig.text(0.5, 0.94,
         "Neftel et al. 2019 Cell (GSE131928)  ·  Zhang et al. 2016 (GSE67835 normal brain)",
         ha="center", va="top", color="#888888", fontsize=10, fontstyle="italic")

# 3 columns: cancer attractor | landscape | normal attractor
gs = fig.add_gridspec(1, 3, left=0.03, right=0.97, bottom=0.05, top=0.90,
                      wspace=0.05, hspace=0.1)

# ────────────────────────────────────────────────────────────────────────────
# PANEL 1 – GBM Cancer Attractor gene grid
# ────────────────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(C_PANEL)
ax1.set_title("GBM Cancer Attractor\n(MES + NPC stem state)", color=C_TEXT,
              fontsize=13, fontweight="bold", pad=10)
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis("off")

all_cancer_genes = CANCER_ON + CANCER_OFF
n_cols = 2
n_rows = -(-len(all_cancer_genes) // n_cols)  # ceiling div
x_pos = [0.15, 0.60]

for i, gene in enumerate(all_cancer_genes):
    row = i // n_cols
    col = i % n_cols
    is_on = gene in CANCER_ON
    color = C_ON if is_on else C_OFF
    label = f"{'▲' if is_on else '▼'}  {gene}"
    is_switch = gene in (ACTIVATE_GENES + REPRESS_GENES)

    y = 0.94 - row * (0.90 / max(n_rows, 1))
    x = x_pos[col]

    if is_switch:
        # Highlight switch genes
        rect = FancyBboxPatch((x - 0.11, y - 0.022), 0.34, 0.044,
                               boxstyle="round,pad=0.005",
                               linewidth=1.5, edgecolor=C_ACCEL if gene in ACTIVATE_GENES else C_REPR,
                               facecolor="#2A2A3A", zorder=2)
        ax1.add_patch(rect)
        tcip = "⬆ ACTIVATE" if gene in ACTIVATE_GENES else "⬇ REPRESS"
        ax1.text(x + 0.22, y, tcip, ha="right", va="center", color=C_ACCEL if gene in ACTIVATE_GENES else C_REPR,
                 fontsize=6.5, fontweight="bold", zorder=3)

    ax1.text(x, y, label, ha="left", va="center", color=color,
             fontsize=9.5, fontweight="bold" if is_switch else "normal",
             fontfamily="monospace", zorder=3)

# Legend
leg_y = 0.035
ax1.add_patch(mpatches.FancyArrow(0.05, leg_y, 0, 0, width=0.015, color=C_ON))
ax1.text(0.10, leg_y, "HIGH / ON", color=C_ON, va="center", fontsize=8)
ax1.add_patch(mpatches.FancyArrow(0.38, leg_y, 0, 0, width=0.015, color=C_OFF))
ax1.text(0.43, leg_y, "LOW / OFF", color=C_OFF, va="center", fontsize=8)

box_leg = FancyBboxPatch((0.05, 0.001), 0.66, 0.06, boxstyle="round,pad=0.005",
                          linewidth=0.8, edgecolor="#555555", facecolor="#1E1E28")
ax1.add_patch(box_leg)

# ────────────────────────────────────────────────────────────────────────────
# PANEL 2 – Landscape / Waddington bowl + reversion trajectory
# ────────────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor(C_BG)
ax2.set_title("Epigenetic Landscape & TCIP Reversion\n(Waddington attractor topology)",
              color=C_TEXT, fontsize=13, fontweight="bold", pad=10)

# Draw Waddington-style landscape using filled contours
x = np.linspace(-3, 3, 300)
y_pos = np.linspace(-1.5, 3, 300)
X, Y = np.meshgrid(x, y_pos)

# Two-basin potential: cancer (left) and normal (right)
# U(x,y) = (x²-1)² + 0.3y² − 0.4x   (asymmetric: cancer basin slightly deeper initially)
U = 0.9 * (X**2 - 1.2)**2 + 0.35 * (Y + 0.2)**2 - 0.3 * X

# Custom colormap: dark gradient
cmap_land = LinearSegmentedColormap.from_list(
    "waddington",
    [(0, "#0F1117"), (0.3, "#1a2a3a"), (0.6, "#1e3a4a"), (1.0, "#2a5a2a")],
)
cf = ax2.contourf(X, Y, U, levels=40, cmap=cmap_land, alpha=0.9)

# Basin labels
ax2.text(-1.6, -0.7, "CANCER\nATTRACTOR", ha="center", color="#FF6B6B",
         fontsize=9, fontweight="bold", alpha=0.9)
ax2.text(1.5, -0.7, "NORMAL\nATTRACTOR", ha="center", color="#69FF69",
         fontsize=9, fontweight="bold", alpha=0.9)

# Draw basin wells
ax2.plot(-1.35, -1.1, "o", color="#E84855", markersize=15, zorder=5, alpha=0.85)
ax2.plot(-1.35, -1.1, "o", color="#FF9999", markersize=7, zorder=6, alpha=0.9)
ax2.plot(1.35, -1.05, "o", color="#2ECC71", markersize=15, zorder=5, alpha=0.85)
ax2.plot(1.35, -1.05, "o", color="#99FFCC", markersize=7, zorder=6, alpha=0.9)

# Barrier ridge
barrier_x = np.array([-0.3, -0.1, 0.0, 0.1, 0.3])
barrier_y = np.array([0.8, 0.85, 0.9, 0.85, 0.8])
ax2.plot(barrier_x, barrier_y, "--", color="#CCCCCC", linewidth=1.5, alpha=0.5, zorder=4)
ax2.text(0.0, 1.05, "epigenetic barrier", ha="center", color="#AAAAAA",
         fontsize=7.5, fontstyle="italic")

# Reversion trajectory (TCIP-assisted)
t = np.linspace(0, 1, 100)
traj_x = -1.35 + 2.7 * t + 0.25 * np.sin(3.5 * np.pi * t)
traj_y = -1.1 + (0.05 + 0.9 * (t - 0.5)**2) - 0.9 * t**2 * (1 - t) * 3
ax2.plot(traj_x, traj_y, "-", color=C_TRAJ, linewidth=2.5, zorder=7, alpha=0.85)
ax2.annotate("", xy=(1.35, -1.05), xytext=(traj_x[-5], traj_y[-5]),
             arrowprops=dict(arrowstyle="->", color=C_TRAJ, lw=2.0), zorder=8)

# TCIP perturbation markers along trajectory
tcip_positions = [0.18, 0.35, 0.55, 0.72]
tcip_names = ["REPRESS\nSOX2·MYC", "REPRESS\nTWIST1·EZH2", "REPRESS\nBRD4", "ACTIVATE\nNEUROD1·RBFOX1·GFAP"]
tcip_colors = [C_REPR, C_REPR, C_REPR, C_ACCEL]
for ti, (pos, label, col) in enumerate(zip(tcip_positions, tcip_names, tcip_colors)):
    tx = -1.35 + 2.7 * pos + 0.25 * np.sin(3.5 * np.pi * pos)
    ty = -1.1 + (0.05 + 0.9 * (pos - 0.5)**2) - 0.9 * pos**2 * (1 - pos) * 3
    ax2.plot(tx, ty, "D", color=col, markersize=11, zorder=9, alpha=0.9)
    ax2.plot(tx, ty, "D", color="#FFFFFF", markersize=5, zorder=10, alpha=0.7)
    offset_y = 0.35 if ti % 2 == 0 else -0.38
    ax2.annotate(label,
                 xy=(tx, ty), xytext=(tx - 0.1, ty + offset_y),
                 fontsize=6.8, color=col, ha="center", fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=col, lw=0.8, alpha=0.7),
                 zorder=11)

# Axis labels
ax2.set_xlabel("Gene Regulatory Network State  (cancer → normal)", color=C_TEXT, fontsize=9)
ax2.set_ylabel("Epigenetic Potential  (U)", color=C_TEXT, fontsize=9)
ax2.tick_params(colors=C_TEXT, labelsize=7)
for spine in ax2.spines.values():
    spine.set_color("#333333")

ax2.set_xlim(-3, 3); ax2.set_ylim(-1.5, 3)

# Legend
legend_elements = [
    mpatches.Patch(color=C_TRAJ, label="TCIP reversion trajectory"),
    mpatches.Patch(color=C_REPR, label="REPRESS (eraser recruited)"),
    mpatches.Patch(color=C_ACCEL, label="ACTIVATE (writer recruited)"),
    mpatches.Patch(color="#E84855", label="Cancer attractor basin"),
    mpatches.Patch(color="#2ECC71", label="Normal brain attractor"),
]
ax2.legend(handles=legend_elements, loc="upper right", fontsize=7.5,
           facecolor="#1A1D25", edgecolor="#444444",
           labelcolor=C_TEXT, framealpha=0.9)

# ────────────────────────────────────────────────────────────────────────────
# PANEL 3 – Normal Brain Attractor + TCIP table
# ────────────────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.set_facecolor(C_PANEL)
ax3.set_title("Normal Brain Attractor +\nTCIP Switch Set & Assembly", color=C_TEXT,
              fontsize=13, fontweight="bold", pad=10)
ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.axis("off")

# Normal attractor genes (top half)
all_normal = NORMAL_ON + NORMAL_OFF
for i, gene in enumerate(all_normal):
    row = i // 2; col = i % 2
    is_on = gene in NORMAL_ON
    color = C_NRM_ON if is_on else C_NRM_OFF
    label = f"{'▲' if is_on else '▼'}  {gene}"
    y = 0.93 - row * 0.088
    x = 0.08 if col == 0 else 0.54
    ax3.text(x, y, label, ha="left", va="center", color=color,
             fontsize=9.5, fontweight="bold", fontfamily="monospace")

# Divider
ax3.axhline(0.44, color="#444455", linewidth=1.2, xmin=0.02, xmax=0.98)
ax3.text(0.5, 0.415, "TCIP SWITCH SET  (8 molecules, all Tier-1 amide, 8/8 pass)",
         ha="center", color=C_TEXT, fontsize=8, fontweight="bold")

# TCIP table
headers = ["Gene", "Action", "Recruiter", "MW", "QED"]
col_x = [0.04, 0.18, 0.38, 0.68, 0.84]
ax3.text(col_x[0], 0.385, "Gene", color="#AAAAAA", fontsize=7.5, fontweight="bold")
ax3.text(col_x[1], 0.385, "Action", color="#AAAAAA", fontsize=7.5, fontweight="bold")
ax3.text(col_x[2], 0.385, "Recruiter", color="#AAAAAA", fontsize=7.5, fontweight="bold")
ax3.text(col_x[3], 0.385, "MW", color="#AAAAAA", fontsize=7.5, fontweight="bold")
ax3.text(col_x[4], 0.385, "QED", color="#AAAAAA", fontsize=7.5, fontweight="bold")
ax3.axhline(0.370, color="#444455", linewidth=0.8, xmin=0.02, xmax=0.98)

table_data = [
    # gene, action, recruiter, MW, QED, color
    ("NEUROD1", "ACTIVATE", "p300 A-485",  "664", "0.116", C_ACCEL),
    ("RBFOX1",  "ACTIVATE", "p300 A-485",  "662", "0.139", C_ACCEL),
    ("GFAP",    "ACTIVATE", "BRD4 JQ1",   "684", "0.168", C_ACCEL),
    ("SOX2",    "REPRESS",  "PRMT5 GSK",  "686", "0.138", C_REPR),
    ("MYC",     "REPRESS",  "PRMT5 GSK",  "713", "0.125", C_REPR),
    ("TWIST1",  "REPRESS",  "HDAC2 Ent.", "642", "0.048", C_REPR),
    ("EZH2",    "REPRESS",  "PRMT5 GSK",  "669", "0.193", C_REPR),
    ("BRD4",    "REPRESS",  "EZH2 EPZ",  "681", "0.193", C_REPR),
]
for row_i, (gene, action, rec, mw, qed, col) in enumerate(table_data):
    y = 0.350 - row_i * 0.040
    # Row highlight
    if row_i % 2 == 0:
        ax3.add_patch(FancyBboxPatch((0.02, y - 0.016), 0.96, 0.032,
                                      boxstyle="square,pad=0", linewidth=0,
                                      facecolor="#1E2030", zorder=1))
    ax3.text(col_x[0], y, gene,   color=col, fontsize=8, fontweight="bold",
             fontfamily="monospace", va="center", zorder=2)
    ax3.text(col_x[1], y, action, color=col, fontsize=7.5, va="center", zorder=2)
    ax3.text(col_x[2], y, rec,    color=C_TEXT, fontsize=7.5, va="center", zorder=2)
    ax3.text(col_x[3], y, mw,     color=C_TEXT, fontsize=7.5, va="center", zorder=2)
    ax3.text(col_x[4], y, qed,    color=C_TEXT, fontsize=7.5, va="center", zorder=2)

# Footer
ax3.text(0.5, 0.012,
         "All molecules: Tier-1 amide  |  MW 640–713 Da  |  SA ≤ 3.5  |  logP 2.1–5.2",
         ha="center", color="#777777", fontsize=7.5, fontstyle="italic")

# ── Epigenetic mark legend (bottom of panel 3) ────────────────────────────
mark_y = 0.040
marks = [("H3K27ac", C_NRM_ON, "activation"), ("H3K27me3", "#8A6FBF", "PRC2 repression"),
         ("H4R3me2s", C_REPR, "PRMT5 repression"), ("−H3K27ac", "#E8A000", "HDAC2 removal")]
for mi, (mark, mc, desc) in enumerate(marks):
    ax3.add_patch(mpatches.Circle((0.08 + mi * 0.24, mark_y), 0.012, color=mc, zorder=3))
    ax3.text(0.10 + mi * 0.24, mark_y, mark, color=mc, fontsize=6.5, va="center")

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight", facecolor=C_BG)
print(f"Saved → {OUT_PATH}")
plt.close()
