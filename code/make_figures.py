"""Generate the two conceptual figures for the paper in a clean
draw.io-like style: Arial text, rounded boxes, lots of whitespace.

Outputs into ../figures/ relative to this script.
Run:  python3 code/make_figures.py
"""

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, PathPatch
from matplotlib.path import Path
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")

BLUE = "#1f78b4"
LT_BLUE = "#dae8fc"
DARK = "#0d2c54"
GREEN = "#d5e8d4"
GREEN_EDGE = "#82b366"
YELLOW = "#fff2cc"
YELLOW_EDGE = "#d6b656"
RED = "#c0392b"
GREY = "#666666"

FS = 10.0        # box text
FS_SM = 9.0      # edge / side labels


def hand_drawn(fn):
    """Run the figure body inside matplotlib's xkcd sketch style with a
    handwriting font, so conceptual figures look hand-drawn."""
    def wrapper():
        with plt.xkcd(randomness=2):
            plt.rcParams["font.family"] = ["Bradley Hand",
                                           "Chalkboard SE",
                                           "Comic Sans MS"]
            return fn()
    return wrapper


def box(ax, x, y, w, h, text, fc=LT_BLUE, ec=BLUE, fs=FS, dashed=False,
        bold=False, square=False):
    style = "square,pad=0.02" if square else \
        "round,pad=0.02,rounding_size=0.12"
    b = FancyBboxPatch((x, y), w, h, boxstyle=style,
                       linewidth=1.2, edgecolor=ec, facecolor=fc,
                       linestyle="--" if dashed else "-")
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=DARK,
            fontweight="bold" if bold else "normal", linespacing=1.3)
    return (x + w / 2, y + h / 2)


def arrow(ax, p1, p2, dashed=False, color=DARK, lw=1.3, label=None,
          fs=FS_SM, rad=0.0, label_dx=0.0, label_dy=0.12,
          label_color=GREY):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                        linewidth=lw, color=color,
                        linestyle="--" if dashed else "-",
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if label:
        mx = (p1[0] + p2[0]) / 2 + label_dx
        my = (p1[1] + p2[1]) / 2 + label_dy
        ax.text(mx, my, label, fontsize=fs, ha="center",
                color=label_color, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.9))


def figure_architecture():
    """High-level design of the wildfire-response system: sensor inputs
    feed the federated fog network, the data-aware scheduler places the
    six workflow tasks on the four fog nodes, and results return to the
    emergency operations centre.  Landscape layout so it reads well as a
    two-column figure.  Node names, task placements and link rates match
    Fig. 2 exactly."""
    fig, ax = plt.subplots(figsize=(13.6, 5.2), dpi=300)
    ax.set_xlim(0, 34); ax.set_ylim(0, 13.8); ax.axis("off")

    EDGE_BLUE = "#2c7fb8"

    def pill(x, y, text, heavy=False, ha="center"):
        ax.text(x, y, text, fontsize=FS_SM, ha=ha, va="center",
                color=RED if heavy else DARK, zorder=6, linespacing=1.2)

    def elbow(pts, dashed=False, color=DARK, lw=1.1, z=3, ls="-"):
        """Axis-aligned (orthogonal) connector: polyline plus arrowhead
        on the final straight segment."""
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=color, lw=lw, zorder=z,
                linestyle=(0, (4, 3)) if dashed else ls)
        a = FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                            mutation_scale=11, linewidth=lw,
                            color=color, zorder=z + 1)
        ax.add_patch(a)

    def cloud(x, y, w, h):
        """The classic cloud glyph (Font Awesome outline) scaled to
        (x, y, w, h).  Cubics are already expanded to absolute coords in
        a 640 x 512 box; y is flipped for matplotlib."""
        pts640 = [
            (537.6, 226.6),
            (594.3, 245.5), (633.6, 299.1), (633.6, 362.2),
            (633.6, 440.9), (569.8, 504.9), (491.1, 504.9),
            (160.0, 504.9),
            (71.6, 504.9), (0.0, 433.3), (0.0, 355.4),
            (0.0, 286.3), (50.0, 228.1), (116.5, 215.0),
            (123.2, 139.0), (191.6, 86.3), (270.1, 86.3),
            (331.4, 86.3), (384.6, 125.2), (405.6, 181.6),
            (420.8, 163.3), (446.2, 151.9), (474.2, 151.9),
            (529.7, 151.9), (574.8, 197.0), (574.8, 252.5),
            (574.8, 254.1), (574.7, 255.7), (574.6, 257.2),
            (0, 0),
        ]
        codes = ([Path.MOVETO] + [Path.CURVE4] * 6 + [Path.LINETO] +
                 [Path.CURVE4] * 21 + [Path.CLOSEPOLY])
        pts = [(x + vx / 640.0 * w, y + (512.0 - vy) / 512.0 * h)
               for vx, vy in pts640]
        ax.add_patch(PathPatch(Path(pts, codes), facecolor="white",
                               edgecolor=GREY, lw=1.3, linestyle="--",
                               zorder=4))

    # ---------- top row: inputs, scheduler, unreachable cloud ----------
    box(ax, 0.8, 11.3, 9.4, 2.2,
        "Wildfire monitoring inputs\ndrone imagery, cameras and\n"
        "environmental sensors", fc="white", ec=DARK, fs=FS,
        square=True)
    box(ax, 12.0, 11.3, 12.4, 2.2,
        "Data-aware scheduler\nassign tasks using compute time,\n"
        "data size and network bandwidth",
        fc="white", ec=DARK, fs=FS, bold=True, square=True)

    cloud(27.9, 10.9, 4.8, 2.8)
    ax.text(30.3, 12.5, "Cloud", ha="center", va="center",
            fontsize=FS, color=DARK, zorder=6)
    ax.plot([28.6, 32.0], [12.5, 12.5], color=RED, lw=1.6,
            solid_capstyle="round", zorder=6)
    ax.text(30.3, 10.35, "unreachable during the event", ha="center",
            va="center", fontsize=FS_SM, color=RED,
            fontstyle="italic", zorder=6)

    # ---------- federated fog container: one horizontal row ----------
    fed = FancyBboxPatch((0.8, 1.5), 27.6, 9.5,
                         boxstyle="square,pad=0.02",
                         linewidth=1.3, edgecolor="#9a9a9a",
                         facecolor="none", zorder=0)
    ax.add_patch(fed)
    ax.text(14.6, 10.4, "Federated fog network", ha="center",
            va="center", fontsize=FS + 0.5, color="#555555",
            fontweight="bold", zorder=5)

    # four nodes in a single row, left to right
    nw, nh = 5.4, 3.4
    fy = 6.0
    f1x, f2x, f3x, f4x = 4.5, 11.0, 17.5, 24.0
    box(ax, f1x - nw/2, fy, nw, nh,
        "f1 — Gateway\nT1: Collect data\nT2: Prepare data\n"
        "T3: Detect smoke", fc="white", ec=EDGE_BLUE, fs=FS,
        square=True)
    box(ax, f2x - nw/2, fy, nw, nh,
        "f2 — Shelter\nT5: Map fire",
        fc="white", ec=EDGE_BLUE, fs=FS, square=True)
    box(ax, f3x - nw/2, fy, nw, nh,
        "f3 — Base station\nT4: Predict fire\nspread",
        fc="white", ec=EDGE_BLUE, fs=FS, square=True)
    box(ax, f4x - nw/2, fy, nw, nh,
        "f4 — Mobile unit\nT6: Assess\ndamage",
        fc="white", ec=EDGE_BLUE, fs=FS, square=True)

    # ---------- links as arcs under the row (same rates as Fig. 2b) ----------
    def link_arc(x1, x2, rad, lbl, heavy=False, dotted=False, ly=3.6):
        a = FancyArrowPatch((x1, fy), (x2, fy), arrowstyle="-|>",
                            mutation_scale=12,
                            linewidth=2.0 if heavy else 1.1,
                            color=RED if heavy else "#555555",
                            linestyle=":" if dotted else "-",
                            connectionstyle=f"arc3,rad={rad}", zorder=3)
        ax.add_patch(a)
        ax.text((x1 + x2) / 2, ly, lbl, fontsize=FS_SM, ha="center",
                va="center", color=RED if heavy else "black",
                fontstyle="italic", linespacing=1.2, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.9))

    edge = nw / 2
    # adjacent pairs: shallow arcs
    link_arc(f1x + edge, f2x - edge, 0.35, "800 Mbp/s\nT3→T5: 0.9 GB",
             ly=4.15)
    link_arc(f2x + edge, f3x - edge, 0.35, "300 Mbp/s\n(unused)",
             dotted=True, ly=4.15)
    link_arc(f3x + edge, f4x - edge, 0.35, "400 Mbp/s\nT4→T6: 0.35 GB",
             ly=4.45)
    # spanning pairs: deeper arcs
    link_arc(f1x, f3x, 0.42, "80 Mbp/s — bottleneck\nT3→T4: 1.8 GB",
             heavy=True, ly=2.5)
    link_arc(f2x, f4x, 0.30, "500 Mbp/s\nT5→T6: 0.7 GB", ly=3.25)

    # ---------- raw data in ----------
    elbow([(4.5, 11.3), (4.5, 9.4)], color=DARK, lw=1.2)
    pill(4.0, 10.6, "Raw data", ha="right")

    # ---------- task placement: dashed drops ----------
    for tx in (f1x, f2x, f3x, f4x):
        elbow([(17.5, 11.3), (17.5, 10.9), (tx, 10.9), (tx, 9.4)],
              dashed=True, color=GREY, lw=0.9, z=2)
    pill(14.4, 10.9, "Task placement", ha="right")

    # ---------- results out ----------
    box(ax, 29.4, 5.9, 4.4, 3.2,
        "Emergency\noperations\ncentre\nfire map,\ndamage",
        fc="white", ec=DARK, fs=FS, square=True)
    elbow([(f4x + edge, 7.7), (29.4, 7.7)], color=DARK, lw=1.2)
    pill(28.2, 8.6, "Response\nresults", ha="center")

    ax.text(1.4, 0.5, "Link labels: rate in Mbp/s; task data in GB "
            "where used", fontsize=FS_SM - 0.5, color="#777777",
            fontstyle="italic", ha="left")

    fig.savefig(os.path.join(FIG, "federated-fog-architecture.png"),
                bbox_inches="tight", transparent=True, pad_inches=0.04)
    plt.close(fig)


def figure_dag():
    """Two-panel wildfire-response figure.  Left: the workflow DAG --
    circular task nodes, edges carry realistic data sizes (hundreds of
    MB).  Right: the fog resource network -- rectangular nodes, links
    carry bandwidths in MB/s.  Task colours mark which fog node runs
    them; the highlighted transfer crosses a slow link and pays D/B."""
    NODE_FC = {1: "#dbe9f6", 2: "#e8e4f3", 3: "#dff0e6", 4: "#fdf3d7"}
    EDGE_BLUE = "#2c7fb8"

    fig, ax = plt.subplots(figsize=(7.6, 4.15), dpi=300)
    ax.set_xlim(0, 16.4); ax.set_ylim(0, 9.4); ax.axis("off")

    def tnode(x, y, name, f, r=0.62):
        c = plt.Circle((x, y), r, facecolor=NODE_FC[f],
                       edgecolor=EDGE_BLUE, lw=1.6, zorder=3)
        ax.add_patch(c)
        ax.text(x, y, name, ha="center", va="center", fontsize=10,
                color="black", zorder=4, linespacing=1.15)
        return (x, y)

    def fnode(x, y, name, f, w=2.5, h=1.1):
        b = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.18",
                           linewidth=1.6, edgecolor=EDGE_BLUE,
                           facecolor=NODE_FC[f])
        ax.add_patch(b)
        ax.text(x, y, name, ha="center", va="center", fontsize=11,
                color="black", zorder=4, linespacing=1.15)
        return (x, y)

    ax.text(3.5, 9.05, "(a) Wildfire response workflow", ha="center",
            fontsize=12, color="black", fontweight="bold")
    ax.text(11.2, 9.05, "(b) Fog resource network", ha="center",
            fontsize=12, color="black", fontweight="bold")

    # ---------- left: workflow DAG (circles) ----------
    T = {}
    T[1] = tnode(3.5, 7.9, "T1\nf1", 1, r=0.6)
    T[2] = tnode(1.4, 5.7, "T2\nf1", 1, r=0.6)
    T[3] = tnode(5.6, 5.7, "T3\nf1", 1, r=0.6)
    T[4] = tnode(1.4, 3.3, "T4\nf3", 3, r=0.6)
    T[5] = tnode(5.6, 3.3, "T5\nf2", 2, r=0.6)
    T[6] = tnode(3.5, 1.3, "T6\nf4", 4, r=0.6)

    edges = [(1, 2, "1.2 GB", (-0.85, 0.05)), (1, 3, "2.4 GB", (0.9, 0.05)),
             (2, 4, "0.45 GB", (-0.75, 0.05)), (3, 4, "1.8 GB", (0.5, -0.1)),
             (3, 5, "0.9 GB", (0.85, 0.05)), (4, 6, "0.35 GB", (-0.75, 0.05)),
             (5, 6, "0.7 GB", (0.85, 0.05))]
    for a, b, lbl, (dx, dy) in edges:
        heavy = (a, b) == (3, 4)
        ux, uy = T[b][0] - T[a][0], T[b][1] - T[a][1]
        d = math.hypot(ux, uy)
        ux, uy = ux / d, uy / d
        p1 = (T[a][0] + ux * 0.57, T[a][1] + uy * 0.62)
        p2 = (T[b][0] - ux * 0.57, T[b][1] - uy * 0.62)
        arrow(ax, p1, p2,
              color="#c0392b" if heavy else DARK,
              lw=2.0 if heavy else 1.1,
              rad=0.0,
              label=lbl, fs=10.5,
              label_dx=dx, label_dy=dy,
              label_color="#c0392b" if heavy else "black")

    # ---------- right: resource network (rectangles) ----------
    F = {}
    F[1] = fnode(9.2, 7.6, "f1", 1, w=1.5, h=1.0)
    F[2] = fnode(13.2, 7.6, "f2", 2, w=1.5, h=1.0)
    F[3] = fnode(9.2, 4.2, "f3", 3, w=1.5, h=1.0)
    F[4] = fnode(13.2, 4.2, "f4", 4, w=1.5, h=1.0)

    # (a, b, label, label offset, directed a -> b?)
    links = [(1, 2, "800 Mbp/s\nWi-Fi 6", (0, 0.9), True, 0),
             (1, 3, "80 Mbp/s\nLTE backup", (-1.45, 0.0), True, 0),
             (2, 4, "500 Mbp/s\nWi-Fi", (1.45, 0.0), True, 0),
             (3, 4, "400 Mbp/s\nWi-Fi", (0.55, 0.72), True, 0)]
    for a, b, lbl, (dx, dy), directed, rot in links:
        heavy = (a, b) == (1, 3)
        if a == 3 and b == 4:                # horizontal, right to right
            p1 = (F[a][0] + 0.78, F[a][1])
            p2 = (F[b][0] - 0.78, F[b][1])
        elif b in (3, 4):                    # vertical, top to bottom
            p1 = (F[a][0], F[a][1] - 0.53)
            p2 = (F[b][0], F[b][1] + 0.53)
        else:                                # horizontal, left to right
            p1 = (F[a][0] + 0.78, F[a][1])
            p2 = (F[b][0] - 0.78, F[b][1])
        col = "#c0392b" if heavy else GREY
        ax.add_patch(FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=13,
            linewidth=2.0 if heavy else 1.1,
            color=col, zorder=2))
        ax.text((p1[0] + p2[0]) / 2 + dx, (p1[1] + p2[1]) / 2 + dy,
                lbl, fontsize=10, ha="center", va="center",
                rotation=rot,
                color="#c0392b" if heavy else "black",
                fontstyle="italic", linespacing=1.25, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.9))

    # unused f2--f3 link: L-shaped dotted elbow, f2 left edge -> f3 top
    elb = [(12.45, 7.6), (11.6, 7.6), (11.6, 4.73), (9.2, 4.73)]
    xs, ys = zip(*elb)
    ax.plot(xs, ys, color=GREY, lw=1.0, ls=":", zorder=2)
    # unused f1--f4 pair: shallow dotted diagonal so all six node
    # pairs are drawn (the model prices every pair; the example
    # placement simply leaves these two unused)
    ax.add_patch(FancyArrowPatch(
        (F[1][0] + 0.35, F[1][1] - 0.5), (F[4][0] - 0.45, F[4][1] + 0.5),
        arrowstyle="-|>", mutation_scale=11, linewidth=1.0,
        color=GREY, zorder=2, linestyle=":",
        connectionstyle="arc3,rad=-0.18"))
    ax.text(10.3, 6.3, "300 Mbp/s\n(microwave)", fontsize=10,
            ha="center",
            va="center", color="black", fontstyle="italic", zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec="none", alpha=0.9))

    # ---------- line legend ----------
    lx, ly = 14.35, 2.9
    ax.add_patch(FancyArrowPatch((lx, ly + 1.6), (lx + 1.0, ly + 1.6),
        arrowstyle="-|>", mutation_scale=11, linewidth=1.1,
        color=GREY, zorder=5))
    ax.text(lx + 1.2, ly + 1.6, "link in use", fontsize=9.5,
            va="center", color="black")
    ax.add_patch(FancyArrowPatch((lx, ly + 0.8), (lx + 1.0, ly + 0.8),
        arrowstyle="-|>", mutation_scale=11, linewidth=2.0,
        color="#c0392b", zorder=5))
    ax.text(lx + 1.2, ly + 0.8, "bottleneck", fontsize=9.5,
            va="center", color="#c0392b")
    ax.plot([lx, lx + 1.0], [ly, ly], color=GREY, lw=1.0, ls=":",
            zorder=5)
    ax.text(lx + 1.2, ly, "unused link", fontsize=9.5,
            va="center", color="black")

    # ---------- payoff note under the network panel ----------
    ax.text(11.2, 1.85, "Highlighted transfer: T3 on f1 → T4 on f3",
            ha="center", fontsize=10.5, color=DARK, zorder=2)
    ax.text(11.2, 1.4, "Delay = data size / link bandwidth",
            ha="center", fontsize=10, color=DARK, zorder=2)
    ax.text(11.2, 0.85, "1.8 GB = 14.4 Gb / 80 Mbp/s = 180 s",
            ha="center", fontsize=11.5, color="#c0392b",
            fontweight="bold", zorder=2)

    ax.text(8.2, 0.0,
            "Task colours and fog IDs show placement.   Edge labels: "
            "data sizes.   Link labels: bandwidth.",
            ha="center", va="bottom", fontsize=9.5, color=GREY)

    fig.savefig(os.path.join(FIG, "disaster-response-dag.png"),
                bbox_inches="tight", transparent=True, pad_inches=0.04)
    plt.close(fig)


def figure_pipeline():
    """Method-pipeline flow diagram: inputs on the left, the four
    schedulers sharing one cost model in the middle, and the produced
    schedule + metrics on the right. Sells the paper's story at a glance."""
    fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=300)
    ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.axis("off")

    # ---------- inputs ----------
    inp = FancyBboxPatch((0.35, 1.7), 3.7, 7.0,
                         boxstyle="round,pad=0.02,rounding_size=0.15",
                         linewidth=1.0, edgecolor="#8fb8de",
                         facecolor="#f4f9fd")
    ax.add_patch(inp)
    ax.text(2.2, 8.15, "inputs", ha="center", fontsize=FS,
            color="#4a6fa5", fontweight="bold")

    i1 = box(ax, 0.7, 6.35, 3.0, 1.0,
             "workflow DAG\n$D_{ji}$ on every edge", fs=FS)
    i2 = box(ax, 0.7, 4.55, 3.0, 1.0,
             "fog network\n$B_{ba}$ on every link", fs=FS)
    i3 = box(ax, 0.7, 2.75, 3.0, 1.0,
             "$t_{ia}, A_i, P_{ia}$\ndeadline $t_{max}$", fs=FS)

    # ---------- shared cost model + schedulers ----------
    mid = FancyBboxPatch((5.0, 1.7), 6.0, 7.0,
                         boxstyle="round,pad=0.02,rounding_size=0.15",
                         linewidth=1.2, edgecolor=YELLOW_EDGE,
                         facecolor="#fffdf5")
    ax.add_patch(mid)
    ax.text(8.0, 8.15, "one data-aware cost model", ha="center",
            fontsize=FS, color="#8a6d1c", fontweight="bold")
    ax.text(8.0, 7.62, "crossed edge costs $c_{ji}=D_{ji}/B_{ba}$",
            ha="center", fontsize=FS_SM, color=GREY, fontstyle="italic")

    box(ax, 5.45, 6.05, 5.1, 0.85, "MILP  (exact reference)", fs=FS,
        bold=True)
    box(ax, 5.45, 4.95, 5.1, 0.85, "LP + randomized rounding", fs=FS)
    box(ax, 5.45, 3.85, 5.1, 0.85, "bandwidth-aware greedy", fs=FS)
    box(ax, 5.45, 2.75, 5.1, 0.85, "genetic algorithm", fs=FS)
    ax.text(8.0, 2.15, "same objective: max accuracy, meet $t_{max}$",
            ha="center", fontsize=FS_SM, color=GREY, fontstyle="italic")

    # ---------- outputs ----------
    out = FancyBboxPatch((11.95, 1.7), 3.7, 7.0,
                         boxstyle="round,pad=0.02,rounding_size=0.15",
                         linewidth=1.0, edgecolor=GREEN_EDGE,
                         facecolor="#f5faf2")
    ax.add_patch(out)
    ax.text(13.8, 8.15, "schedule", ha="center", fontsize=FS,
            color=GREEN_EDGE, fontweight="bold")

    o1 = box(ax, 12.3, 6.35, 3.0, 1.0,
             "placement $e_{ia}$\nwhich node runs it", fs=FS,
             fc=GREEN, ec=GREEN_EDGE)
    o2 = box(ax, 12.3, 4.55, 3.0, 1.0,
             "fraction $l_{ia}$\nhow much of it runs", fs=FS,
             fc=GREEN, ec=GREEN_EDGE)
    o3 = box(ax, 12.3, 2.75, 3.0, 1.0,
             "score, makespan,\ndeadline feasibility", fs=FS,
             fc=GREEN, ec=GREEN_EDGE)

    # flow arrows
    for src in (i1, i2, i3):
        arrow(ax, (src[0] + 1.5, src[1]), (5.0, src[1]), color=BLUE, lw=1.2)
    for dst in (o1, o2, o3):
        arrow(ax, (11.0, dst[1]), (dst[0] - 1.5, dst[1]), color=GREEN_EDGE,
              lw=1.2)

    fig.savefig(os.path.join(FIG, "scheduling-pipeline.png"),
                bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    figure_architecture()
    figure_dag()
    figure_pipeline()
    print("wrote federated-fog-architecture.png, disaster-response-dag.png, "
          "scheduling-pipeline.png")
