"""Fig. 1: YOLO11s-SS2D macro architecture and WarehouseSS2DBlock internals.

Drawn at the final printed size (IEEE double-column width, 7.16 in) so that the smallest text is
6 pt in print, with TrueType font embedding (pdf.fonttype 42) instead of Type 3 fonts.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'mathtext.fontset': 'stixsans',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

WIDTH_IN = 7.16          # IEEE \textwidth
X_MAX, Y_MAX = 130, 48   # drawing units

BG = "#FFFFFF"
PANEL = "#BDBDBD"
BOX = "#424242"
ARROW = "#212121"
ACCENT = "#1565C0"
MAIN = "#212121"
MUTED = "#555555"

T_SEC = {'fontsize': 7.5, 'fontweight': 'bold', 'color': MAIN}
T_SUB = {'fontsize': 6.0, 'color': MUTED}
B_MAIN = {'fontsize': 6.8, 'fontweight': 'bold', 'color': MAIN}
B_SUB = {'fontsize': 6.0, 'color': MUTED}


def box(ax, x, y, w, h, ec=BOX, lw=0.8, z=2, pad=0.15):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad}", fc=BG, ec=ec, lw=lw, zorder=z))


def arrow(ax, xy, xytext, color=ARROW, ls='-', lw=0.9, **kw):
    ax.annotate('', xy=xy, xytext=xytext, arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, linestyle=ls,
                                                          mutation_scale=7, **kw))


def generate(block_only=False, out="paper/figures/fig_architecture_diagram"):
    """block_only=True draws panel (b) alone at 4.76 in, which fits the 12.2 cm LNCS text width
    with the same 6-7.5 pt text sizes."""
    x0 = 43.3 if block_only else 0
    fig, ax = plt.subplots(figsize=(WIDTH_IN * (X_MAX - x0) / X_MAX, WIDTH_IN * Y_MAX / X_MAX))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(x0, X_MAX)
    ax.set_ylim(0, Y_MAX)
    ax.axis('off')
    if not block_only:
        draw_macro(ax)
    draw_block(ax, title="WarehouseSS2DBlock" if block_only else "(b) WarehouseSS2DBlock")
    fig.savefig(out + ".pdf", format="pdf")
    fig.savefig(out + ".png", format="png", dpi=300)
    print("Saved " + out + ".pdf")


def draw_macro(ax):
    # ---------------- (a) macro architecture ----------------
    box(ax, 1, 1, 39, 46, ec=PANEL, z=1, pad=0.3)
    ax.text(20.5, 44.3, "(a) YOLO11s-SS2D", ha='center', **T_SEC)
    ax.text(20.5, 42.0, "Case B: Layer 10 replaced at $P_5$", ha='center', **T_SUB)

    box(ax, 4, 33.5, 33, 6.0)
    ax.text(20.5, 37.4, "Input image", ha='center', **B_MAIN)
    ax.text(20.5, 35.0, "[b, 3, 640, 640]", ha='center', **B_SUB)
    arrow(ax, (20.5, 29.6), (20.5, 33.4))

    box(ax, 4, 23.0, 33, 6.5)
    ax.text(20.5, 27.2, "Backbone, Layers 0–9", ha='center', **B_MAIN)
    ax.text(20.5, 24.6, "Conv + C3k2 + SPPF → [b, 512, 20, 20]", ha='center', **B_SUB)
    arrow(ax, (20.5, 19.1), (20.5, 22.9))

    box(ax, 3, 11.5, 35, 7.5, ec=ACCENT, lw=1.1, pad=0.2)
    ax.text(20.5, 16.4, "Layer 10: WarehouseSS2DBlock", ha='center', fontsize=6.8, fontweight='bold', color=ACCENT)
    ax.text(20.5, 14.0, "replaces C2PSA (−63.9% block parameters)", ha='center', **B_SUB)
    ax.text(20.5, 12.1, "[b, 512, 20, 20], L = 400 tokens", ha='center', **B_SUB)
    arrow(ax, (20.5, 7.6), (20.5, 11.4))

    box(ax, 4, 3.0, 33, 4.5)
    ax.text(20.5, 5.7, "FPN/PAN neck + decoupled head", ha='center', **B_MAIN)
    ax.text(20.5, 3.7, "boxes and 4 classes", ha='center', **B_SUB)

    # link to the detail panel
    arrow(ax, (44.2, 24.0), (38.2, 15.2), color=ACCENT, ls='--',
          connectionstyle="angle,angleA=0,angleB=90,rad=4")


def draw_block(ax, title):
    # ---------------- (b) block internals ----------------
    box(ax, 44, 1, 85, 46, ec=PANEL, z=1, pad=0.3)
    ax.text(86.5, 44.3, title, ha='center', **T_SEC)
    ax.text(86.5, 42.0, "Four-direction selective scan from standard operators (ONNX opset 17)", ha='center', **T_SUB)

    box(ax, 47, 32.0, 18, 6.0)
    ax.text(56, 35.8, "Input F", ha='center', **B_MAIN)
    ax.text(56, 33.4, "[b, 512, 20, 20]", ha='center', **B_SUB)

    # residual connection
    ax.plot([65.2, 127.6, 127.6], [35.0, 35.0, 9.8], color=ARROW, lw=0.9, ls='--', zorder=3)
    arrow(ax, (119.4, 9.8), (127.6, 9.8), ls='--')
    ax.text(95, 36.0, "residual connection", ha='center', **B_SUB)

    arrow(ax, (56, 28.1), (56, 31.9))
    box(ax, 46.5, 17.0, 19, 11.0)
    ax.text(56, 25.6, "Local projection", ha='center', **B_MAIN)
    ax.text(56, 23.2, "1×1 Conv 512→256", ha='center', **B_SUB)
    ax.text(56, 21.2, "BN, SiLU", ha='center', **B_SUB)
    ax.text(56, 19.0, "3×3 DWConv, BN, SiLU", ha='center', **B_SUB)
    arrow(ax, (68.3, 22.5), (65.7, 22.5))

    # scan core
    box(ax, 68.5, 4.5, 36, 29.5, ec=ACCENT, lw=0.9, pad=0.2)
    ax.text(86.5, 31.6, "Selective scan core (SS2D)", ha='center', fontsize=6.8, fontweight='bold', color=ACCENT)
    ax.text(86.5, 29.2, r"shared $\mathbf{W}_{\mathrm{proj}}$: $[\mathbf{B}_k, \mathbf{C}_k, \mathbf{d}_k] = \mathbf{u}_k \mathbf{W}_{\mathrm{proj}}$, $\Delta_k = \mathrm{softplus}(\mathbf{d}_k)$",
            ha='center', **B_SUB)
    ax.text(86.5, 26.9, r"$\mathbf{h}_t = \bar{\mathbf{A}}_t \odot \mathbf{h}_{t-1} + (\bar{\mathbf{B}}\mathbf{u})_t$",
            ha='center', **B_SUB)
    ax.text(86.5, 24.6, "L = 400 sequential steps per direction", ha='center', **B_SUB)
    for i, label in enumerate(["Dir 1: row-major", "Dir 2: row-major, reversed",
                               "Dir 3: column-major", "Dir 4: column-major, reversed"]):
        y = 19.0 - i * 4.2
        box(ax, 72, y, 29, 3.3, ec="#9E9E9E", lw=0.6, z=3, pad=0.1)
        ax.text(86.5, y + 1.65, label, ha='center', va='center', fontsize=6.0, color=MAIN, zorder=4)

    arrow(ax, (107.3, 19.5), (104.9, 19.5))
    box(ax, 107.5, 13.6, 18.5, 11.4)
    ax.text(116.75, 22.7, "Output projection", ha='center', **B_MAIN)
    ax.text(116.75, 20.5, "restore + sum 4 dirs", ha='center', **B_SUB)
    ax.text(116.75, 18.5, "1×1 Conv 256→512, BN", ha='center', **B_SUB)
    ax.text(116.75, 15.0, r"reported: $\mathbf{W}_{\mathrm{out}} = \mathbf{0}$ at init", ha='center', fontsize=6.0,
            color="#C62828")

    arrow(ax, (117.5, 11.8), (117.5, 13.5))
    ax.add_patch(patches.Circle((117.5, 9.8), radius=1.8, fc=BG, ec=ARROW, lw=0.9, zorder=4))
    ax.text(117.5, 9.75, "+", ha='center', va='center', fontsize=8, fontweight='bold', color=MAIN, zorder=5)
    arrow(ax, (117.5, 6.0), (117.5, 7.9))
    box(ax, 107.2, 2.4, 20.6, 3.5)
    ax.text(117.5, 4.15, r"$\mathbf{Y} = \mathbf{F} + \mathrm{BN}(\mathbf{W}_{\mathrm{out}} * \mathbf{F}_{\mathrm{ssm}})$",
            ha='center', va='center', fontsize=6.0, color=MAIN)


if __name__ == '__main__':
    import sys
    if '--lncs' in sys.argv:
        generate(block_only=True, out="paper_lncs/figures/fig_ss2d_block")
    else:
        generate()
