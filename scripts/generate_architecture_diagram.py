import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_transparent_white_diagram():
    # Pure white / transparent background, minimalist black & crisp line drawings (Nature/IEEE style)
    fig, ax = plt.subplots(figsize=(13.0, 4.8), dpi=300)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 48)
    ax.axis('off')

    # Color Palette: Pure monochrome lines with very subtle, clean slate borders
    BG_BOX = "#FFFFFF"        # Pure clean white
    BORDER_PANEL = "#BDBDBD"  # Thin gray boundary for panels
    BORDER_BOX = "#424242"    # Sharp dark gray for component boxes
    LINE_ARROW = "#212121"    # Dark crisp arrows
    ACCENT_LINE = "#1565C0"   # Subtle clean IEEE blue for the SS2D component outline
    
    TEXT_MAIN = "#212121"
    TEXT_MUTED = "#616161"

    # Font dictionary helpers
    t_sec = {'fontsize': 10.0, 'fontweight': 'bold', 'color': TEXT_MAIN}
    t_sub = {'fontsize': 8.0, 'color': TEXT_MUTED}
    b_main = {'fontsize': 8.8, 'fontweight': 'bold', 'color': TEXT_MAIN}
    b_sub = {'fontsize': 7.4, 'color': TEXT_MUTED}

    # ==========================================
    # 1. LEFT PANEL: Macro Architecture
    # ==========================================
    rect_macro = patches.FancyBboxPatch((2, 2), 38, 44, boxstyle="round,pad=0.3", 
                                        fc=BG_BOX, ec=BORDER_PANEL, lw=0.9, zorder=1)
    ax.add_patch(rect_macro)
    ax.text(21, 43.5, "(a) YOLO11s-SS2D Macro Architecture", ha='center', **t_sec)
    ax.text(21, 41.5, "Stage-wise Local Convolutions & Selective Context", ha='center', **t_sub)

    # Input image
    b_in = patches.FancyBboxPatch((5, 33.0), 32, 6.0, boxstyle="round,pad=0.15", fc=BG_BOX, ec=BORDER_BOX, lw=0.9, zorder=2)
    ax.add_patch(b_in)
    ax.text(21, 36.5, "Input Image [3, 640, 640]", ha='center', **b_main)
    ax.text(21, 34.6, "Warehouse Camera RGB Stream", ha='center', **b_sub)

    ax.annotate('', xy=(21, 29.0), xytext=(21, 33.0), arrowprops=dict(arrowstyle="-|>", lw=1.1, color=LINE_ARROW))

    # P1-P4 Backbone
    b_bb = patches.FancyBboxPatch((5, 22.5), 32, 6.5, boxstyle="round,pad=0.15", fc=BG_BOX, ec=BORDER_BOX, lw=0.9, zorder=2)
    ax.add_patch(b_bb)
    ax.text(21, 26.5, "Backbone Layers 0-9 (Local Conv)", ha='center', **b_main)
    ax.text(21, 24.6, "Conv + C3k2 (P1-P5) + SPPF -> [512, 20, 20]", ha='center', **b_sub)

    ax.annotate('', xy=(21, 18.5), xytext=(21, 22.5), arrowprops=dict(arrowstyle="-|>", lw=1.1, color=LINE_ARROW))

    # Stage P5 Context Replacement
    b_p5 = patches.FancyBboxPatch((4, 11.0), 34, 7.5, boxstyle="round,pad=0.2", fc=BG_BOX, ec=ACCENT_LINE, lw=1.3, zorder=2)
    ax.add_patch(b_p5)
    ax.text(21, 15.8, "Stage P5: WarehouseSS2DBlock", ha='center', fontsize=9.0, fontweight='bold', color=ACCENT_LINE)
    ax.text(21, 14.0, "Replaces C2PSA Attention (-63.9% parameters)", ha='center', **b_sub)
    ax.text(21, 12.3, "Tensor: [512, 20, 20] (400 spatial tokens)", ha='center', **b_sub)

    ax.annotate('', xy=(21, 7.0), xytext=(21, 11.0), arrowprops=dict(arrowstyle="-|>", lw=1.1, color=LINE_ARROW))

    # Neck & Detect Head
    b_hd = patches.FancyBboxPatch((5, 3.2), 32, 3.8, boxstyle="round,pad=0.15", fc=BG_BOX, ec=BORDER_BOX, lw=0.9, zorder=2)
    ax.add_patch(b_hd)
    ax.text(21, 5.5, "PANet Neck + Decoupled Head", ha='center', **b_main)
    ax.text(21, 4.0, "Class Probabilities (4) & Bounding Boxes", ha='center', **b_sub)

    # Connector to micro block
    ax.annotate('', xy=(44, 23.5), xytext=(38, 14.8), 
                arrowprops=dict(arrowstyle="-|>", connectionstyle="angle,angleA=0,angleB=90,rad=6", 
                                lw=1.1, color=ACCENT_LINE, linestyle="--"))
    ax.text(41, 18.5, "Detail", ha='center', va='center', fontsize=7.2, color=ACCENT_LINE, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=ACCENT_LINE, lw=0.6))

    # ==========================================
    # 2. RIGHT PANEL: Micro-Architecture
    # ==========================================
    rect_micro = patches.FancyBboxPatch((44, 2), 84, 44, boxstyle="round,pad=0.3", 
                                        fc=BG_BOX, ec=BORDER_PANEL, lw=0.9, zorder=1)
    ax.add_patch(rect_micro)
    ax.text(86, 43.5, "(b) WarehouseSS2DBlock Micro-Architecture & 4-Direction Selective Scan", ha='center', **t_sec)
    ax.text(86, 41.5, "Sequential Pure-PyTorch Selective Scan (Standard ONNX Opset 17 Operators)", ha='center', **t_sub)

    # Input Tensor
    b_tin = patches.FancyBboxPatch((48, 32.0), 16, 6.0, boxstyle="round,pad=0.15", fc=BG_BOX, ec=BORDER_BOX, lw=0.9, zorder=2)
    ax.add_patch(b_tin)
    ax.text(56, 35.8, "Input Tensor F", ha='center', **b_main)
    ax.text(56, 33.8, "[b, 512, 20, 20]", ha='center', **b_sub)

    # Residual path across top
    # Routed right of the out-proj block so that it enters the addition node, not the block
    ax.plot([64, 125.5, 125.5], [35.0, 35.0, 9.8], color=LINE_ARROW, lw=1.1, linestyle="--", zorder=3)
    ax.annotate('', xy=(119.0, 9.8), xytext=(125.5, 9.8), arrowprops=dict(arrowstyle="-|>", lw=1.1, color=LINE_ARROW, linestyle="--"))
    ax.text(90, 36.2, "Residual Identity Connection (Exact Pass-Through)", ha='center', fontsize=7.6, color=TEXT_MUTED)

    # Main branch downward
    ax.annotate('', xy=(56, 28.0), xytext=(56, 32.0), arrowprops=dict(arrowstyle="-|>", lw=1.1, color=LINE_ARROW))

    # Local Convolutions block
    b_lconv = patches.FancyBboxPatch((48, 19.5), 16, 8.5, boxstyle="round,pad=0.15", fc=BG_BOX, ec=BORDER_BOX, lw=0.9, zorder=2)
    ax.add_patch(b_lconv)
    ax.text(56, 25.5, "Local Projections", ha='center', **b_main)
    ax.text(56, 23.6, "1x1 Conv (512 -> 256)", ha='center', **b_sub)
    ax.text(56, 21.9, "BatchNorm2d + SiLU", ha='center', **b_sub)
    ax.text(56, 20.3, "3x3 DWConv + BN + SiLU", ha='center', **b_sub)

    # Arrow to scan core
    ax.annotate('', xy=(68, 23.5), xytext=(64, 23.5), arrowprops=dict(arrowstyle="-|>", lw=1.1, color=LINE_ARROW))

    # 4-Direction Selective Scan Core Panel
    rect_score = patches.FancyBboxPatch((68, 8.0), 35, 26.0, boxstyle="round,pad=0.2", fc=BG_BOX, ec=ACCENT_LINE, lw=1.0, zorder=2)
    ax.add_patch(rect_score)
    ax.text(85.5, 31.5, "4-Way Selective Scan Core (SS2D)", ha='center', fontsize=8.8, fontweight='bold', color=ACCENT_LINE)
    ax.text(85.5, 29.5, r"Discrete Sequential Recurrence: $\mathbf{h}_t = \mathbf{\bar{A}}_t \odot \mathbf{h}_{t-1} + (\mathbf{\bar{B}}\mathbf{u})_t$", ha='center', fontsize=7.4, color=TEXT_MUTED)

    # 4 Scan Directions
    dir_boxes = [
        ("Dir 1: Standard Row-Major (Top -> Btm, L -> R)", 24.5),
        ("Dir 2: Reversed Row-Major (Btm -> Top, R -> L)", 19.8),
        ("Dir 3: Column-Major (L -> R, Top -> Btm)", 15.1),
        ("Dir 4: Reversed Column-Major (R -> L, Btm -> Top)", 10.4)
    ]
    for label, y_c in dir_boxes:
        b_d = patches.FancyBboxPatch((70.5, y_c), 30, 3.8, boxstyle="round,pad=0.1", fc=BG_BOX, ec="#9E9E9E", lw=0.7, zorder=3)
        ax.add_patch(b_d)
        ax.text(85.5, y_c + 1.9, label, ha='center', va='center', fontsize=7.2, color=TEXT_MAIN)

    # Arrow from Scan to Fusion
    ax.annotate('', xy=(107, 21.0), xytext=(103, 21.0), arrowprops=dict(arrowstyle="-|>", lw=1.1, color=LINE_ARROW))

    # Re-orientation & Out-proj block
    b_fuse = patches.FancyBboxPatch((107, 15.0), 16, 9.0, boxstyle="round,pad=0.15", fc=BG_BOX, ec=BORDER_BOX, lw=0.9, zorder=2)
    ax.add_patch(b_fuse)
    ax.text(115, 22.4, "Re-Orientation", ha='center', **b_main)
    ax.text(115, 20.9, "Fold 4 Directions + Sum", ha='center', **b_sub)
    ax.text(115, 19.4, "1x1 Conv (256 -> 512)", ha='center', **b_sub)
    ax.text(115, 17.9, "BatchNorm2d", ha='center', **b_sub)
    # Initialization of the reported runs; zero-gamma (gamma=beta=0) is the corrected scheme, not yet evaluated
    ax.text(115, 16.2, "Reported runs: Conv W = 0", ha='center', fontsize=6.6, color="#C62828", fontweight='bold')

    # Arrow to Addition Node
    ax.annotate('', xy=(117, 11.8), xytext=(115, 15.0), arrowprops=dict(arrowstyle="-|>", lw=1.1, color=LINE_ARROW))

    # Addition Node Circle
    circle = patches.Circle((117, 9.8), radius=1.9, fc=BG_BOX, ec=LINE_ARROW, lw=1.1, zorder=4)
    ax.add_patch(circle)
    # zorder above the circle (4), otherwise the patch hides the sign
    ax.text(117, 9.8, "+", ha='center', va='center', fontsize=11, fontweight='bold', color=TEXT_MAIN, zorder=5)

    # Final Output Arrow & Box
    # Addition circle is at y=9.8 with radius 1.9 -> bottom edge is at y=7.9
    # Output box spans from y=2.2 to y=5.8 (height=3.6) -> top edge is at y=5.8
    # Arrow runs cleanly from y=7.8 down to y=5.9, terminating right at the top border
    ax.annotate('', xy=(117, 5.9), xytext=(117, 7.8), arrowprops=dict(arrowstyle="-|>", lw=1.2, color=LINE_ARROW))

    b_out = patches.FancyBboxPatch((107, 2.2), 20, 3.6, boxstyle="round,pad=0.15", fc=BG_BOX, ec=BORDER_BOX, lw=1.0, zorder=2)
    ax.add_patch(b_out)
    ax.text(117, 4.0, r"$\mathbf{Y} = \mathbf{F} + \text{SS2D}(\mathbf{F})$", ha='center', va='center', color=TEXT_MAIN, fontweight='bold', fontsize=8.2)

    plt.tight_layout()
    output_pdf = "paper/figures/fig_architecture_diagram.pdf"
    output_png = "paper/figures/fig_architecture_diagram.png"
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight", transparent=True)
    plt.savefig(output_png, format="png", bbox_inches="tight", dpi=300, transparent=True)
    print(f"Generated clean monochrome/transparent diagram: {output_pdf}")

if __name__ == '__main__':
    generate_transparent_white_diagram()
