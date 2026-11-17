#!/usr/bin/env python3
"""
Automated Convergence Plotting & Analysis Utility for 3-Arm Comparative Study:
- Arm 1: Control Baseline YOLO11s (COCO Transfer)
- Arm 2: Case B Single-Stage YOLO11s-SS2D (Direct Fine-Tuning)
- Arm 3: Case B Two-Stage YOLO11s-SS2D (Warmup + Fine-Tuning)

Reads `results.csv` from runs/detect/runs/two_stage_protocol_study/ and generates:
1. `paper/figures/fig_convergence_map.pdf` (mAP50 and mAP50-95 curves)
2. `paper/figures/fig_convergence_loss.pdf` (Box loss and Cls loss curves)
3. Prints a clean Markdown summary table ready to be copied into the paper.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sized for a single IEEE column (3.5 in) so that text stays >= 7 pt in print.
# TrueType font embedding (fonttype 42) avoids Type 3 fonts, which IEEE PDF eXpress rejects.
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300
})

BASE_DIR = os.environ.get("RUNS_DIR", "runs/detect/runs/two_stage_protocol_study")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "paper/figures")
# FIG_LAYOUT=wide: side-by-side panels for the 12.2 cm LNCS text width
WIDE = os.environ.get("FIG_LAYOUT") == "wide"

def find_arm_csv(pattern):
    matches = glob.glob(os.path.join(BASE_DIR, pattern, "results.csv"))
    if not matches:
        # Search recursively in runs/ in case path differs slightly
        matches = glob.glob(f"runs/**/{pattern}/results.csv", recursive=True)
    return matches[0] if matches else None

def load_clean_df(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df

def plot_convergence():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_arm1 = find_arm_csv("arm1_baseline_control")
    csv_arm2 = find_arm_csv("arm2_case_b_single_stage")
    csv_arm3_s1 = find_arm_csv("arm3_stage1_warmup")
    csv_arm3_s2 = find_arm_csv("arm3_stage2_finetune")

    csv_arm4 = find_arm_csv("arm4_case_c_dual_scale")
    if csv_arm4 is None and os.path.exists("results/tables/arm4_case_c_dual_results.csv"):
        csv_arm4 = "results/tables/arm4_case_c_dual_results.csv"

    # Combine arm 3 stages if both exist
    df_arm1 = load_clean_df(csv_arm1)
    df_arm2 = load_clean_df(csv_arm2)
    df_arm3_s1 = load_clean_df(csv_arm3_s1)
    df_arm3_s2 = load_clean_df(csv_arm3_s2)
    df_arm4 = load_clean_df(csv_arm4)

    df_arm3 = None
    if df_arm3_s1 is not None and df_arm3_s2 is not None:
        # Re-index epochs sequentially for Stage 2
        df_arm3_s2_shifted = df_arm3_s2.copy()
        offset = len(df_arm3_s1)
        if 'epoch' in df_arm3_s2_shifted.columns:
            df_arm3_s2_shifted['epoch'] += offset
        df_arm3 = pd.concat([df_arm3_s1, df_arm3_s2_shifted], ignore_index=True)
    elif df_arm3_s2 is not None:
        df_arm3 = df_arm3_s2
    elif df_arm3_s1 is not None:
        df_arm3 = df_arm3_s1

    available_data = {}
    if df_arm1 is not None: available_data['Arm 1: baseline'] = (df_arm1, '#2b5c8f', '-')
    if df_arm2 is not None: available_data['Arm 2: SS2D, single-stage'] = (df_arm2, '#d95f02', '--')
    if df_arm3 is not None: available_data['Arm 3: SS2D, two-stage'] = (df_arm3, '#1b9e77', '-.')
    if df_arm4 is not None: available_data['Arm 4: SS2D, dual-scale'] = (df_arm4, '#7570b3', ':')

    if not available_data:
        print(f"[!] No results.csv found under {BASE_DIR} yet.")
        print(f"    Expected paths:")
        print(f"    - {BASE_DIR}/arm1_baseline_control/results.csv")
        print(f"    - {BASE_DIR}/arm2_case_b_single_stage/results.csv")
        print(f"    - {BASE_DIR}/arm3_stage2_finetune/results.csv")
        return

    print(f"\nLoaded {len(available_data)} experimental arms:")
    for name in available_data.keys():
        print(f"  * {name}")

    # --- Plot 1: Detection Accuracy (mAP50 & mAP50-95) ---
    if WIDE:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.75, 2.35))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.2), sharex=True)

    for label, (df, color, ls) in available_data.items():
        epochs = df['epoch'] if 'epoch' in df.columns else np.arange(1, len(df) + 1)
        
        map50 = df.get('metrics/mAP50(B)', None)
        if map50 is not None:
            ax1.plot(epochs, map50 * 100.0, label=label, color=color, linestyle=ls, linewidth=2.0)

        map50_95 = df.get('metrics/mAP50-95(B)', None)
        if map50_95 is not None:
            ax2.plot(epochs, map50_95 * 100.0, label=label, color=color, linestyle=ls, linewidth=2.0)

    ax1.set_ylabel(r'(a) mAP$_{50}$ (%)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(r'(b) mAP$_{50-95}$ (%)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    for ax in ((ax1, ax2) if WIDE else (ax2,)):
        ax.set_xticks(range(2, 21, 2) if WIDE else range(1, 21))
        ax.set_xlim(0.5, 20.5)
        ax.set_xlabel('Epoch')
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.84 if WIDE else 0.9))
    map_fig_path = os.path.join(OUTPUT_DIR, "fig_convergence_map.pdf")
    plt.savefig(map_fig_path, format="pdf", bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_convergence_map.png"), format="png", bbox_inches="tight")
    print(f"\n[+] Saved mAP convergence figure: {map_fig_path}")

    # --- Plot 2: Training Loss Dynamics (Box Loss & Cls Loss) ---
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for label, (df, color, ls) in available_data.items():
        epochs = df['epoch'] if 'epoch' in df.columns else np.arange(1, len(df) + 1)

        box_loss = df.get('train/box_loss', None)
        if box_loss is not None:
            ax3.plot(epochs, box_loss, label=label, color=color, linestyle=ls, linewidth=2.0)

        cls_loss = df.get('train/cls_loss', None)
        if cls_loss is not None:
            ax4.plot(epochs, cls_loss, label=label, color=color, linestyle=ls, linewidth=2.0)

    ax3.set_title('(a) Bounding Box Regression Loss (train/box_loss)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Box Loss')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper right')

    ax4.set_title('(b) Classification Loss (train/cls_loss)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Cls Loss')
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend(loc='upper right')

    plt.tight_layout()
    loss_fig_path = os.path.join(OUTPUT_DIR, "fig_convergence_loss.pdf")
    plt.savefig(loss_fig_path, format="pdf", bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_convergence_loss.png"), format="png", bbox_inches="tight")
    print(f"[+] Saved Loss convergence figure: {loss_fig_path}")

    # --- Print Summary Table ---
    print("\n" + "=" * 70)
    print("FINAL CONVERGENCE SUMMARY TABLE")
    print("=" * 70)
    summary_rows = []
    for label, (df, _, _) in available_data.items():
        last = df.iloc[-1]
        summary_rows.append({
            "Experimental Arm": label,
            "Total Epochs": len(df),
            "mAP50 (%)": f"{last.get('metrics/mAP50(B)', 0.0) * 100:.2f}%",
            "mAP50-95 (%)": f"{last.get('metrics/mAP50-95(B)', 0.0) * 100:.2f}%",
            "Final Box Loss": f"{last.get('train/box_loss', 0.0):.4f}",
            "Final Cls Loss": f"{last.get('train/cls_loss', 0.0):.4f}"
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("=" * 70 + "\n")

if __name__ == "__main__":
    plot_convergence()
