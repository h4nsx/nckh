#!/usr/bin/env python3
"""
Generate publication-quality charts for the LaTeX paper:
1. Architectural footprint & Layer 10 parameter comparison (Bar chart)
2. Edge device deployment: Memory footprint & latency profile across execution providers
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style for academic paper
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.autolayout': True,
    'savefig.dpi': 300
})

def generate_architectural_comparison_chart():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))

    # Subplot 1: Total Parameters & Layer 10 Parameters
    models = ['Baseline\nYOLO11s', 'Ours\nYOLO11s-SS2D']
    total_params = [9.414, 9.005] # Millions
    l10_params = [0.991, 0.579]    # Millions (990,976 vs 578,816)

    x = np.arange(len(models))
    width = 0.35

    rects1 = ax1.bar(x - width/2, total_params, width, label='Total Params (M)', color='#2b5c8f', edgecolor='black', alpha=0.9)
    rects2 = ax1.bar(x + width/2, l10_params, width, label='Layer 10 Params (M)', color='#d95f02', edgecolor='black', alpha=0.9)

    ax1.set_ylabel('Parameters (Millions)')
    ax1.set_title('(a) Parameter Footprint Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0, 11)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Add labels on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f'{height:.3f}M',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    for rect in rects2:
        height = rect.get_height()
        pct_diff = "-41.6%" if height < 0.8 else "C2PSA"
        ax1.annotate(f'{height:.3f}M\n({pct_diff})',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # Subplot 2: Computational Complexity (GFLOPs) & Checkpoint Size (MB)
    gflops = [21.4, 21.0]
    weights_mb = [19.2, 18.3]

    rects3 = ax2.bar(x - width/2, gflops, width, label='Complexity (GFLOPs)', color='#7570b3', edgecolor='black', alpha=0.9)
    rects4 = ax2.bar(x + width/2, weights_mb, width, label='Weights Size (MB)', color='#1b9e77', edgecolor='black', alpha=0.9)

    ax2.set_ylabel('GFLOPs / Megabytes')
    ax2.set_title('(b) Complexity and Storage Size')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylim(0, 26)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    for rect in rects3:
        height = rect.get_height()
        ax2.annotate(f'{height:.1f}G',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    for rect in rects4:
        height = rect.get_height()
        ax2.annotate(f'{height:.1f}MB',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    output_path = "paper/figures/fig_architectural_comparison.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.savefig("paper/figures/fig_architectural_comparison.png", format="png", bbox_inches="tight")
    print(f"Generated: {output_path}")

def generate_edge_deployment_chart():
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    categories = ['Host CPU\n(PyTorch)', 'Host CPU\n(ONNX Runtime)', 'Edge Budget Limit\n(1.0 GB Target)']
    rss_values = [420.0, 405.2, 1024.0] # Peak RSS in MB
    colors = ['#4575b4', '#74add1', '#d73027']

    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, rss_values, align='center', color=colors, edgecolor='black', height=0.5, alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Peak Physical Process Memory RSS (MB)')
    ax.set_title('Inference Memory RSS vs. Constrained Edge Budget')
    ax.set_xlim(0, 1150)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Add vertical line for 1GB boundary
    ax.axvline(1024, color='#d73027', linestyle='--', linewidth=1.5, alpha=0.7)

    # Value annotations
    for bar in bars:
        width = bar.get_width()
        label = f"{width:.1f} MB"
        if width == 1024.0:
            label += " (Target Ceiling)"
        elif width == 405.2:
            label += " (-60.4% under ceiling)"
        ax.annotate(label,
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontweight='bold', fontsize=9.5)

    plt.tight_layout()
    output_path = "paper/figures/fig_edge_memory_footprint.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.savefig("paper/figures/fig_edge_memory_footprint.png", format="png", bbox_inches="tight")
    print(f"Generated: {output_path}")

if __name__ == '__main__':
    generate_architectural_comparison_chart()
    generate_edge_deployment_chart()
