#!/usr/bin/env python3
"""
Qualitative Inference Comparison Generator:
Generates publication-ready comparative visualization figures (PDF/PNG)
contrasting predictions from:
- Ground Truth (Annotated labels)
- Arm 1: Baseline YOLO11s (COCO Transfer)
- Arm 2: Case B YOLO11s-SS2D (Single-Stage)
- Arm 3: Case B YOLO11s-SS2D (Two-Stage Protocol)

Outputs:
  paper/figures/fig_qualitative_comparison.pdf
  paper/figures/fig_qualitative_comparison.png
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure modules are registered
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from models.warehouse_ss2d import WarehouseSS2DBlock

tasks.WarehouseSS2DBlock = WarehouseSS2DBlock
modules.WarehouseSS2DBlock = WarehouseSS2DBlock
from ultralytics import YOLO

# Palette definition (RGB for matplotlib)
# Class 0: worker (Blue), Class 1: forklift (Amber), Class 2: amr (Green), Class 3: pallet_truck (Purple)
CLASS_NAMES = ["worker", "forklift", "amr", "pallet_truck"]
CLASS_COLORS = {
    0: (0.12, 0.47, 0.71),  # Blue
    1: (0.89, 0.45, 0.13),  # Orange/Amber
    2: (0.17, 0.63, 0.17),  # Green
    3: (0.58, 0.40, 0.74),  # Purple
}

ARM1_PATH = "runs/detect/runs/two_stage_protocol_study/arm1_baseline_control/weights/best.pt"
ARM2_PATH = "runs/detect/runs/two_stage_protocol_study/arm2_case_b_single_stage/weights/best.pt"
ARM3_PATH = "runs/detect/runs/two_stage_protocol_study/arm3_stage2_finetune/weights/best.pt"

# Select 2 distinct complex warehouse test scenes
SELECTED_IMAGES = [
    "data/smartspaces_micro/images/val/Warehouse_015_Camera_01_f000360.jpg",
    "data/smartspaces_micro/images/val/Warehouse_015_Camera_01_f000840.jpg",
]

def load_ground_truth(img_path, orig_w, orig_h):
    label_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    x1 = (xc - w / 2) * orig_w
                    y1 = (yc - h / 2) * orig_h
                    x2 = (xc + w / 2) * orig_w
                    y2 = (yc + h / 2) * orig_h
                    boxes.append((cls_id, x1, y1, x2, y2, 1.0))
    return boxes

def get_predictions(model, img_path, conf_thresh=0.25):
    results = model.predict(img_path, conf=conf_thresh, iou=0.45, verbose=False)[0]
    preds = []
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].cpu().numpy()
        preds.append((cls_id, xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf))
    return preds

def render_scene(ax, img_rgb, boxes, title, is_gt=False):
    ax.imshow(img_rgb)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.axis("off")

    for cls_id, x1, y1, x2, y2, conf in boxes:
        color = CLASS_COLORS.get(cls_id, (1.0, 1.0, 0.0))
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle(
            (x1, y1), w, h, linewidth=1.5, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        # Only render labels for key machinery or high-confidence person detections to prevent overlapping clutter
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        # Prominently label non-workers (forklift, amr, pallet_truck) or top-confidence workers
        if cls_id != 0 or conf >= 0.70 or is_gt:
            lbl_str = f"{cls_name}" if is_gt else f"{cls_name} {conf:.2f}"
            ax.text(
                x1, max(y1 - 4, 12),
                lbl_str,
                fontsize=8,
                color="white",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.9, edgecolor="none")
            )

def generate_comparison_figure():
    print("[1/3] Loading trained models...")
    m1 = YOLO(ARM1_PATH)
    m2 = YOLO(ARM2_PATH)
    m3 = YOLO(ARM3_PATH)

    os.makedirs("paper/figures", exist_ok=True)
    num_scenes = len(SELECTED_IMAGES)
    
    # 2 scenes x 4 columns (Ground Truth, Baseline YOLO11s, Case B Single-Stage, Case B Two-Stage)
    fig, axes = plt.subplots(num_scenes, 4, figsize=(16, 4.2 * num_scenes), dpi=300)

    print("[2/3] Running inference on benchmark test scenes...")
    for row_idx, img_p in enumerate(SELECTED_IMAGES):
        img_bgr = cv2.imread(img_p)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        gt_boxes = load_ground_truth(img_p, orig_w, orig_h)
        p1_boxes = get_predictions(m1, img_p)
        p2_boxes = get_predictions(m2, img_p)
        p3_boxes = get_predictions(m3, img_p)

        scene_name = Path(img_p).stem

        # Column 1: Ground Truth
        render_scene(
            axes[row_idx, 0], img_rgb, gt_boxes,
            f"Ground Truth ({scene_name})\nTotal Objects: {len(gt_boxes)}",
            is_gt=True
        )

        # Column 2: Arm 1 (Baseline YOLO11s)
        render_scene(
            axes[row_idx, 1], img_rgb, p1_boxes,
            f"(a) Baseline YOLO11s Control\nDetected: {len(p1_boxes)} | Recall: {len(p1_boxes)/max(1, len(gt_boxes))*100:.1f}%"
        )

        # Column 3: Arm 2 (Case B Single-Stage)
        render_scene(
            axes[row_idx, 2], img_rgb, p2_boxes,
            f"(b) Case B SS2D (Single-Stage)\nDetected: {len(p2_boxes)} | Recall: {len(p2_boxes)/max(1, len(gt_boxes))*100:.1f}%"
        )

        # Column 4: Arm 3 (Case B Two-Stage)
        render_scene(
            axes[row_idx, 3], img_rgb, p3_boxes,
            f"(c) Case B SS2D (Two-Stage Protocol)\nDetected: {len(p3_boxes)} | Recall: {len(p3_boxes)/max(1, len(gt_boxes))*100:.1f}%"
        )

    plt.tight_layout()

    out_pdf = "paper/figures/fig_qualitative_comparison.pdf"
    out_png = "paper/figures/fig_qualitative_comparison.png"
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(out_png, format="png", bbox_inches="tight")
    print(f"[3/3] Generated qualitative figures:")
    print(f"      - {out_pdf}")
    print(f"      - {out_png}")

if __name__ == "__main__":
    generate_comparison_figure()
