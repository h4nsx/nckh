#!/usr/bin/env python3
"""
SmartSpaces Micro-Subset Pipeline:
Downloads, samples keyframes, converts annotations to YOLO format,
and creates the data configuration file.
"""

import os
import sys
import json
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
from huggingface_hub import hf_hub_download

REPO_ID = "nvidia/PhysicalAI-SmartSpaces"
REPO_TYPE = "dataset"

# Unified warehouse taxonomy mapping
# SmartSpaces uses: 'Person', 'Forklift', 'Transporter', 'NovaCarter'
CLASS_MAP = {
    "Person": 0,       # worker
    "Forklift": 1,     # forklift
    "Transporter": 2,  # pallet_truck / transporter
    "NovaCarter": 3    # amr / agv
}

CLASS_NAMES = ["worker", "forklift", "pallet_truck", "amr"]

def build_subset(
    split_name: str,
    scene_id: str,
    camera_ids: List[str],
    subsample_interval: int, # in frames (e.g. 30 = 1 FPS for 30 FPS video)
    max_frames_per_camera: int,
    output_dir: Path
):
    print(f"\n{'='*60}")
    print(f"PROCESSING SPLIT: {split_name} (Scene: {scene_id})")
    print(f"{'='*60}")

    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download ground_truth.json
    gt_repo_path = f"MTMC_Tracking_2025/{split_name}/{scene_id}/ground_truth.json"
    print(f"[HF] Downloading {gt_repo_path}...")
    local_gt_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=gt_repo_path,
        local_dir=str(output_dir / ".cache")
    )

    with open(local_gt_path, "r") as f:
        gt_data = json.load(f)
    print(f"[INFO] Loaded ground truth: {len(gt_data)} frames indexed.")

    # 2. Process each camera stream
    total_saved = 0
    for cam_id in camera_ids:
        video_repo_path = f"MTMC_Tracking_2025/{split_name}/{scene_id}/videos/{cam_id}.mp4"
        print(f"\n[HF] Downloading {video_repo_path}...")
        local_video_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=video_repo_path,
            local_dir=str(output_dir / ".cache")
        )

        cap = cv2.VideoCapture(local_video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[OpenCV] Video opened: {frame_count} frames, {orig_w}x{orig_h}")

        saved_for_cam = 0
        frame_idx = 0
        while cap.isOpened() and saved_for_cam < max_frames_per_camera:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % subsample_interval == 0:
                frame_str = str(frame_idx)
                # Check annotations for this frame and camera
                yolo_boxes = []
                if frame_str in gt_data:
                    for obj in gt_data[frame_str]:
                        obj_type = obj.get("object type")
                        if obj_type not in CLASS_MAP:
                            continue
                        vis_dict = obj.get("2d bounding box visible", {})
                        if cam_id in vis_dict:
                            coords = vis_dict[cam_id]
                            # coords format: [x1, y1, x2, y2]
                            x1, y1, x2, y2 = coords
                            # Clip to image boundaries
                            x1 = max(0, min(orig_w - 1, x1))
                            x2 = max(0, min(orig_w - 1, x2))
                            y1 = max(0, min(orig_h - 1, y1))
                            y2 = max(0, min(orig_h - 1, y2))

                            bw = x2 - x1
                            bh = y2 - y1
                            if bw <= 4 or bh <= 4:
                                continue # skip degenerate tiny noise boxes

                            # Convert to normalized YOLO format [xc, yc, w, h]
                            xc = (x1 + bw / 2.0) / orig_w
                            yc = (y1 + bh / 2.0) / orig_h
                            wn = bw / orig_w
                            hn = bh / orig_h

                            cls_id = CLASS_MAP[obj_type]
                            yolo_boxes.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

                # Save frame only if it has positive detections (or keep 5% background)
                if len(yolo_boxes) > 0 or (saved_for_cam % 20 == 0):
                    # Resize to 640x640 for training efficiency
                    resized_frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)

                    base_name = f"{scene_id}_{cam_id}_f{frame_idx:06d}"
                    img_path = images_dir / f"{base_name}.jpg"
                    lbl_path = labels_dir / f"{base_name}.txt"

                    cv2.imwrite(str(img_path), resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    with open(lbl_path, "w") as lf:
                        lf.write("\n".join(yolo_boxes) + "\n" if yolo_boxes else "")

                    saved_for_cam += 1
                    total_saved += 1

            frame_idx += 1

        cap.release()
        print(f"[INFO] {cam_id}: Extracted and annotated {saved_for_cam} frames.")

        # Cleanup cached MP4 to protect laptop disk space
        if os.path.exists(local_video_path):
            os.remove(local_video_path)
            print(f"[CLEANUP] Removed temporary MP4: {os.path.basename(local_video_path)} to free disk.")

    print(f"[DONE] Total {split_name} frames: {total_saved}")
    return total_saved


def main():
    root_data = Path("/home/hans/Documents/nckh/data/smartspaces_micro")
    root_data.mkdir(parents=True, exist_ok=True)

    # Train: Warehouse_000, 2 cameras, 1 frame every 60 frames (every 2s), max 150 frames/camera -> ~300 frames
    train_count = build_subset(
        split_name="train",
        scene_id="Warehouse_000",
        camera_ids=["Camera_0000", "Camera_0001"],
        subsample_interval=60,
        max_frames_per_camera=150,
        output_dir=root_data
    )

    # Val: Warehouse_015, 2 cameras, 1 frame every 60 frames, max 50 frames/camera -> ~100 frames
    val_count = build_subset(
        split_name="val",
        scene_id="Warehouse_015",
        camera_ids=["Camera.mp4".replace(".mp4", ""), "Camera_01"],
        subsample_interval=60,
        max_frames_per_camera=50,
        output_dir=root_data
    )

    # Generate data.yaml
    yaml_content = f"""# NVIDIA PhysicalAI-SmartSpaces Micro-Benchmark Dataset
path: {root_data.resolve()}
train: images/train
val: images/val

# Class Taxonomy
names:
  0: worker
  1: forklift
  2: pallet_truck
  3: amr
"""
    yaml_path = root_data / "smartspaces_micro.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  - Train frames: {train_count}")
    print(f"  - Val frames:   {val_count}")
    print(f"  - Dataset YAML: {yaml_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
