import json

with open(r"d:\nckh\Reports\deep_analysis.json", encoding="utf-8") as f:
    data = json.load(f)

core_keys = [
    r"Related work\MambaNeXt-YOLO A Hybrid State Space Model for Real-time Object.pdf",
    r"CNN - Mamba\A hybrid architecture based on structured state space sequence model and.pdf",
    r"CNN - Mamba\UAVDet A CNN–Mamba hybrid network for efficient small object detection.pdf",
    r"Related work\LEAF-YOLO Lightweight Edge-Real-Time Small Object Detection on Aerial.pdf",
    r"Related work\Edge-YOLO Lightweight Infrared Object Detection Method Deployed on Edge Devices.pdf",
    r"Related work\A comprehensive survey of deep learning-based lightweight object detection models for edge devices.pdf",
    r"Yolo\Ultralytics YOLO Evolution An Overview of YOLO27, YOLO26, YOLO11, YOLOv8, and YOLOv5 Object Detectors for Computer Vision and Pattern Recognition.pdf"
]

with open(r"d:\nckh\Reports\core_papers_summary.txt", "w", encoding="utf-8") as out:
    for k in core_keys:
        if k in data:
            out.write("==================================================\n")
            out.write(f"FILE: {k} ({data[k]['pages']} pages)\n")
            out.write("--- INTRO / ABSTRACT ---\n")
            out.write(data[k]["intro_preview"][:1200] + "\n")
            out.write("--- CONCLUSION ---\n")
            out.write(data[k]["concl_preview"][:800] + "\n\n")

print("Saved core papers summary.")
