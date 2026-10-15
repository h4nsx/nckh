from pypdf import PdfReader

def get_text(path, max_p=4):
    reader = PdfReader(path)
    txt = ""
    for i in range(min(max_p, len(reader.pages))):
        txt += f"\n--- Page {i+1} ---\n" + (reader.pages[i].extract_text() or "")
    return txt

papers = {
    "Hybrid-YOLO (Chen & Er)": r"d:\nckh\Reports\CNN - Mamba\A hybrid architecture based on structured state space sequence model and.pdf",
    "UAVDet": r"d:\nckh\Reports\CNN - Mamba\UAVDet A CNN–Mamba hybrid network for efficient small object detection.pdf",
    "VSS-SpatioNet": r"d:\nckh\Reports\CNN - Mamba\VMamba\VSS-SpatioNet a multi-scale feature fusion network for multimodal image integrations.pdf",
    "Vision Mamba RS Survey": r"d:\nckh\Reports\CNN - Mamba\Vision Mamba in Remote Sensing A Comprehensive Survey of Techniques, Applications and Outlook.pdf",
    "Edge-YOLO": r"d:\nckh\Reports\Related work\Edge-YOLO Lightweight Infrared Object Detection Method Deployed on Edge Devices.pdf",
    "LEAF-YOLO": r"d:\nckh\Reports\Related work\LEAF-YOLO Lightweight Edge-Real-Time Small Object Detection on Aerial.pdf",
    "Ultralytics YOLO Evolution": r"d:\nckh\Reports\Yolo\Ultralytics YOLO Evolution An Overview of YOLO27, YOLO26, YOLO11, YOLOv8, and YOLOv5 Object Detectors for Computer Vision and Pattern Recognition.pdf"
}

with open(r"d:\nckh\Reports\key_papers_detail.txt", "w", encoding="utf-8") as out:
    for name, p in papers.items():
        out.write(f"\n=======================================================\n")
        out.write(f"=== {name} ===\n")
        out.write(f"=======================================================\n")
        try:
            txt = get_text(p)
            out.write(txt[:4000] + "\n")
        except Exception as e:
            out.write(f"Error reading: {e}\n")

print("Key papers extracted.")
