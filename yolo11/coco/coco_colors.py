# yolo11/coco/coco_colors.py

# Palette centralisée (BGR pour OpenCV)
COCO_COLORS = {
    "joint": (0, 180, 255),   # orange-ish (B,G,R)
    "bone":  (0, 120, 255),   # darker orange
    "box":   (0, 200, 0),     # green
    "mask":  (0, 0, 255),     # red
    "label_bg": (0, 0, 0),    # background for label text
    "label_txt": (255, 255, 255)  # white text
}
