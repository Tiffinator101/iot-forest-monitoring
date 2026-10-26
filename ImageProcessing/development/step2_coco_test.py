# Step 2: run a standard COCO-trained YOLO11 model on one tree photo.
# Purpose: learn the Ultralytics workflow and check whether a normal
# pretrained model detects the tree at all (COCO has no "tree" class).

import os
from pathlib import Path
from ultralytics import YOLO

# Settings
TEST_DIR = Path(r"C:\Users\tiffa\OneDrive\Documents\University\Engineering\Year 4"
                r"\Semester 2\Project 448\Project\Testing\Tree Photo Test\Analysis Images")
IMAGE_PATH = TEST_DIR / "Position 5 - 10m" / "TreeCam1_2026-10-04_18-35-20.jpg"
CONF_VALUES = [0.25, 0.05]   # default threshold, then a low one to show weak guesses
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "step2"

# Load the nano model (weights download automatically on first run)
# The model files, including YOLOE's text encoder (mobileclip*.ts), are in ../models, and
# Ultralytics looks for them in the working folder
os.chdir(Path(__file__).resolve().parent.parent / "models")
model = YOLO("yolo11n.pt")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for conf in CONF_VALUES:
    print(f"\n===== conf = {conf} =====")
    r = model.predict(source=str(IMAGE_PATH), conf=conf)[0]

    print(f"Boxes found: {len(r.boxes)}")
    for box in r.boxes:
        name = model.names[int(box.cls[0])]
        score = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"  {name:15s} {score:.2f}   x1={x1:6.0f} y1={y1:6.0f} x2={x2:6.0f} y2={y2:6.0f}")

    r.save(filename=str(OUTPUT_DIR / f"step2_conf{conf}.jpg"))
