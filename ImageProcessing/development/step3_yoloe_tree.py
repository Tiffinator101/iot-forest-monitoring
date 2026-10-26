# Step 3: detect the tree with an open-vocabulary model (YOLOE).
# YOLOE takes a text prompt as its class list, so it can look for "tree"
# without any training. Runs on all photos in Analysis Images and saves
# an annotated copy of each plus a CSV of the best tree box per photo.

import csv
import os
from pathlib import Path
from ultralytics import YOLOE

# Settings
TEST_DIR = Path(r"C:\Users\tiffa\OneDrive\Documents\University\Engineering\Year 4"
                r"\Semester 2\Project 448\Project\Testing\Tree Photo Test\Analysis Images")
CLASSES = ["tree"]           # text prompt(s) the model looks for
CONF = 0.25
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "step3"

# Load the small YOLOE model (YOLO11 based) and set its classes from the text prompt
# The model files, including YOLOE's text encoder (mobileclip*.ts), are in ../models, and
# Ultralytics looks for them in the working folder
os.chdir(Path(__file__).resolve().parent.parent / "models")
model = YOLOE("yoloe-11s-seg.pt")
model.set_classes(CLASSES)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
for image_path in sorted(TEST_DIR.glob("*/*.jpg")):
    position = image_path.parent.name
    r = model.predict(source=str(image_path), conf=CONF, verbose=False)[0]

    print(f"\n{position} / {image_path.name}: {len(r.boxes)} box(es)")
    for box in r.boxes:
        name = model.names[int(box.cls[0])]
        score = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"  {name:6s} {score:.2f}   x1={x1:6.0f} y1={y1:6.0f} x2={x2:6.0f} y2={y2:6.0f}"
              f"   w={x2 - x1:5.0f} h={y2 - y1:5.0f}")

    # Keep the most confident box as "the tree" for the summary
    if len(r.boxes) > 0:
        best = int(r.boxes.conf.argmax())
        x1, y1, x2, y2 = r.boxes.xyxy[best].tolist()
        rows.append([position, image_path.name, len(r.boxes), round(float(r.boxes.conf[best]), 3),
                     round(x1), round(y1), round(x2), round(y2), round(x2 - x1), round(y2 - y1)])
    else:
        rows.append([position, image_path.name, 0, "", "", "", "", "", "", ""])

    r.save(filename=str(OUTPUT_DIR / f"{position} - {image_path.name}"))

with open(OUTPUT_DIR / "tree_boxes.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["position", "image", "boxes_found", "best_conf",
                     "x1", "y1", "x2", "y2", "width_px", "height_px"])
    writer.writerows(rows)
print(f"\nSaved annotated images and tree_boxes.csv to {OUTPUT_DIR}")
