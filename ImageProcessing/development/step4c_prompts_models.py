# Step 4c: does a different text prompt or a bigger YOLOE model find the whole tree
# (canopy AND trunk) consistently?
# For every model + prompt pair, each photo is run and the bottom of the combined
# tree mask is recorded. Success means: small spread between the 4 identical photos
# at a position, AND the bottom sits at the trunk base at every position.

import csv
import os
from pathlib import Path
import numpy as np
from ultralytics import YOLOE

# Settings
TEST_DIR = Path(r"C:\Users\tiffa\OneDrive\Documents\University\Engineering\Year 4"
                r"\Semester 2\Project 448\Project\Testing\Tree Photo Test\Analysis Images")
CONF = 0.25
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "step4c"

# The model files, including YOLOE's text encoder (mobileclip*.ts), are in ../models, and
# Ultralytics looks for them in the working folder
os.chdir(Path(__file__).resolve().parent.parent / "models")
MODELS = ["yoloe-11s-seg.pt", "yoloe-11l-seg.pt", "yoloe-26s-seg.pt", "yoloe-26l-seg.pt"]
PROMPTS = ["tree", "whole tree with trunk", "tree including its trunk", "full tree from ground to top"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
images = sorted(TEST_DIR.glob("*/*.jpg"))
positions = sorted({p.parent.name for p in images})

rows = []
summary = []
for model_name in MODELS:
    for prompt in PROMPTS:
        model = YOLOE(model_name)
        model.set_classes([prompt])
        bottoms = {pos: [] for pos in positions}
        for image_path in images:
            r = model.predict(source=str(image_path), conf=CONF, retina_masks=True, verbose=False)[0]
            if r.masks is None:
                bottom = None
            else:
                mask = r.masks.data.cpu().numpy().any(axis=0)
                bottom = int(np.where(mask.any(axis=1))[0].max())
                bottoms[image_path.parent.name].append(bottom)
            rows.append({"model": model_name, "prompt": prompt, "position": image_path.parent.name,
                         "image": image_path.name, "mask_bottom": bottom if bottom is not None else ""})

        # One summary line per model + prompt: mean bottom and spread at each position
        cells = []
        for pos in positions:
            b = bottoms[pos]
            if len(b) < 4:
                cells.append(f"{len(b)}/4 found")
            else:
                cells.append(f"{np.mean(b):5.0f} ±{max(b) - min(b):3d}")
        summary.append((model_name, prompt, cells))
        print(f"{model_name:18s} {prompt:30s} " + " | ".join(cells))

with open(OUTPUT_DIR / "prompt_model_bottoms.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "prompt", "position", "image", "mask_bottom"])
    writer.writeheader()
    writer.writerows(rows)

print("\nMean mask bottom ± spread (px) per position: " + " | ".join(positions))
for model_name, prompt, cells in summary:
    print(f"{model_name:18s} {prompt:30s} " + " | ".join(cells))
