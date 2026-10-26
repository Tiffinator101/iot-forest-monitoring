# Step 4b: does changing the image before YOLOE make the trunk base consistent?
# Each variant is run on all photos with the prompt "tree". For each photo the
# bottom of the combined tree mask is recorded. The 4 photos at one position have
# identical framing, so a good variant gives the same bottom in all 4 (small spread)
# and that bottom should sit at the trunk base, not at the bottom of the crown.

import csv
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLOE

# Settings
TEST_DIR = Path(r"C:\Users\tiffa\OneDrive\Documents\University\Engineering\Year 4"
                r"\Semester 2\Project 448\Project\Testing\Tree Photo Test\Analysis Images")
CONF = 0.25
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "step4b"

# The model files, including YOLOE's text encoder (mobileclip*.ts), are in ../models, and
# Ultralytics looks for them in the working folder
os.chdir(Path(__file__).resolve().parent.parent / "models")
model = YOLOE("yoloe-11s-seg.pt")
model.set_classes(["tree"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Image changes (each takes and returns a BGR image) ----------

def grayscale(img):
    """Grey copy, stacked back to 3 channels because YOLO expects colour input."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def white_balance(img):
    """Grey-world white balance: scale each channel so its mean equals the overall mean."""
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    f *= means.mean() / means
    return np.clip(f, 0, 255).astype(np.uint8)


def clahe(img):
    """Local contrast boost on the lightness channel only, so colours are kept."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def gaussian_blur(img):
    """Gaussian blur (workshop Part 3): weighted average of neighbours, smooths noise."""
    return cv2.GaussianBlur(img, (5, 5), 0)


def median_blur(img):
    """Median blur (workshop Part 3): each pixel becomes the median of its neighbours."""
    return cv2.medianBlur(img, 5)


def bilateral(img):
    """Bilateral filter (workshop Part 3): smooths flat areas but keeps edges sharp."""
    return cv2.bilateralFilter(img, 9, 75, 75)


# ---------- Running the model ----------

def mask_extent(img, imgsz=640):
    """Run YOLOE and return (top, bottom) of all tree masks combined, or None."""
    r = model.predict(source=img, conf=CONF, imgsz=imgsz, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return None
    mask = r.masks.data.cpu().numpy().any(axis=0)
    rows = np.where(mask.any(axis=1))[0]
    return int(rows.min()), int(rows.max())


def crop_around_tree(img):
    """Two passes: find the crown on the full photo, then crop a region around it
    that reaches down towards the ground, run again, and shift the result back."""
    r = model.predict(source=img, conf=CONF, verbose=False)[0]
    if len(r.boxes) == 0:
        return None
    x1, y1, x2, y2 = r.boxes.xyxy[int(r.boxes.conf.argmax())].tolist()
    w, h = x2 - x1, y2 - y1
    cx1 = int(max(0, x1 - 0.25 * w))
    cx2 = int(min(img.shape[1], x2 + 0.25 * w))
    cy1 = int(max(0, y1 - 0.1 * h))
    cy2 = int(min(img.shape[0], y2 + 1.0 * h))      # one crown height below the box
    ext = mask_extent(img[cy1:cy2, cx1:cx2])
    if ext is None:
        return None
    return ext[0] + cy1, ext[1] + cy1


VARIANTS = {
    "original":      lambda img: mask_extent(img),
    "grayscale":     lambda img: mask_extent(grayscale(img)),
    "white_balance": lambda img: mask_extent(white_balance(img)),
    "clahe":         lambda img: mask_extent(clahe(img)),
    "gaussian":      lambda img: mask_extent(gaussian_blur(img)),
    "median":        lambda img: mask_extent(median_blur(img)),
    "bilateral":     lambda img: mask_extent(bilateral(img)),
    "imgsz_1280":    lambda img: mask_extent(img, imgsz=1280),
    "crop":          lambda img: crop_around_tree(img),
}

# Save one example of each image change so you can see what the model was given
example = cv2.imread(str(sorted(TEST_DIR.glob("Position 5*/*.jpg"))[0]))
for name, fn in (("grayscale", grayscale), ("white_balance", white_balance), ("clahe", clahe),
                 ("gaussian", gaussian_blur), ("median", median_blur), ("bilateral", bilateral)):
    cv2.imwrite(str(OUTPUT_DIR / f"example_{name}.jpg"), fn(example))

rows = []
for image_path in sorted(TEST_DIR.glob("*/*.jpg")):
    img = cv2.imread(str(image_path))
    row = {"position": image_path.parent.name, "image": image_path.name}
    for name, run in VARIANTS.items():
        ext = run(img)
        row[f"{name}_top"] = ext[0] if ext else ""
        row[f"{name}_bottom"] = ext[1] if ext else ""
    rows.append(row)
    print(f"{row['position']:20s} {image_path.stem[-8:]}  bottoms: " +
          "  ".join(f"{n}={row[f'{n}_bottom']}" for n in VARIANTS))

with open(OUTPUT_DIR / "preprocessing_bottoms.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("\nSpread of the mask bottom across the 4 photos at each position (px, smaller is better)")
print(f"{'position':20s}" + "".join(f"{n:>15s}" for n in VARIANTS))
for position in sorted({r["position"] for r in rows}):
    line = f"{position:20s}"
    for name in VARIANTS:
        b = [r[f"{name}_bottom"] for r in rows if r["position"] == position and r[f"{name}_bottom"] != ""]
        line += f"{(max(b) - min(b)) if len(b) == 4 else 'missing':>15}"
    print(line)
