# Step 4: compare two ways of getting a consistent tree height from YOLOE.
#   Method A: prompt "tree", measure the top and bottom of the segmentation mask
#             (all tree masks in the photo combined), not the box.
#   Method B: prompt "tree" and "tree trunk", take the top of the best tree box
#             and the bottom of the best trunk box.
# The 4 photos at each position have identical framing, so a good method gives
# (almost) the same pixel height for all 4. Results go to a CSV plus annotated images.

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
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "step4"

# Two copies of the model, one per prompt list
# The model files, including YOLOE's text encoder (mobileclip*.ts), are in ../models, and
# Ultralytics looks for them in the working folder
os.chdir(Path(__file__).resolve().parent.parent / "models")
model_a = YOLOE("yoloe-11s-seg.pt")
model_a.set_classes(["tree"])
model_b = YOLOE("yoloe-11s-seg.pt")
model_b.set_classes(["tree", "tree trunk"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def method_a(image_path):
    """Combine every tree mask into one and return its extent in pixels."""
    # retina_masks=True gives masks at the full photo resolution
    r = model_a.predict(source=str(image_path), conf=CONF, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return None, r
    mask = r.masks.data.cpu().numpy().any(axis=0)       # True where any tree mask covers the pixel
    rows = np.where(mask.any(axis=1))[0]                 # image rows that contain tree pixels
    cols = np.where(mask.any(axis=0))[0]                 # image columns that contain tree pixels
    return {"top": int(rows.min()), "bottom": int(rows.max()),
            "left": int(cols.min()), "right": int(cols.max()), "mask": mask}, r


def method_b(image_path):
    """Top of the best tree box and bottom of the best trunk box."""
    r = model_b.predict(source=str(image_path), conf=CONF, verbose=False)[0]
    best = {}
    for box in r.boxes:
        name = model_b.names[int(box.cls[0])]
        score = float(box.conf[0])
        if name not in best or score > best[name][0]:
            best[name] = (score, box.xyxy[0].tolist())
    if "tree" not in best or "tree trunk" not in best:
        return None, r, best
    tree_top = best["tree"][1][1]
    trunk_bottom = best["tree trunk"][1][3]
    return {"top": round(tree_top), "bottom": round(trunk_bottom)}, r, best


def save_method_a_image(image_path, a, out_path):
    """Original photo with the combined mask shaded and lines at its top and bottom."""
    img = cv2.imread(str(image_path))
    overlay = img.copy()
    overlay[a["mask"]] = (255, 0, 0)
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    for y in (a["top"], a["bottom"]):
        cv2.line(img, (0, y), (img.shape[1], y), (0, 0, 255), 3)
    cv2.imwrite(str(out_path), img)


rows = []
for image_path in sorted(TEST_DIR.glob("*/*.jpg")):
    position = image_path.parent.name
    tag = f"{position} - {image_path.stem}"

    a, ra = method_a(image_path)
    b, rb, best_b = method_b(image_path)

    row = {"position": position, "image": image_path.name}
    if a:
        row.update(a_top=a["top"], a_bottom=a["bottom"], a_height=a["bottom"] - a["top"],
                   a_width=a["right"] - a["left"], a_masks=len(ra.masks))
        save_method_a_image(image_path, a, OUTPUT_DIR / f"{tag} - A mask.jpg")
    if b:
        row.update(b_top=b["top"], b_bottom=b["bottom"], b_height=b["bottom"] - b["top"],
                   b_tree_conf=round(best_b["tree"][0], 2), b_trunk_conf=round(best_b["tree trunk"][0], 2))
    rb.save(filename=str(OUTPUT_DIR / f"{tag} - B boxes.jpg"))
    rows.append(row)

    print(f"{tag}:  A height={row.get('a_height', '-'):>5}  "
          f"(top {row.get('a_top', '-')}, bottom {row.get('a_bottom', '-')}, {row.get('a_masks', 0)} mask(s))   "
          f"B height={row.get('b_height', '-'):>5}  "
          f"(top {row.get('b_top', '-')}, bottom {row.get('b_bottom', '-')}, "
          f"found {sorted(best_b)})")

fields = ["position", "image", "a_top", "a_bottom", "a_height", "a_width", "a_masks",
          "b_top", "b_bottom", "b_height", "b_tree_conf", "b_trunk_conf"]
with open(OUTPUT_DIR / "height_methods.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

# Spread of the 4 heights at each position: smaller is better (same framing in all 4)
print("\nPer position: mean height and spread (max - min) in px")
for position in sorted({r["position"] for r in rows}):
    for m in ("a", "b"):
        h = [r[f"{m}_height"] for r in rows if r["position"] == position and f"{m}_height" in r]
        if h:
            print(f"  {position:20s} method {m.upper()}: mean {np.mean(h):6.0f}  spread {max(h) - min(h):4d}  (n={len(h)})")
