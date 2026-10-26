# Tree Measurement app: pick an example or upload your own tree photos, give the camera-to-tree
# distance, and get the tree height, crown width, trunk width, crown length and height to crown base,
# with the annotated photos.
#
# Two screens on one page:
#   start screen   - example photos (one click measures them) or your own photos and their distance
#   results screen - per-photo table, summary (mean, smallest and largest, measured values, % error),
#                    result photos, CSV. "Measure all distances" shows one row per distance instead, with
#                    the % error at each distance and the lowest and highest % error.
#                    "Save to database" saves the per-photo results to the base station's tree_measurements
#                    table, only for photos already in its images table, and only when you press it.
#                    "Save result photos" opens a Windows folder window and copies all the run's result
#                    photos into the folder you choose.
#
# Double-click "Start Tree Measurement.bat", or run from this folder:  .venv\Scripts\python.exe app.py
# It opens in the browser at http://127.0.0.1:7860 and runs only on this computer.
# The measuring itself is in tree_measure.py.

import csv
import shutil
import sys
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
import cv2
import gradio as gr
from tree_measure import measure, load_model, MeasureError, F_PX, WIDTH_HEIGHT_M

# The base station's database.py, which owns database.db and its tables
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "BaseStation"))
import database

# Example photos: copies of the 2026-10-04 photo test, one folder per position named "... - <distance>m".
# Each entry is (distance in m, the photos at that distance), farthest first.
EXAMPLES = sorted(((float(d.name.split(" - ")[1].rstrip("m")), sorted(str(p) for p in d.glob("*.jpg")))
                   for d in (Path(__file__).parent / "examples").iterdir() if d.is_dir()), reverse=True)

RESULT_ORDER = ["Tree height (m)", "Crown width (m)", "Trunk width (mm)", "Crown length (m)",
                "Height to crown base (m)"]
PIXEL_ORDER = ["Tree height (px)", "Crown width (px)", "Trunk width (px)", "Crown length (px)",
               "Height to crown base (px)", "Pixel size at the tree (mm)"]

# Measured values of the test tree, used as the defaults for the comparison
REF_HEIGHT_M = 4.0              # tape
REF_CROWN_M = 3.5              # tape
REF_TRUNK_MM = 121.0            # calliper in the camera's direction (133.7 mm from the 420 mm circumference)
REF_CROWN_BASE_M = 1.1        # ground to where the lowest leaves start: bottom of 1.1 to 1.2 m, top of the 1.05 to 1.10 m leaf reading


def crown_reference(ref_height, ref_crown_base):
    """Crown length to compare with: measured tree height minus measured height to crown base."""
    return round(ref_height - ref_crown_base, 2) if ref_height and ref_crown_base else None

# The summary table gets an outlined, tinted box so it stands apart from the per-photo table, and the
# example photos fill their boxes (no dark bands above and below) and show a hand cursor
CSS = """
.summary-box { border: 2px solid var(--color-accent); border-radius: 10px; padding: 10px;
               background: var(--color-accent-soft); }
.example-photo img { cursor: pointer; width: 100% !important; height: 100% !important; object-fit: cover; }
.save-row { align-items: center !important; }
.save-row .prose p { margin: 0; }
"""


def save_result(out, photo_path, folder):
    """Save an annotated photo as <photo name>_measured.jpg, so a downloaded result keeps its photo's name."""
    path = Path(folder) / f"{Path(photo_path).stem}_measured.jpg"
    cv2.imwrite(str(path), out)
    return str(path)


def percent_error(value, ref):
    """Measured from the photo against the measured value, in percent."""
    return f"{100 * (value / ref - 1):+.1f}%" if ref else ""


def measure_photos(paths, distance, width_height, folder, label=""):
    """Measure each photo. Returns the result photos, per-photo rows, pixel rows, values, problems and
    the records for the database (photo file name, its values and the height the trunk width was measured at)."""
    gallery, per_photo, pixels, values, problems, records = [], [], [], [], [], []
    for path in paths:
        name = label + Path(path).name
        img = cv2.imread(path)
        if img is None:
            problems.append(f"{name}: not an image.")
            continue
        try:
            results, out, warnings = measure(img, float(distance), float(width_height))
        except MeasureError as e:
            problems.append(f"{name}: {e}")
            continue
        gallery.append((save_result(out, path, folder), name))
        per_photo.append([name] + [results[k] for k in RESULT_ORDER])
        pixels.append([name] + [results[k] for k in PIXEL_ORDER])
        values.append([results[k] for k in RESULT_ORDER])
        records.append((Path(path).name, [results[k] for k in RESULT_ORDER] + [float(width_height)]))
        problems += [f"{name}: {w}" for w in warnings]
    return gallery, per_photo, pixels, values, problems, records


def tree_id_from_name(file_name):
    """Tree ID of the camera that took the photo, or None. The camera ID is the start of the photo's name
    (TreeCam1_2026-10-04_18-21-49.jpg), and database.py says which tree each camera is at."""
    return database.CAMERA_TREE_IDS.get(file_name.split("_")[0])


def save_to_database(records):
    """Save each measured photo to tree_measurements. Only photos already in the images table are saved,
    and a photo that is already saved is skipped. The width height is saved with each photo, so trunk widths
    measured at different heights can be told apart."""
    if not records:
        return "**Nothing to save.**"
    saved, already, not_in_db, no_tree, failed = [], [], [], [], []
    for file_name, vals in records:
        image_id = database.image_id_lookup(file_name)
        tree_id = tree_id_from_name(file_name)
        if image_id is None:
            not_in_db.append(file_name)
        elif tree_id is None:
            no_tree.append(file_name)
        else:
            row = database.insert_tree_measurement_record(image_id, tree_id, *vals)
            (already if row == -1 else failed if row is None else saved).append(file_name)
    lines = [f"**Saved to the database:** {len(saved)} of {len(records)} photo{'s' if len(records) > 1 else ''}."]
    for title, names in (("Already in the database, not saved again", already),
                         ("Not in the camera node database, not saved", not_in_db),
                         ("No tree set for this camera in database.py, not saved", no_tree),
                         ("Could not be saved (see the console)", failed)):
        if names:
            lines.append(f"**{title}:** " + ", ".join(names))
    return "\n\n".join(lines)


# Where the folder window opens: the photo test's folder for result photos, one folder per position
ANALYSED_DIR = Path(__file__).resolve().parents[2] / "Testing" / "Tree Photo Test" / "Analysed Images"


def save_photos(paths):
    """Ask for a folder with the Windows folder window and copy the run's result photos into it.
    This works because the app runs on this computer, so the window opens on its desktop."""
    if not paths:
        return "**No result photos to save.**"
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)                           # keep the window in front of the browser
    folder = filedialog.askdirectory(parent=root, title="Choose a folder for the result photos",
                                     initialdir=str(ANALYSED_DIR) if ANALYSED_DIR.is_dir() else None)
    root.destroy()
    if not folder:
        return "**No folder chosen, nothing saved.**"
    for path in paths:
        shutil.copy(path, Path(folder) / Path(path).name)
    return f"**Saved {len(paths)} result photo{'s' if len(paths) > 1 else ''} to** {folder}"


def column_stat(values, stat):
    """Mean, smallest or largest of each measurement, rounded like the measurements themselves."""
    return [round(stat(c), 1 if "mm" in key else 2) for c, key in zip(zip(*values), RESULT_ORDER)]


def mean(c):
    return sum(c) / len(c)


def references(ref_height, ref_crown, ref_trunk, ref_crown_base):
    """The measured values in RESULT_ORDER; crown length is worked out from the height and the height to crown base."""
    return [ref_height, ref_crown, ref_trunk, crown_reference(ref_height, ref_crown_base), ref_crown_base]


def write_csv(folder, header, rows, summary, photo_rows, settings_row):
    """CSV with the main table, the summary, the per-photo rows (all distances only) and the settings."""
    path = Path(folder) / "tree_measurements.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
        writer.writerow([])
        writer.writerows(summary)
        if photo_rows:
            writer.writerow([])
            writer.writerow(["Photo"] + RESULT_ORDER)
            writer.writerows(photo_rows)
        writer.writerow([])
        writer.writerow(settings_row)
    return str(path)


def stay(text):
    """Stay on the start screen with a message."""
    return [gr.update(visible=True), gr.update(visible=False), text] + [gr.update()] * (len(OUTPUTS) - 3)


def show(info, problems, heading, table, summary, pixels, csv_path, gallery, records):
    """Switch to the results screen and fill it. heading None leaves out the first table, which would
    only repeat the summary for a single photo. records are kept for the Save to database button, and the
    result photo paths for the Save result photos button."""
    message = "\n\n".join(f"**Warning.** {p}" for p in problems)
    return [gr.update(visible=False), gr.update(visible=True), "", info, message,
            gr.update(visible=heading is not None), heading or "", table, summary, pixels, csv_path,
            gr.update(value=gallery, columns=min(2, len(gallery))), records, "", [p for p, _ in gallery], ""]


def run(paths, distance, width_height, compare, ref_height, ref_crown, ref_trunk, ref_crown_base, source):
    """Photos at one distance: per-photo table, and a summary with the mean, the smallest and largest
    values and their % errors."""
    if not paths:
        return stay("**Upload one or more photos first.**")
    if not distance:
        return stay("**Enter the distance from the camera to the trunk.**")
    folder = tempfile.mkdtemp()
    gallery, per_photo, pixels, values, problems, records = measure_photos(paths, distance, width_height, folder)
    if not values:
        return stay("**No photo could be measured.**\n\n" + "\n\n".join(problems))

    several = len(values) > 1
    rows = [("Mean of the photos" if several else "Result", column_stat(values, mean))]
    if several:
        rows += [("Smallest", column_stat(values, min)), ("Largest", column_stat(values, max))]
    summary = [[name] + vals for name, vals in rows]
    if compare:
        refs = references(ref_height, ref_crown, ref_trunk, ref_crown_base)
        summary.append(["Measured"] + [r if r else "" for r in refs])
        labels = {"Mean of the photos": "% error (mean)", "Result": "% error",
                  "Smallest": "% error (smallest)", "Largest": "% error (largest)"}
        for name, vals in rows:
            summary.append([labels[name]] + [percent_error(v, r) for v, r in zip(vals, refs)])

    csv_path = write_csv(folder, ["Photo"] + RESULT_ORDER, per_photo, summary, None,
                         ["Distance (m)", distance, "Width height (m)", width_height])
    info = (f"**{source}** · {len(values)} of {len(paths)} photo{'s' if len(paths) > 1 else ''} measured · "
            f"distance {float(distance):g} m · trunk width measured {float(width_height):g} m above the trunk base")
    return show(info, problems, "### Measurements per photo" if several else None,
                gr.update(value=per_photo, headers=["Photo"] + RESULT_ORDER), summary, pixels, csv_path, gallery,
                records)


def run_all(width_height, compare, ref_height, ref_crown, ref_trunk, ref_crown_base, progress=gr.Progress()):
    """All example distances: one row per distance (the mean of its photos), and a summary with the
    % error at each distance and the lowest and highest % error."""
    folder = tempfile.mkdtemp()
    gallery, photo_rows, pixels, problems, per_distance, means_by_distance, records = [], [], [], [], [], [], []
    for i, (distance, photos) in enumerate(EXAMPLES):
        progress(i / len(EXAMPLES), desc=f"Measuring the {distance:g} m photos")
        g, per_photo, px, values, p, r = measure_photos(photos, distance, width_height, folder, f"{distance:g} m · ")
        gallery += g
        photo_rows += per_photo
        pixels += px
        problems += p
        records += r
        if values:
            means = column_stat(values, mean)
            per_distance.append([f"{distance:g} m"] + means)
            means_by_distance.append((distance, means))
    if not means_by_distance:
        return stay("**No photo could be measured.**\n\n" + "\n\n".join(problems))

    if compare:
        refs = references(ref_height, ref_crown, ref_trunk, ref_crown_base)
        summary = [["Measured"] + [r if r else "" for r in refs]]
        errors = []                                           # % error per distance, per measurement
        for distance, means in means_by_distance:
            row = [100 * (m / r - 1) if r else None for m, r in zip(means, refs)]
            errors.append(row)
            summary.append([f"% error at {distance:g} m"] + [f"{e:+.1f}%" if e is not None else "" for e in row])
        columns = list(zip(*errors))
        summary.append(["Lowest % error"] + [f"{min(c):+.1f}%" if None not in c else "" for c in columns])
        summary.append(["Highest % error"] + [f"{max(c):+.1f}%" if None not in c else "" for c in columns])
    else:
        values = [means for _, means in means_by_distance]
        summary = [["Smallest"] + column_stat(values, min), ["Largest"] + column_stat(values, max)]

    csv_path = write_csv(folder, ["Distance"] + RESULT_ORDER, per_distance, summary, photo_rows,
                         ["Width height (m)", width_height])
    info = (f"**Example: all distances** · {len(photo_rows)} photos at {len(means_by_distance)} distances "
            f"measured · trunk width measured {float(width_height):g} m above the trunk base")
    return show(info, problems, "### Measurements per distance (mean of the photos at each distance)",
                gr.update(value=per_distance, headers=["Distance"] + RESULT_ORDER), summary, pixels, csv_path,
                gallery, records)


def example_handler(i):
    """Click handler for example i: all its photos at its distance. Each example is its own image
    because an image reports every click, while a gallery ignores a click on the photo it has
    already selected (so the same example couldn't be measured twice in a row)."""
    distance, photos = EXAMPLES[i]

    def run_example(width_height, compare, ref_height, ref_crown, ref_trunk, ref_crown_base):
        return run(photos, distance, width_height, compare, ref_height, ref_crown, ref_trunk, ref_crown_base,
                   f"Example: photo test at {distance:g} m")
    return run_example


def run_own(files, distance, width_height, compare, ref_height, ref_crown, ref_trunk, ref_crown_base):
    return run(files, distance, width_height, compare, ref_height, ref_crown, ref_trunk, ref_crown_base,
               "Your photos")


with gr.Blocks(title="Tree Measurement", analytics_enabled=False, css=CSS) as demo:
    gr.Markdown("# Tree Measurement\n"
                "Measures a tree's height, crown width, trunk width, crown length (tree top to crown base) and "
                "height to crown base (trunk base to crown base) from a camera node photo. The crown is found with YOLOE and the trunk with classical image processing, and "
                f"pixels are turned into metres with the pinhole model (focal length {F_PX} px, calibrated "
                "with the reference stick).")

    # Start screen
    with gr.Column(visible=True) as start:
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Try an example\nPhotos from the photo test, 4 at each distance. "
                            "Click one to measure it.")
                with gr.Row():
                    example_photos = [gr.Image(photos[0], label=f"{distance:g} m", interactive=False,
                                               show_download_button=False, show_fullscreen_button=False,
                                               show_share_button=False, min_width=80, height=150,
                                               elem_classes="example-photo")
                                      for distance, photos in EXAMPLES]
                all_button = gr.Button(f"Measure all {len(EXAMPLES)} distances "
                                       f"({sum(len(p) for _, p in EXAMPLES)} photos)", variant="secondary")
                gr.Markdown("### Or use your own photos\nOne photo or several, all taken at the same distance.")
                files_in = gr.File(show_label=False, file_count="multiple", file_types=["image"], type="filepath",
                                   height=140)
                distance_in = gr.Number(label="Distance from camera to trunk (m)", value=None)
                measure_button = gr.Button("Measure", variant="primary")
                start_message = gr.Markdown()
            with gr.Column(scale=1):
                gr.Markdown("### Settings\nUsed for the examples and your own photos.")
                width_in = gr.Number(label="Height of the trunk width above the trunk base (m)",
                                     value=WIDTH_HEIGHT_M)
                compare = gr.Checkbox(label="Compare with measured values", value=True)
                with gr.Group(visible=True) as refs_group:
                    ref_h = gr.Number(label="Measured tree height (m)", value=REF_HEIGHT_M,
                                      info="Test tree: 4.0 m (tape)")
                    ref_c = gr.Number(label="Measured crown width (m)", value=REF_CROWN_M,
                                      info="Test tree: 3.5 m (tape)")
                    ref_t = gr.Number(label="Measured trunk width (mm)", value=REF_TRUNK_MM,
                                      info="Test tree: 121 mm (calliper), or 133.7 mm from the 420 mm circumference")
                    ref_cb = gr.Number(label="Measured height to crown base (m)", value=REF_CROWN_BASE_M,
                                       info="Test tree: 1.1 m")
                    ref_cl = gr.Number(label="Crown length to compare with (m)", interactive=False,
                                       value=crown_reference(REF_HEIGHT_M, REF_CROWN_BASE_M),
                                       info="Worked out: measured tree height minus measured height to crown base")
                for box in (ref_h, ref_cb):
                    box.change(crown_reference, [ref_h, ref_cb], ref_cl)
                compare.change(lambda on: gr.update(visible=on), compare, refs_group)

    # Results screen
    with gr.Column(visible=False) as results:
        with gr.Row():
            back_button = gr.Button("← Back", size="sm", scale=0, min_width=100)
            info_out = gr.Markdown()
        message_out = gr.Markdown()
        with gr.Column() as photos_section:
            photos_heading = gr.Markdown("### Measurements per photo")
            photos_table = gr.Dataframe(headers=["Photo"] + RESULT_ORDER, interactive=False)
        with gr.Column(elem_classes="summary-box"):
            gr.Markdown("### Summary")
            summary_table = gr.Dataframe(headers=[""] + RESULT_ORDER, interactive=False)
        with gr.Accordion("Pixel values per photo", open=False):
            pixels_table = gr.Dataframe(headers=["Photo"] + PIXEL_ORDER, interactive=False)
        csv_out = gr.File(label="Results as CSV")
        with gr.Row(elem_classes="save-row"):        # note centred beside the button
            save_button = gr.Button("Save to database", variant="secondary", scale=0, min_width=180)
            gr.Markdown("Saves each photo's results to the base station database. Only photos the camera node "
                        "sent to the base station can be saved, and each photo only once.")
        save_message = gr.Markdown()
        records_state = gr.State([])
        with gr.Row(elem_classes="save-row"):
            save_photos_button = gr.Button("Save result photos", variant="secondary", scale=0, min_width=180)
            gr.Markdown("Opens a folder window (check the taskbar if it does not appear) and saves all the "
                        "result photos below into the folder you choose.")
        save_photos_message = gr.Markdown()
        photos_state = gr.State([])
        gallery_out = gr.Gallery(label="Result photos (click one, then the download button to save it)",
                                 columns=2, height="auto", show_download_button=True)

    OUTPUTS = [start, results, start_message, info_out, message_out,
               photos_section, photos_heading, photos_table, summary_table, pixels_table, csv_out, gallery_out,
               records_state, save_message, photos_state, save_photos_message]
    SETTINGS = [width_in, compare, ref_h, ref_c, ref_t, ref_cb]
    for i, photo in enumerate(example_photos):
        photo.select(example_handler(i), SETTINGS, OUTPUTS)
    all_button.click(run_all, SETTINGS, OUTPUTS)
    measure_button.click(run_own, [files_in, distance_in] + SETTINGS, OUTPUTS)
    back_button.click(lambda: (gr.update(visible=True), gr.update(visible=False)), None, [start, results])
    save_button.click(save_to_database, records_state, save_message)
    save_photos_button.click(save_photos, photos_state, save_photos_message)

if __name__ == "__main__":
    print("Loading the YOLOE model...")
    load_model()                                # load it now, so the first Measure is as quick as the rest
    demo.launch(inbrowser=True)
