"""
Organize Microscopy Images by Well Condition (B10_1, B10_2, C10_1, C10_2, D10_1, D10_2)
------------------------------------------------------------------------------------------
Collects ALL FOV images for each well series, sorts them chronologically by the
timestamp embedded in the filename (ddDhhHmmM), and stores each well's images
into its own dedicated folder under 'sorted_by_well/'.

This is separate from the project — stored outside the main app directories.

Output structure:
  sorted_by_well/
    B10_1/   <- all FOV1+FOV2+FOV3+FOV4 images for well B10_1, sorted by time
    B10_2/
    C10_1/
    C10_2/
    D10_1/
    D10_2/
    summary.csv
    README.txt
"""

import os
import re
import shutil
import csv

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
OUTPUT_DIR  = os.path.join(BASE_DIR, "sorted_by_well")


def parse_filename(fn):
    """
    Parse time and well from filename like:
      SHSY5Y_Phase_B10_1_01d04h00m_2.png
    Returns None if the filename doesn't match the expected pattern.
    """
    well_match = re.search(r'Phase_([A-Za-z]\d+_\d+)', fn, re.IGNORECASE)
    if not well_match:
        return None
    well = well_match.group(1).upper()

    time_match = re.search(r'(\d+)d(\d+)h(\d+)m', fn, re.IGNORECASE)
    if not time_match:
        return None
    d, h, m = map(int, time_match.groups())
    total_minutes = d * 1440 + h * 60 + m

    fov_match = re.search(r'_(\d+)\.png$', fn, re.IGNORECASE)
    fov = int(fov_match.group(1)) if fov_match else 0

    return {
        "well":          well,
        "days":          d,
        "hours":         h,
        "minutes":       m,
        "total_minutes": total_minutes,
        "fov":           fov,
        "time_str":      f"{d:02d}d{h:02d}h{m:02d}m",
        "filename":      fn,
    }


def collect_images_from_archive(archive_root):
    """Walk archive, collect unique original microscopy PNGs."""
    seen = {}

    for root, _, files in os.walk(archive_root):
        is_original_folder = "original" in root.lower()
        for fn in files:
            if not fn.lower().endswith(".png"):
                continue
            if "_mask" in fn.lower():
                continue

            meta = parse_filename(fn)
            if meta is None:
                continue

            full_path = os.path.join(root, fn)
            meta["src_path"] = full_path

            if fn not in seen or is_original_folder:
                seen[fn] = meta

    return list(seen.values())


def main():
    print("=" * 65)
    print("  Sorting Microscopy Images by Well  (B10_1 to D10_2)")
    print("=" * 65)

    if os.path.exists(OUTPUT_DIR):
        print(f"\nRemoving existing '{OUTPUT_DIR}' ...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    print(f"Created fresh output directory: {OUTPUT_DIR}\n")

    all_images = collect_images_from_archive(ARCHIVE_DIR)
    print(f"Found {len(all_images)} unique original microscopy images.\n")

    if not all_images:
        print("ERROR: No images found. Check 'archive' directory.")
        return

    wells = {}
    for img in all_images:
        wells.setdefault(img["well"], []).append(img)

    manifest_rows = []

    for well_name in sorted(wells.keys()):
        images = wells[well_name]
        images.sort(key=lambda x: (x["total_minutes"], x["fov"], x["filename"]))

        well_dir = os.path.join(OUTPUT_DIR, well_name)
        os.makedirs(well_dir, exist_ok=True)

        print(f"  Well {well_name:8s}  ->  {len(images):3d} images  ->  {well_dir}")

        for idx, img in enumerate(images, start=1):
            new_name = f"{idx:04d}_{img['time_str']}_FOV{img['fov']}_{img['filename']}"
            dest = os.path.join(well_dir, new_name)
            shutil.copy2(img["src_path"], dest)

            manifest_rows.append({
                "well":              well_name,
                "order_in_well":     idx,
                "days":              img["days"],
                "hours":             img["hours"],
                "minutes":           img["minutes"],
                "total_minutes":     img["total_minutes"],
                "fov":               img["fov"],
                "sorted_filename":   new_name,
                "original_filename": img["filename"],
            })

    csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "well", "order_in_well", "days", "hours", "minutes",
            "total_minutes", "fov", "sorted_filename", "original_filename"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\n  Summary CSV -> {csv_path}")

    well_summary_lines = "\n".join(
        f"  {w}: {len(wells[w])} images" for w in sorted(wells.keys())
    )
    readme_path = os.path.join(OUTPUT_DIR, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"""MICROSCOPY IMAGES SORTED BY WELL CONDITION
==========================================
Total Images : {len(manifest_rows)}
Total Wells  : {len(wells)}

Each well folder contains images from ALL FOV slices (1-4),
sorted strictly by the timestamp in the filename (ddDhhHmmM).
Within the same timestamp, images are further sorted by FOV slice number.

Well Counts:
{well_summary_lines}

File naming inside each folder:
  <order>_<timestamp>_FOV<n>_<original_filename>.png
  e.g. 0001_00d00h00m_FOV1_SHSY5Y_Phase_B10_1_00d00h00m_1.png

This folder is standalone and NOT part of the main application.
""")

    print(f"  README      -> {readme_path}")
    print("\n" + "=" * 65)
    print(f"  Done! {len(manifest_rows)} images sorted into {len(wells)} well folders.")
    print("=" * 65)


if __name__ == "__main__":
    main()
