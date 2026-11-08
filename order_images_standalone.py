"""
Order Original Microscopy Images Chronologically by Time & Date from Filename
Stores them directly into a clean, dedicated separate folder 'ordered_images'.
"""

import os
import re
import shutil
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
OUTPUT_DIR = os.path.join(BASE_DIR, "ordered_images")

def parse_filename_time(fn: str):
    """
    Extracts time (days, hours, minutes) and condition from filename.
    Example: 'SHSY5Y_Phase_B10_1_01d04h00m_2.png'
    """
    time_match = re.search(r'(\d+)d(\d+)h(\d+)m', fn, re.IGNORECASE)
    if time_match:
        d, h, m = map(int, time_match.groups())
        total_minutes = (d * 1440) + (h * 60) + m
    else:
        d, h, m, total_minutes = 0, 0, 0, 0
        
    well_match = re.search(r'Phase_([A-Za-z0-9]+_\d+)', fn, re.IGNORECASE)
    well = well_match.group(1).upper() if well_match else "UNKNOWN"
    
    slice_match = re.search(r'_(\d+)\.png', fn, re.IGNORECASE)
    fov = int(slice_match.group(1)) if slice_match else 1
    
    return {
        "days": d,
        "hours": h,
        "minutes": m,
        "total_minutes": total_minutes,
        "well": well,
        "fov": fov,
        "time_str": f"{d:02d}d{h:02d}h{m:02d}m",
        "time_label": f"Day {d}, {h:02d}:{m:02d} ({d*24 + h}h)"
    }

def main():
    print("=" * 60)
    print(" Ordering Original Microscopy Images by Date & Time")
    print("=" * 60)

    # Clean destination
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_chrono_dir = os.path.join(OUTPUT_DIR, "all_chronological")
    by_series_dir = os.path.join(OUTPUT_DIR, "by_cell_series")
    os.makedirs(all_chrono_dir, exist_ok=True)
    os.makedirs(by_series_dir, exist_ok=True)

    # 1. Collect ONLY unique original raw micrographs (ignore processed/masks to avoid duplicates)
    original_images = {}
    
    for root, _, files in os.walk(ARCHIVE_DIR):
        # Look for original images folders first
        is_original_folder = "original" in root.lower()
        for f in files:
            if f.lower().endswith(".png") and not "_mask" in f.lower():
                # Prefer files inside 'original' folders
                full_path = os.path.join(root, f)
                if f not in original_images or is_original_folder:
                    original_images[f] = full_path

    print(f"Found {len(original_images)} unique raw microscopy images.")

    # 2. Parse time for each file
    parsed_items = []
    for fn, src_path in original_images.items():
        meta = parse_filename_time(fn)
        meta["filename"] = fn
        meta["src_path"] = src_path
        parsed_items.append(meta)

    # 3. Sort chronologically by:
    # 1st: Total Elapsed Minutes (0m, 240m, 480m, 720m, 960m, 1200m, 1440m...)
    # 2nd: Well Condition (B10_1, B10_2, C10_1, C10_2, D10_1, D10_2)
    # 3rd: FOV Slice (1, 2, 3, 4)
    parsed_items.sort(key=lambda x: (x["total_minutes"], x["well"], x["fov"], x["filename"]))

    # 4. Copy into 'all_chronological' with clear ordering prefix
    manifest = []
    for idx, item in enumerate(parsed_items, start=1):
        # Sequential filename: e.g. '001_Day00_00h00m_B10_1_FOV1_SHSY5Y_Phase_B10_1_00d00h00m_1.png'
        chrono_name = f"{idx:04d}_Day{item['days']:02d}_{item['hours']:02d}h{item['minutes']:02d}m_{item['well']}_FOV{item['fov']}_{item['filename']}"
        dest_path = os.path.join(all_chrono_dir, chrono_name)
        shutil.copy2(item["src_path"], dest_path)

        # Also store into by_cell_series subfolder
        well_dir = os.path.join(by_series_dir, f"{item['well']}_FOV{item['fov']}")
        os.makedirs(well_dir, exist_ok=True)
        shutil.copy2(item["src_path"], os.path.join(well_dir, f"{item['time_str']}_{item['filename']}"))

        manifest.append({
            "order_number": idx,
            "days": item["days"],
            "hours": item["hours"],
            "minutes": item["minutes"],
            "total_hours": item["days"] * 24 + item["hours"],
            "total_minutes": item["total_minutes"],
            "time_label": item["time_label"],
            "cell_condition": item["well"],
            "fov_slice": item["fov"],
            "chronological_filename": chrono_name,
            "original_filename": item["filename"]
        })

    # 5. Write clean summary index
    csv_file = os.path.join(OUTPUT_DIR, "images_chronological_order.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "order_number", "days", "hours", "minutes", "total_hours",
            "total_minutes", "time_label", "cell_condition", "fov_slice",
            "chronological_filename", "original_filename"
        ])
        writer.writeheader()
        writer.writerows(manifest)

    # 6. Readme in output folder
    with open(os.path.join(OUTPUT_DIR, "README.txt"), "w", encoding="utf-8") as f:
        f.write(f"""CHRONOLOGICALLY ORDERED MICROSCOPY IMAGES
==========================================
Total Images: {len(manifest)}
Ordered strictly by the date and time in the filename:
dd (days), hh (hours), mm (minutes).

Folders:
1. 'all_chronological/': All {len(manifest)} images in a single folder, prefixed 0001_, 0002_...
   Sorted from Day 0, 00:00 (0h) to Day 3, 12:00 (84h).
2. 'by_cell_series/': Grouped by cell well and FOV slice (B10_1_FOV1, B10_2_FOV2, etc.).
3. 'images_chronological_order.csv': Complete list of images with exact timeline order.
""")

    print(f"\nSuccessfully stored {len(manifest)} images into '{OUTPUT_DIR}'")
    print(f"Folder 1 (Flat Chronological): {all_chrono_dir}")
    print(f"Folder 2 (Grouped by Series):  {by_series_dir}")
    print(f"CSV Summary Table:             {csv_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
