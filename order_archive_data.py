"""
Standalone Utility: Chronological Ordering & Storage of Archive Image Data
Extracts date and time (days, hours, minutes) embedded in LIVECell microscopy filenames,
sorts all frames chronologically into time-series sequences, and stores them in a dedicated folder.
"""

import os
import re
import shutil
import json
import csv
from glob import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
OUTPUT_DIR = os.path.join(BASE_DIR, "ordered_archive")

def parse_livecell_filename(filename: str):
    """
    Parses metadata and timestamp from LIVECell phase-contrast filenames.
    Example: 'SHSY5Y_Phase_B10_1_00d04h00m_2.png'
    - Cell line: SHSY5Y
    - Condition/Well: B10_1
    - Days: 0
    - Hours: 4
    - Minutes: 0 -> total_minutes = 240
    - FOV / Slice: 2
    """
    fn = os.path.basename(filename)
    
    # 1. Extract Condition / Well (e.g. B10_1, B10_2, C10_1, etc.)
    well_match = re.search(r'Phase_([A-Za-z0-9]+_\d+)', fn, re.IGNORECASE)
    well_id = well_match.group(1).upper() if well_match else "UNKNOWN_SERIES"
    
    # 2. Extract Date & Time (days, hours, minutes)
    time_match = re.search(r'(\d+)d(\d+)h(\d+)m', fn, re.IGNORECASE)
    if time_match:
        days, hours, minutes = map(int, time_match.groups())
        total_minutes = (days * 1440) + (hours * 60) + minutes
    else:
        days, hours, minutes, total_minutes = 0, 0, 0, 0
        
    # 3. Extract FOV / Slice suffix (e.g. _1, _2, _3, _4)
    slice_match = re.search(r'_(\d+)(?:_mask)?\.png', fn, re.IGNORECASE)
    fov_slice = int(slice_match.group(1)) if slice_match else 1
    
    is_mask = "_mask" in fn.lower()
    
    return {
        "filename": fn,
        "well_condition": well_id,
        "fov_slice": fov_slice,
        "series_key": f"{well_id}_FOV{fov_slice}",
        "days": days,
        "hours": hours,
        "minutes": minutes,
        "total_minutes": total_minutes,
        "time_formatted": f"Day {days}, {hours:02d}:{minutes:02d} ({days*24 + hours}h total)",
        "is_mask": is_mask
    }

def process_and_order_archive():
    print("=" * 65)
    print("  Ordering LIVECell Archive Images by Filename Date & Time")
    print("=" * 65)
    
    if not os.path.exists(ARCHIVE_DIR):
        print(f"Error: Archive directory not found at: {ARCHIVE_DIR}")
        return

    # Create destination output directory structure
    images_out_dir = os.path.join(OUTPUT_DIR, "ordered_micrographs")
    masks_out_dir = os.path.join(OUTPUT_DIR, "ordered_masks")
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(masks_out_dir, exist_ok=True)

    # 1. Scan all PNG files in archive
    all_files = []
    for root, _, files in os.walk(ARCHIVE_DIR):
        for f in files:
            if f.lower().endswith(".png"):
                full_path = os.path.join(root, f)
                all_files.append(full_path)

    print(f"Discovered {len(all_files)} total images in {ARCHIVE_DIR}\n")

    # 2. Parse metadata & timestamps
    records = []
    for path in all_files:
        meta = parse_livecell_filename(path)
        meta["source_path"] = path
        # Categorize category based on folder
        rel_path = os.path.relpath(path, ARCHIVE_DIR)
        meta["archive_rel_path"] = rel_path
        records.append(meta)

    # 3. Separate Micrographs and Masks
    micrographs = [r for r in records if not r["is_mask"]]
    masks = [r for r in records if r["is_mask"]]

    print(f"Found {len(micrographs)} raw microscopy images and {len(masks)} mask images.")

    # 4. Group by Series (Condition + FOV Slice) and Sort Chronologically by Time
    def sort_and_save_group(record_list, target_base_dir, group_type_label):
        series_groups = {}
        for r in record_list:
            s_key = r["series_key"]
            series_groups.setdefault(s_key, []).append(r)

        manifest_entries = []
        total_copied = 0

        for s_key in sorted(series_groups.keys()):
            items = series_groups[s_key]
            # CHRONOLOGICAL SORT BY DATE & TIME IN FILENAME
            # Primary key: total_minutes (days * 1440 + hours * 60 + minutes)
            # Secondary key: filename
            sorted_items = sorted(items, key=lambda x: (x["total_minutes"], x["filename"]))

            # Create specific series directory
            dest_series_dir = os.path.join(target_base_dir, s_key)
            os.makedirs(dest_series_dir, exist_ok=True)

            print(f"  • [{group_type_label}] {s_key:15s}: {len(sorted_items):2d} frames ordered from {sorted_items[0]['time_formatted']} to {sorted_items[-1]['time_formatted']}")

            for seq_num, item in enumerate(sorted_items, start=1):
                # Format a sequential chronological filename:
                # e.g., 'seq_01_T0000m_00d00h00m_SHSY5Y_Phase_B10_1_00d00h00m_1.png'
                new_fn = f"step_{seq_num:03d}_time_{item['total_minutes']:05d}m_{item['days']:02d}d{item['hours']:02d}h_{item['filename']}"
                dest_path = os.path.join(dest_series_dir, new_fn)

                # Copy file preserving timestamp
                shutil.copy2(item["source_path"], dest_path)
                total_copied += 1

                manifest_entries.append({
                    "sequence_step": seq_num,
                    "series_key": s_key,
                    "condition": item["well_condition"],
                    "fov_slice": item["fov_slice"],
                    "days": item["days"],
                    "hours": item["hours"],
                    "minutes": item["minutes"],
                    "total_minutes": item["total_minutes"],
                    "time_label": item["time_formatted"],
                    "ordered_filename": new_fn,
                    "stored_path": os.path.relpath(dest_path, BASE_DIR),
                    "original_path": os.path.relpath(item["source_path"], BASE_DIR),
                    "is_mask": item["is_mask"]
                })

        return manifest_entries, total_copied

    print("\nOrganizing Micrographs into Chronological Time-Series:")
    micro_manifest, num_micro = sort_and_save_group(micrographs, images_out_dir, "Micrograph")

    print("\nOrganizing Masks into Chronological Time-Series:")
    mask_manifest, num_masks = sort_and_save_group(masks, masks_out_dir, "Mask")

    all_manifest = micro_manifest + mask_manifest

    # 5. Export Master CSV Manifest
    csv_path = os.path.join(OUTPUT_DIR, "ordered_archive_manifest.csv")
    fieldnames = [
        "series_key", "sequence_step", "days", "hours", "minutes", "total_minutes",
        "time_label", "condition", "fov_slice", "ordered_filename",
        "stored_path", "original_path", "is_mask"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_manifest:
            writer.writerow(row)

    # 6. Export Hierarchical JSON Index
    json_hierarchy = {}
    for entry in all_manifest:
        s_key = entry["series_key"]
        t_type = "masks" if entry["is_mask"] else "micrographs"
        json_hierarchy.setdefault(s_key, {}).setdefault(t_type, []).append({
            "step": entry["sequence_step"],
            "total_minutes": entry["total_minutes"],
            "days": entry["days"],
            "hours": entry["hours"],
            "time_label": entry["time_label"],
            "ordered_filename": entry["ordered_filename"],
            "file_path": entry["stored_path"]
        })

    json_path = os.path.join(OUTPUT_DIR, "ordered_archive_hierarchy.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(json_hierarchy, jf, indent=2)

    # 7. Summary Readme
    readme_path = os.path.join(OUTPUT_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as rf:
        rf.write(f"""# Chronologically Ordered Archive Dataset

This directory contains image data extracted from the `archive` directory, chronologically ordered by the date and time (`dd`d`hh`h`mm`m) embedded in the filenames.

## Summary:
- **Total Micrographs Stored:** {num_micro}
- **Total Masks Stored:** {num_masks}
- **Master CSV Manifest:** [`ordered_archive_manifest.csv`](ordered_archive_manifest.csv)
- **JSON Hierarchy:** [`ordered_archive_hierarchy.json`](ordered_archive_hierarchy.json)

## Directory Structure:
```
ordered_archive/
├── ordered_micrographs/
│   ├── B10_1_FOV1/
│   │   ├── step_001_time_00000m_00d00h_...png
│   │   ├── step_002_time_00240m_00d04h_...png
│   │   └── ...
│   ├── B10_1_FOV2/
│   └── ...
├── ordered_masks/
│   ├── B10_1_FOV1/
│   └── ...
├── ordered_archive_manifest.csv
└── ordered_archive_hierarchy.json
```

## Timestamp Parsing Scheme:
Each file format `SHSY5Y_Phase_{{Condition}}_{{dd}}d{{hh}}h{{mm}}m_{{FOV}}.png` is mapped to:
- `Total Elapsed Minutes` = `(Days * 1440) + (Hours * 60) + Minutes`
- Sequential step prefix (`step_001_`, `step_002_`, ...) guarantees that standard alphabetical file listings match true chronological biological sequence.
""")

    print("\n" + "=" * 65)
    print(f"SUCCESS: Stored {num_micro + num_masks} ordered images in '{OUTPUT_DIR}'")
    print(f"Master CSV Index: {csv_path}")
    print(f"Hierarchical JSON: {json_path}")
    print("=" * 65)

if __name__ == "__main__":
    process_and_order_archive()
