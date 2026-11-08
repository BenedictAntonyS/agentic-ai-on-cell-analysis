"""
Select images at positions 1, 4, 8, 12, 16, ... from each sorted well folder.
──────────────────────────────────────────────────────────────────────────────
Source : sorted_by_well/<WELL>/   (88 images per well, 4 FOVs x 22 timepoints)
Output : sorted_multiples_of_4/<WELL>/

Selection rule:
  - Image at position  1          (FOV1 of timepoint 0)
  - Images at positions 4, 8, 12, 16, 20, 24, ... (multiples of 4)
  → picks the last FOV of every timepoint block, plus the very first image.

Per well with 88 images: positions 1, 4, 8, 12, ..., 88  → 23 images each.
"""

import os
import shutil
import csv

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "sorted_by_well")
OUTPUT_DIR = os.path.join(BASE_DIR, "sorted_multiples_of_4")


def get_selected_positions(total: int) -> list[int]:
    """
    Returns 1-based positions: [1, 4, 8, 12, 16, ..., last_multiple_of_4]
    """
    positions = [1]                          # always include image #1
    positions += list(range(4, total + 1, 4))  # multiples of 4
    # Remove duplicates (if total < 4, pos 1 may already be covered)
    return sorted(set(positions))


def main():
    print("=" * 65)
    print("  Extracting positions 1, 4, 8, 12, ... for every well class")
    print("=" * 65)

    # ── Clean output directory ─────────────────────────────────────
    if os.path.exists(OUTPUT_DIR):
        print(f"\nRemoving existing '{OUTPUT_DIR}' ...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    print(f"Created fresh output directory: {OUTPUT_DIR}\n")

    wells = sorted([
        d for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
    ])

    if not wells:
        print("ERROR: No well folders found in sorted_by_well/")
        return

    manifest = []

    for well in wells:
        src_well_dir = os.path.join(SOURCE_DIR, well)
        # Get sorted list of image files only
        all_files = sorted([
            f for f in os.listdir(src_well_dir)
            if f.lower().endswith(".png")
        ])

        total = len(all_files)
        positions = get_selected_positions(total)

        out_well_dir = os.path.join(OUTPUT_DIR, well)
        os.makedirs(out_well_dir, exist_ok=True)

        print(f"  Well {well:8s}  |  total={total:3d}  |  selected={len(positions):2d}  "
              f"|  positions: 1, 4, 8, ... (multiples of 4)")

        for rank, pos in enumerate(positions, start=1):
            src_file = all_files[pos - 1]          # convert 1-based -> 0-based
            # Rename with a clean sequential prefix
            dest_name = f"{rank:04d}_pos{pos:04d}_{src_file}"
            shutil.copy2(
                os.path.join(src_well_dir, src_file),
                os.path.join(out_well_dir, dest_name)
            )
            manifest.append({
                "well":            well,
                "rank_in_output":  rank,
                "original_pos":    pos,
                "source_filename": src_file,
                "output_filename": dest_name,
            })

    # ── Summary CSV ────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "well", "rank_in_output", "original_pos",
            "source_filename", "output_filename"
        ])
        writer.writeheader()
        writer.writerows(manifest)
    print(f"\n  Summary CSV → {csv_path}")

    # ── README ─────────────────────────────────────────────────────
    well_counts = {}
    for row in manifest:
        well_counts[row["well"]] = well_counts.get(row["well"], 0) + 1

    readme_path = os.path.join(OUTPUT_DIR, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"""MICROSCOPY IMAGES — MULTIPLES-OF-4 SELECTION
=============================================
Source      : sorted_by_well/
Selection   : positions 1, 4, 8, 12, 16, ... (image #1 + every multiple of 4)
Total output: {len(manifest)} images across {len(well_counts)} wells

Per-well counts:
""")
        for w in sorted(well_counts):
            f.write(f"  {w}: {well_counts[w]} images\n")
        f.write("""
File naming:
  <rank>_pos<original_position>_<original_sorted_filename>
  e.g. 0001_pos0001_0001_00d00h00m_FOV1_SHSY5Y_Phase_B10_1_00d00h00m_1.png
       0002_pos0004_0004_00d00h00m_FOV4_SHSY5Y_Phase_B10_1_00d00h00m_4.png
       0003_pos0008_0008_00d04h00m_FOV4_SHSY5Y_Phase_B10_1_00d04h00m_4.png

This is a standalone folder — NOT part of the main application.
""")

    print(f"  README      → {readme_path}")
    print("\n" + "=" * 65)
    print(f"  Done! {len(manifest)} images stored into {len(well_counts)} well folders.")
    print("=" * 65)


if __name__ == "__main__":
    main()
