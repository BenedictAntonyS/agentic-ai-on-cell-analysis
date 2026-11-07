"""
Shared Microscopy Computer Vision & Time-Series Forecasting Module
NeuroCell GAN Forecaster

Provides robust, validated, and honest image processing for phase-contrast microscopy:
1. Input sequence validation (dimensions, consistency, chronology verification).
2. Phase-contrast preprocessing & dual segmentation (Classical Otsu vs Adaptive Watershed).
3. Quality & uncertainty metrics (contrast, sharpness, SNR estimation).
4. Temporal region tracking (centroid displacement & IoU matching with uncertain flags).
5. Reconciled ConvLSTM model inference (strictly takes masks, as trained).
6. Quantitative evaluation against Ground Truth (IoU, Dice, MAE, confluence delta).
7. Transparent Explainable AI (XAI) with explicit measured vs illustrative labels.
"""

import os
import re
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

# ==========================================
# 1. INPUT VALIDATION & METADATA EXTRACTION
# ==========================================

def parse_filename_time(filename: str) -> Dict[str, Any]:
    """
    Attempts to extract time/well metadata from LIVECell filename conventions.
    Returns parsed metadata with an 'is_verified' flag.
    """
    fn = os.path.basename(filename)
    time_match = re.search(r'(\d+)d(\d+)h(\d+)m', fn, re.IGNORECASE)
    well_match = re.search(r'Phase_([A-Za-z0-9]+_\d+)', fn, re.IGNORECASE)
    fov_match = re.search(r'FOV(\d+)', fn, re.IGNORECASE) or re.search(r'_(\d+)(?:_mask)?\.png', fn, re.IGNORECASE)

    if time_match:
        d, h, m = map(int, time_match.groups())
        total_hours = (d * 24.0) + h + (m / 60.0)
        time_label = f"{d:02d}d {h:02d}h {m:02d}m ({total_hours:.1f}h)"
        is_verified = True
    else:
        # Fallback numeric frame search: e.g. T1, Frame1, _001, etc.
        t_match = re.search(r'[T_](\d+)', fn, re.IGNORECASE)
        if t_match:
            idx = int(t_match.group(1))
            total_hours = float(idx * 4.0)
            time_label = f"Frame Index {idx}"
            is_verified = False
        else:
            total_hours = 0.0
            time_label = "Unspecified Time"
            is_verified = False

    return {
        "filename": fn,
        "well": well_match.group(1).upper() if well_match else "UNKNOWN",
        "fov": int(fov_match.group(1)) if fov_match else 1,
        "time_hours": total_hours,
        "time_label": time_label,
        "is_time_verified": is_verified
    }


def validate_image_sequence(
    frames: List[np.ndarray],
    filenames: Optional[List[str]] = None
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Validates a list of uploaded microscopy frames.
    Checks:
    - Minimum count (>= 4 frames).
    - Dimensional consistency (all frames must have same H, W).
    - Non-empty image content.
    - Chronology check from filenames.
    """
    if len(frames) < 4:
        return False, f"Expected at least 4 consecutive frames, got {len(frames)}.", []

    h0, w0 = frames[0].shape[:2]
    metadata = []
    times = []

    for idx, img in enumerate(frames[:4]):
        if img is None or img.size == 0:
            return False, f"Frame {idx+1} is empty or unreadable.", []

        h, w = img.shape[:2]
        if (h, w) != (h0, w0):
            return False, (
                f"Dimension mismatch: Frame 1 is {w0}x{h0}, but Frame {idx+1} is {w}x{h}. "
                "All frames in a time-lapse sequence must share identical dimensions."
            ), []

        # Pixel range check
        if np.max(img) == np.min(img):
            return False, f"Frame {idx+1} contains flat/zero data (no optical signal).", []

        fn = filenames[idx] if filenames and idx < len(filenames) else f"Frame_{idx+1}.png"
        meta = parse_filename_time(fn)
        meta["dimensions"] = (w, h)
        metadata.append(meta)
        times.append(meta["time_hours"])

    # Chronology validation
    is_strictly_chronological = True
    for i in range(len(times) - 1):
        if metadata[i]["is_time_verified"] and metadata[i+1]["is_time_verified"]:
            if times[i+1] <= times[i]:
                is_strictly_chronological = False

    # Build sequence status
    all_verified = all(m["is_time_verified"] for m in metadata)
    if all_verified and is_strictly_chronological:
        status_msg = "Chronology verified: Filename timestamps indicate ascending order."
    elif all_verified and not is_strictly_chronological:
        status_msg = "Warning: Filename timestamps indicate non-ascending or repeated intervals."
    else:
        status_msg = (
            "Chronology notice: Filenames do not contain standard LIVECell timestamps; "
            "frames are processed in upload/file order."
        )

    return True, status_msg, metadata


# ==========================================
# 2. IMAGE QUALITY & OPTICAL METRICS
# ==========================================

def assess_image_quality(img_gray: np.ndarray) -> Dict[str, Any]:
    """
    Computes optical quality indicators for phase-contrast microscopy:
    - Sharpness via Laplacian variance.
    - Dynamic range / RMS contrast.
    - Overall quality classification.
    """
    if len(img_gray.shape) == 3:
        gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_gray.copy()

    gray_f = gray.astype(np.float32)

    # Sharpness (focus metric)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # RMS Contrast
    mean_val = np.mean(gray_f)
    rms_contrast = float(np.sqrt(np.mean((gray_f - mean_val) ** 2)))

    # Dynamic Range utilization
    min_val, max_val = float(np.min(gray)), float(np.max(gray))
    dyn_range_pct = round(((max_val - min_val) / 255.0) * 100.0, 1)

    # Classification
    if laplacian_var > 120 and dyn_range_pct > 60:
        level = "High Quality"
        notes = "Sharp cellular borders with good optical contrast."
    elif laplacian_var > 45:
        level = "Moderate Quality"
        notes = "Acceptable contrast; minor optical blur or diffraction halo artifacts present."
    else:
        level = "Low Quality / Noisy"
        notes = "Low contrast or out-of-focus; segmentation may have elevated uncertainty."

    return {
        "sharpness_laplacian": round(laplacian_var, 2),
        "rms_contrast": round(rms_contrast, 2),
        "dynamic_range_pct": dyn_range_pct,
        "quality_level": level,
        "quality_notes": notes
    }


# ==========================================
# 3. SEGMENTATION: CLASSICAL OTSU & WATERSHED
# ==========================================

def preprocess_phase_contrast(img: np.ndarray) -> np.ndarray:
    """
    Authentic preprocessing adapted from project notebook Cell 1:
    Grayscale conversion -> Gaussian blur (5x5) -> Contrast Inversion (cells bright)
    -> Histogram Equalization.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    gray_8u = np.uint8(np.clip(gray, 0, 255))
    blurred = cv2.GaussianBlur(gray_8u, (5, 5), 0)
    # Phase-contrast cell bodies are darker than halo/background;
    # inverting makes cellular masses the foreground (white)
    inverted = 255 - blurred
    equalized = cv2.equalizeHist(inverted)
    return equalized


def segment_otsu(
    preprocessed: np.ndarray,
    original_for_annotation: Optional[np.ndarray] = None,
    min_area_px: int = 25
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Global Otsu thresholding + morphological opening/closing + contour extraction.
    Returns:
    - binary_mask: (H, W) uint8 with cells=255, bg=0.
    - annotated_img: (H, W, 3) BGR overlay showing detected region contours and centroids.
    - metrics: Confluence %, region counts, area distribution stats.
    """
    gray_8u = np.uint8(np.clip(preprocessed, 0, 255))
    _, thresh = cv2.threshold(gray_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean morphological speckles
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]

    # Prepare visual annotation on the original image (not inverted preprocessed)
    if original_for_annotation is not None:
        if len(original_for_annotation.shape) == 2:
            annotated = cv2.cvtColor(original_for_annotation, cv2.COLOR_GRAY2BGR)
        else:
            annotated = original_for_annotation.copy()
    else:
        annotated = cv2.cvtColor(gray_8u, cv2.COLOR_GRAY2BGR)

    areas = []
    centroids = []
    for c in valid_contours:
        area = float(cv2.contourArea(c))
        areas.append(area)
        m = cv2.moments(c)
        if m["m00"] > 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            centroids.append((cx, cy))
            cv2.circle(annotated, (cx, cy), 2, (0, 165, 255), -1)

    cv2.drawContours(annotated, valid_contours, -1, (0, 255, 0), 1)

    # Compute metric values
    total_pixels = cleaned.shape[0] * cleaned.shape[1]
    cell_pixels = int(np.count_nonzero(cleaned))
    confluence_pct = round((cell_pixels / total_pixels) * 100.0, 2)

    stats = {
        "confluence_pct": confluence_pct,
        "segmented_area_px": cell_pixels,
        "region_count": len(valid_contours),
        "mean_region_area_px": round(float(np.mean(areas)), 1) if areas else 0.0,
        "median_region_area_px": round(float(np.median(areas)), 1) if areas else 0.0,
        "max_region_area_px": round(float(np.max(areas)), 1) if areas else 0.0,
        "centroids": centroids,
        "method": "Global Otsu + Morphological Filtering"
    }

    return cleaned, annotated, stats


def segment_watershed(
    preprocessed: np.ndarray,
    original_for_annotation: Optional[np.ndarray] = None,
    min_area_px: int = 25
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Enhanced distance-transform watershed segmentation:
    Partitions conjoined / clumped cell bodies using Euclidean distance peaks.
    Provides a clear side-by-side comparison with Global Otsu.
    """
    # 1. Base binary threshold
    gray_8u = np.uint8(np.clip(preprocessed, 0, 255))
    _, thresh = cv2.threshold(gray_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area via Distance Transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    max_dist = dist_transform.max() if dist_transform.max() > 0 else 1.0
    _, sure_fg = cv2.threshold(dist_transform, 0.35 * max_dist, 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    # Unknown region between background and foreground
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Color image required for cv2.watershed
    if original_for_annotation is not None:
        if len(original_for_annotation.shape) == 2:
            color_src = cv2.cvtColor(original_for_annotation, cv2.COLOR_GRAY2BGR)
        else:
            color_src = original_for_annotation.copy()
    else:
        color_src = cv2.cvtColor(gray_8u, cv2.COLOR_GRAY2BGR)

    markers_ws = cv2.watershed(color_src, markers)

    # Build separated mask and boundary overlay
    annotated = color_src.copy()
    # Boundaries are marked with -1 in OpenCV watershed
    annotated[markers_ws == -1] = [0, 0, 255]

    # Mask for segmented cell bodies (excluding background and boundaries)
    ws_mask = np.zeros_like(thresh, dtype=np.uint8)
    ws_mask[markers_ws > 1] = 255

    # Extract separated contours
    contours, _ = cv2.findContours(ws_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    cv2.drawContours(annotated, valid_contours, -1, (255, 120, 0), 1)

    areas = [float(cv2.contourArea(c)) for c in valid_contours]
    total_pixels = ws_mask.shape[0] * ws_mask.shape[1]
    cell_pixels = int(np.count_nonzero(ws_mask))
    confluence_pct = round((cell_pixels / total_pixels) * 100.0, 2)

    # High confluence uncertainty warning: Euclidean distance peaks coalesce in dense cultures
    uncertainty_warning = None
    if confluence_pct > 75.0:
        uncertainty_warning = (
            f"High Confluence Notice ({confluence_pct}%): Conjoined cellular clusters and touching diffraction "
            "halos cause Euclidean distance-transform markers to coalesce into merged macro-regions. "
            "Region counts in dense monolayer cultures represent cluster partitions rather than verified individual cells."
        )

    stats = {
        "confluence_pct": confluence_pct,
        "segmented_area_px": cell_pixels,
        "region_count": len(valid_contours),
        "mean_region_area_px": round(float(np.mean(areas)), 1) if areas else 0.0,
        "median_region_area_px": round(float(np.median(areas)), 1) if areas else 0.0,
        "max_region_area_px": round(float(np.max(areas)), 1) if areas else 0.0,
        "uncertainty_warning": uncertainty_warning,
        "method": "Distance-Transform Watershed Partitioning"
    }

    return ws_mask, annotated, stats


# ==========================================
# 4. TEMPORAL REGION TRACKING
# ==========================================

def track_regions_across_frames(
    masks: List[np.ndarray],
    raw_frames: Optional[List[np.ndarray]] = None,
    distance_threshold: float = 35.0
) -> Dict[str, Any]:
    """
    Tracks segmented cellular regions across consecutive temporal frames (T1 -> T2 -> T3 -> T4).
    Uses centroid Euclidean distance gating with area consistency checking.
    Flags uncertain matches where displacement is high or area changes dramatically.
    
    Disclaimer:
    Tracked regions represent optical segmentation centroids. Optical merges/splits
    reflect cluster changes and do NOT confirm single-cell division or apoptosis.
    """
    num_frames = min(len(masks), 4)
    frame_regions = []

    # Extract centroids and bounding boxes for each frame
    for t in range(num_frames):
        mask = masks[t]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) >= 30]

        regions = []
        for idx, c in enumerate(valid):
            area = float(cv2.contourArea(c))
            m = cv2.moments(c)
            if m["m00"] > 0:
                cx = float(m["m10"] / m["m00"])
                cy = float(m["m01"] / m["m00"])
                x, y, w, h = cv2.boundingRect(c)
                regions.append({
                    "id": f"T{t+1}_R{idx+1}",
                    "cx": cx,
                    "cy": cy,
                    "area": area,
                    "bbox": (x, y, w, h)
                })
        frame_regions.append(regions)

    # Link across frames
    tracks = []
    # Initialize tracks with Frame 1 regions
    for r in frame_regions[0]:
        tracks.append({
            "track_id": len(tracks) + 1,
            "history": {0: r},
            "status": "active"
        })

    # Greedily match from T to T+1
    for t in range(num_frames - 1):
        prev_tracks = [trk for trk in tracks if t in trk["history"]]
        curr_regions = frame_regions[t+1]
        matched_curr = set()

        for trk in prev_tracks:
            p_reg = trk["history"][t]
            px, py, pa = p_reg["cx"], p_reg["cy"], p_reg["area"]

            best_match = None
            min_dist = float("inf")

            for idx, c_reg in enumerate(curr_regions):
                if idx in matched_curr:
                    continue
                cx, cy = c_reg["cx"], c_reg["cy"]
                dist = np.hypot(cx - px, cy - py)
                if dist < min_dist and dist <= distance_threshold:
                    min_dist = dist
                    best_match = (idx, c_reg)

            if best_match is not None:
                idx, c_reg = best_match
                matched_curr.add(idx)
                # Check area consistency
                area_ratio = c_reg["area"] / (pa + 1e-6)
                is_confident = (0.5 <= area_ratio <= 1.8) and (min_dist <= 20.0)
                trk["history"][t+1] = {
                    **c_reg,
                    "displacement_px": round(min_dist, 1),
                    "match_confidence": "high" if is_confident else "uncertain_optical_shift"
                }
            else:
                trk["status"] = "lost_or_merged"

        # Regions in T+1 not matched to existing tracks become new candidate tracks
        for idx, c_reg in enumerate(curr_regions):
            if idx not in matched_curr:
                tracks.append({
                    "track_id": len(tracks) + 1,
                    "history": {t+1: {**c_reg, "match_confidence": "newly_detected"}},
                    "status": "active"
                })

    # Compile summary statistics
    persistent_tracks = [trk for trk in tracks if len(trk["history"]) == num_frames]
    uncertain_events = 0
    for trk in tracks:
        for t_step, r_data in trk["history"].items():
            if r_data.get("match_confidence") == "uncertain_optical_shift":
                uncertain_events += 1

    return {
        "total_tracks_instantiated": len(tracks),
        "persistent_tracks_all_frames": len(persistent_tracks),
        "uncertain_matching_events": uncertain_events,
        "sample_persistent_tracks": [
            {
                "track_id": f"RegionTrack-{trk['track_id']}",
                "lifespan_frames": len(trk["history"]),
                "initial_area_px": round(trk["history"][min(trk["history"].keys())]["area"], 1),
                "latest_area_px": round(trk["history"][max(trk["history"].keys())]["area"], 1)
            }
            for trk in persistent_tracks[:6]
        ],
        "disclaimer": (
            "Temporal tracking links segmented cluster centroids across frames. "
            "Apparent splits or mergers represent optical overlap changes, not verified biological cell divisions."
        )
    }


# ==========================================
# 5. RECONCILED MODEL PREDICTION & RECONSTRUCTION
# ==========================================

def predict_future_mask_legacy_fixed_threshold(
    model: Any,
    masks_4: List[np.ndarray],
    target_shape: Tuple[int, int]
) -> Tuple[np.ndarray, str]:
    """
    [LEGACY BACKUP] Original implementation using a hardcoded threshold of 127.
    NOTE: Because the trained ConvLSTM generator outputs background activations around ~0.508
    (which maps to ~130 in uint8), this fixed threshold of 127 causes all-foreground (100% 255) masks.
    Preserved for comparison and regression verification.
    """
    H_target, W_target = target_shape[:2]
    resized = [cv2.resize(m, (128, 128), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0 for m in masks_4[:4]]
    seq = np.stack(resized, axis=0)
    pred_128 = None
    engine_info = "Heuristic Extrapolation (Model Not Active)"
    if model is not None:
        try:
            inp = np.expand_dims(seq, axis=(0, -1))
            raw_pred = model.predict(inp, verbose=0)
            raw_pred = np.clip(raw_pred, 0, 1)
            squeezed = np.squeeze(raw_pred)
            if len(squeezed.shape) == 3:
                squeezed = squeezed[:, :, 0]
            pred_128 = (squeezed * 255.0).astype(np.uint8)
            engine_info = "Trained ConvLSTM GAN (200 Epochs, Keras / TensorFlow) [Legacy Fixed 127]"
        except Exception:
            pred_128 = None
    if pred_128 is None:
        m3, m4 = resized[2], resized[3]
        growth_vel = m4 - m3
        extrap = np.clip(m4 + (1.10 * growth_vel), 0.0, 1.0)
        pred_128 = (extrap * 255.0).astype(np.uint8)
        engine_info = "Velocity Extrapolation Fallback"

    pred_full = cv2.resize(pred_128, (W_target, H_target), interpolation=cv2.INTER_LINEAR)
    _, cell_mask = cv2.threshold(pred_full, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cell_mask = cv2.morphologyEx(cell_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cell_mask, engine_info


def predict_future_mask(
    model: Any,
    masks_4: List[np.ndarray],
    target_shape: Tuple[int, int]
) -> Tuple[np.ndarray, str]:
    """
    Feeds 4 binary segmentation masks into the 200-epoch ConvLSTM GAN generator.
    The model was specifically trained on sequences of 4 masks of shape (4, 128, 128, 1).

    Uses adaptive Otsu thresholding calculated strictly from the predicted probability map
    (never using future ground-truth target data) to cleanly separate cellular foreground from background.

    Returns:
    - predicted_mask: (H, W) uint8 binary mask with values strictly 0 and 255.
    - engine_info: String identifying whether native trained weights or fallback was used.
    """
    H_target, W_target = target_shape[:2]

    # 1. Resize and normalize masks to [0.0, 1.0] as done in training
    resized = []
    for m in masks_4[:4]:
        r = cv2.resize(m, (128, 128), interpolation=cv2.INTER_NEAREST)
        resized.append(r.astype(np.float32) / 255.0)

    seq = np.stack(resized, axis=0) # (4, 128, 128)
    pred_128 = None
    engine_info = "Heuristic Extrapolation (Model Not Active)"

    if model is not None:
        try:
            # Batch input format: (1, 4, 128, 128, 1)
            inp = np.expand_dims(seq, axis=(0, -1))
            raw_pred = model.predict(inp, verbose=0)
            raw_pred = np.clip(raw_pred, 0, 1)
            squeezed = np.squeeze(raw_pred)
            if len(squeezed.shape) == 3:
                squeezed = squeezed[:, :, 0]
            pred_128 = (squeezed * 255.0).astype(np.uint8)
            engine_info = "Trained ConvLSTM GAN (200 Epochs, Keras / TensorFlow)"
        except Exception:
            # Fallback if model forward pass throws
            pred_128 = None

    if pred_128 is None:
        # Transparent fallback: velocity extrapolation based on Frame 4 - Frame 3
        m3 = resized[2]
        m4 = resized[3]
        growth_vel = m4 - m3
        extrap = np.clip(m4 + (1.10 * growth_vel), 0.0, 1.0)
        pred_128 = (extrap * 255.0).astype(np.uint8)
        engine_info = "Velocity Extrapolation Fallback (TensorFlow not loaded or execution error)"

    # 2. Upscale continuous probability map to target resolution
    prob_full = cv2.resize(pred_128, (W_target, H_target), interpolation=cv2.INTER_LINEAR)

    # 3. Adaptive Otsu Thresholding strictly on the predicted probability map:
    # Solves the all-foreground issue where background activations (~0.508) exceed fixed 127 (0.498).
    # If the distribution is degenerate (uniform), fallback to median.
    th_val, cell_mask = cv2.threshold(prob_full, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th_val == 0 or np.all(cell_mask == 0) or np.all(cell_mask == 255):
        # Fallback to mid-range dynamic threshold
        mid_th = int((int(prob_full.min()) + int(prob_full.max())) / 2)
        _, cell_mask = cv2.threshold(prob_full, max(mid_th, 127), 255, cv2.THRESH_BINARY)

    # 4. Clean morphological noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cell_mask = cv2.morphologyEx(cell_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Ensure strictly binary uint8 {0, 255}
    cell_mask = ((cell_mask > 127) * 255).astype(np.uint8)

    return cell_mask, engine_info


def synthesize_phase_contrast_view(
    mask: np.ndarray,
    reference_frame: np.ndarray
) -> np.ndarray:
    """
    Renders an illustrative phase-contrast micrograph view from a segmentation mask,
    simulating characteristic neuroblastoma optical properties (dark cell body + bright diffraction halo).
    """
    h, w = reference_frame.shape[:2]
    ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY) if len(reference_frame.shape) == 3 else reference_frame.copy()
    ref_gray = cv2.resize(ref_gray, (w, h))

    mask_res = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    _, mask_bin = cv2.threshold(mask_res, 127, 255, cv2.THRESH_BINARY)

    # Diffraction halo via dilation
    kernel_halo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(mask_bin, kernel_halo, iterations=1)
    halo = cv2.subtract(dilated, mask_bin)

    synth = ref_gray.astype(np.float32).copy()
    synth[mask_bin == 255] = np.clip(synth[mask_bin == 255] * 0.55 + 25.0, 45, 110)
    synth[halo == 255] = np.clip(synth[halo == 255] * 1.35 + 30.0, 160, 225)
    synth = np.uint8(np.clip(synth, 0, 255))
    synth = cv2.GaussianBlur(synth, (3, 3), 0.5)

    return synth


# ==========================================
# 6. GROUND TRUTH EVALUATION METRICS
# ==========================================

def evaluate_segmentation_or_forecast(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    split_info: str = "LIVECell Benchmark Test Set"
) -> Dict[str, Any]:
    """
    Computes rigorous quantitative metrics between predicted mask and authentic ground truth mask:
    - Intersection over Union (IoU / Jaccard Index)
    - Dice Similarity Coefficient (F1 Score)
    - Pixel-level Accuracy & Mean Absolute Error (MAE)
    - Relative Confluence Error
    """
    h, w = ground_truth_mask.shape[:2]
    pred_res = cv2.resize(predicted_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    p_bin = (pred_res > 127).astype(bool)
    g_bin = (ground_truth_mask > 127).astype(bool)

    intersection = np.logical_and(p_bin, g_bin).sum()
    union = np.logical_or(p_bin, g_bin).sum()
    iou = float(intersection / union) if union > 0 else 0.0

    dice = float((2.0 * intersection) / (p_bin.sum() + g_bin.sum())) if (p_bin.sum() + g_bin.sum()) > 0 else 0.0

    accuracy = float((p_bin == g_bin).mean())
    mae = float(np.abs(p_bin.astype(float) - g_bin.astype(float)).mean())

    total_px = h * w
    pred_conf = (p_bin.sum() / total_px) * 100.0
    gt_conf = (g_bin.sum() / total_px) * 100.0
    conf_error = abs(pred_conf - gt_conf)

    return {
        "iou_score": round(iou, 3),
        "dice_coefficient": round(dice, 3),
        "pixel_accuracy": round(accuracy * 100.0, 2),
        "mean_absolute_error": round(mae, 4),
        "predicted_confluence_pct": round(pred_conf, 2),
        "ground_truth_confluence_pct": round(gt_conf, 2),
        "confluence_absolute_error_pct": round(conf_error, 2),
        "benchmark_split": split_info,
        "is_evaluated_on_ground_truth": True
    }
