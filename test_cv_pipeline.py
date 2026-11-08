import os
import sys
import tempfile
import cv2
import numpy as np

# Ensure environment directories
os.environ.setdefault("KERAS_HOME", os.path.join(tempfile.gettempdir(), ".keras"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), ".mpl"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import microscopy_cv as mcv
import vlm_client as vlm

print("=== 1. TESTING B10_1 DATA LOADING & VALIDATION ===")
b10_dir = os.path.join(BASE_DIR, "sorted_multiples_of_4", "B10_1")
files = [
    "0002_pos0004_0004_00d00h00m_FOV4_SHSY5Y_Phase_B10_1_00d00h00m_4.png",
    "0003_pos0008_0008_00d04h00m_FOV4_SHSY5Y_Phase_B10_1_00d04h00m_4.png",
    "0004_pos0012_0012_00d08h00m_FOV4_SHSY5Y_Phase_B10_1_00d08h00m_4.png",
    "0005_pos0016_0016_00d12h00m_FOV4_SHSY5Y_Phase_B10_1_00d12h00m_4.png"
]
frames = [cv2.imread(os.path.join(b10_dir, fn), cv2.IMREAD_GRAYSCALE) for fn in files]
ok, msg, meta = mcv.validate_image_sequence(frames, files)
print(f"Validation OK: {ok}, Message: {msg}")
assert ok, "Validation failed!"

print("=== 2. TESTING OPTICAL QUALITY ASSESSMENT ===")
q = mcv.assess_image_quality(frames[0])
print(f"Quality Level: {q['quality_level']}, Focus Sharpness: {q['sharpness_laplacian']}, Contrast: {q['rms_contrast']}")

print("=== 3. TESTING DUAL SEGMENTATION (OTSU vs WATERSHED) ===")
prep0 = mcv.preprocess_phase_contrast(frames[0])
m_otsu, a_otsu, s_otsu = mcv.segment_otsu(prep0, original_for_annotation=frames[0])
m_ws, a_ws, s_ws = mcv.segment_watershed(prep0, original_for_annotation=frames[0])
print(f"Otsu Confluence: {s_otsu['confluence_pct']}%, Regions: {s_otsu['region_count']}")
print(f"Watershed Confluence: {s_ws['confluence_pct']}%, Regions: {s_ws['region_count']}")

print("=== 4. TESTING TEMPORAL REGION TRACKING ===")
masks = []
for f in frames:
    p = mcv.preprocess_phase_contrast(f)
    m, _, _ = mcv.segment_otsu(p)
    masks.append(m)
trk = mcv.track_regions_across_frames(masks, raw_frames=frames)
print(f"Total tracks: {trk['total_tracks_instantiated']}, Persistent: {trk['persistent_tracks_all_frames']}, Displaced/shifts: {trk['uncertain_matching_events']}")

print("=== 5. TESTING GAN MODEL INFERENCE (NATIVE KERAS) ===")
import tensorflow as tf
m_path = os.path.join(BASE_DIR, "livecell_gan_200epochs_final.h5")
model = tf.keras.models.load_model(m_path, compile=False)
pred_mask, engine = mcv.predict_future_mask(model, masks, frames[0].shape)
print(f"Inference engine: {engine}")
print(f"Predicted mask shape: {pred_mask.shape}, non-zero px: {np.count_nonzero(pred_mask)}")

print("=== 6. TESTING ALGORITHM REFERENCE EVALUATION ON ACTUAL T5 ===")
tgt_fn = "0006_pos0020_0020_00d16h00m_FOV4_SHSY5Y_Phase_B10_1_00d16h00m_4.png"
tgt_img = cv2.imread(os.path.join(b10_dir, tgt_fn), cv2.IMREAD_GRAYSCALE)
prep_tgt = mcv.preprocess_phase_contrast(tgt_img)
ref_tgt_mask, _, ref_tgt_stats = mcv.segment_otsu(prep_tgt)
eval_res = mcv.evaluate_segmentation_or_forecast(pred_mask, ref_tgt_mask, split_info="B10_1 Algorithm Reference")
print(f"Reference IoU Agreement: {eval_res['iou_score']}, Dice: {eval_res['dice_coefficient']}, Pixel Accuracy: {eval_res['pixel_accuracy']}%")

print("=== 7. TESTING VLM DATA PREPARATION & ENCODING ===")
b64 = vlm.compress_image_to_base64_data_url(frames[0], max_dimension=512)
print(f"Encoded base64 string length: {len(b64)} chars, valid header: {b64.startswith('data:image/jpeg;base64,')}")

print("\n>>> ALL TEST VERIFICATIONS COMPLETED SUCCESSFULLY! <<<")
