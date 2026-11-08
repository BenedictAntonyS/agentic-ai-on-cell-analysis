import os
import sys
import io
import json
import tempfile
import numpy as np
from PIL import Image
import cv2
import streamlit as st

_app_dir = os.path.dirname(os.path.abspath(__file__))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

# Ensure safe writable cache directories
_temp_dir = tempfile.gettempdir()
os.environ["KERAS_HOME"] = os.path.join(_temp_dir, ".keras")
os.environ["MPLCONFIGDIR"] = os.path.join(_temp_dir, ".mpl")

# Import shared robust microscopy CV and VLM modules
import microscopy_cv as mcv
import vlm_client as vlm
import groq_client as groq

# ==========================================
# PAGE SETUP & STYLING
# ==========================================
st.set_page_config(
    page_title="NeuroCell GAN & VLM - Microscopy Forecaster",
    page_icon="🔬",
    layout="wide"
)

def safe_image(img, caption=None):
    """Safely renders image across all Streamlit versions without TypeError or deprecation warnings."""
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        try:
            st.image(img, caption=caption, use_column_width=True)
        except Exception:
            st.image(img, caption=caption)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        color: #00d2ff;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .section-header {
        color: #f1f5f9;
        font-size: 1.25rem;
        font-weight: 600;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #1e293b;
        margin-top: 1.4rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .disclaimer-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #38bdf8;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #cbd5e1;
        margin: 0.8rem 0;
    }
    .warning-box {
        background: rgba(120, 53, 15, 0.25);
        border-left: 4px solid #f59e0b;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #fde68a;
        margin: 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "livecell_gan_200epochs_final.h5")
SAMPLE_DATA_DIR = os.path.join(BASE_DIR, "sample_data", "synthetic_shsy5y_sequence")

# ==========================================
# MODEL LOADER
# ==========================================
@st.cache_resource(show_spinner=False)
def load_generator_model():
    if not os.path.exists(MODEL_PATH):
        return None, "Model file not found on disk."
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model, "Loaded successfully (200-Epoch ConvLSTM GAN)"
    except Exception as e:
        return None, f"TensorFlow load notice: {str(e)}"

model, model_msg = load_generator_model()

# ==========================================
# SIDEBAR SETTINGS
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Model & System")
    if model is not None:
        st.success(f"✅ GAN Model: Active\n`{os.path.basename(MODEL_PATH)}`")
    else:
        st.info(f"ℹ️ Engine Mode: Fallback / Heuristic ({model_msg})")

    st.markdown("---")
    st.markdown("### 🔑 API Configuration (Optional)")
    st.caption("Core CV, segmentation, tracking, and GAN forecasting run **100% locally** without an API key.")
    user_api_key = st.text_input(
        "Groq API Key (for VLM & Copilot)",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Used only when calling Groq VLM image understanding or text copilot."
    )
    vlm_model_choice = st.selectbox(
        "Multimodal VLM Model",
        [vlm.DEFAULT_VISION_MODEL] + vlm.FALLBACK_VISION_MODELS,
        index=0
    )
    text_model_choice = st.selectbox(
        "Text Copilot Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0
    )

    st.markdown("---")
    st.markdown("### 🔬 Segmentation Settings")
    seg_method = st.radio(
        "Segmentation Algorithm",
        ["Global Otsu + Morphology", "Distance-Transform Watershed"],
        index=0,
        help="Compare standard project thresholding vs adaptive watershed partitioning."
    )

# ==========================================
# HEADER
# ==========================================
st.markdown('<div class="main-title">🔬 NeuroCell: Microscopy CV & Vision-Language AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Spatiotemporal time-lapse forecasting, dual segmentation, temporal tracking, and direct Multimodal Vision-Language (VLM) image understanding for SH-SY5Y Neuroblastoma cultures.</div>', unsafe_allow_html=True)

B10_1_DIR = os.path.join(BASE_DIR, "sorted_multiples_of_4", "B10_1")

# ==========================================
# STEP 1: SEQUENCE INPUT & VALIDATION
# ==========================================
st.markdown('<div class="section-header">Step 1: Input Time-Lapse Microscopy Sequence</div>', unsafe_allow_html=True)

input_mode = st.radio(
    "Choose input source:",
    [
        "Real LIVECell B10_1 (FOV 4: 00h to 16h Real Microscopy)",
        "Bundled Demo Sequence (Synthetic Demo)",
        "Upload Custom Microscopy Frames"
    ],
    horizontal=True
)

raw_frames = []
frame_names = []
ground_truth_target = None
ground_truth_target_mask = None
is_synthetic_demo = False
is_real_b10_1 = False

if input_mode == "Real LIVECell B10_1 (FOV 4: 00h to 16h Real Microscopy)":
    is_real_b10_1 = True
    b10_files = [
        "0002_pos0004_0004_00d00h00m_FOV4_SHSY5Y_Phase_B10_1_00d00h00m_4.png",
        "0003_pos0008_0008_00d04h00m_FOV4_SHSY5Y_Phase_B10_1_00d04h00m_4.png",
        "0004_pos0012_0012_00d08h00m_FOV4_SHSY5Y_Phase_B10_1_00d08h00m_4.png",
        "0005_pos0016_0016_00d12h00m_FOV4_SHSY5Y_Phase_B10_1_00d12h00m_4.png"
    ]
    frame_names = b10_files
    for fn in b10_files:
        fp = os.path.join(B10_1_DIR, fn)
        if os.path.exists(fp):
            raw_frames.append(cv2.imread(fp, cv2.IMREAD_GRAYSCALE))
    
    tgt_fn = "0006_pos0020_0020_00d16h00m_FOV4_SHSY5Y_Phase_B10_1_00d16h00m_4.png"
    tgt_fp = os.path.join(B10_1_DIR, tgt_fn)
    if os.path.exists(tgt_fp):
        ground_truth_target = cv2.imread(tgt_fp, cv2.IMREAD_GRAYSCALE)

    st.success("🔬 Loaded **Real LIVECell SH-SY5Y B10_1 Sequence (FOV 4)** from uploaded archive. FOV 1 image is excluded. Note: Raw microscopy only; comparison against Frame 5 represents an **algorithm-generated reference**, not ground truth.")

elif input_mode == "Bundled Demo Sequence (Synthetic Demo)":
    is_synthetic_demo = True
    meta_path = os.path.join(SAMPLE_DATA_DIR, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as mf:
            meta = json.load(mf)
        frame_names = meta.get("frames", [])
        for fn in frame_names:
            fp = os.path.join(SAMPLE_DATA_DIR, fn)
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            raw_frames.append(img)

        tgt_fp = os.path.join(SAMPLE_DATA_DIR, meta.get("target_frame", ""))
        tgt_m_fp = os.path.join(SAMPLE_DATA_DIR, meta.get("target_mask", ""))
        if os.path.exists(tgt_fp):
            ground_truth_target = cv2.imread(tgt_fp, cv2.IMREAD_GRAYSCALE)
        if os.path.exists(tgt_m_fp):
            ground_truth_target_mask = cv2.imread(tgt_m_fp, cv2.IMREAD_GRAYSCALE)

        st.info("🧪 Loaded **Synthetic SH-SY5Y Demonstration Sequence**. Note: This is labeled synthetic demo data for pipeline verification, NOT authentic clinical LIVECell benchmark data.")
    else:
        st.error("Sample demonstration folder missing. Please upload custom frames.")

else:
    uploaded = st.file_uploader(
        "Upload 4 consecutive microscopy frames (Hold Ctrl/Shift to select all 4 files):",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        accept_multiple_files=True
    )
    if uploaded and len(uploaded) >= 4:
        sorted_files = sorted(uploaded, key=lambda f: f.name)[:4]
        frame_names = [f.name for f in sorted_files]
        for f in sorted_files:
            pil_img = Image.open(f).convert("L")
            raw_frames.append(np.array(pil_img))
    elif uploaded and len(uploaded) < 4:
        st.warning(f"⚠️ You selected {len(uploaded)} file(s). Exactly 4 consecutive frames are required.")

# Validation check
if len(raw_frames) >= 4:
    valid_ok, valid_msg, frame_meta = mcv.validate_image_sequence(raw_frames, frame_names)
    if not valid_ok:
        st.error(f"❌ Input Validation Error: {valid_msg}")
        st.stop()
    else:
        st.success(f"✅ Sequence validated: 4 frames ({raw_frames[0].shape[1]}x{raw_frames[0].shape[0]} px). {valid_msg}")

        # Optical Quality Assessment on Frame 1
        q_info = mcv.assess_image_quality(raw_frames[0])
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            st.metric("Optical Quality", q_info["quality_level"])
        with col_q2:
            st.metric("Focus Sharpness (Laplacian)", f"{q_info['sharpness_laplacian']:.1f}")
        with col_q3:
            st.metric("Dynamic Range", f"{q_info['dynamic_range_pct']}%")
else:
    st.warning("Please provide 4 sequential frames to proceed.")
    st.stop()

# ==========================================
# STEP 2: PREPROCESSING & DUAL SEGMENTATION
# ==========================================
st.markdown('<div class="section-header">Step 2: Dual Segmentation & Temporal Region Tracking</div>', unsafe_allow_html=True)

processed_masks = []
annotated_frames = []
confluences = []
region_counts = []

cols_display = st.columns(4)

for i in range(4):
    frame = raw_frames[i]
    prep = mcv.preprocess_phase_contrast(frame)
    if seg_method == "Distance-Transform Watershed":
        mask_i, annot_i, stats_i = mcv.segment_watershed(prep, original_for_annotation=frame)
    else:
        mask_i, annot_i, stats_i = mcv.segment_otsu(prep, original_for_annotation=frame)

    processed_masks.append(mask_i)
    annotated_frames.append(annot_i)
    confluences.append(stats_i["confluence_pct"])
    region_counts.append(stats_i["region_count"])

    with cols_display[i]:
        safe_image(annot_i, caption=f"T{i+1}: {frame_names[i]}\nConfluence: {stats_i['confluence_pct']:.1f}% | Regions: {stats_i['region_count']}")

# Temporal Region Tracking
tracking_res = mcv.track_regions_across_frames(processed_masks, raw_frames=raw_frames)
st.markdown(f"""
<div class="disclaimer-box">
    <b>Temporal Region Tracking:</b> Tracked <b>{tracking_res['total_tracks_instantiated']}</b> region centroids across 4 frames (<b>{tracking_res['persistent_tracks_all_frames']}</b> persistent across all timepoints, {tracking_res['uncertain_matching_events']} optical shifts flagged).<br>
    <i>{tracking_res['disclaimer']}</i>
</div>
""", unsafe_allow_html=True)

# ==========================================
# STEP 3: MODEL INFERENCE & FORECASTING
# ==========================================
st.markdown('<div class="section-header">Step 3: Future Frame (T+1 / T5) GAN Forecast</div>', unsafe_allow_html=True)

h_orig, w_orig = raw_frames[0].shape[:2]
# Pass MASKS to the model (strictly matching how it was trained)
pred_mask, engine_name = mcv.predict_future_mask(model, processed_masks, (h_orig, w_orig))

# Reconstruct phase-contrast appearance and annotate
synth_view = mcv.synthesize_phase_contrast_view(pred_mask, reference_frame=raw_frames[3])
_, pred_annot, pred_stats = mcv.segment_otsu(pred_mask, original_for_annotation=raw_frames[3])
delta_conf = pred_stats["confluence_pct"] - confluences[3]
delta_regions = pred_stats["region_count"] - region_counts[3]

# Metrics Row
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{pred_stats['confluence_pct']:.1f}%</div>
        <div class="metric-label">Predicted Confluence (T5)</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value" style="color: {'#10b981' if delta_conf >= 0 else '#ef4444'}">{'+' if delta_conf >= 0 else ''}{delta_conf:.1f}%</div>
        <div class="metric-label">Confluence Growth Delta (Δ)</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{pred_stats['region_count']}</div>
        <div class="metric-label">Segmented Regions (Contours)</div>
    </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value" style="color: #a855f7;">{'+' if delta_regions >= 0 else ''}{delta_regions}</div>
        <div class="metric-label">Net Region Count Delta</div>
    </div>
    """, unsafe_allow_html=True)

st.caption(f"Inference Engine: **{engine_name}** | Input: 4 segmentation masks scaled to $128 \\times 128 \\times 1$")

# Visual Display Row
c_p1, c_p2, c_p3, c_p4 = st.columns(4)
with c_p1:
    st.markdown("#### 🎯 Synthesized Micrograph")
    safe_image(synth_view, caption="Simulated Phase-Contrast View")
with c_p2:
    st.markdown("#### 🔳 Forecast Mask")
    safe_image(pred_mask, caption=f"Predicted Binary Mask ({pred_stats['confluence_pct']:.1f}%)")
with c_p3:
    st.markdown("#### 🔬 Segmented Contours")
    safe_image(pred_annot, caption=f"Overlay on Reference Frame ({pred_stats['region_count']} regions)")
with c_p4:
    st.markdown("#### 🧬 Proliferation Hotspots")
    diff_map = cv2.absdiff(raw_frames[3], synth_view)
    diff_heatmap = cv2.applyColorMap(np.uint8(np.clip(diff_map * 3.5, 0, 255)), cv2.COLORMAP_INFERNO)
    safe_image(diff_heatmap, caption="Frame 4 vs Predicted Frame 5 Delta")

# Quantitative Validation
if ground_truth_target_mask is not None:
    st.markdown("##### 📊 Ground-Truth Quantitative Validation")
    eval_metrics = mcv.evaluate_segmentation_or_forecast(
        pred_mask,
        ground_truth_target_mask,
        split_info="Ground Truth Annotation Mask (T5)"
    )
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.metric("IoU (Jaccard)", f"{eval_metrics['iou_score']:.3f}")
    with e2:
        st.metric("Dice (F1 Score)", f"{eval_metrics['dice_coefficient']:.3f}")
    with e3:
        st.metric("Pixel Accuracy", f"{eval_metrics['pixel_accuracy']:.1f}%")
    with e4:
        st.metric("Confluence Abs Error", f"{eval_metrics['confluence_absolute_error_pct']:.1f}%")

elif ground_truth_target is not None:
    st.markdown("##### 🔬 Algorithm-Generated Reference Comparison (Actual T5 Micrograph)")
    st.caption("ℹ️ The B10_1 archive provides authentic raw microscopy only (no manual ground-truth masks). Below is an algorithm-generated reference comparison evaluating model agreement with the Otsu segmentation of the actual Frame 5 micrograph, NOT manual ground-truth.")
    prep_actual_t5 = mcv.preprocess_phase_contrast(ground_truth_target)
    ref_t5_mask, _, ref_stats = mcv.segment_otsu(prep_actual_t5, original_for_annotation=ground_truth_target)
    ref_eval = mcv.evaluate_segmentation_or_forecast(
        pred_mask,
        ref_t5_mask,
        split_info="Algorithm-Generated Reference (Otsu on Actual T5)"
    )
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.metric("Reference IoU Agreement", f"{ref_eval['iou_score']:.3f}")
    with e2:
        st.metric("Reference Dice Score", f"{ref_eval['dice_coefficient']:.3f}")
    with e3:
        st.metric("Actual T5 Confluence (Otsu)", f"{ref_stats['confluence_pct']:.1f}%")
    with e4:
        st.metric("Confluence Delta vs Ref", f"{abs(pred_stats['confluence_pct'] - ref_stats['confluence_pct']):.1f}%")

else:
    st.info("ℹ️ Target verification horizon enabled when a 5th frame or ground-truth mask is available.")

# Explainable AI (XAI) Sensitivity
st.markdown("##### 🧠 Explainable AI: Spatiotemporal Occlusion Profile")
st.markdown("""
<div class="disclaimer-box">
    <b>XAI Status: [Illustrative Reference Profile]</b> — Derived from notebook patch occlusion benchmark ablation (Frame 1: 35.8%, Frame 2: 5.9%, Frame 3: 5.1%, Frame 4: 3.0%). Frame 1 acts as the dominant spatial coordinate anchor for the ConvLSTM sequence.
</div>
""", unsafe_allow_html=True)

# ==========================================
# STEP 4: MULTIMODAL VLM & LLM COPILOT
# ==========================================
st.markdown('<div class="section-header">Step 4: Vision-Language Model (VLM) & Bio-Oncology Copilot</div>', unsafe_allow_html=True)

tab_vlm, tab_report, tab_chat = st.tabs([
    "👁️ Multimodal VLM (Image Understanding)",
    "📋 Automated Clinical Growth Report",
    "💬 Interactive Research Chat"
])

# Package CV metrics dictionary
cv_summary_dict = {
    "confluence_t1": confluences[0],
    "confluence_t4": confluences[3],
    "forecast_confluence": pred_stats["confluence_pct"],
    "delta_confluence": delta_conf,
    "forecast_regions": pred_stats["region_count"],
    "optical_quality": q_info["quality_level"],
    "inference_engine": engine_name,
    "chronology_status": valid_msg
}

with tab_vlm:
    st.markdown("#### 👁️ Direct Microscopy Image Understanding (VLM)")
    st.markdown("""
    Unlike text-only models that only see numbers, this **Multimodal Vision-Language Model** receives the actual phase-contrast micrograph images alongside verified Computer Vision metrics.
    """)
    st.markdown("""
    <div class="warning-box">
        <b>Data Privacy & External API Notice:</b> Analyzing with VLM sends a compressed, downsampled base64 image representation to the configured Groq Vision endpoint. If strict on-premise data isolation is required, do not submit confidential patient samples to cloud APIs.
    </div>
    """, unsafe_allow_html=True)

    vlm_frame_choice = st.selectbox(
        "Select frame(s) to send to Vision Model:",
        [
            "Frame 4 (Latest Known Micrograph) + Predicted Frame 5",
            "Frame 1 (Initial Horizon) + Frame 4 (Latest Known)",
            "Frame 4 Only (High-Resolution Single Micrograph)",
            "Segmentation Overlay (Contour Map)"
        ]
    )

    preset_questions = [
        "Describe visible optical evidence, cell density, and spatial distribution.",
        "Where are the largest confluent clusters and spatial growth hotspots?",
        "Inspect optical quality, diffraction halos, and potential segmentation artifacts.",
        "Compare Frame 4 against the predicted Frame 5 and evaluate biological plausibility."
    ]
    selected_preset = st.selectbox("Choose a research question or type below:", preset_questions)
    custom_query = st.text_input("Or enter a custom question for the Vision Model:", value=selected_preset)

    if st.button("🚀 Analyze Micrograph with VLM", type="primary"):
        with st.spinner(f"Querying Vision Model ({vlm_model_choice})..."):
            # Select images based on user choice
            if "Frame 4 (Latest Known Micrograph) + Predicted Frame 5" in vlm_frame_choice:
                vlm_imgs = [raw_frames[3], synth_view]
                vlm_lbls = [f"Frame 4 ({frame_names[3]})", "Predicted Frame 5 (GAN Forecast)"]
            elif "Frame 1" in vlm_frame_choice:
                vlm_imgs = [raw_frames[0], raw_frames[3]]
                vlm_lbls = [f"Frame 1 ({frame_names[0]})", f"Frame 4 ({frame_names[3]})"]
            elif "Segmentation Overlay" in vlm_frame_choice:
                vlm_imgs = [annotated_frames[3], pred_annot]
                vlm_lbls = ["Frame 4 Segmented Contours", "Predicted Frame 5 Contours"]
            else:
                vlm_imgs = [raw_frames[3]]
                vlm_lbls = [f"Frame 4 ({frame_names[3]})"]

            vlm_ok, vlm_resp, vlm_meta = vlm.query_vision_language_model(
                images=vlm_imgs,
                user_query=custom_query,
                cv_metrics=cv_summary_dict,
                api_key=user_api_key,
                model=vlm_model_choice,
                frame_labels=vlm_lbls
            )

            if vlm_ok:
                st.markdown(f"**VLM Response ({vlm_meta.get('model_used', vlm_model_choice)})**:")
                st.markdown(vlm_resp)
                st.caption(f"Frames transmitted: {', '.join(vlm_meta.get('frames_analyzed', []))} | Provider: {vlm_meta.get('provider', 'Groq')}")
            else:
                st.error(vlm_resp)

with tab_report:
    st.markdown("#### 📋 Automated Clinical & Proliferation Report (Text LLM)")
    if st.button("📄 Generate Comprehensive Report"):
        with st.spinner(f"Dr. NeuroCell AI is synthesizing report ({text_model_choice})..."):
            metrics_summary_text = f"""
Cell Line: Neuroblastoma (SH-SY5Y)
Sequence Chronology: {valid_msg}
Optical Quality: {q_info['quality_level']} (Laplacian focus {q_info['sharpness_laplacian']:.1f})
Initial Confluence (T1): {confluences[0]:.2f}%
Latest Confluence (T4): {confluences[3]:.2f}%
GAN Predicted Confluence (T5): {pred_stats['confluence_pct']:.2f}%
Net Confluence Delta: {delta_conf:+.2f}%
Segmented Regions: {pred_stats['region_count']} (Delta: {delta_regions:+d})
Temporal Tracking: {tracking_res['persistent_tracks_all_frames']} persistent clusters
Inference Engine: {engine_name}
Occlusion Sensitivity Anchor: Frame 1 carries ~35.8% impact
"""
            prompt = f"Please provide an expert Bio-Oncology evaluation based on these verified metrics:\n{metrics_summary_text}\nInclude Proliferation Kinetics, Confluence Assessment, Uncertainty/Method Limitations, and Recommended Experimental Next Steps."
            ok, rep = groq.ask_groq(prompt, api_key=user_api_key, model=text_model_choice)
            if ok:
                st.markdown(rep)
            else:
                st.error(f"Report Error: {rep}")

with tab_chat:
    st.markdown("#### 💬 Interactive Bio-Oncology Copilot Chat")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! I am Dr. NeuroCell AI. Ask me about your sequence metrics, GAN spatiotemporal dynamics, Otsu vs Watershed segmentation, or LIVECell SH-SY5Y biology."}
        ]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_chat := st.chat_input("Ask a question about the sequence, metrics, or cell line..."):
        st.session_state.chat_history.append({"role": "user", "content": user_chat})
        with st.chat_message("user"):
            st.markdown(user_chat)

        with st.chat_message("assistant"):
            with st.spinner("Dr. NeuroCell AI is responding..."):
                ctx = f"Context: Latest Confluence is {confluences[3]:.1f}%, Predicted T5 is {pred_stats['confluence_pct']:.1f}% ({delta_conf:+.1f}%), {pred_stats['region_count']} regions."
                ok, reply = groq.ask_groq(f"{ctx}\nQuestion: {user_chat}", api_key=user_api_key, model=text_model_choice)
                if ok:
                    st.markdown(reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                else:
                    st.error(f"Error: {reply}")
