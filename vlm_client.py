"""
Multimodal Vision-Language Model (VLM) Client for Microscopy Analysis
NeuroCell GAN Forecaster

Provides direct image-input visual understanding for phase-contrast microscopy:
1. Encodes real microscopy frames (downsampled and compressed for fast, quota-friendly transmission).
2. Sends real image arrays + verified CV metrics to vision-capable models (e.g. Groq Llama 3.2 Vision or OpenAI-compatible endpoints).
3. Enforces responsible scientific AI principles:
   - Separates direct visual evidence from biological interpretation.
   - Communicates uncertainty (no speculative clinical diagnoses of apoptosis or cytopathology without assays).
   - Provides clear user notice when images are sent to cloud providers.
   - Includes a full offline / no-API-key fallback path.
"""

import os
import io
import json
import base64
import urllib.request
import urllib.error
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any

# Default Vision Models on Groq
DEFAULT_VISION_MODEL = "llama-3.2-11b-vision-preview"
FALLBACK_VISION_MODELS = [
    "llama-3.2-90b-vision-preview",
    "qwen/qwen3.6-27b"
]

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT_MICROSCOPY_VLM = """You are Dr. NeuroCell Vision AI, an expert Senior Computational Oncologist and Bio-imaging Specialist analyzing phase-contrast time-lapse microscopy of neuroblastoma (SH-SY5Y) cells.

You will be provided with:
1. One or more actual microscopy micrographs or segmentation overlays.
2. Verified quantitative Computer Vision metrics (confluence %, segmented areas, temporal tracking info).

Strict Guidelines for Analysis:
- Ground all statements in observable optical features (cellular clusters, boundaries, diffraction halos, confluence density, foreground vs background).
- Separate VISIBLE EVIDENCE from BIOLOGICAL INTERPRETATION:
  * Visible Evidence: What is directly observable in the image pixels (e.g. 'high optical density in lower-left quadrant', 'halo diffraction artifacts surrounding dense clusters').
  * Biological Interpretation: Inferred cellular kinetics based on verified metrics (e.g. 'exponential expansion phase indicated by +14.2% confluence delta').
- Explicit Uncertainty: Do NOT diagnose apoptosis, cell death, mitotic division, or genetic subtype purely from optical phase-contrast appearance without biochemical markers (e.g. caspase assays). Label visual signs of retraction or rounding as 'potential morphological changes requiring validation'.
- Keep your answer clear, scientific, and concise. Format with clean Markdown.
"""

def compress_image_to_base64_data_url(
    img_arr: np.ndarray,
    max_dimension: int = 512,
    quality: int = 85
) -> str:
    """
    Downsamples and compresses a numpy image array to a lightweight JPEG/PNG base64 URL.
    Limits payload size to conserve bandwidth and prevent API rate-limit errors.
    """
    if img_arr is None or img_arr.size == 0:
        return ""

    if len(img_arr.shape) == 2:
        # Grayscale
        pil_img = Image.fromarray(img_arr).convert("L")
    elif len(img_arr.shape) == 3:
        if img_arr.shape[2] == 4:
            pil_img = Image.fromarray(img_arr).convert("RGB")
        elif img_arr.shape[2] == 3:
            pil_img = Image.fromarray(img_arr)
        else:
            pil_img = Image.fromarray(img_arr[:, :, 0]).convert("L")
    else:
        return ""

    # Scale down preserving aspect ratio if larger than max_dimension
    w, h = pil_img.size
    if max(w, h) > max_dimension:
        scale = max_dimension / float(max(w, h))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    # Save as JPEG for efficient vision API transport
    pil_img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def build_vlm_prompt(
    user_query: str,
    cv_metrics: Dict[str, Any],
    frame_context: Optional[str] = None
) -> str:
    """Formats verified CV outputs alongside the user question."""
    metrics_summary = f"""Verified Computer Vision Metrics:
- Initial Confluence (T1): {cv_metrics.get('confluence_t1', 'N/A')}%
- Latest Observed Confluence (T4): {cv_metrics.get('confluence_t4', 'N/A')}%
- Predicted Confluence (T5 Forecast): {cv_metrics.get('forecast_confluence', 'N/A')}%
- Net Confluence Change (Δ): {cv_metrics.get('delta_confluence', 'N/A')}%
- Segmented Regions (Contours): {cv_metrics.get('forecast_regions', 'N/A')}
- Optical Quality Index: {cv_metrics.get('optical_quality', 'Moderate Quality')}
- Sequence Chronology Status: {cv_metrics.get('chronology_status', 'Order from file sequence')}
- Inference Engine: {cv_metrics.get('inference_engine', 'Trained ConvLSTM GAN (200 Epochs)')}
"""
    if frame_context:
        metrics_summary += f"\nFrame Context: {frame_context}\n"

    prompt = f"""{metrics_summary}

User Research Question:
{user_query}

Please examine the attached micrograph(s) alongside the verified metrics above. 
Structure your answer into:
1. 👁️ Visible Optical Evidence (Direct Observations)
2. 🔬 Quantitative Metrics Interpretation
3. ⚠️ Uncertainty & Methodological Nuances
4. 📋 Recommended Experimental Validation
"""
    return prompt


def query_vision_language_model(
    images: List[np.ndarray],
    user_query: str,
    cv_metrics: Dict[str, Any],
    api_key: Optional[str] = None,
    model: str = DEFAULT_VISION_MODEL,
    base_url: str = GROQ_API_URL,
    frame_labels: Optional[List[str]] = None
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Submits microscopy image(s) and structured CV metrics to a multimodal Vision-Language Model.
    
    Supports:
    - Groq Vision Models (Llama 3.2 Vision, Qwen multimodal).
    - Generic OpenAI-compatible multimodal endpoints (e.g. self-hosted Ollama or vLLM).
    
    Returns:
    - (success: bool, response_text: str, metadata: dict)
    """
    actual_key = (api_key or os.environ.get("GROQ_API_KEY", "")).strip()

    # 1. Check for API Key
    if not actual_key and "localhost" not in base_url and "127.0.0.1" not in base_url:
        return (
            False,
            "**API Key Required for Cloud VLM**: No API key was provided. To use the Vision-Language Model, "
            "set `GROQ_API_KEY` in your environment or enter your key in the dashboard sidebar. "
            "\n\n*Note: All core Computer Vision, Otsu/Watershed segmentation, GAN forecasting, and temporal tracking features run 100% locally without an API key.*",
            {"status": "missing_api_key"}
        )

    # 2. Encode images (limit to max 2 images per request to prevent token/quota blowout)
    encoded_images = []
    frames_used = []
    for i, img in enumerate(images[:2]):
        data_url = compress_image_to_base64_data_url(img, max_dimension=512)
        if data_url:
            lbl = frame_labels[i] if frame_labels and i < len(frame_labels) else f"Frame {i+1}"
            encoded_images.append(data_url)
            frames_used.append(lbl)

    if not encoded_images:
        return False, "Error: No valid images could be encoded for vision analysis.", {}

    # 3. Construct Multimodal Content Payload
    frame_ctx = f"Attached {len(encoded_images)} frame(s): {', '.join(frames_used)}"
    full_text_prompt = build_vlm_prompt(user_query, cv_metrics, frame_context=frame_ctx)

    content_items = [{"type": "text", "text": full_text_prompt}]
    for url in encoded_images:
        content_items.append({
            "type": "image_url",
            "image_url": {
                "url": url,
                "detail": "low" # Low-detail mode saves tokens and stays within free rate limits
            }
        })

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_MICROSCOPY_VLM},
            {"role": "user", "content": content_items}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {actual_key}"
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base_url, data=data, headers=headers, method="POST")

    # 4. Execute HTTP Request
    try:
        with urllib.request.urlopen(req, timeout=35) as res:
            res_body = res.read().decode("utf-8")
            res_json = json.loads(res_body)
            answer = res_json["choices"][0]["message"]["content"]
            used_model = res_json.get("model", model)
            return True, answer, {
                "model_used": used_model,
                "frames_analyzed": frames_used,
                "provider": "Groq Cloud Vision API"
            }
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        # Try fallback model if default preview model is deprecated or rate-limited
        for fb_model in FALLBACK_VISION_MODELS:
            if model != fb_model:
                try:
                    payload["model"] = fb_model
                    data_fb = json.dumps(payload).encode("utf-8")
                    req_fb = urllib.request.Request(base_url, data=data_fb, headers=headers, method="POST")
                    with urllib.request.urlopen(req_fb, timeout=35) as res_fb:
                        res_json_fb = json.loads(res_fb.read().decode("utf-8"))
                        answer_fb = res_json_fb["choices"][0]["message"]["content"]
                        return True, answer_fb, {
                            "model_used": fb_model,
                            "frames_analyzed": frames_used,
                            "provider": "Groq Cloud Vision API (Fallback)"
                        }
                except Exception:
                    continue

        return False, f"Vision API HTTP {e.code}: {err_msg[:300]}", {"error_code": e.code}
    except Exception as e:
        return False, f"Vision API Error: {str(e)}", {}
