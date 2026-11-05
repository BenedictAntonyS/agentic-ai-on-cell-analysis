# NeuroCell: Microscopy Computer Vision & Multimodal VLM Forecaster

A research and clinical time-lapse bio-imaging suite for **SH-SY5Y Neuroblastoma** phase-contrast microscopy.

This platform integrates:
1. **200-Epoch ConvLSTM GAN Generator** (`livecell_gan_200epochs_final.h5`): Spatiotemporal deep learning model trained to forecast future cellular morphology and confluence from 4 prior timepoints.
2. **Dual Phase-Contrast Segmentation & Metrics**: Classical Global Otsu thresholding with morphological filtering alongside **Distance-Transform Watershed Partitioning** for separated colony clustering.
3. **Temporal Region Tracking & Optical Quality Index**: Centroid tracking across time ($T_1 \to T_4$) and focus sharpness/contrast assessment.
4. **Authentic LIVECell B10_1 Time-Series Sequence**: Preloaded 22-frame sequence spanning 00h to 84h at 4-hour intervals from Field of View (FOV) 4.
5. **Multimodal Vision-Language Model (VLM)**: Direct image-input inspection powered by Groq Llama 3.2 Vision (`llama-3.2-11b-vision-preview`), separating visible optical evidence from biological interpretation.
6. **Multi-Step ReAct AI Agent & Text Copilot**: Autonomous research planning and tool execution for quantitative metric interpretation and experimental protocols.

---

## 🏗️ Architecture & Entry Points

```
cv_llm-main/
├── app.py                         # Primary Production Web Server (FastAPI + Uvicorn)
├── streamlit_app.py               # Interactive Research Dashboard (Streamlit)
├── microscopy_cv.py               # Shared Core Microscopy CV & Forecasting Pipeline
├── vlm_client.py                  # Multimodal Vision-Language Model (VLM) Client
├── groq_client.py                 # Text LLM Copilot & Clinical Report Engine
├── livecell_gan_200epochs_final.h5 # Trained 200-Epoch ConvLSTM GAN Generator
├── sorted_multiples_of_4/
│   └── B10_1/                     # Authentic LIVECell Phase-Contrast Micrographs (FOV 4)
├── sample_data/
│   └── synthetic_shsy5y_sequence/ # Labeled Synthetic Demo Data (Offline Fallback)
├── static/                        # Frontend Styles (CSS) & Interactive Logic (JS)
├── templates/
│   └── index.html                 # Modern Full-Stack Web Dashboard
├── requirements.txt               # Verified Python Dependencies
├── .env.example                   # API Configuration Template
└── run_web_app.bat / run_streamlit.bat
```

### Recommended Primary Entry Point
- **FastAPI Modern Web Application**:
  ```bash
  python -m uvicorn app:app --host 127.0.0.1 --port 8000
  ```
  Accessible at: `http://localhost:8000`
- **Streamlit Quick Dashboard**:
  ```bash
  python -m streamlit run streamlit_app.py
  ```
  Accessible at: `http://localhost:8501`

---

## ⚡ Local vs. Cloud Feature Execution

| Feature | Execution Mode | Requires API Key? | External Transmission |
| :--- | :--- | :--- | :--- |
| **Input Sequence Validation** | 100% Local (NumPy / OpenCV) | No | None |
| **Optical Quality Assessment** | 100% Local (Laplacian / RMS) | No | None |
| **Otsu & Watershed Segmentation** | 100% Local (OpenCV) | No | None |
| **Temporal Region Tracking** | 100% Local (Centroid Matching) | No | None |
| **ConvLSTM GAN Forecasting** | 100% Local (TensorFlow / Keras) | No | None |
| **Hotspot Heatmaps & Overlay** | 100% Local (OpenCV) | No | None |
| **Multimodal VLM Analysis** | Cloud Vision API (Groq) | Yes (`GROQ_API_KEY`) | Downsampled Base64 JPEG |
| **Clinical Growth Report** | Cloud Text API (Groq) | Yes (`GROQ_API_KEY`) | Numerical Metrics Only |
| **ReAct Research Agent** | Cloud Text API (Groq) | Yes (`GROQ_API_KEY`) | Structured Tool Outputs |

---

## 🔬 Dataset & Sequences

### 1. Real LIVECell B10_1 Sequence (FOV 4)
- **Included in `sorted_multiples_of_4/B10_1/`**: 22 authentic phase-contrast micrographs captured at 4-hour intervals covering 00h to 84h.
- **FOV Filtering**: Uses strictly **FOV 4** frames (`0002_pos0004...` through `0023_pos0088...`). The single FOV 1 image (`0001_pos0001...`) is excluded from time-series analysis because it belongs to a different spatial coordinate frame.
- **Evaluation Status**: This archive provides raw microscopy images without ground-truth masks. The application compares the forecasted mask against an **algorithm-generated reference** (Otsu segmentation of actual Frame 5). It is explicitly **not** labeled as manual ground truth.

### 2. Sample Demo Sequence (`sample_data/`)
- A labeled synthetic demonstration sequence (`frame_00h.png` to `frame_16h_target.png`) with true synthetic ground-truth masks for offline testing.

---

## 👁️ Multimodal Vision-Language Model (VLM) Setup

1. **Free Tier Setup**:
   - Register for a free API key at [GroqCloud Console](https://console.groq.com/).
   - Set the environment variable:
     ```bash
     export GROQ_API_KEY="your_api_key_here"       # Linux / macOS
     set GROQ_API_KEY=your_api_key_here          # Windows Command Prompt
     $env:GROQ_API_KEY="your_api_key_here"       # Windows PowerShell
     ```
   - Alternatively, enter your API key directly in the top navigation bar of the web app or the Streamlit sidebar at runtime.
2. **Supported Vision Models**:
   - `llama-3.2-11b-vision-preview` (Default, low latency, free-tier eligible)
   - `llama-3.2-90b-vision-preview` (Higher capacity)
   - `qwen/qwen3.6-27b` (Multimodal)
3. **Scientific Guardrails**:
   - The model is guided to separate direct visible optical evidence (diffraction halos, density, borders) from biological interpretations.
   - It explicitly communicates uncertainty and avoids speculative diagnoses of apoptosis or cytopathology without biochemical assays.

---

## 🏃 Quick Start Guide

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch FastAPI Web Application
python -m uvicorn app:app --port 8000

# 3. Open browser at http://localhost:8000
```

---

## ⚠️ Known Limitations
1. **Model Resolution**: The ConvLSTM generator operates on $128 \times 128 \times 1$ binary masks. Bicubic upscaling to native $520 \times 704$ introduces boundary smoothing.
2. **Optical Clumping**: In high-confluence neuroblastoma cultures (>70%), touching diffraction halos can cause classical Otsu to merge neighboring cells. The Watershed feature provides partition estimates, but cannot substitute for fluorescent nuclear staining (e.g. DAPI/Hoechst).
3. **Ground-Truth Masks**: The uploaded B10_1 archive contains raw phase-contrast images only. True IoU/Dice evaluation requires expert manual polygon annotations.
