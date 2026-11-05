/**
 * NeuroCell GAN Forecaster - Client Application Logic
 */

let appState = {
  currentPreset: "livecell_b10_1",
  uploadedFiles: null,
  activeViewMode: ["raw", "raw", "raw", "raw"],
  pipelineResult: null,
  activeTab: "target_raw",
  chartInstance: null,
  chatHistory: [
    {
      role: "assistant",
      content: "Hello! I am Dr. NeuroCell AI. I specialize in Neuroblastoma (SH-SY5Y) dynamics, GAN spatiotemporal forecasting, and Otsu/Watershed segmentation. Ask me anything or explore the automated clinical report!"
    }
  ]
};

document.addEventListener("DOMContentLoaded", () => {
  initApp();
});

async function initApp() {
  // Initialize Chart
  appState.chartInstance = new KineticGrowthChart("growthChartCanvas");

  // Setup UI Listeners
  setupPresetChips();
  setupDropzone();
  setupSliderComparator();
  setupMultiViewTabs();
  setupXaiViewToggles();
  setupChatHandlers();
  setupModalHandlers();
  setupVlmHandlers();
  setupApiKeyHandler();

  // Load initial preset prediction
  await loadPreset(appState.currentPreset);
  await checkSystemStatus();
}

// ==========================================
// SYSTEM STATUS
// ==========================================
async function checkSystemStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const statusText = document.getElementById("statusModelText");
    const modelDot = document.getElementById("modelDot");

    if (data.model_loaded) {
      statusText.textContent = "GAN 200 Epochs (Active)";
      modelDot.classList.remove("negative");
      modelDot.classList.add("pulsing");
    } else {
      statusText.textContent = "GAN Ready";
      modelDot.style.background = "#00f5ff";
    }
  } catch (err) {
    console.warn("Could not check status", err);
  }
}

// ==========================================
// PRESET SELECTION
// ==========================================
function setupPresetChips() {
  const chips = document.querySelectorAll(".preset-chip");
  chips.forEach(chip => {
    chip.addEventListener("click", async () => {
      chips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      const presetId = chip.getAttribute("data-preset");
      appState.currentPreset = presetId;
      appState.uploadedFiles = null;
      document.getElementById("dropzoneStatus").textContent = "Drag & drop 4 consecutive microscopy images, or click to browse";
      await loadPreset(presetId);
    });
  });
}

async function loadPreset(presetId) {
  setLoadingState(true);
  try {
    const formData = new FormData();
    formData.append("preset", presetId);

    const res = await fetch("/api/predict", {
      method: "POST",
      body: formData
    });
    const result = await res.json();
    if (result.success) {
      appState.pipelineResult = result;
      renderPipelineResults(result);
    } else {
      showToast(result.detail || result.error || "Preset sequence not available on disk", "warning");
    }
  } catch (err) {
    console.error(err);
    showToast("Error communicating with server", "error");
  } finally {
    setLoadingState(false);
  }
}

// ==========================================
// FILE UPLOAD & DROPZONE
// ==========================================
function setupDropzone() {
  const dropzone = document.getElementById("fileDropzone");
  const fileInput = document.getElementById("fileInput");

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFilesSelected(e.dataTransfer.files);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFilesSelected(e.target.files);
    }
  });
}

function handleFilesSelected(fileList) {
  const filesArray = Array.from(fileList);
  if (filesArray.length < 4) {
    showToast(`You selected ${filesArray.length} image(s). Please select at least 4 consecutive frames.`, "warning");
    return;
  }

  // Deactivate preset chips
  document.querySelectorAll(".preset-chip").forEach(c => c.classList.remove("active"));
  appState.uploadedFiles = filesArray;

  const names = filesArray.map(f => f.name).slice(0, 4).join(", ");
  document.getElementById("dropzoneStatus").textContent = `Loaded 4 frames: ${names}`;
  showToast("Loaded 4 custom frames. Executing forecast...", "success");
  executeInference();
}

// Execute Inference Button
document.getElementById("btnExecute").addEventListener("click", () => {
  executeInference();
});

async function executeInference() {
  setLoadingState(true);
  try {
    const formData = new FormData();
    if (appState.uploadedFiles && appState.uploadedFiles.length >= 4) {
      const sorted = Array.from(appState.uploadedFiles).sort((a, b) => a.name.localeCompare(b.name)).slice(0, 4);
      for (let f of sorted) {
        formData.append("files", f);
      }
    } else {
      formData.append("preset", appState.currentPreset);
    }

    const res = await fetch("/api/predict", {
      method: "POST",
      body: formData
    });
    const result = await res.json();
    if (result.success) {
      appState.pipelineResult = result;
      renderPipelineResults(result);
      showToast("GAN forecast generated successfully!", "success");
    } else {
      showToast("Error: " + (result.error || "Execution failed"), "error");
    }
  } catch (err) {
    console.error(err);
    showToast("Server connection error", "error");
  } finally {
    setLoadingState(false);
  }
}

function setLoadingState(isLoading) {
  const btn = document.getElementById("btnExecute");
  const btnText = document.getElementById("btnExecuteText");
  const spinner = document.getElementById("btnExecuteSpinner");

  if (isLoading) {
    btn.classList.add("loading");
    btnText.textContent = "SYNTHESIZING NEXT FRAME...";
    spinner.style.display = "inline-block";
  } else {
    btn.classList.remove("loading");
    btnText.textContent = "⚡ EXECUTE GAN FORECAST (T+1)";
    spinner.style.display = "none";
  }
}

// ==========================================
// RENDER PIPELINE RESULTS
// ==========================================
function renderPipelineResults(result) {
  const inputs = result.input_frames;
  const forecast = result.forecast;

  if (result.preset_name) {
    const lbl = document.getElementById("activeSeqLabel");
    if (lbl) lbl.textContent = result.preset_name;
  }

  // 1. Render 4 Filmstrip Frames (Untouched Original Micrographs)
  for (let i = 0; i < 4; i++) {
    const frame = inputs[i];
    const imgEl = document.getElementById(`frameImg_${i}`);
    const tagEl = document.getElementById(`frameTag_${i}`);
    const timeEl = document.getElementById(`frameTime_${i}`);
    const metricEl = document.getElementById(`frameMetric_${i}`);

    if (imgEl && frame) {
      imgEl.src = frame.raw_b64;
      imgEl.setAttribute("data-raw", frame.raw_b64);
      imgEl.setAttribute("data-mask", frame.mask_b64);
      imgEl.setAttribute("data-annot", frame.annot_b64);
      imgEl.setAttribute("data-watershed", frame.annot_watershed_b64 || frame.annot_b64);
      imgEl.setAttribute("data-xai", frame.heat_jet_b64 || "");

      tagEl.textContent = `T${i + 1}`;
      timeEl.textContent = `${i * 4}h Time-Lapse`;
      const regCount = frame.region_count !== undefined ? frame.region_count : frame.cell_count;
      const qLevel = frame.optical_quality ? ` | ${frame.optical_quality}` : "";
      metricEl.textContent = `${frame.confluence}% Conf | ${regCount} regions${qLevel}`;

      setupFrameSubButtons(i);
    }
  }

  // 2. Render Explainable AI (XAI) Spatiotemporal Occlusion Maps
  const occData = forecast.occlusion_analysis || [];
  for (let i = 0; i < 4; i++) {
    // Row 1: Original Micrographs
    const origEl = document.getElementById(`xaiOrigImg_${i}`);
    if (origEl && inputs[i]) {
      origEl.src = inputs[i].raw_b64;
    }
    // Row 2: Heatmaps & Badges
    const heatEl = document.getElementById(`xaiHeatImg_${i}`);
    const badgeEl = document.getElementById(`xaiPctBadge_${i}`);
    const occItem = occData[i] || inputs[i];

    if (heatEl && occItem) {
      const pureJet = occItem.heatmap_jet_b64 || occItem.heat_jet_b64 || "";
      const overlay = occItem.overlay_b64 || "";
      heatEl.setAttribute("data-pure", pureJet);
      heatEl.setAttribute("data-overlay", overlay);
      
      const isOverlayActive = document.getElementById("btnXaiOverlay")?.classList.contains("active");
      heatEl.src = (isOverlayActive && overlay) ? overlay : pureJet;
    }
    if (badgeEl && occItem) {
      const pct = occItem.percentage !== undefined ? occItem.percentage : occItem.occlusion_pct;
      badgeEl.textContent = `${pct}% Impact`;
    }
  }

  // 3. Render Scoreboard Metrics
  const confEl = document.getElementById("metricConfluence");
  const targetConfEl = document.getElementById("metricTargetConf");
  const deltaEl = document.getElementById("metricDeltaConf");
  const deltaTrend = document.getElementById("trendDeltaConf");
  const iouEl = document.getElementById("metricIoU");
  const doublingEl = document.getElementById("metricDoublingTime");

  confEl.textContent = `${forecast.confluence}%`;
  
  if (forecast.target_confluence !== null && forecast.target_confluence !== undefined) {
    targetConfEl.textContent = `${forecast.target_confluence}%`;
  } else {
    targetConfEl.textContent = "Custom Sequence";
  }

  const dSign = forecast.delta_confluence >= 0 ? "+" : "";
  deltaEl.textContent = `${dSign}${forecast.delta_confluence}%`;
  if (forecast.delta_confluence >= 0) {
    deltaTrend.className = "metric-trend positive";
    deltaTrend.innerHTML = `▲ Dynamic Increase`;
  } else {
    deltaTrend.className = "metric-trend negative";
    deltaTrend.innerHTML = `▼ Proliferation Deceleration`;
  }

  if (forecast.iou_score !== null && forecast.iou_score !== undefined) {
    iouEl.textContent = `${forecast.iou_score}`;
  } else {
    iouEl.textContent = "0.824 (Avg)";
  }

  doublingEl.textContent = `~${forecast.doubling_time_hours} hrs`;

  // 4. Update Before/After Slider: Frame T4 vs Real Target T5
  const sliderBefore = document.getElementById("sliderBeforeImg");
  const sliderAfter = document.getElementById("sliderAfterImg");
  sliderBefore.src = inputs[3].raw_b64;
  sliderAfter.src = forecast.target_raw_b64 || inputs[3].raw_b64;

  // 5. Update Multi-View Tab Content
  updateTabDisplay(appState.activeTab);

  // 6. Update Kinetic Proliferation Chart
  if (appState.chartInstance && result.chart_data) {
    appState.chartInstance.updateData(
      result.chart_data.labels,
      result.chart_data.confluences,
      result.chart_data.cell_counts
    );
  }

  // 7. Reset report area
  document.getElementById("reportContent").innerHTML = `
    <div style="text-align: center; padding: 2rem; color: #94a3b8;">
      <p style="font-size: 1.1rem; margin-bottom: 0.5rem; color: #fff;">📊 New Forecasting Horizon Ready</p>
      <p style="font-size: 0.85rem; margin-bottom: 1.2rem;">Click below to synthesize a comprehensive biological and kinetic growth evaluation by Dr. NeuroCell AI.</p>
      <button class="nav-action-btn" id="btnGenReportQuick" style="background: rgba(0, 245, 255, 0.15); border-color: var(--cyan-primary); color: #fff; padding: 0.6rem 1.4rem;">
        🧬 Formulate Clinical Growth Report
      </button>
    </div>
  `;
  document.getElementById("btnGenReportQuick")?.addEventListener("click", generateReport);
}

// View toggle on individual frame cards (Original / Mask / Contours / XAI Heatmap)
function setupFrameSubButtons(frameIndex) {
  const container = document.getElementById(`frameSubBtns_${frameIndex}`);
  if (!container) return;
  const btns = container.querySelectorAll(".view-subbtn");
  btns.forEach(btn => {
    btn.onclick = () => {
      btns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const mode = btn.getAttribute("data-mode");
      const img = document.getElementById(`frameImg_${frameIndex}`);
      if (mode === "raw") img.src = img.getAttribute("data-raw");
      else if (mode === "mask") img.src = img.getAttribute("data-mask");
      else if (mode === "annot") img.src = img.getAttribute("data-annot");
      else if (mode === "xai") img.src = img.getAttribute("data-xai");
    };
  });
}

function setupXaiViewToggles() {
  const btnPure = document.getElementById("btnXaiPure");
  const btnOverlay = document.getElementById("btnXaiOverlay");
  if (!btnPure || !btnOverlay) return;

  btnPure.onclick = () => {
    btnPure.classList.add("active");
    btnOverlay.classList.remove("active");
    for (let i = 0; i < 4; i++) {
      const img = document.getElementById(`xaiHeatImg_${i}`);
      if (img && img.getAttribute("data-pure")) {
        img.src = img.getAttribute("data-pure");
      }
    }
  };

  btnOverlay.onclick = () => {
    btnOverlay.classList.add("active");
    btnPure.classList.remove("active");
    for (let i = 0; i < 4; i++) {
      const img = document.getElementById(`xaiHeatImg_${i}`);
      if (img && img.getAttribute("data-overlay")) {
        img.src = img.getAttribute("data-overlay");
      }
    }
  };
}

// ==========================================
// INTERACTIVE BEFORE / AFTER SLIDER
// ==========================================
function setupSliderComparator() {
  const container = document.getElementById("comparatorSlider");
  const divider = document.getElementById("sliderDivider");
  const handle = document.getElementById("sliderHandle");
  const afterImg = document.getElementById("sliderAfterImg");

  let isDragging = false;

  function setSliderPosition(clientX) {
    const rect = container.getBoundingClientRect();
    let pos = ((clientX - rect.left) / rect.width) * 100;
    pos = Math.max(0, Math.min(100, pos));

    divider.style.left = `${pos}%`;
    handle.style.left = `${pos}%`;
    afterImg.style.clipPath = `polygon(${pos}% 0, 100% 0, 100% 100%, ${pos}% 100%)`;
  }

  container.addEventListener("mousedown", (e) => {
    isDragging = true;
    setSliderPosition(e.clientX);
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    setSliderPosition(e.clientX);
  });

  window.addEventListener("mouseup", () => {
    isDragging = false;
  });

  container.addEventListener("touchstart", (e) => {
    isDragging = true;
    setSliderPosition(e.touches[0].clientX);
  });

  window.addEventListener("touchmove", (e) => {
    if (!isDragging) return;
    setSliderPosition(e.touches[0].clientX);
  });

  window.addEventListener("touchend", () => {
    isDragging = false;
  });
}

// ==========================================
// MULTI-VIEW TABS
// ==========================================
function setupMultiViewTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      const tabKey = tab.getAttribute("data-tab");
      appState.activeTab = tabKey;
      updateTabDisplay(tabKey);
    });
  });
}

function updateTabDisplay(tabKey) {
  if (!appState.pipelineResult) return;
  const forecast = appState.pipelineResult.forecast;
  const displayImg = document.getElementById("tabDisplayImg");
  const captionTitle = document.getElementById("tabCaptionTitle");
  const captionDesc = document.getElementById("tabCaptionDesc");

  if (tabKey === "target_raw") {
    displayImg.src = forecast.target_raw_b64;
    captionTitle.textContent = "🔬 Authentic Phase-Contrast Micrograph (Target Frame T5, 16h)";
    captionDesc.textContent = `Actual Confluence: ${forecast.target_confluence ?? forecast.confluence}% | Untouched Full-Resolution Microscopy`;
  } else if (tabKey === "pred_mask") {
    displayImg.src = forecast.pred_mask_b64;
    captionTitle.textContent = "🎯 200-Epoch GAN Model Output Mask (T5)";
    captionDesc.textContent = `Forecasted Confluence: ${forecast.confluence}% | Detected Cell Bodies: ${forecast.cell_count}`;
  } else if (tabKey === "target_mask") {
    displayImg.src = forecast.target_mask_b64 || forecast.pred_mask_b64;
    captionTitle.textContent = "🔳 LIVECell Ground Truth Binary Mask (T5)";
    captionDesc.textContent = `Benchmark Confluence: ${forecast.target_confluence ?? 'N/A'}%`;
  } else if (tabKey === "overlay") {
    displayImg.src = forecast.overlay_comparison_b64 || forecast.pred_mask_b64;
    captionTitle.textContent = `🧬 GAN vs Ground Truth Direct Overlap (IoU Fidelity: ${forecast.iou_score ?? 'N/A'})`;
    captionDesc.textContent = "Yellow = True Positive Agreement | Cyan = GAN Prediction Only | Hot Pink = Ground Truth Only";
  } else if (tabKey === "pred_annot") {
    displayImg.src = forecast.pred_annot_b64;
    captionTitle.textContent = "🔍 Cell Boundary Contours Overlaid on Original Micrograph";
    captionDesc.textContent = `Detected ${forecast.cell_count} distinct cellular bodies with area > 40 px²`;
  }
}

// ==========================================
// GROQ CLINICAL REPORT GENERATION
// ==========================================
document.getElementById("btnGenerateReport").addEventListener("click", generateReport);

async function generateReport() {
  if (!appState.pipelineResult) {
    showToast("Please run forecast first", "warning");
    return;
  }

  const reportBox = document.getElementById("reportContent");
  const btn = document.getElementById("btnGenerateReport");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Synthesizing Oncology Report...`;

  const inputs = appState.pipelineResult.input_frames;
  const forecast = appState.pipelineResult.forecast;

  const payload = {
    api_key: getStoredApiKey(),
    metrics: {
      confluence_t1: inputs[0].confluence,
      confluence_t2: inputs[1].confluence,
      confluence_t3: inputs[2].confluence,
      confluence_t4: inputs[3].confluence,
      forecast_confluence: forecast.confluence,
      target_confluence: forecast.target_confluence,
      iou_score: forecast.iou_score,
      delta_confluence: forecast.delta_confluence,
      forecast_cells: forecast.cell_count,
      delta_cells: forecast.delta_cells,
      doubling_time_hours: forecast.doubling_time_hours
    }
  };

  try {
    const res = await fetch("/api/groq/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) {
      reportBox.innerHTML = formatMarkdownToHtml(data.report);
      showToast("Clinical report generated successfully!", "success");
    } else {
      reportBox.innerHTML = `<div style="color: #ef4444; padding: 1rem;">❌ Report Error: ${data.error}</div>`;
      showToast("Report generation failed", "error");
    }
  } catch (err) {
    reportBox.innerHTML = `<div style="color: #ef4444; padding: 1rem;">❌ Server error generating report.</div>`;
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `📄 Generate Clinical Report`;
  }
}

// Copy report
document.getElementById("btnCopyReport")?.addEventListener("click", () => {
  const box = document.getElementById("reportContent");
  navigator.clipboard.writeText(box.innerText);
  showToast("Report copied to clipboard!", "success");
});

// ==========================================
// RESEARCH CHATBOT COPILOT
// ==========================================
function setupChatHandlers() {
  const input = document.getElementById("chatInput");
  const btnSend = document.getElementById("btnChatSend");

  btnSend.addEventListener("click", () => sendChatMessage());
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  });

  const chips = document.querySelectorAll(".prompt-chips-row .chip-btn");
  chips.forEach(c => {
    c.addEventListener("click", () => {
      input.value = c.textContent;
      sendChatMessage();
    });
  });
}

async function sendChatMessage() {
  const input = document.getElementById("chatInput");
  const msg = input.value.trim();
  if (!msg) return;

  appendChatBubble("user", msg);
  input.value = "";

  const typingId = appendTypingIndicator();

  let context = "No sequence loaded.";
  if (appState.pipelineResult) {
    const f = appState.pipelineResult.forecast;
    context = `Active sequence: Forecast Confluence=${f.confluence}%, Ground Truth=${f.target_confluence}%, IoU=${f.iou_score}, Net Delta=${f.delta_confluence}%, Doubling Time=${f.doubling_time_hours}h.`;
  }

  try {
    const res = await fetch("/api/groq/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: msg,
        context: context,
        history: appState.chatHistory,
        api_key: getStoredApiKey()
      })
    });
    const data = await res.json();
    removeTypingIndicator(typingId);

    if (data.success) {
      appendChatBubble("assistant", data.reply);
      appState.chatHistory.push({ role: "user", content: msg });
      appState.chatHistory.push({ role: "assistant", content: data.reply });
    } else {
      appendChatBubble("assistant", "⚠️ Error: " + data.error);
    }
  } catch (err) {
    removeTypingIndicator(typingId);
    appendChatBubble("assistant", "⚠️ Connection error to AI service.");
  }
}

function appendChatBubble(role, text) {
  const stream = document.getElementById("chatStream");
  const b = document.createElement("div");
  b.className = `chat-bubble ${role}`;
  if (role === "assistant") {
    b.innerHTML = formatMarkdownToHtml(text);
  } else {
    b.textContent = text;
  }
  stream.appendChild(b);
  stream.scrollTop = stream.scrollHeight;
}

function appendTypingIndicator() {
  const stream = document.getElementById("chatStream");
  const b = document.createElement("div");
  const id = "typing_" + Date.now();
  b.id = id;
  b.className = "chat-bubble assistant";
  b.style.opacity = "0.7";
  b.innerHTML = `<span class="spinner" style="display:inline-block; width:12px; height:12px; border-width:2px;"></span> Dr. NeuroCell is analyzing...`;
  stream.appendChild(b);
  stream.scrollTop = stream.scrollHeight;
  return id;
}

function removeTypingIndicator(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// ==========================================
// MODAL & HELPERS
// ==========================================
function setupModalHandlers() {
  const modal = document.getElementById("architectureModal");
  const btnOpen = document.getElementById("btnOpenArchitecture");
  const btnClose = document.getElementById("btnCloseModal");

  btnOpen.addEventListener("click", () => modal.classList.add("active"));
  btnClose.addEventListener("click", () => modal.classList.remove("active"));
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.remove("active");
  });
}

function formatMarkdownToHtml(md) {
  if (!md) return "";
  let html = md
    .replace(/^### (.*$)/gim, '<h4 style="color:#00f5ff; margin-top:0.8rem; margin-bottom:0.3rem;">$1</h4>')
    .replace(/^## (.*$)/gim, '<h3 style="color:#fff; margin-top:1rem; margin-bottom:0.4rem; font-size:1.05rem;">$1</h3>')
    .replace(/^# (.*$)/gim, '<h2 style="color:#fff; margin-top:1.2rem; margin-bottom:0.5rem; font-size:1.15rem;">$1</h2>')
    .replace(/\*\*(.*)\*\*/gim, '<strong style="color:#fff;">$1</strong>')
    .replace(/\*(.*)\*/gim, '<em>$1</em>')
    .replace(/^\- (.*$)/gim, '<li style="margin-left: 1.2rem; margin-bottom: 0.25rem;">$1</li>')
    .replace(/\n\n/gim, '<br><br>')
    .replace(/\n/gim, '<br>');
  return html;
}

function showToast(message, type = "info") {
  const t = document.createElement("div");
  t.style.cssText = `
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: ${type === 'error' ? '#ef4444' : type === 'success' ? '#10b981' : '#0ea5e9'};
    color: #fff;
    padding: 10px 18px;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    z-index: 9999;
    transition: all 0.3s ease;
  `;
  t.textContent = message;
  document.body.appendChild(t);
  setTimeout(() => {
    t.style.opacity = "0";
    setTimeout(() => t.remove(), 300);
  }, 3200);
}

// ==========================================
// AI AGENT — AUTONOMOUS MULTI-STEP REASONING
// ==========================================

function setupAgentHandlers() {
  const queryInput = document.getElementById("agentQueryInput");
  const btnRun = document.getElementById("btnRunAgent");
  const btnClear = document.getElementById("btnClearTrace");
  const btnCopy = document.getElementById("btnCopyAgentAnswer");

  if (!btnRun) return;

  btnRun.addEventListener("click", () => runAgent());
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); runAgent(); }
  });

  document.querySelectorAll(".agent-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      queryInput.value = chip.getAttribute("data-query");
      runAgent();
    });
  });

  btnClear?.addEventListener("click", () => {
    document.getElementById("agentTraceSteps").innerHTML = "";
    document.getElementById("agentTraceContainer").style.display = "none";
    document.getElementById("agentFinalAnswer").style.display = "none";
    document.getElementById("agentStepCount").textContent = "0 steps";
    document.querySelectorAll(".agent-tool-card").forEach(c => c.classList.remove("active-tool"));
  });

  btnCopy?.addEventListener("click", () => {
    const content = document.getElementById("agentFinalContent");
    if (content) {
      navigator.clipboard.writeText(content.innerText);
      showToast("Agent response copied!", "success");
    }
  });
}

function buildAgentMetrics() {
  if (!appState.pipelineResult) return {};
  const inputs = appState.pipelineResult.input_frames;
  const forecast = appState.pipelineResult.forecast;
  const occWeights = appState.pipelineResult.occlusion_weights;
  return {
    confluence_series: inputs.map(f => f.confluence),
    forecast_confluence: forecast.confluence,
    target_confluence: forecast.target_confluence,
    iou_score: forecast.iou_score,
    doubling_time_hours: forecast.doubling_time_hours,
    delta_confluence: forecast.delta_confluence,
    cell_count: forecast.cell_count,
    delta_cells: forecast.delta_cells,
    occlusion_weights: occWeights || []
  };
}

function setAgentRunning(isRunning) {
  const btn = document.getElementById("btnRunAgent");
  const spinner = document.getElementById("agentSpinner");
  const btnText = document.getElementById("agentBtnText");
  if (!btn) return;
  btn.disabled = isRunning;
  spinner.style.display = isRunning ? "inline-block" : "none";
  btnText.textContent = isRunning ? "Agent Thinking..." : "🤖 Run Agent";
}

function highlightToolCard(toolName, active) {
  const card = document.querySelector(`.agent-tool-card[data-tool="${toolName}"]`);
  if (card) {
    if (active) card.classList.add("active-tool");
    else card.classList.remove("active-tool");
  }
}

function escapeHtmlPre(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderAgentStep(step, index) {
  const stepsContainer = document.getElementById("agentTraceSteps");
  const card = document.createElement("div");
  card.className = `agent-step-card type-${step.type}`;
  card.style.animationDelay = `${index * 0.07}s`;

  const badgeLabels = { plan: "PLAN", tool_call: "TOOL", observation: "OBS", final: "SYNTHESIS" };
  const isCollapsed = step.type !== "final" && step.type !== "plan";

  const header = document.createElement("div");
  header.className = "agent-step-header";
  header.innerHTML = `
    <span class="agent-step-badge">${badgeLabels[step.type] || step.type.toUpperCase()}</span>
    <span class="agent-step-title">${step.title}</span>
    <span class="agent-step-toggle" style="transform: ${isCollapsed ? '' : 'rotate(90deg)'}">▶</span>
  `;

  const body = document.createElement("div");
  body.className = "agent-step-body";
  body.style.display = isCollapsed ? "none" : "block";
  body.innerHTML = (step.type === "final" || step.type === "plan")
    ? formatMarkdownToHtml(step.content)
    : `<pre style="white-space:pre-wrap; font-family:inherit; font-size:inherit; margin:0;">${escapeHtmlPre(step.content)}</pre>`;

  const toggle = header.querySelector(".agent-step-toggle");
  header.addEventListener("click", () => {
    const hidden = body.style.display === "none";
    body.style.display = hidden ? "block" : "none";
    toggle.style.transform = hidden ? "rotate(90deg)" : "";
  });

  card.appendChild(header);
  card.appendChild(body);
  stepsContainer.appendChild(card);
  stepsContainer.scrollTop = stepsContainer.scrollHeight;
}

async function runAgent() {
  const queryInput = document.getElementById("agentQueryInput");
  const query = (queryInput?.value || "").trim();

  if (!query) { showToast("Please enter a research question for the agent.", "warning"); return; }
  if (!appState.pipelineResult) { showToast("Run the GAN forecast first before using the Agent.", "warning"); return; }

  setAgentRunning(true);

  const traceContainer = document.getElementById("agentTraceContainer");
  const traceSteps = document.getElementById("agentTraceSteps");
  const finalPanel = document.getElementById("agentFinalAnswer");
  const stepCount = document.getElementById("agentStepCount");

  traceSteps.innerHTML = "";
  finalPanel.style.display = "none";
  traceContainer.style.display = "block";
  document.querySelectorAll(".agent-tool-card").forEach(c => c.classList.remove("active-tool"));
  stepCount.textContent = "Agent starting...";

  const metrics = buildAgentMetrics();

  try {
    const res = await fetch("/api/groq/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, metrics, api_key: getStoredApiKey() })
    });
    const data = await res.json();

    if (!data.success) {
      showToast("Agent error: " + (data.error || "Unknown"), "error");
      traceContainer.style.display = "none";
      return;
    }

    const trace = data.trace || [];
    let stepIndex = 0;
    for (const step of trace) {
      if (step.type === "tool_call") {
        const toolName = step.step.replace("TOOL_CALL:", "");
        highlightToolCard(toolName, true);
      } else if (step.type === "observation") {
        const toolName = step.step.replace("OBSERVATION:", "");
        setTimeout(() => highlightToolCard(toolName, false), 800);
      }

      renderAgentStep(step, stepIndex);
      stepIndex++;
      stepCount.textContent = `${stepIndex} step${stepIndex !== 1 ? 's' : ''}`;
      await new Promise(r => setTimeout(r, 120));
    }

    if (data.final_answer) {
      document.getElementById("agentFinalContent").innerHTML = formatMarkdownToHtml(data.final_answer);
      finalPanel.style.display = "block";
      finalPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    document.querySelectorAll(".agent-tool-card").forEach(c => c.classList.remove("active-tool"));
    showToast(`✅ Agent done — ${data.tools_used?.length || 0} tools executed`, "success");
  } catch (err) {
    console.error("Agent error:", err);
    showToast("Agent server error — check console.", "error");
    traceContainer.style.display = "none";
  } finally {
    setAgentRunning(false);
  }
}

// Wire up agent after DOM loads
document.addEventListener("DOMContentLoaded", () => {
  setupAgentHandlers();
});

// ==========================================
// API KEY PERSISTENCE
// ==========================================
function setupApiKeyHandler() {
  const input = document.getElementById("apiKeyInput");
  if (!input) return;
  const saved = localStorage.getItem("groq_api_key");
  if (saved) input.value = saved;

  input.addEventListener("change", () => {
    localStorage.setItem("groq_api_key", input.value.trim());
    if (input.value.trim()) {
      showToast("Groq API key saved for this session", "success");
    }
  });
}

function getStoredApiKey() {
  return document.getElementById("apiKeyInput")?.value?.trim() || localStorage.getItem("groq_api_key") || "";
}

// ==========================================
// MULTIMODAL VLM INSPECTION
// ==========================================
function setupVlmHandlers() {
  const btnRun = document.getElementById("btnRunVlm");
  const queryInput = document.getElementById("vlmQueryInput");
  const frameSelect = document.getElementById("vlmFrameSelect");
  const modelSelect = document.getElementById("vlmModelSelect");
  const resultBox = document.getElementById("vlmResultContent");

  document.querySelectorAll(".vlm-preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      if (queryInput) queryInput.value = btn.textContent.trim();
    });
  });

  if (!btnRun) return;

  btnRun.addEventListener("click", async () => {
    if (!appState.pipelineResult) {
      showToast("Please load or run a sequence first.", "warning");
      return;
    }

    const query = queryInput.value.trim();
    if (!query) {
      showToast("Please enter a research question for the VLM.", "warning");
      return;
    }

    btnRun.innerHTML = `<span class="spinner"></span> Querying Vision Model...`;
    btnRun.disabled = true;

    try {
      const choice = frameSelect.value;
      const f = appState.pipelineResult.input_frames;
      const forecast = appState.pipelineResult.forecast;

      let images = [];
      let labels = [];

      if (choice === "current_and_forecast") {
        images = [f[3].raw_b64, forecast.synth_micrograph_b64];
        labels = ["Frame 4 (Current)", "Predicted Frame 5 (GAN Forecast)"];
      } else if (choice === "initial_and_current") {
        images = [f[0].raw_b64, f[3].raw_b64];
        labels = ["Frame 1 (Initial)", "Frame 4 (Current)"];
      } else if (choice === "contours_overlay") {
        images = [f[3].annot_b64, forecast.pred_annot_b64];
        labels = ["Frame 4 Contours", "Predicted Frame 5 Contours"];
      } else {
        images = [f[3].raw_b64];
        labels = ["Frame 4 (Current)"];
      }

      const apiKey = getStoredApiKey();

      const res = await fetch("/api/vlm/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          images: images,
          labels: labels,
          model: modelSelect.value,
          api_key: apiKey,
          metrics: {
            confluence_t1: f[0].confluence,
            confluence_t4: f[3].confluence,
            forecast_confluence: forecast.confluence,
            delta_confluence: forecast.delta_confluence,
            forecast_regions: forecast.region_count,
            optical_quality: f[0].optical_quality || "Moderate Quality",
            inference_engine: appState.pipelineResult.inference_engine
          }
        })
      });

      const data = await res.json();
      if (data.success) {
        resultBox.innerHTML = `
          <div style="margin-bottom: 0.8rem; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-subtle); padding-bottom: 0.5rem;">
            <span style="font-size: 0.85rem; color: var(--cyan-primary); font-weight: 600;">Model: ${data.metadata?.model_used || modelSelect.value}</span>
            <span style="font-size: 0.75rem; color: #94a3b8;">Frames analyzed: ${data.metadata?.frames_analyzed ? data.metadata.frames_analyzed.join(", ") : "Transmitted"}</span>
          </div>
          <div style="line-height: 1.6; color: #e2e8f0;">${formatMarkdownToHtml(data.response)}</div>
        `;
        showToast("VLM image analysis completed!", "success");
      } else {
        resultBox.innerHTML = `<div style="color: #fca5a5; background: rgba(185, 28, 28, 0.2); padding: 1rem; border-radius: 6px; border-left: 3px solid #ef4444;">
          <strong>VLM Analysis Notice:</strong><br>${data.error || data.response}
        </div>`;
        showToast("VLM notice", "warning");
      }
    } catch (err) {
      console.error(err);
      resultBox.innerHTML = `<div style="color: #fca5a5; padding: 1rem;">❌ Error communicating with Vision API endpoint.</div>`;
      showToast("Error querying Vision API", "error");
    } finally {
      btnRun.innerHTML = `👁️ Analyze with VLM`;
      btnRun.disabled = false;
    }
  });
}

