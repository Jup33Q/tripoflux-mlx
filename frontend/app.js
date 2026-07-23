import SplatViewer from '/frontend/splat-viewer.js';

(() => {
  const $ = (id) => document.getElementById(id);
  const promptEl = $("prompt");
  const negativePromptEl = $("negative_prompt");
  const seedEl = $("seed");
  const widthEl = $("width");
  const heightEl = $("height");
  const numGaussiansEl = $("num_gaussians");
  const splatStepsEl = $("splat_steps");
  const fluxQuantizeEl = $("flux_quantize");
  const generateBtn = $("generate");
  const progressBar = $("progressBar");
  const progressText = $("progressText");
  const statusEl = $("status");
  const generatedImage = $("generatedImage");
  const rgbaImage = $("rgbaImage");
  const downloadPly = $("downloadPly");
  const downloadSpz = $("downloadSpz");
  const splatCanvas = $("splatCanvas");
  const terminalEl = $("terminal");

  const STAGE_ORDER = ["flux", "birefnet", "triposplat"];
  const STAGE_LABELS = {
    flux: "FLUX Image",
    birefnet: "Background Removal",
    triposplat: "TripoSplat 3D",
  };
  const STAGES = {
    flux: { el: $("stageFlux"), fill: $("stageFluxFill"), pct: $("stageFluxPct"), indeterminate: false },
    birefnet: { el: $("stageBirefnet"), fill: $("stageBirefnetFill"), pct: $("stageBirefnetPct"), indeterminate: true },
    triposplat: { el: $("stageTriposplat"), fill: $("stageTriposplatFill"), pct: $("stageTriposplatPct"), indeterminate: false },
  };

  let splatViewer = null;

  function setProgress(frac, stage) {
    progressBar.style.width = `${Math.round(frac * 100)}%`;
    progressText.textContent = `${stage} ${Math.round(frac * 100)}%`;
  }

  function setStageState(s, state, frac) {
    s.el.dataset.state = state;
    const f = Math.max(0, Math.min(1, frac || 0));
    if (state === "done") {
      s.fill.style.width = "100%";
      s.pct.textContent = "100%";
    } else {
      s.fill.style.width = `${Math.round(f * 100)}%`;
      s.pct.textContent = state === "active" ? `${Math.round(f * 100)}%` : "0%";
    }
    if (s.indeterminate && state === "active" && f < 1) {
      s.el.setAttribute("data-indeterminate", "");
    } else {
      s.el.removeAttribute("data-indeterminate");
    }
  }

  function resetStages() {
    for (const key of STAGE_ORDER) {
      setStageState(STAGES[key], "pending", 0);
    }
  }

  function updateStages(stage, frac) {
    const idx = STAGE_ORDER.indexOf(stage);
    if (idx === -1) return;
    STAGE_ORDER.forEach((key, i) => {
      const s = STAGES[key];
      if (i < idx) setStageState(s, "done", 1);
      else if (i === idx) setStageState(s, "active", frac);
    });
  }

  function finishStages() {
    for (const key of STAGE_ORDER) {
      setStageState(STAGES[key], "done", 1);
    }
  }

  function setStatus(msg, isError = false) {
    statusEl.textContent = msg;
    statusEl.className = isError ? "status error" : "status";
  }

  function log(msg) {
    const line = document.createElement("div");
    line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    terminalEl.appendChild(line);
    terminalEl.scrollTop = terminalEl.scrollHeight;
  }

  function resetOutputs() {
    generatedImage.src = "";
    rgbaImage.src = "";
    downloadPly.removeAttribute("href");
    downloadSpz.removeAttribute("href");
    terminalEl.innerHTML = "";
    if (splatViewer) {
      splatViewer.dispose();
      splatViewer = null;
    }
  }

  async function startGeneration() {
    const prompt = promptEl.value.trim();
    if (!prompt) {
      setStatus("Please enter a prompt", true);
      return;
    }

    generateBtn.disabled = true;
    resetOutputs();
    resetStages();
    setProgress(0, "starting");
    setStatus("Submitting job...");

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          negative_prompt: negativePromptEl.value.trim() || null,
          seed: seedEl.value ? parseInt(seedEl.value, 10) : null,
          width: parseInt(widthEl.value, 10),
          height: parseInt(heightEl.value, 10),
          num_gaussians: parseInt(numGaussiansEl.value, 10),
          splat_steps: parseInt(splatStepsEl.value, 10),
          flux_quantize: fluxQuantizeEl.value ? parseInt(fluxQuantizeEl.value, 10) : null,
        }),
      });
      if (!res.ok) throw new Error(`generate failed: ${res.status}`);
      const { job_id } = await res.json();
      setStatus(`Job ${job_id} started`);

      const evtSource = new EventSource(`/api/stream/${job_id}`);
      evtSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.log) {
          log(data.log);
        }
        if (data.image_ready === "flux") {
          generatedImage.src = `/api/result/${job_id}/stage/flux?t=${Date.now()}`;
          log("FLUX image preview updated");
        }
        if (data.image_ready === "rgba") {
          rgbaImage.src = `/api/result/${job_id}/stage/rgba?t=${Date.now()}`;
          log("RGBA image preview updated");
        }
        if (data.status === "failed") {
          setStatus(`Failed: ${data.error || "unknown"}`, true);
          evtSource.close();
          generateBtn.disabled = false;
          return;
        }
        if (data.status === "completed") {
          setProgress(1, "done");
          finishStages();
          setStatus("Completed");
          loadResults(job_id);
          evtSource.close();
          generateBtn.disabled = false;
          return;
        }
        if (data.stage && STAGE_ORDER.includes(data.stage)) {
          updateStages(data.stage, data.stage_progress || 0);
        }
        const label = STAGE_LABELS[data.stage] || data.stage || "running";
        setProgress(data.progress || 0, label);
      };
      evtSource.onerror = () => {
        setStatus("Stream error / connection closed", true);
        evtSource.close();
        generateBtn.disabled = false;
      };
    } catch (err) {
      setStatus(err.message, true);
      generateBtn.disabled = false;
    }
  }

  function loadResults(jobId) {
    generatedImage.src = `/api/result/${jobId}/image?t=${Date.now()}`;
    rgbaImage.src = `/api/result/${jobId}/rgba?t=${Date.now()}`;
    downloadPly.href = `/api/result/${jobId}/ply`;
    downloadPly.download = `${jobId}.ply`;
    downloadSpz.href = `/api/result/${jobId}/spz`;
    downloadSpz.download = `${jobId}.spz`;
    initSplatViewer(jobId);
  }

  async function initSplatViewer(jobId) {
    try {
      const res = await fetch(`/api/result/${jobId}/splat`);
      if (!res.ok) throw new Error("splat fetch failed");
      const buffer = await res.arrayBuffer();
      splatViewer = new SplatViewer(splatCanvas, buffer);
      splatViewer.render();
    } catch (err) {
      console.error("splat preview error", err);
    }
  }

  generateBtn.addEventListener("click", startGeneration);
})();
