(() => {
  const $ = (id) => document.getElementById(id);
  const promptEl = $("prompt");
  const negativePromptEl = $("negative_prompt");
  const seedEl = $("seed");
  const widthEl = $("width");
  const heightEl = $("height");
  const numGaussiansEl = $("num_gaussians");
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

  let splatViewer = null;

  function setProgress(frac, stage) {
    progressBar.style.width = `${Math.round(frac * 100)}%`;
    progressText.textContent = `${stage} ${Math.round(frac * 100)}%`;
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
          setStatus("Completed");
          loadResults(job_id);
          evtSource.close();
          generateBtn.disabled = false;
          return;
        }
        setProgress(data.progress || 0, data.stage || "running");
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

  // Very small .splat parser / renderer (antialiasing-agnostic).
  class SplatViewer {
    constructor(canvas, buffer) {
      this.canvas = canvas;
      this.buffer = buffer;
      this.scene = new THREE.Scene();
      this.camera = new THREE.PerspectiveCamera(60, 1, 0.01, 100);
      this.camera.position.set(0, 0, 3);
      this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
      this.renderer.setSize(canvas.clientWidth, canvas.clientWidth, false);
      this.controls = null;
      this.points = null;
      this._parse();
      this._build();
    }

    _parse() {
      const dv = new DataView(this.buffer);
      const count = dv.byteLength / 32;
      const positions = new Float32Array(count * 3);
      const colors = new Float32Array(count * 3);
      for (let i = 0; i < count; i++) {
        const off = i * 32;
        positions[i * 3 + 0] = dv.getFloat32(off + 0, true);
        positions[i * 3 + 1] = dv.getFloat32(off + 4, true);
        positions[i * 3 + 2] = dv.getFloat32(off + 8, true);
        colors[i * 3 + 0] = dv.getUint8(off + 24) / 255;
        colors[i * 3 + 1] = dv.getUint8(off + 25) / 255;
        colors[i * 3 + 2] = dv.getUint8(off + 26) / 255;
      }
      this.positions = positions;
      this.colors = colors;
    }

    _build() {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(this.colors, 3));
      const mat = new THREE.PointsMaterial({
        size: 0.02,
        vertexColors: true,
        sizeAttenuation: true,
      });
      this.points = new THREE.Points(geo, mat);
      this.scene.add(this.points);
      this.scene.background = new THREE.Color(0x0d1017);
    }

    render() {
      this.renderer.render(this.scene, this.camera);
    }

    dispose() {
      if (this.points) {
        this.points.geometry.dispose();
        this.points.material.dispose();
      }
      this.renderer.dispose();
    }
  }

  generateBtn.addEventListener("click", startGeneration);
})();
