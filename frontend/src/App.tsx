import { useEffect, useRef, useState } from "react";
import {
  generate,
  stream,
  resultUrl,
  stageUrl,
  type GenerateRequest,
  type StreamEvent,
} from "./api";
import Terminal, { type LogLine } from "./components/Terminal";
import ProgressPanel, { STAGE_LABELS } from "./components/ProgressPanel";
import ImagePreview from "./components/ImagePreview";
import SplatViewport from "./components/SplatViewport";

export default function App() {
  // Control panel fields (defaults match the old UI).
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [seed, setSeed] = useState("");
  const [width, setWidth] = useState("1024");
  const [height, setHeight] = useState("1024");
  const [numGaussians, setNumGaussians] = useState("262144");
  const [splatSteps, setSplatSteps] = useState("28");
  const [fluxQuantize, setFluxQuantize] = useState("8");

  // Job state.
  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState("starting");
  const [stageProgress, setStageProgress] = useState(0);
  const [statusText, setStatusText] = useState("");
  const [isError, setIsError] = useState(false);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [fluxUrl, setFluxUrl] = useState<string | null>(null);
  const [rgbaUrl, setRgbaUrl] = useState<string | null>(null);
  const [resultJobId, setResultJobId] = useState<string | null>(null);
  // The splat currently shown in the viewport. It is only swapped when a
  // generation completes: pressing Generate keeps the old model on screen,
  // and the viewport lerp-morphs to the final splat once it is ready.
  const [currentSplatUrl, setCurrentSplatUrl] = useState<string | null>(null);

  // Debug: load a splat directly from ?splat=<url> (e.g. an existing
  // /api/result/<job>/splat URL) without running a generation.
  // ?splat2=<url> adds a "Swap splat" button that hides the viewport for 1s
  // and then loads the other URL — mimics the generate-again transition.
  const [debugSplatUrl] = useState<string | null>(
    () => new URLSearchParams(window.location.search).get("splat"),
  );
  const [debugSplatUrl2] = useState<string | null>(
    () => new URLSearchParams(window.location.search).get("splat2"),
  );
  const [swapAlt, setSwapAlt] = useState(false);
  const [swapHidden, setSwapHidden] = useState(false);
  const activeDebugUrl =
    debugSplatUrl && debugSplatUrl2
      ? swapAlt
        ? debugSplatUrl2
        : debugSplatUrl
      : debugSplatUrl;
  const shownDebugUrl = swapHidden ? null : activeDebugUrl;
  const swapSplat = () => {
    setSwapHidden(true);
    setSwapAlt((a) => !a);
    setTimeout(() => setSwapHidden(false), 1000);
  };

  const evtSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    return () => evtSourceRef.current?.close();
  }, []);

  const pushLog = (msg: string) => {
    setLogs((prev) => [...prev, { time: new Date().toLocaleTimeString(), msg }]);
  };

  const setStatus = (msg: string, error = false) => {
    setStatusText(msg);
    setIsError(error);
  };

  const handleEvent = (jobId: string, data: StreamEvent) => {
    if (data.log) {
      pushLog(data.log);
    }
    if (data.image_ready === "flux") {
      setFluxUrl(`${stageUrl(jobId, "flux")}?t=${Date.now()}`);
      pushLog("FLUX image preview updated");
    }
    if (data.image_ready === "rgba") {
      setRgbaUrl(`${stageUrl(jobId, "rgba")}?t=${Date.now()}`);
      pushLog("RGBA image preview updated");
    }
    if (data.status === "failed") {
      setStatus(`Failed: ${data.error || "unknown"}`, true);
      pushLog(`Job failed: ${data.error || "unknown"}`);
      evtSourceRef.current?.close();
      setGenerating(false);
      return;
    }
    if (data.status === "completed") {
      setProgress(1);
      setStage("done");
      setStageProgress(1);
      setStatus("Completed");
      loadResults(jobId);
      evtSourceRef.current?.close();
      setGenerating(false);
      return;
    }
    if (data.stage) {
      setStage(data.stage);
      setStageProgress(data.stage_progress || 0);
    }
    if (typeof data.progress === "number") {
      setProgress(data.progress);
    }
    const label = STAGE_LABELS[data.stage || ""] || data.stage || "running";
    setStatus(
      `${label} ${Math.round((data.progress || 0) * 100)}%`,
    );
  };

  const loadResults = (jobId: string) => {
    setFluxUrl(`${resultUrl(jobId, "image")}?t=${Date.now()}`);
    setRgbaUrl(`${resultUrl(jobId, "rgba")}?t=${Date.now()}`);
    setResultJobId(jobId);
    setCurrentSplatUrl(resultUrl(jobId, "splat"));
    pushLog("Results ready (ply / spz / splat)");
  };

  const startGeneration = async () => {
    const trimmed = prompt.trim();
    if (!trimmed) {
      setStatus("Please enter a prompt", true);
      return;
    }

    evtSourceRef.current?.close();
    setGenerating(true);
    setFluxUrl(null);
    setRgbaUrl(null);
    setResultJobId(null);
    setLogs([]);
    setProgress(0);
    setStage("starting");
    setStageProgress(0);
    setStatus("Submitting job...");

    const req: GenerateRequest = {
      prompt: trimmed,
      negative_prompt: negativePrompt.trim() || null,
      seed: seed ? parseInt(seed, 10) : null,
      width: parseInt(width, 10),
      height: parseInt(height, 10),
      num_gaussians: parseInt(numGaussians, 10),
      splat_steps: parseInt(splatSteps, 10),
      flux_quantize: fluxQuantize ? parseInt(fluxQuantize, 10) : null,
    };

    try {
      const jobId = await generate(req);
      setStatus(`Job ${jobId} started`);
      pushLog(`Job ${jobId} submitted`);
      evtSourceRef.current = stream(jobId, {
        onEvent: (data) => handleEvent(jobId, data),
        onError: () => {
          setStatus("Stream error / connection closed", true);
          pushLog("Stream error / connection closed");
          evtSourceRef.current?.close();
          setGenerating(false);
        },
      });
    } catch (err) {
      setStatus(err instanceof Error ? err.message : String(err), true);
      setGenerating(false);
    }
  };

  const downloads = resultJobId
    ? {
        ply: resultUrl(resultJobId, "ply"),
        spz: resultUrl(resultJobId, "spz"),
        splat: resultUrl(resultJobId, "splat"),
      }
    : null;

  return (
    <>
      <header>
        <h1>TripoFlux MLX</h1>
        <p>
          FLUX.2-klein-9B → BiRefNet → TripoSplat Gaussian Splatting on Apple
          Silicon
        </p>
      </header>
      <main>
        <section className="panel">
          <label htmlFor="prompt">Prompt</label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="A red sports car, studio lighting, product photography..."
          />

          <label htmlFor="negative_prompt">Negative Prompt (optional)</label>
          <textarea
            id="negative_prompt"
            value={negativePrompt}
            onChange={(e) => setNegativePrompt(e.target.value)}
            placeholder="blurry, low quality, text, watermark..."
          />

          <label htmlFor="seed">Seed (optional)</label>
          <input
            id="seed"
            type="number"
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
            placeholder="42"
          />

          <label htmlFor="width">Width</label>
          <input
            id="width"
            type="number"
            value={width}
            onChange={(e) => setWidth(e.target.value)}
          />

          <label htmlFor="height">Height</label>
          <input
            id="height"
            type="number"
            value={height}
            onChange={(e) => setHeight(e.target.value)}
          />

          <label htmlFor="num_gaussians">Number of Gaussians</label>
          <select
            id="num_gaussians"
            value={numGaussians}
            onChange={(e) => setNumGaussians(e.target.value)}
          >
            <option value="32768">32,768 (fast)</option>
            <option value="65536">65,536</option>
            <option value="131072">131,072</option>
            <option value="262144">262,144 (best quality)</option>
          </select>

          <label htmlFor="splat_steps">TripoSplat Steps</label>
          <select
            id="splat_steps"
            value={splatSteps}
            onChange={(e) => setSplatSteps(e.target.value)}
          >
            <option value="20">20 (fast)</option>
            <option value="24">24</option>
            <option value="28">28 (best quality)</option>
          </select>

          <label htmlFor="flux_quantize">FLUX Quantization</label>
          <select
            id="flux_quantize"
            value={fluxQuantize}
            onChange={(e) => setFluxQuantize(e.target.value)}
          >
            <option value="">None (largest, best quality)</option>
            <option value="8">8-bit (balanced)</option>
            <option value="4">4-bit (smallest, fastest)</option>
          </select>

          <button
            id="generate"
            type="button"
            onClick={startGeneration}
            disabled={generating}
          >
            {generating ? "Generating..." : "Generate Splat"}
          </button>

          <ProgressPanel
            visible={generating || stage === "done" || isError}
            progress={progress}
            stage={stage}
            stageProgress={stageProgress}
            statusText={statusText}
            isError={isError}
            lastLog={logs.length > 0 ? logs[logs.length - 1].msg : null}
          />
        </section>

        <section>
          <div className="preview-grid">
            <ImagePreview
              fluxUrl={fluxUrl}
              rgbaUrl={rgbaUrl}
              downloads={downloads}
              jobId={resultJobId}
            />
            <div className="preview-card viewport-card">
              <h3>Gaussian Splat Preview</h3>
              <SplatViewport splatUrl={shownDebugUrl ?? currentSplatUrl} />
              {debugSplatUrl2 && (
                <button type="button" onClick={swapSplat}>
                  Swap splat
                </button>
              )}
            </div>
          </div>
        </section>

        <Terminal logs={logs} onClear={() => setLogs([])} />
      </main>
    </>
  );
}
