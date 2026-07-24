export const STAGE_ORDER = ["flux", "birefnet", "triposplat"] as const;

export const STAGE_LABELS: Record<string, string> = {
  flux: "FLUX Image",
  birefnet: "Background Removal",
  triposplat: "TripoSplat 3D",
};

// birefnet has no fine-grained progress signal from the backend.
const INDETERMINATE: Record<string, boolean> = {
  flux: false,
  birefnet: true,
  triposplat: false,
};

type StageState = "pending" | "active" | "done";

interface ProgressPanelProps {
  visible: boolean;
  progress: number;
  stage: string;
  stageProgress: number;
  statusText: string;
  isError: boolean;
  lastLog: string | null;
}

// Inline progress display embedded in the control panel (replaces the old
// modal popup).
export default function ProgressPanel({
  visible,
  progress,
  stage,
  stageProgress,
  statusText,
  isError,
  lastLog,
}: ProgressPanelProps) {
  if (!visible) return null;

  const currentIdx = STAGE_ORDER.indexOf(stage as (typeof STAGE_ORDER)[number]);
  const allDone = stage === "done";

  const stateFor = (i: number): StageState => {
    if (allDone) return "done";
    if (currentIdx === -1) return "pending";
    if (i < currentIdx) return "done";
    if (i === currentIdx) return "active";
    return "pending";
  };

  const fracFor = (i: number): number => {
    const state = stateFor(i);
    if (state === "done") return 1;
    if (state === "active") return Math.max(0, Math.min(1, stageProgress || 0));
    return 0;
  };

  return (
    <div className="progress-inline">
      <div className="progress">
        <div
          className="progress-bar"
          style={{ width: `${Math.round(progress * 100)}%` }}
        />
        <div className="progress-text">{statusText || "starting"}</div>
      </div>

      <div className="stages">
        {STAGE_ORDER.map((key, i) => {
          const state = stateFor(i);
          const frac = fracFor(i);
          const indeterminate =
            INDETERMINATE[key] && state === "active" && frac < 1;
          return (
            <div
              key={key}
              className="stage"
              data-state={state}
              {...(indeterminate ? { "data-indeterminate": "" } : {})}
            >
              <div className="stage-head">
                <span className="stage-name">
                  {i + 1} · {STAGE_LABELS[key]}
                </span>
                <span className="stage-pct">
                  {state === "done"
                    ? "100%"
                    : state === "active"
                      ? `${Math.round(frac * 100)}%`
                      : "0%"}
                </span>
              </div>
              <div className="stage-track">
                <div
                  className="stage-fill"
                  style={{ width: `${Math.round(frac * 100)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className={isError ? "status error" : "status"}>{statusText}</div>
      {lastLog && <div className="modal-log">{lastLog}</div>}
    </div>
  );
}
