import { useEffect, useRef, useState } from "react";

export interface LogLine {
  time: string;
  msg: string;
}

interface TerminalProps {
  logs: LogLine[];
  onClear: () => void;
}

export default function Terminal({ logs, onClear }: TerminalProps) {
  const [collapsed, setCollapsed] = useState(false);
  const bodyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = bodyRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs, collapsed]);

  return (
    <section className="terminal panel">
      <div className="terminal-head">
        <span className="terminal-title">Terminal</span>
        <div className="terminal-actions">
          <button
            type="button"
            className="terminal-btn"
            onClick={onClear}
            disabled={logs.length === 0}
          >
            Clear
          </button>
          <button
            type="button"
            className="terminal-btn"
            onClick={() => setCollapsed((c) => !c)}
          >
            {collapsed ? "Expand" : "Collapse"}
          </button>
        </div>
      </div>
      {!collapsed && (
        <div className="terminal-body" ref={bodyRef}>
          {logs.map((line, i) => (
            <div key={i} className="terminal-line">
              [{line.time}] {line.msg}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
