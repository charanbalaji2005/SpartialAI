import { useEffect, useRef, useState } from "react";
import { api, type JobStatus } from "./api";

const TERMINAL = new Set(["completed", "failed", "cancelled"]);

/** Live job status over WebSocket, with HTTP polling as a fallback. */
export function useJobProgress(projectId: string | null, active: boolean) {
  const [status, setStatus] = useState<JobStatus | null>(null);
  const timer = useRef<number>();

  useEffect(() => {
    if (!projectId || !active) return;
    let closed = false;
    let ws: WebSocket | null = null;

    const poll = async () => {
      if (closed) return;
      try {
        const s = await api.status(projectId);
        setStatus(s);
        if (TERMINAL.has(s.status)) return;
      } catch { /* backend restarting; keep trying */ }
      timer.current = window.setTimeout(poll, 800);
    };

    try {
      ws = new WebSocket(api.progressSocketUrl(projectId));
      ws.onmessage = (e) => setStatus(JSON.parse(e.data));
      ws.onerror = () => { ws?.close(); };
      ws.onclose = () => { if (!closed) poll(); };
    } catch {
      poll();
    }
    return () => { closed = true; ws?.close(); window.clearTimeout(timer.current); };
  }, [projectId, active]);

  return status;
}
