import { cn } from "../lib/format";
import type { JobStatus } from "../types/api";

const statusClasses: Record<JobStatus, string> = {
  queued: "bg-white/5 text-muted border-white/10",
  running: "bg-success/10 text-success border-success/20",
  stopping: "bg-warning/10 text-warning border-warning/20",
  stopped: "bg-white/5 text-muted border-white/10",
  failed: "bg-danger/10 text-danger border-danger/20",
  finished: "bg-accent/10 text-accent border-accent/20",
};

export function StatusBadge({ status }: { status: JobStatus }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium uppercase tracking-[0.08em]",
        statusClasses[status],
      )}
    >
      {status}
    </span>
  );
}
