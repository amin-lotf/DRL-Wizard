import type { JobResponse, MetricRow } from "../types/api";

export function cn(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}

export function formatDateTime(value?: string | null): string {
  if (!value) {
    return "Unavailable";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

export function formatNumber(
  value?: number | null,
  options: Intl.NumberFormatOptions = {},
): string {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "Unavailable";
  }
  return new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 4,
    ...options,
  }).format(value);
}

export function formatCompactNumber(value?: number | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "Unavailable";
  }
  return new Intl.NumberFormat(undefined, {
    notation: "compact",
    maximumFractionDigits: 2,
  }).format(value);
}

export function titleFromKey(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function getMetricKeys(rows: MetricRow[]): string[] {
  const keys = new Set<string>();
  rows.forEach((row) => {
    Object.keys(row).forEach((key) => {
      if (key !== "step") {
        keys.add(key);
      }
    });
  });
  return [...keys];
}

export function getLastMetricValue(rows: MetricRow[], metric: string): number | null {
  const row = rows.length ? rows[rows.length - 1] : undefined;
  if (!row) {
    return null;
  }
  const value = row[metric];
  return typeof value === "number" ? value : null;
}

export function getJobDisplayName(job?: JobResponse | null): string {
  if (!job?.job_id) {
    return "No active run";
  }
  const algo = job.algo?.algo_name ?? "Unknown";
  const env = job.env?.env_name ?? job.env?.env_id ?? "Environment";
  return `Run ${job.job_id} · ${algo} · ${env}`;
}

export function toVideoDataUrl(base64Data?: string | null, mimeType?: string | null): string | null {
  if (!base64Data) {
    return null;
  }
  return `data:${mimeType ?? "video/mp4"};base64,${base64Data}`;
}
