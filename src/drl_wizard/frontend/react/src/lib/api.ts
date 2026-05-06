import type {
  AlgoMetadata,
  ConfigRecord,
  HealthResponse,
  JobResponse,
  MetricRow,
  ResultType,
  SavedRunDetailsResponse,
  SavedRunDiscoveryResponse,
  SavedRunEvaluationRequest,
  SavedRunEvaluationResponse,
  TrainingRequest,
  WrappedConfig,
} from "../types/api";
import type { EnvMetadata } from "../types/api";

const configuredBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000").trim();
export const API_BASE_URL = configuredBaseUrl.replace(/\/+$/, "");

function buildApiUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${normalizedPath}`;
}

async function readErrorMessage(response: Response): Promise<string> {
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        return payload.detail;
      }
    } catch {
      return `${response.status} ${response.statusText}`;
    }
  }

  try {
    const text = await response.text();
    return text || `${response.status} ${response.statusText}`;
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(buildApiUrl(path), {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

async function requestText(path: string): Promise<string> {
  const response = await fetch(buildApiUrl(path), {
    headers: {
      Accept: "application/x-ndjson, text/plain",
    },
  });

  if (response.status === 404) {
    return "";
  }

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.text();
}

function flattenMetricRecord(record: Record<string, unknown>): MetricRow | null {
  const general = record.general;
  if (typeof general !== "object" || general === null || !("step" in general)) {
    return null;
  }

  const stepValue = (general as { step?: unknown }).step;
  if (typeof stepValue !== "number") {
    return null;
  }

  const row: MetricRow = { step: stepValue };
  Object.entries(record).forEach(([key, value]) => {
    if (key === "general" || typeof value !== "object" || value === null) {
      return;
    }
    const metricValue = (value as { mean?: unknown }).mean;
    if (typeof metricValue === "number") {
      row[key] = metricValue;
    }
  });

  return Object.keys(row).length > 1 ? row : null;
}

function parseMetricRows(input: string): MetricRow[] {
  if (!input) {
    return [];
  }

  const rows = input
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      try {
        return JSON.parse(line) as Record<string, unknown>;
      } catch {
        return null;
      }
    })
    .filter((record): record is Record<string, unknown> => record !== null)
    .map(flattenMetricRecord)
    .filter((record): record is MetricRow => record !== null)
    .sort((left, right) => left.step - right.step);

  const deduped = new Map<number, MetricRow>();
  rows.forEach((row) => {
    deduped.set(row.step, row);
  });
  return [...deduped.values()];
}

export const api = {
  getHealth: () => requestJson<HealthResponse>("/healthz"),
  getEnvironments: () => requestJson<EnvMetadata[]>("/training_service/environments"),
  getAlgorithms: () => requestJson<AlgoMetadata[]>("/training_service/algorithms"),
  getSupportedAlgorithms: (envId: string) =>
    requestJson<AlgoMetadata[]>(
      `/training_service/environments/${encodeURIComponent(envId)}/supported_algorithms`,
    ),
  getGeneralConfig: (envId: string) =>
    requestJson<WrappedConfig<ConfigRecord>>(
      `/training_service/environments/${encodeURIComponent(envId)}/general_config`,
    ),
  getLogConfig: () =>
    requestJson<WrappedConfig<ConfigRecord>>("/training_service/logs/log_config"),
  getAlgoConfig: (algoId: string) =>
    requestJson<WrappedConfig<ConfigRecord>>(`/training_service/algorithms/${algoId}/config`),
  getJobs: () => requestJson<JobResponse[]>("/training_service/all"),
  getJob: (jobId: number) => requestJson<JobResponse>(`/training_service/${jobId}`),
  startTraining: (payload: TrainingRequest) =>
    requestJson<JobResponse>("/training_service/train", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  stopJob: (jobId: number) =>
    requestJson<void>(`/training_service/${jobId}/stop`, {
      method: "PATCH",
    }),
  deleteJob: (jobId: number) =>
    requestJson<void>(`/training_service/${jobId}`, {
      method: "DELETE",
    }),
  getJobMetrics: async (jobId: number, resultType: ResultType) =>
    parseMetricRows(await requestText(`/training_service/${jobId}/results/${resultType}/stream`)),
  getSavedRuns: (envId: string) =>
    requestJson<SavedRunDiscoveryResponse>(
      `/training_service/saved_runs?env_id=${encodeURIComponent(envId)}`,
    ),
  getSavedRunDetails: (runId: string) =>
    requestJson<SavedRunDetailsResponse>(`/training_service/saved_runs/${encodeURIComponent(runId)}`),
  evaluateSavedRun: (runId: string, payload: SavedRunEvaluationRequest) =>
    requestJson<SavedRunEvaluationResponse>(
      `/training_service/saved_runs/${encodeURIComponent(runId)}/evaluate`,
      {
        method: "POST",
        body: JSON.stringify(payload),
      },
    ),
  getJobDownloadUrl: (jobId: number) => buildApiUrl(`/training_service/${jobId}/data/zip`),
};
