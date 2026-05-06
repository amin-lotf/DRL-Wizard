import { useCallback, useEffect, useRef, useState } from "react";
import type { Dispatch, SetStateAction } from "react";

import { api } from "../lib/api";
import type {
  AlgoMetadata,
  ConfigRecord,
  HealthResponse,
  JobResponse,
  MetricRow,
  ResultType,
  SavedRunDetailsResponse,
  SavedRunDiscoveryResponse,
  WrappedConfig,
} from "../types/api";
import type { EnvMetadata } from "../types/api";

interface QueryOptions {
  enabled?: boolean;
  intervalMs?: number;
  clearOnChange?: boolean;
}

interface QueryState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  setData: Dispatch<SetStateAction<T | null>>;
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return "Request failed";
}

export function useQueryResource<T>(
  loader: (() => Promise<T>) | null,
  dependencies: readonly unknown[],
  options: QueryOptions = {},
): QueryState<T> {
  const enabled = options.enabled ?? true;
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(enabled);
  const [error, setError] = useState<string | null>(null);
  const loaderRef = useRef(loader);
  const requestIdRef = useRef(0);

  useEffect(() => {
    loaderRef.current = loader;
  }, [loader]);

  const refresh = useCallback(async () => {
    const currentLoader = loaderRef.current;
    const requestId = ++requestIdRef.current;

    if (!currentLoader || !enabled) {
      setData(null);
      setError(null);
      setLoading(false);
      return;
    }

    setLoading(true);
    try {
      const next = await currentLoader();
      if (requestIdRef.current !== requestId) {
        return;
      }
      setData(next);
      setError(null);
    } catch (err) {
      if (requestIdRef.current !== requestId) {
        return;
      }
      setError(getErrorMessage(err));
    } finally {
      if (requestIdRef.current === requestId) {
        setLoading(false);
      }
    }
  }, [enabled]);

  useEffect(() => {
    if (options.clearOnChange) {
      setData(null);
      setError(null);
    }
    void refresh();
  }, [enabled, options.clearOnChange, refresh, ...dependencies]);

  useEffect(() => {
    if (!enabled || !options.intervalMs) {
      return undefined;
    }

    const interval = window.setInterval(() => {
      void refresh();
    }, options.intervalMs);
    return () => window.clearInterval(interval);
  }, [enabled, options.intervalMs, refresh]);

  return { data, loading, error, refresh, setData };
}

export function useHealth(): QueryState<HealthResponse> {
  return useQueryResource(() => api.getHealth(), []);
}

export function useEnvironments(): QueryState<EnvMetadata[]> {
  return useQueryResource(() => api.getEnvironments(), []);
}

export function useAlgorithms(): QueryState<AlgoMetadata[]> {
  return useQueryResource(() => api.getAlgorithms(), []);
}

export function useSupportedAlgorithms(
  envId: string | null,
): QueryState<AlgoMetadata[]> {
  return useQueryResource(
    envId ? () => api.getSupportedAlgorithms(envId) : null,
    [envId],
    { enabled: Boolean(envId), clearOnChange: true },
  );
}

export function useGeneralConfig(
  envId: string | null,
): QueryState<WrappedConfig<ConfigRecord>> {
  return useQueryResource(
    envId ? () => api.getGeneralConfig(envId) : null,
    [envId],
    { enabled: Boolean(envId), clearOnChange: true },
  );
}

export function useAlgoConfig(
  algoId: string | null,
): QueryState<WrappedConfig<ConfigRecord>> {
  return useQueryResource(
    algoId ? () => api.getAlgoConfig(algoId) : null,
    [algoId],
    { enabled: Boolean(algoId), clearOnChange: true },
  );
}

export function useLogConfig(): QueryState<WrappedConfig<ConfigRecord>> {
  return useQueryResource(() => api.getLogConfig(), []);
}

export function useJobs(intervalMs = 0): QueryState<JobResponse[]> {
  return useQueryResource(() => api.getJobs(), [], { intervalMs });
}

export function useJob(
  jobId: number | null,
  intervalMs = 0,
): QueryState<JobResponse> {
  return useQueryResource(
    jobId ? () => api.getJob(jobId) : null,
    [jobId],
    { enabled: Boolean(jobId), intervalMs, clearOnChange: true },
  );
}

export function useJobMetrics(
  jobId: number | null,
  resultType: ResultType,
  intervalMs = 0,
): QueryState<MetricRow[]> {
  return useQueryResource(
    jobId ? () => api.getJobMetrics(jobId, resultType) : null,
    [jobId, resultType],
    { enabled: Boolean(jobId), intervalMs, clearOnChange: true },
  );
}

export function useSavedRuns(
  envId: string | null,
): QueryState<SavedRunDiscoveryResponse> {
  return useQueryResource(
    envId ? () => api.getSavedRuns(envId) : null,
    [envId],
    { enabled: Boolean(envId), clearOnChange: true },
  );
}

export function useSavedRunDetails(
  runId: string | null,
): QueryState<SavedRunDetailsResponse> {
  return useQueryResource(
    runId ? () => api.getSavedRunDetails(runId) : null,
    [runId],
    { enabled: Boolean(runId), clearOnChange: true },
  );
}
