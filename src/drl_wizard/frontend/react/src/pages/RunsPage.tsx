import { useState } from "react";
import { Download, LoaderCircle, Square, Trash2 } from "lucide-react";

import { MetricChartGrid } from "../components/MetricChartGrid";
import { SectionCard } from "../components/SectionCard";
import { StatusBadge } from "../components/StatusBadge";
import { api } from "../lib/api";
import {
  formatDateTime,
  formatNumber,
  getJobDisplayName,
  getLastMetricValue,
} from "../lib/format";
import { useJob, useJobMetrics, useJobs } from "../hooks/useApiData";
import type { JobResponse } from "../types/api";

function canStop(job?: JobResponse | null): boolean {
  return job?.status === "queued" || job?.status === "running";
}

export function RunsPage() {
  const jobs = useJobs(10000);
  const [selectedJobId, setSelectedJobId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyJobId, setBusyJobId] = useState<number | null>(null);

  const selectedJob = useJob(selectedJobId, selectedJobId ? 7000 : 0);
  const trainingMetrics = useJobMetrics(selectedJobId, "train", selectedJobId ? 7000 : 0);
  const evaluationMetrics = useJobMetrics(selectedJobId, "evaluate", selectedJobId ? 7000 : 0);

  async function handleStop(jobId: number) {
    setBusyJobId(jobId);
    setError(null);
    try {
      await api.stopJob(jobId);
      await Promise.all([jobs.refresh(), selectedJob.refresh()]);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Failed to stop the job");
    } finally {
      setBusyJobId(null);
    }
  }

  async function handleDelete(jobId: number) {
    const confirmed = window.confirm(`Delete run ${jobId} and its persisted data?`);
    if (!confirmed) {
      return;
    }
    setBusyJobId(jobId);
    setError(null);
    try {
      await api.deleteJob(jobId);
      if (selectedJobId === jobId) {
        setSelectedJobId(null);
      }
      await jobs.refresh();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Failed to delete the job");
    } finally {
      setBusyJobId(null);
    }
  }

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <div className="eyebrow">Runs</div>
        <h1 className="text-3xl font-semibold text-text">Runs and artifacts</h1>
        <p className="max-w-3xl text-sm text-muted">
          Select a run from the list and inspect its summary, artifacts, and metric history in one place.
        </p>
      </header>

      {error ? (
        <div className="rounded-lg border border-danger/20 bg-danger/10 p-4 text-sm text-danger">
          {error}
        </div>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-[420px_minmax(0,1fr)]">
        <SectionCard
          eyebrow="Runs"
          title="Saved jobs"
          description="Choose a run to inspect. The right side updates only for the selected run."
        >
          <div className="space-y-3">
            {(jobs.data ?? []).map((job) => (
              <button
                key={job.job_id}
                type="button"
                onClick={() => setSelectedJobId(job.job_id ?? null)}
                className={`w-full rounded-lg border p-4 text-left transition ${
                  selectedJobId === job.job_id
                    ? "border-accent bg-accent/10"
                    : "border-border bg-page/60 hover:border-accent/40"
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-text">{getJobDisplayName(job)}</div>
                    <div className="mt-1 text-xs text-muted">{formatDateTime(job.created_at)}</div>
                    <div className="mt-2 text-xs text-muted">
                      {job.env?.env_id ?? "Unknown env"} · {job.algo?.algo_name ?? "Unknown algo"}
                    </div>
                  </div>
                  <StatusBadge status={job.status} />
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  <span className="rounded-full border border-border bg-page/70 px-2.5 py-1 text-xs text-muted">
                    Run {job.job_id ?? "-"}
                  </span>
                  {job.detail ? (
                    <span className="rounded-full border border-border bg-page/70 px-2.5 py-1 text-xs text-muted">
                      {job.detail}
                    </span>
                  ) : null}
                </div>
              </button>
            ))}
            {!jobs.loading && !jobs.data?.length ? (
              <div className="rounded-lg border border-border bg-page/50 p-4 text-sm text-muted">
                No jobs are currently stored.
              </div>
            ) : null}
          </div>
        </SectionCard>

        <div className="space-y-4">
          <SectionCard
            eyebrow="Selection"
            title="Selected run"
            description={
              selectedJobId
                ? "Run details, actions, and metrics for the selected job."
                : "Select a run from the left to load its details and metrics."
            }
            action={
              selectedJobId ? (
                <button
                  className="btn-secondary"
                  type="button"
                  onClick={() => {
                    void Promise.all([
                      jobs.refresh(),
                      selectedJob.refresh(),
                      trainingMetrics.refresh(),
                      evaluationMetrics.refresh(),
                    ]);
                  }}
                >
                  <LoaderCircle className="h-4 w-4" />
                  Refresh
                </button>
              ) : null
            }
          >
            {selectedJobId ? (
              <div className="space-y-4">
                <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                  <div className="rounded-lg border border-border bg-page/60 p-4">
                    <div className="label mb-2">Run</div>
                    <div className="text-base font-semibold text-text">
                      {getJobDisplayName(selectedJob.data)}
                    </div>
                    <div className="mt-2 text-sm text-muted">
                      Created {formatDateTime(selectedJob.data?.created_at)}
                    </div>
                  </div>
                  <div className="rounded-lg border border-border bg-page/60 p-4">
                    <div className="label mb-2">Status</div>
                    {selectedJob.data ? <StatusBadge status={selectedJob.data.status} /> : null}
                    <div className="mt-2 text-sm text-muted">
                      {selectedJob.data?.detail ?? "No extra detail from the backend."}
                    </div>
                  </div>
                  <div className="rounded-lg border border-border bg-page/60 p-4">
                    <div className="label mb-2">Step reward</div>
                    <div className="text-3xl font-semibold text-text">
                      {formatNumber(getLastMetricValue(trainingMetrics.data ?? [], "average_step_rewards"))}
                    </div>
                  </div>
                  <div className="rounded-lg border border-border bg-page/60 p-4">
                    <div className="label mb-2">Eval reward</div>
                    <div className="text-3xl font-semibold text-text">
                      {formatNumber(
                        getLastMetricValue(
                          evaluationMetrics.data ?? [],
                          "eval_average_episode_rewards",
                        ),
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex flex-wrap gap-3">
                  <button
                    className="btn-secondary"
                    type="button"
                    disabled={!canStop(selectedJob.data) || busyJobId === selectedJobId}
                    onClick={() => {
                      if (selectedJobId) {
                        void handleStop(selectedJobId);
                      }
                    }}
                  >
                    {busyJobId === selectedJobId ? (
                      <LoaderCircle className="h-4 w-4 animate-spin" />
                    ) : (
                      <Square className="h-4 w-4" />
                    )}
                    Stop run
                  </button>

                  {selectedJobId ? (
                    <a className="btn-secondary" href={api.getJobDownloadUrl(selectedJobId)}>
                      <Download className="h-4 w-4" />
                      Download archive
                    </a>
                  ) : null}

                  <button
                    className="btn-secondary text-danger hover:border-danger/40 hover:text-danger"
                    type="button"
                    disabled={!selectedJobId || busyJobId === selectedJobId}
                    onClick={() => {
                      if (selectedJobId) {
                        void handleDelete(selectedJobId);
                      }
                    }}
                  >
                    {busyJobId === selectedJobId ? (
                      <LoaderCircle className="h-4 w-4 animate-spin" />
                    ) : (
                      <Trash2 className="h-4 w-4" />
                    )}
                    Delete run
                  </button>
                </div>

                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-lg border border-border bg-page/60 p-4 text-sm text-muted">
                    <div>Started {formatDateTime(selectedJob.data?.started_at)}</div>
                    <div className="mt-2">Finished {formatDateTime(selectedJob.data?.finished_at)}</div>
                  </div>
                  <div className="rounded-lg border border-border bg-page/60 p-4 text-sm text-muted">
                    Archive downloads use the existing backend ZIP endpoint and reflect whatever artifacts are available for the selected run.
                  </div>
                </div>
              </div>
            ) : (
              <div className="rounded-lg border border-border bg-page/50 p-4 text-sm text-muted">
                Select a run from the left column.
              </div>
            )}
          </SectionCard>

          {selectedJobId ? (
            <>
              <MetricChartGrid
                title="Training metric history"
                rows={trainingMetrics.data ?? []}
                emptyMessage="Training metrics are not available for the selected run."
                preferredMetrics={[
                  "average_step_rewards",
                  "value_loss",
                  "policy_loss",
                  "dist_entropy",
                  "actor_grad_norm",
                  "critic_grad_norm",
                  "ratio",
                ]}
                maxCharts={8}
              />

              <MetricChartGrid
                title="Evaluation metric history"
                rows={evaluationMetrics.data ?? []}
                emptyMessage="Evaluation metrics are not available for the selected run."
                preferredMetrics={["eval_average_episode_rewards"]}
                maxCharts={4}
              />
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}
