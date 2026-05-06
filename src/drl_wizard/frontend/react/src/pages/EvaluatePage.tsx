import { useEffect, useMemo, useState } from "react";
import { LoaderCircle, Play } from "lucide-react";

import { ActionTypeSelector } from "../components/ActionTypeSelector";
import { ReadonlyConfigPanel } from "../components/ConfigEditor";
import { SectionCard } from "../components/SectionCard";
import { api } from "../lib/api";
import { formatNumber, toVideoDataUrl } from "../lib/format";
import { useEnvironments, useSavedRunDetails, useSavedRuns } from "../hooks/useApiData";
import type { ActionType, SavedRunEvaluationResponse } from "../types/api";

export function EvaluatePage() {
  const environments = useEnvironments();
  const [selectedActionType, setSelectedActionType] = useState<ActionType | "">("");
  const [selectedEnvId, setSelectedEnvId] = useState<string>("");
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [episodes, setEpisodes] = useState<number>(1);
  const [renderSample, setRenderSample] = useState(false);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<SavedRunEvaluationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const availableActionTypes = useMemo(
    () =>
      Array.from(
        new Set((environments.data ?? []).map((environment) => environment.supported_action)),
      ),
    [environments.data],
  );

  const environmentOptions = useMemo(
    () =>
      (environments.data ?? []).filter(
        (environment) => environment.supported_action === selectedActionType,
      ),
    [environments.data, selectedActionType],
  );

  const savedRuns = useSavedRuns(selectedEnvId || null);
  const runDetails = useSavedRunDetails(selectedRunId || null);

  useEffect(() => {
    if (!selectedActionType) {
      setSelectedEnvId("");
      setSelectedRunId("");
      setResult(null);
      return;
    }

    const currentEnvStillValid = environmentOptions.some(
      (environment) => environment.env_id === selectedEnvId,
    );
    if (!currentEnvStillValid) {
      setSelectedEnvId("");
      setSelectedRunId("");
      setResult(null);
    }
  }, [environmentOptions, selectedActionType, selectedEnvId]);

  useEffect(() => {
    setSelectedRunId("");
    setResult(null);
    setError(null);
  }, [selectedEnvId]);

  useEffect(() => {
    setResult(null);
    setError(null);
  }, [selectedRunId]);

  useEffect(() => {
    if (runDetails.data?.eval_episodes_default) {
      setEpisodes(runDetails.data.eval_episodes_default);
    }
  }, [runDetails.data]);

  async function handleEvaluate() {
    if (!selectedRunId || !selectedEnvId) {
      return;
    }

    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const nextResult = await api.evaluateSavedRun(selectedRunId, {
        env_id: selectedEnvId,
        episodes: Math.max(1, episodes),
        render: renderSample,
      });
      setResult(nextResult);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Evaluation failed");
    } finally {
      setRunning(false);
    }
  }

  const selectedEnvironment =
    environmentOptions.find((environment) => environment.env_id === selectedEnvId) ?? null;
  const videoUrl = toVideoDataUrl(
    result?.rendered_video_base64 ?? null,
    result?.rendered_video_mime_type ?? null,
  );

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <div className="eyebrow">Evaluation</div>
        <h1 className="text-3xl font-semibold text-text">Evaluate saved models</h1>
        <p className="max-w-3xl text-sm text-muted">
          Use the same environment-first selection flow as the Streamlit reference, then load a saved run and start evaluation.
        </p>
      </header>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.35fr)_360px]">
        <div className="space-y-4">
          <ActionTypeSelector
            value={selectedActionType}
            options={availableActionTypes}
            onChange={(value) => setSelectedActionType(value)}
            title="Choose action type"
            description="Start by narrowing evaluation to the matching environment family."
          />

          <SectionCard
            eyebrow="Step 2"
            title="Choose environment"
            description="Saved runs will be filtered using the selected environment."
          >
            <label className="space-y-2">
              <span className="text-sm font-medium text-text">Environment</span>
              <select
                className="input"
                value={selectedEnvId}
                onChange={(event) => setSelectedEnvId(event.target.value)}
                disabled={!selectedActionType}
              >
                <option value="">
                  {selectedActionType
                    ? "Select an environment"
                    : "Select an action type first"}
                </option>
                {environmentOptions.map((environment) => (
                  <option key={environment.env_id} value={environment.env_id}>
                    {environment.env_name} ({environment.env_id})
                  </option>
                ))}
              </select>
            </label>
            {selectedEnvironment ? (
              <div className="mt-4 rounded-lg border border-border bg-page/60 p-4 text-sm text-muted">
                <div className="font-medium text-text">{selectedEnvironment.env_name}</div>
                <div className="mt-1">{selectedEnvironment.env_id}</div>
              </div>
            ) : null}
          </SectionCard>

          <SectionCard
            eyebrow="Step 3"
            title="Choose saved run"
            description="Only runs matching the selected environment are listed."
          >
            <label className="space-y-2">
              <span className="text-sm font-medium text-text">Saved run</span>
              <select
                className="input"
                value={selectedRunId}
                onChange={(event) => setSelectedRunId(event.target.value)}
                disabled={!selectedEnvId}
              >
                <option value="">
                  {selectedEnvId ? "Select a saved run" : "Select an environment first"}
                </option>
                {(savedRuns.data?.runs ?? []).map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    Run {run.run_id} · {run.algo_id} · {run.checkpoint_label}
                  </option>
                ))}
              </select>
            </label>

            {savedRuns.data?.warnings.length ? (
              <div className="mt-4 space-y-2">
                {savedRuns.data.warnings.map((warning) => (
                  <div
                    key={warning}
                    className="rounded-lg border border-warning/20 bg-warning/10 px-3 py-2 text-sm text-warning"
                  >
                    {warning}
                  </div>
                ))}
              </div>
            ) : null}

            {!savedRuns.loading && selectedEnvId && !savedRuns.data?.runs.length ? (
              <div className="mt-4 rounded-lg border border-border bg-page/60 p-4 text-sm text-muted">
                No saved runs match the selected environment.
              </div>
            ) : null}
          </SectionCard>

          <div className="grid gap-4 xl:grid-cols-2">
            <ReadonlyConfigPanel
              title="Environment config"
              description="Loaded from the saved app config and kept read-only here."
              config={runDetails.data?.env_config ?? null}
            />
            <ReadonlyConfigPanel
              title="Agent config"
              description="The algorithm config saved with the selected run."
              config={runDetails.data?.algo_config ?? null}
            />
          </div>

          {result ? (
            <div className="grid gap-4 md:grid-cols-2">
              <div className="panel p-5">
                <div className="label mb-2">Average step reward</div>
                <div className="text-3xl font-semibold text-text">
                  {formatNumber(result.average_step_reward)}
                </div>
              </div>
              <div className="panel p-5">
                <div className="label mb-2">Average episode reward</div>
                <div className="text-3xl font-semibold text-text">
                  {formatNumber(result.average_episode_reward)}
                </div>
              </div>
            </div>
          ) : null}

          {result?.render_warning ? (
            <div className="rounded-lg border border-warning/20 bg-warning/10 p-4 text-sm text-warning">
              {result.render_warning}
            </div>
          ) : null}

          {videoUrl ? (
            <section className="panel p-5">
              <div className="mb-4 space-y-1">
                <div className="eyebrow">Sample render</div>
                <h2 className="text-lg font-semibold text-text">Single rendered episode</h2>
                <p className="text-sm text-muted">
                  Rendering stays limited to one episode while reward metrics still cover the full evaluation request.
                </p>
              </div>
              <video
                className="w-full rounded-lg border border-border bg-page/70"
                controls
                src={videoUrl}
              />
            </section>
          ) : null}
        </div>

        <div className="space-y-4">
          <SectionCard
            eyebrow="Controls"
            title="Evaluation request"
            description="The React UI submits the request parameters and the backend performs the actual evaluation."
          >
            <div className="space-y-4">
              <div className="rounded-lg border border-border bg-page/60 p-4">
                <div className="label mb-2">Checkpoint</div>
                <div className="space-y-1 text-sm text-text">
                  <div>{runDetails.data?.summary.checkpoint_label ?? "Unavailable"}</div>
                  <div className="break-words text-muted">
                    {runDetails.data?.summary.checkpoint_path ?? "Unavailable"}
                  </div>
                </div>
              </div>

              <label className="space-y-2">
                <span className="text-sm font-medium text-text">Evaluation episodes</span>
                <input
                  className="input"
                  type="number"
                  min={1}
                  value={episodes}
                  onChange={(event) => setEpisodes(Math.max(1, Number(event.target.value) || 1))}
                />
              </label>

              <label className="flex items-center justify-between rounded-lg border border-border bg-page/60 px-4 py-3">
                <div>
                  <div className="text-sm font-medium text-text">Render sample episode</div>
                  <div className="text-xs text-muted">
                    When enabled, only one evaluation episode is rendered.
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={renderSample}
                  onChange={(event) => setRenderSample(event.target.checked)}
                />
              </label>

              {error ? (
                <div className="rounded-lg border border-danger/20 bg-danger/10 p-3 text-sm text-danger">
                  {error}
                </div>
              ) : null}

              <button
                className="btn-primary w-full"
                onClick={() => void handleEvaluate()}
                type="button"
                disabled={running || !selectedRunId}
              >
                {running ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                Start evaluation
              </button>
            </div>
          </SectionCard>
        </div>
      </div>
    </div>
  );
}
