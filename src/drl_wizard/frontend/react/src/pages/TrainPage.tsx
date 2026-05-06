import { useEffect, useMemo, useState } from "react";
import { Eye, LoaderCircle, Play, Plus, Square, TimerReset } from "lucide-react";

import { ActionTypeSelector } from "../components/ActionTypeSelector";
import { ConfigEditor } from "../components/ConfigEditor";
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
import {
  useAlgoConfig,
  useEnvironments,
  useGeneralConfig,
  useJob,
  useJobMetrics,
  useJobs,
  useLogConfig,
  useSupportedAlgorithms,
} from "../hooks/useApiData";
import type {
  ActionType,
  AlgoType,
  ConfigRecord,
  JobResponse,
  TrainingRequest,
} from "../types/api";

type TrainTab = "general" | "algorithm" | "logging";
type WizardStep = 1 | 2 | 3 | 4;

function isActiveJob(job?: JobResponse | null): boolean {
  return job?.status === "queued" || job?.status === "running" || job?.status === "stopping";
}

function StepHeader({
  currentStep,
}: {
  currentStep: WizardStep;
}) {
  const steps: Array<{ id: WizardStep; label: string }> = [
    { id: 1, label: "Action type" },
    { id: 2, label: "Environment" },
    { id: 3, label: "Algorithm" },
    { id: 4, label: "Config" },
  ];

  return (
    <div className="flex flex-wrap gap-2">
      {steps.map((step) => (
        <div
          key={step.id}
          className={`rounded-full px-3 py-1.5 text-xs font-medium uppercase tracking-[0.08em] ${
            currentStep === step.id
              ? "bg-accent/12 text-accent"
              : currentStep > step.id
                ? "bg-page/70 text-text"
                : "bg-page/50 text-muted"
          }`}
        >
          {step.id}. {step.label}
        </div>
      ))}
    </div>
  );
}

function StepFooter({
  canGoBack,
  canGoNext,
  onBack,
  onNext,
  nextLabel = "Next",
}: {
  canGoBack: boolean;
  canGoNext: boolean;
  onBack: () => void;
  onNext: () => void;
  nextLabel?: string;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <button
        className="btn-secondary"
        type="button"
        disabled={!canGoBack}
        onClick={onBack}
      >
        Back
      </button>
      <button
        className="btn-primary"
        type="button"
        disabled={!canGoNext}
        onClick={onNext}
      >
        {nextLabel}
      </button>
    </div>
  );
}

export function TrainPage() {
  const environments = useEnvironments();
  const jobs = useJobs(12000);
  const logTemplate = useLogConfig();

  const [wizardStep, setWizardStep] = useState<WizardStep>(1);
  const [isCreatingRun, setIsCreatingRun] = useState(true);
  const [selectedActionType, setSelectedActionType] = useState<ActionType | "">("");
  const [selectedEnvId, setSelectedEnvId] = useState<string>("");
  const [selectedAlgoId, setSelectedAlgoId] = useState<AlgoType | "">("");
  const [generalConfig, setGeneralConfig] = useState<ConfigRecord | null>(null);
  const [logConfig, setLogConfig] = useState<ConfigRecord | null>(null);
  const [algoConfig, setAlgoConfig] = useState<ConfigRecord | null>(null);
  const [activeTab, setActiveTab] = useState<TrainTab>("general");
  const [trackedJobId, setTrackedJobId] = useState<number | null>(null);
  const [latestTargetSteps, setLatestTargetSteps] = useState<number | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);

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

  const supportedAlgorithms = useSupportedAlgorithms(selectedEnvId || null);
  const generalTemplate = useGeneralConfig(selectedEnvId || null);
  const algoTemplate = useAlgoConfig(selectedAlgoId || null);
  const trackedJob = useJob(trackedJobId, trackedJobId ? 5000 : 0);
  const trainingMetrics = useJobMetrics(
    trackedJobId,
    "train",
    trackedJobId && isActiveJob(trackedJob.data) ? 5000 : 0,
  );
  const evaluationMetrics = useJobMetrics(
    trackedJobId,
    "evaluate",
    trackedJobId && isActiveJob(trackedJob.data) ? 7000 : 0,
  );

  useEffect(() => {
    if (logTemplate.data) {
      setLogConfig(logTemplate.data.config);
    }
  }, [logTemplate.data]);

  useEffect(() => {
    if (!selectedActionType) {
      setSelectedEnvId("");
      setSelectedAlgoId("");
      setGeneralConfig(null);
      setAlgoConfig(null);
      return;
    }

    const currentEnvStillValid = environmentOptions.some(
      (environment) => environment.env_id === selectedEnvId,
    );
    if (!currentEnvStillValid) {
      setSelectedEnvId("");
      setSelectedAlgoId("");
      setGeneralConfig(null);
      setAlgoConfig(null);
    }
  }, [environmentOptions, selectedActionType, selectedEnvId]);

  useEffect(() => {
    setSelectedAlgoId("");
    setGeneralConfig(null);
    setAlgoConfig(null);
    setSubmitError(null);
    if (selectedEnvId && wizardStep < 2) {
      setWizardStep(2);
    }
  }, [selectedEnvId]);

  useEffect(() => {
    if (selectedEnvId && generalTemplate.data) {
      setGeneralConfig(generalTemplate.data.config);
    }
  }, [generalTemplate.data, selectedEnvId]);

  useEffect(() => {
    if (!selectedAlgoId) {
      setAlgoConfig(null);
      return;
    }

    if (algoTemplate.data) {
      setAlgoConfig(algoTemplate.data.config);
    }
  }, [algoTemplate.data, selectedAlgoId]);

  useEffect(() => {
    if (!selectedAlgoId || !supportedAlgorithms.data?.length) {
      return;
    }
    const currentStillValid = supportedAlgorithms.data.some(
      (algorithm) => algorithm.algo_id === selectedAlgoId,
    );
    if (!currentStillValid) {
      setSelectedAlgoId("");
      setAlgoConfig(null);
    }
  }, [selectedAlgoId, supportedAlgorithms.data]);

  const selectedEnvironment =
    environmentOptions.find((environment) => environment.env_id === selectedEnvId) ?? null;
  const selectedAlgorithm =
    supportedAlgorithms.data?.find((algorithm) => algorithm.algo_id === selectedAlgoId) ?? null;

  const runningJobs = useMemo(
    () => (jobs.data ?? []).filter((job) => isActiveJob(job)),
    [jobs.data],
  );
  const recentJobs = useMemo(() => (jobs.data ?? []).slice(0, 6), [jobs.data]);

  const latestStep = useMemo(() => {
    if (!trainingMetrics.data?.length) {
      return 0;
    }
    return trainingMetrics.data[trainingMetrics.data.length - 1]?.step ?? 0;
  }, [trainingMetrics.data]);

  const progressPercent = latestTargetSteps
    ? Math.min(100, Math.round((latestStep / latestTargetSteps) * 100))
    : null;

  const evaluationReward = getLastMetricValue(
    evaluationMetrics.data ?? [],
    "eval_average_episode_rewards",
  );

  const canProceedFromStep1 = Boolean(selectedActionType);
  const canProceedFromStep2 = Boolean(selectedEnvId);
  const canProceedFromStep3 = Boolean(selectedAlgoId);
  const canStartTraining = Boolean(selectedEnvId && selectedAlgoId && generalConfig && logConfig && algoConfig);

  function resetWizard() {
    setWizardStep(1);
    setSelectedActionType("");
    setSelectedEnvId("");
    setSelectedAlgoId("");
    setGeneralConfig(null);
    setAlgoConfig(null);
    setActiveTab("general");
    setSubmitError(null);
    if (logTemplate.data) {
      setLogConfig(logTemplate.data.config);
    }
  }

  function startAnotherRun() {
    resetWizard();
    setIsCreatingRun(true);
  }

  async function handleStartTraining() {
    if (!selectedEnvId || !selectedAlgoId || !generalConfig || !logConfig || !algoConfig) {
      return;
    }

    const payload: TrainingRequest = {
      env_id: selectedEnvId,
      algo_id: selectedAlgoId,
      general_cfg: {
        ...generalConfig,
        env_id: selectedEnvId,
      },
      log_cfg: logConfig,
      algo_cfg: {
        ...algoConfig,
        algo_id: selectedAlgoId,
      },
    };

    setStarting(true);
    setSubmitError(null);
    try {
      const job = await api.startTraining(payload);
      setTrackedJobId(job.job_id ?? null);
      const totalSteps = Number(payload.general_cfg.total_steps ?? 0);
      setLatestTargetSteps(Number.isFinite(totalSteps) ? totalSteps : null);
      setIsCreatingRun(false);
      await jobs.refresh();
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "Failed to start training");
    } finally {
      setStarting(false);
    }
  }

  async function handleStopTrackedRun() {
    if (!trackedJobId) {
      return;
    }
    setStopping(true);
    try {
      await api.stopJob(trackedJobId);
      await Promise.all([jobs.refresh(), trackedJob.refresh()]);
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "Failed to stop training");
    } finally {
      setStopping(false);
    }
  }

  const showTrackedRunView = !isCreatingRun && trackedJobId !== null;
  const trackedRunIsActive = isActiveJob(trackedJob.data);

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <div className="eyebrow">Training</div>
        <h1 className="text-3xl font-semibold text-text">Training wizard</h1>
        <p className="max-w-3xl text-sm text-muted">
          The training flow now follows the Streamlit reference more closely and only shows metrics for runs you explicitly track.
        </p>
      </header>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.25fr)_380px]">
        <div className="space-y-4">
          {showTrackedRunView ? (
            <>
              <SectionCard
                eyebrow="Tracked run"
                title="Current monitored run"
                description="Config editors are hidden while you are monitoring this run. Start another run to open a fresh wizard."
                action={
                  <div className="flex gap-2">
                    <button
                      className="btn-secondary"
                      type="button"
                      onClick={() => {
                        void Promise.all([
                          jobs.refresh(),
                          trackedJob.refresh(),
                          trainingMetrics.refresh(),
                          evaluationMetrics.refresh(),
                        ]);
                      }}
                    >
                      <TimerReset className="h-4 w-4" />
                      Refresh
                    </button>
                    <button className="btn-primary" type="button" onClick={startAnotherRun}>
                      <Plus className="h-4 w-4" />
                      Start another run
                    </button>
                  </div>
                }
              >
                <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                  <div className="rounded-lg border border-border bg-page/60 p-4">
                    <div className="label mb-2">Run</div>
                    <div className="text-base font-semibold text-text">
                      {getJobDisplayName(trackedJob.data)}
                    </div>
                    <div className="mt-2 text-sm text-muted">
                      Created {formatDateTime(trackedJob.data?.created_at)}
                    </div>
                  </div>
                  <div className="rounded-lg border border-border bg-page/60 p-4">
                    <div className="label mb-2">Status</div>
                    {trackedJob.data ? <StatusBadge status={trackedJob.data.status} /> : null}
                    <div className="mt-2 text-sm text-muted">{trackedJob.data?.detail ?? "No detail"}</div>
                  </div>
                  <div className="rounded-lg border border-border bg-page/60 p-4">
                    <div className="label mb-2">Latest step</div>
                    <div className="text-3xl font-semibold text-text">
                      {formatNumber(latestStep, { maximumFractionDigits: 0 })}
                    </div>
                  </div>
                  <div className="rounded-lg border border-border bg-page/60 p-4">
                    <div className="label mb-2">Average step reward</div>
                    <div className="text-3xl font-semibold text-text">
                      {formatNumber(getLastMetricValue(trainingMetrics.data ?? [], "average_step_rewards"))}
                    </div>
                  </div>
                </div>

                <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:w-[420px]">
                  <button
                    className="btn-primary"
                    type="button"
                    onClick={startAnotherRun}
                  >
                    <Play className="h-4 w-4" />
                    New training run
                  </button>
                  <button
                    className="btn-secondary"
                    type="button"
                    disabled={!trackedRunIsActive || stopping}
                    onClick={() => void handleStopTrackedRun()}
                  >
                    {stopping ? (
                      <LoaderCircle className="h-4 w-4 animate-spin" />
                    ) : (
                      <Square className="h-4 w-4" />
                    )}
                    Stop tracked run
                  </button>
                </div>
              </SectionCard>

              <MetricChartGrid
                title="Training metrics"
                rows={trainingMetrics.data ?? []}
                emptyMessage="Training metrics are not available for the tracked run yet."
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
                title="Evaluation metrics"
                rows={evaluationMetrics.data ?? []}
                emptyMessage="Evaluation metrics are not available for the tracked run yet."
                preferredMetrics={["eval_average_episode_rewards"]}
                maxCharts={4}
              />

              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                <div className="panel p-5">
                  <div className="label mb-2">Progress</div>
                  <div className="text-3xl font-semibold text-text">
                    {progressPercent !== null ? `${progressPercent}%` : "Unavailable"}
                  </div>
                </div>
                <div className="panel p-5">
                  <div className="label mb-2">Eval episode reward</div>
                  <div className="text-3xl font-semibold text-text">
                    {formatNumber(evaluationReward)}
                  </div>
                </div>
                <div className="panel p-5">
                  <div className="label mb-2">Started</div>
                  <div className="text-lg font-semibold text-text">
                    {formatDateTime(trackedJob.data?.started_at)}
                  </div>
                </div>
                <div className="panel p-5">
                  <div className="label mb-2">Finished</div>
                  <div className="text-lg font-semibold text-text">
                    {formatDateTime(trackedJob.data?.finished_at)}
                  </div>
                </div>
              </div>
            </>
          ) : (
            <>
              <SectionCard
                eyebrow="Wizard"
                title="Create a training run"
                description="Only the current step is shown. Move forward with Next and go back if you need to adjust the earlier selections."
              >
                <div className="space-y-5">
                  <StepHeader currentStep={wizardStep} />

                  {wizardStep === 1 ? (
                    <>
                      <ActionTypeSelector
                        value={selectedActionType}
                        options={availableActionTypes}
                        onChange={(value) => setSelectedActionType(value)}
                        title="Choose action type"
                        description="Start by selecting the environment action space."
                      />
                      <StepFooter
                        canGoBack={false}
                        canGoNext={canProceedFromStep1}
                        onBack={() => undefined}
                        onNext={() => setWizardStep(2)}
                      />
                    </>
                  ) : null}

                  {wizardStep === 2 ? (
                    <>
                      <section className="panel p-5">
                        <div className="mb-4 space-y-1">
                          <div className="eyebrow">Step 2</div>
                          <h2 className="text-lg font-semibold text-text">Choose environment</h2>
                          <p className="text-sm text-muted">
                            Only environments matching the selected action type are shown.
                          </p>
                        </div>
                        <label className="space-y-2">
                          <span className="text-sm font-medium text-text">Environment</span>
                          <select
                            className="input"
                            value={selectedEnvId}
                            onChange={(event) => setSelectedEnvId(event.target.value)}
                          >
                            <option value="">Select an environment</option>
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
                            <div className="mt-1">Origin: {selectedEnvironment.origin}</div>
                          </div>
                        ) : null}
                      </section>
                      <StepFooter
                        canGoBack
                        canGoNext={canProceedFromStep2}
                        onBack={() => setWizardStep(1)}
                        onNext={() => setWizardStep(3)}
                      />
                    </>
                  ) : null}

                  {wizardStep === 3 ? (
                    <>
                      <section className="panel p-5">
                        <div className="mb-4 space-y-1">
                          <div className="eyebrow">Step 3</div>
                          <h2 className="text-lg font-semibold text-text">Choose algorithm</h2>
                          <p className="text-sm text-muted">
                            The available algorithms are filtered by the selected environment.
                          </p>
                        </div>
                        <label className="space-y-2">
                          <span className="text-sm font-medium text-text">Algorithm</span>
                          <select
                            className="input"
                            value={selectedAlgoId}
                            onChange={(event) => setSelectedAlgoId(event.target.value as AlgoType | "")}
                          >
                            <option value="">Select an algorithm</option>
                            {(supportedAlgorithms.data ?? []).map((algorithm) => (
                              <option key={algorithm.algo_id} value={algorithm.algo_id}>
                                {algorithm.algo_name}
                              </option>
                            ))}
                          </select>
                        </label>
                        {selectedAlgorithm ? (
                          <div className="mt-4 rounded-lg border border-border bg-page/60 p-4 text-sm text-muted">
                            <div className="font-medium text-text">{selectedAlgorithm.algo_name}</div>
                            <div className="mt-1">
                              Supported actions: {selectedAlgorithm.action_type.join(", ")}
                            </div>
                          </div>
                        ) : null}
                      </section>
                      <StepFooter
                        canGoBack
                        canGoNext={canProceedFromStep3}
                        onBack={() => setWizardStep(2)}
                        onNext={() => setWizardStep(4)}
                      />
                    </>
                  ) : null}

                  {wizardStep === 4 ? (
                    <>
                      <div className="flex flex-wrap gap-2">
                        {([
                          ["general", "General"],
                          ["algorithm", "Algorithm"],
                          ["logging", "Logging"],
                        ] as const).map(([tab, label]) => (
                          <button
                            key={tab}
                            className={activeTab === tab ? "btn-primary" : "btn-secondary"}
                            onClick={() => setActiveTab(tab)}
                            type="button"
                          >
                            {label}
                          </button>
                        ))}
                      </div>

                      {activeTab === "general" ? (
                        <ConfigEditor
                          title="General config"
                          description="Environment-level settings returned by the backend."
                          config={generalConfig}
                          meta={generalTemplate.data?.meta}
                          onChange={setGeneralConfig}
                          lockedKeys={["env_id"]}
                        />
                      ) : null}

                      {activeTab === "algorithm" ? (
                        <ConfigEditor
                          title="Algorithm config"
                          description="Algorithm-specific parameters for the selected run."
                          config={algoConfig}
                          meta={algoTemplate.data?.meta}
                          onChange={setAlgoConfig}
                          lockedKeys={["algo_id"]}
                        />
                      ) : null}

                      {activeTab === "logging" ? (
                        <ConfigEditor
                          title="Logging config"
                          description="Shared logging and segmentation settings."
                          config={logConfig}
                          meta={logTemplate.data?.meta}
                          onChange={setLogConfig}
                        />
                      ) : null}

                      {submitError ? (
                        <div className="rounded-lg border border-danger/20 bg-danger/10 p-3 text-sm text-danger">
                          {submitError}
                        </div>
                      ) : null}

                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <button
                          className="btn-secondary"
                          type="button"
                          onClick={() => setWizardStep(3)}
                        >
                          Back
                        </button>
                        <div className="flex flex-wrap gap-3">
                          <button
                            className="btn-secondary"
                            type="button"
                            disabled={!trackedRunIsActive || stopping}
                            onClick={() => void handleStopTrackedRun()}
                          >
                            {stopping ? (
                              <LoaderCircle className="h-4 w-4 animate-spin" />
                            ) : (
                              <Square className="h-4 w-4" />
                            )}
                            Stop tracked run
                          </button>
                          <button
                            className="btn-primary"
                            type="button"
                            disabled={!canStartTraining || starting}
                            onClick={() => void handleStartTraining()}
                          >
                            {starting ? (
                              <LoaderCircle className="h-4 w-4 animate-spin" />
                            ) : (
                              <Play className="h-4 w-4" />
                            )}
                            Start training
                          </button>
                        </div>
                      </div>
                    </>
                  ) : null}
                </div>
              </SectionCard>
            </>
          )}
        </div>

        <div className="space-y-4">
          <SectionCard
            eyebrow="Runs"
            title="Running and recent runs"
            description="Nothing is auto-selected. Use the controls below to explicitly monitor a run."
            className="side-panel-shell"
          >
            <div className="side-panel-scroll space-y-4">
              <div>
                <div className="label mb-3">Running now</div>
                <div className="space-y-2">
                  {runningJobs.length ? (
                    runningJobs.map((job) => (
                      <button
                        key={job.job_id}
                        type="button"
                        className={`w-full rounded-lg border px-3 py-3 text-left transition ${
                          trackedJobId === job.job_id
                            ? "border-accent bg-accent/10"
                            : "border-border bg-page/60 hover:border-accent/40"
                        }`}
                        onClick={() => {
                          setTrackedJobId(job.job_id ?? null);
                          setLatestTargetSteps(null);
                          setIsCreatingRun(false);
                        }}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="text-sm font-medium text-text">{getJobDisplayName(job)}</div>
                            <div className="mt-1 text-xs text-muted">{formatDateTime(job.created_at)}</div>
                          </div>
                          <StatusBadge status={job.status} />
                        </div>
                      </button>
                    ))
                  ) : (
                    <div className="rounded-lg border border-border bg-page/50 p-4 text-sm text-muted">
                      No active training jobs right now.
                    </div>
                  )}
                </div>
              </div>
              <div>
                <div className="label mb-3">Recent runs</div>
                <div className="space-y-2">
                  {recentJobs.length ? (
                    recentJobs.map((job) => (
                      <div
                        key={job.job_id}
                        className="rounded-lg border border-border bg-page/60 p-3"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="text-sm font-medium text-text">{getJobDisplayName(job)}</div>
                            <div className="mt-1 text-xs text-muted">{formatDateTime(job.created_at)}</div>
                          </div>
                          <StatusBadge status={job.status} />
                        </div>
                        <div className="mt-3 flex gap-2">
                          <button
                            className="btn-secondary h-9 px-3"
                            type="button"
                            onClick={() => {
                              setTrackedJobId(job.job_id ?? null);
                              setLatestTargetSteps(null);
                              setIsCreatingRun(false);
                            }}
                          >
                            <Eye className="h-4 w-4" />
                            Monitor
                          </button>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="rounded-lg border border-border bg-page/50 p-4 text-sm text-muted">
                      {jobs.loading ? "Loading runs…" : "No runs recorded yet."}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </SectionCard>
        </div>
      </div>
    </div>
  );
}
