import { ArrowRight, Boxes, Cpu, FlaskConical, Gauge, Rocket } from "lucide-react";
import { Link } from "react-router-dom";

import { SectionCard } from "../components/SectionCard";
import { StatusBadge } from "../components/StatusBadge";
import { formatCompactNumber, formatDateTime, getJobDisplayName } from "../lib/format";
import { useAlgorithms, useEnvironments, useJobs } from "../hooks/useApiData";

function StatTile({
  icon: Icon,
  label,
  value,
  detail,
}: {
  icon: typeof Boxes;
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="panel p-5">
      <div className="mb-4 inline-flex h-11 w-11 items-center justify-center rounded-lg bg-accent/10 text-accent">
        <Icon className="h-5 w-5" />
      </div>
      <div className="label mb-2">{label}</div>
      <div className="text-3xl font-semibold text-text">{value}</div>
      <div className="mt-2 text-sm text-muted">{detail}</div>
    </div>
  );
}

function QuickLink({
  to,
  title,
  description,
  icon: Icon,
}: {
  to: string;
  title: string;
  description: string;
  icon: typeof Rocket;
}) {
  return (
    <Link
      to={to}
      className="rounded-lg border border-border bg-page/60 p-4 transition hover:border-accent/40 hover:bg-page/80"
    >
      <div className="mb-3 inline-flex h-10 w-10 items-center justify-center rounded-lg bg-accent/10 text-accent">
        <Icon className="h-5 w-5" />
      </div>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-text">{title}</div>
          <div className="mt-1 text-xs leading-5 text-muted">{description}</div>
        </div>
        <ArrowRight className="mt-0.5 h-4 w-4 shrink-0 text-muted" />
      </div>
    </Link>
  );
}

export function DashboardPage() {
  const environments = useEnvironments();
  const algorithms = useAlgorithms();
  const jobs = useJobs(15000);

  const runningJobs = jobs.data?.filter((job) => job.status === "running").length ?? 0;
  const recentJobs = jobs.data?.slice(0, 5) ?? [];

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <div className="eyebrow">Dashboard</div>
        <h1 className="text-3xl font-semibold text-text">DRL-Wizard overview</h1>
        <p className="max-w-3xl text-sm text-muted">
          Keep the landing page focused on navigation and backend status. Training and evaluation details stay on their own pages.
        </p>
      </header>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatTile
          icon={Boxes}
          label="Environments"
          value={formatCompactNumber(environments.data?.length ?? 0)}
          detail="Environment catalog discovered by the backend."
        />
        <StatTile
          icon={Cpu}
          label="Algorithms"
          value={formatCompactNumber(algorithms.data?.length ?? 0)}
          detail="Algorithms exposed through FastAPI."
        />
        <StatTile
          icon={Gauge}
          label="Running jobs"
          value={formatCompactNumber(runningJobs)}
          detail="Current training jobs with active backend status."
        />
        <StatTile
          icon={Rocket}
          label="Recorded runs"
          value={formatCompactNumber(jobs.data?.length ?? 0)}
          detail="Persisted jobs returned by the training service."
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(320px,0.9fr)]">
        <SectionCard
          eyebrow="Workflows"
          title="Primary routes"
          description="Use the dedicated pages for the full workflows instead of putting everything on the dashboard."
        >
          <div className="grid gap-3 md:grid-cols-3">
            <QuickLink
              to="/train"
              title="Train"
              description="Choose action type, environment, and algorithm, then start a run."
              icon={Rocket}
            />
            <QuickLink
              to="/evaluate"
              title="Evaluate"
              description="Load saved runs for a selected environment and test a checkpoint."
              icon={FlaskConical}
            />
            <QuickLink
              to="/runs"
              title="Runs"
              description="Inspect saved jobs, status, artifacts, and backend metrics."
              icon={Gauge}
            />
          </div>
        </SectionCard>

        <SectionCard
          eyebrow="Recent jobs"
          title="Latest backend activity"
          description="A concise summary of recent jobs without surfacing charts or evaluation results here."
        >
          <div className="space-y-3">
            {recentJobs.length ? (
              recentJobs.map((job) => (
                <div
                  key={job.job_id}
                  className="flex flex-col gap-3 rounded-lg border border-border bg-page/50 p-4 lg:flex-row lg:items-center lg:justify-between"
                >
                  <div className="space-y-1">
                    <div className="text-sm font-medium text-text">{getJobDisplayName(job)}</div>
                    <div className="text-xs text-muted">
                      Created {formatDateTime(job.created_at)}
                    </div>
                  </div>
                  <StatusBadge status={job.status} />
                </div>
              ))
            ) : (
              <div className="rounded-lg border border-border bg-page/50 p-4 text-sm text-muted">
                {jobs.loading ? "Loading jobs…" : "No jobs have been recorded yet."}
              </div>
            )}
          </div>
        </SectionCard>
      </div>
    </div>
  );
}
