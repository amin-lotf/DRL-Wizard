import { Activity, Database, ShieldCheck } from "lucide-react";

import { SectionCard } from "../components/SectionCard";
import { API_BASE_URL } from "../lib/api";
import { formatCompactNumber } from "../lib/format";
import { useAlgorithms, useEnvironments, useHealth, useJobs } from "../hooks/useApiData";

function StatusTile({
  icon: Icon,
  label,
  value,
  detail,
}: {
  icon: typeof Activity;
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

export function ApiStatusPage() {
  const health = useHealth();
  const environments = useEnvironments();
  const algorithms = useAlgorithms();
  const jobs = useJobs();

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <div className="eyebrow">API status</div>
        <h1 className="text-3xl font-semibold text-text">Connectivity and backend availability</h1>
        <p className="max-w-3xl text-sm text-muted">
          This page verifies the configured API base URL and shows a small snapshot of the backend surface the React app depends on.
        </p>
      </header>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatusTile
          icon={ShieldCheck}
          label="Health"
          value={health.data?.ok ? "Online" : "Unknown"}
          detail={health.error ?? "Reported by `/healthz`."}
        />
        <StatusTile
          icon={Database}
          label="Environments"
          value={formatCompactNumber(environments.data?.length ?? 0)}
          detail="Available environment metadata."
        />
        <StatusTile
          icon={Activity}
          label="Algorithms"
          value={formatCompactNumber(algorithms.data?.length ?? 0)}
          detail="Available algorithm metadata."
        />
        <StatusTile
          icon={Activity}
          label="Jobs"
          value={formatCompactNumber(jobs.data?.length ?? 0)}
          detail="Persisted training jobs."
        />
      </div>

      <SectionCard
        eyebrow="Configuration"
        title="Frontend API base URL"
        description="The React app never hardcodes localhost inside page components; all requests resolve through the configured Vite environment variable."
      >
        <div className="code-block">{API_BASE_URL}</div>
      </SectionCard>
    </div>
  );
}
