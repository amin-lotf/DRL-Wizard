import { ActivitySquare, Bot, FlaskConical, LayoutDashboard, Server } from "lucide-react";
import { NavLink, Route, Routes, useLocation } from "react-router-dom";

import { RouteErrorBoundary } from "./components/RouteErrorBoundary";
import { API_BASE_URL } from "./lib/api";
import { cn } from "./lib/format";
import { ApiStatusPage } from "./pages/ApiStatusPage";
import { DashboardPage } from "./pages/DashboardPage";
import { EvaluatePage } from "./pages/EvaluatePage";
import { RunsPage } from "./pages/RunsPage";
import { TrainPage } from "./pages/TrainPage";

const navigation = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/train", label: "Train", icon: Bot },
  { to: "/evaluate", label: "Evaluate", icon: FlaskConical },
  { to: "/runs", label: "Runs", icon: ActivitySquare },
  { to: "/status", label: "API", icon: Server },
];

function NavigationLink({
  to,
  label,
  icon: Icon,
  end = false,
}: {
  to: string;
  label: string;
  icon: typeof LayoutDashboard;
  end?: boolean;
}) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        cn("sidebar-link", isActive ? "sidebar-link-active" : "sidebar-link-idle")
      }
    >
      <Icon className="h-4 w-4" />
      <span>{label}</span>
    </NavLink>
  );
}

function AppContent() {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-page text-text">
      <div className="mx-auto max-w-[1500px] px-4 py-4 lg:px-6">
        <div className="grid gap-4 lg:grid-cols-[260px_minmax(0,1fr)]">
          <aside className="panel p-4 lg:sticky lg:top-4 lg:h-[calc(100vh-2rem)]">
            <div className="flex h-full flex-col">
              <div className="mb-6">
                <div className="mb-3 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-accent/10 text-accent">
                  <Bot className="h-6 w-6" />
                </div>
                <h1 className="text-xl font-semibold text-text">DRL-Wizard</h1>
                <p className="mt-2 text-sm text-muted">
                  Frontend for the current FastAPI backend and Streamlit reference workflow.
                </p>
              </div>

              <nav className="grid gap-2 sm:grid-cols-2 lg:grid-cols-1">
                {navigation.map((item) => (
                  <NavigationLink key={item.to} {...item} />
                ))}
              </nav>

              <div className="mt-auto rounded-lg border border-border bg-page/60 p-4">
                <div className="label mb-2">API base</div>
                <div className="break-all text-sm text-text">{API_BASE_URL}</div>
              </div>
            </div>
          </aside>

          <main className="min-w-0 py-2">
            <RouteErrorBoundary routeKey={location.pathname}>
              <Routes>
                <Route path="/" element={<DashboardPage />} />
                <Route path="/train" element={<TrainPage />} />
                <Route path="/evaluate" element={<EvaluatePage />} />
                <Route path="/runs" element={<RunsPage />} />
                <Route path="/status" element={<ApiStatusPage />} />
              </Routes>
            </RouteErrorBoundary>
          </main>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  return <AppContent />;
}
