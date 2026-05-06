import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  formatNumber,
  getLastMetricValue,
  getMetricKeys,
  titleFromKey,
} from "../lib/format";
import type { MetricRow } from "../types/api";

interface MetricChartGridProps {
  title: string;
  rows: MetricRow[];
  emptyMessage: string;
  preferredMetrics?: string[];
  maxCharts?: number;
}

const palette = ["#2dd4bf", "#34d399", "#fb923c", "#60a5fa", "#facc15"];

export function MetricChartGrid({
  title,
  rows,
  emptyMessage,
  preferredMetrics,
  maxCharts = 6,
}: MetricChartGridProps) {
  const metrics = useMemo(() => {
    const available = getMetricKeys(rows);
    if (!available.length) {
      return [];
    }
    if (preferredMetrics?.length) {
      const filtered = preferredMetrics.filter((metric) => available.includes(metric));
      const remaining = available.filter((metric) => !filtered.includes(metric));
      return [...filtered, ...remaining].slice(0, maxCharts);
    }
    return available.slice(0, maxCharts);
  }, [maxCharts, preferredMetrics, rows]);

  return (
    <section className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-text">{title}</h2>
      </div>
      {metrics.length ? (
        <div className="grid gap-4 xl:grid-cols-2">
          {metrics.map((metric, index) => (
            <div key={metric} className="panel p-4">
              <div className="mb-4 flex items-start justify-between gap-4">
                <div>
                  <div className="label mb-1">Metric</div>
                  <h3 className="text-base font-semibold text-text">{titleFromKey(metric)}</h3>
                </div>
                <div className="text-right">
                  <div className="label mb-1">Latest</div>
                  <div className="text-sm font-medium text-text">
                    {formatNumber(getLastMetricValue(rows, metric))}
                  </div>
                </div>
              </div>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={rows}>
                    <CartesianGrid stroke="rgba(142, 163, 154, 0.12)" vertical={false} />
                    <XAxis
                      dataKey="step"
                      tick={{ fill: "#8ea39a", fontSize: 12 }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <YAxis
                      tick={{ fill: "#8ea39a", fontSize: 12 }}
                      tickLine={false}
                      axisLine={false}
                      width={72}
                      tickFormatter={(value: number) => formatNumber(value)}
                    />
                    <Tooltip
                      contentStyle={{
                        borderRadius: 12,
                        border: "1px solid rgba(36, 49, 44, 1)",
                        backgroundColor: "rgba(18, 26, 24, 0.96)",
                      }}
                      formatter={(value: number) => formatNumber(value)}
                    />
                    <Line
                      type="monotone"
                      dataKey={metric}
                      stroke={palette[index % palette.length]}
                      strokeWidth={2.5}
                      dot={false}
                      activeDot={{ r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="panel p-5 text-sm text-muted">{emptyMessage}</div>
      )}
    </section>
  );
}
