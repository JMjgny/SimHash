import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { ResultStatus } from "./ResultCard";

interface SystemResult {
  status: ResultStatus;
  processingTime?: number;
  hammingDistance?: number;
}

interface ComparisonChartsProps {
  improved: SystemResult;
  previous: SystemResult;
}

const COLORS = {
  improved: "hsl(217, 91%, 60%)",
  previous: "hsl(215, 15%, 65%)",
};

const ComparisonCharts = ({ improved, previous }: ComparisonChartsProps) => {
  const bothFound = improved.status === "found" && previous.status === "found";
  if (!bothFound) return null;

  const hasTime = improved.processingTime !== undefined && previous.processingTime !== undefined;
  const hasHamming = improved.hammingDistance !== undefined && previous.hammingDistance !== undefined;

  if (!hasTime && !hasHamming) return null;

  const timeData = hasTime
    ? [
        { name: "Improved", value: improved.processingTime!, fill: COLORS.improved },
        { name: "Previous", value: previous.processingTime!, fill: COLORS.previous },
      ]
    : [];

  const hammingData = hasHamming
    ? [
        { name: "Improved", value: improved.hammingDistance!, fill: COLORS.improved },
        { name: "Previous", value: previous.hammingDistance!, fill: COLORS.previous },
      ]
    : [];

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.4 }}
      className="rounded-lg border border-border bg-card shadow-card overflow-hidden"
    >
      <div className="px-4 py-3 border-b border-border bg-muted/50 flex items-center justify-between">
        <h3 className="font-semibold text-sm text-foreground">📈 Visual Comparison</h3>
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full" style={{ background: COLORS.improved }} />
            Improved
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full" style={{ background: COLORS.previous }} />
            Previous
          </span>
        </div>
      </div>

      <div className={`grid ${hasTime && hasHamming ? "sm:grid-cols-2" : ""} divide-y sm:divide-y-0 sm:divide-x divide-border`}>
        {hasTime && (
          <div className="p-4">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide text-center mb-3">
              Processing Time (ms)
            </p>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={timeData} barSize={48}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(214, 20%, 90%)" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 12, fill: "hsl(215, 15%, 50%)" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: "hsl(215, 15%, 50%)" }} axisLine={false} tickLine={false} width={45} />
                <Tooltip
                  contentStyle={{ borderRadius: 8, border: "1px solid hsl(214, 20%, 90%)", fontSize: 12 }}
                  formatter={(value: number) => [`${value.toFixed(1)} ms`, "Time"]}
                />
                <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                  {timeData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {hasHamming && (
          <div className="p-4">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide text-center mb-3">
              Hamming Distance <span className="normal-case">(lower = better)</span>
            </p>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={hammingData} barSize={48}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(214, 20%, 90%)" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 12, fill: "hsl(215, 15%, 50%)" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: "hsl(215, 15%, 50%)" }} axisLine={false} tickLine={false} width={45} />
                <Tooltip
                  contentStyle={{ borderRadius: 8, border: "1px solid hsl(214, 20%, 90%)", fontSize: 12 }}
                  formatter={(value: number) => [value, "Distance"]}
                />
                <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                  {hammingData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default ComparisonCharts;
