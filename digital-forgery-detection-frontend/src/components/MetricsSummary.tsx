import { motion } from "framer-motion";
import { TrendingUp, TrendingDown, Minus, Zap, Target, Clock } from "lucide-react";
import type { ResultStatus } from "./ResultCard";

interface SystemResult {
  status: ResultStatus;
  processingTime?: number;
  hammingDistance?: number;
}

interface MetricsSummaryProps {
  improved: SystemResult;
  previous: SystemResult;
}

const MetricsSummary = ({ improved, previous }: MetricsSummaryProps) => {
  const bothFound = improved.status === "found" && previous.status === "found";
  const bothDone = improved.status !== "idle" && improved.status !== "loading" &&
                   previous.status !== "idle" && previous.status !== "loading";

  if (!bothDone) return null;

  const speedImprovement =
    improved.processingTime && previous.processingTime
      ? ((previous.processingTime - improved.processingTime) / previous.processingTime) * 100
      : null;

  const accuracyBetter =
    improved.hammingDistance !== undefined && previous.hammingDistance !== undefined
      ? previous.hammingDistance - improved.hammingDistance
      : null;

  const improvedFound = improved.status === "found";
  const previousFound = previous.status === "found";

  const metrics = [
    {
      icon: Clock,
      label: "Speed",
      value: speedImprovement !== null ? `${Math.abs(speedImprovement).toFixed(1)}%` : "N/A",
      detail: speedImprovement !== null
        ? speedImprovement > 0 ? "faster" : speedImprovement < 0 ? "slower" : "same"
        : "Incomplete data",
      trend: speedImprovement !== null ? (speedImprovement > 0 ? "up" : speedImprovement < 0 ? "down" : "neutral") : "neutral",
    },
    {
      icon: Target,
      label: "Hamming Distance",
      value: bothFound && improved.hammingDistance !== undefined && previous.hammingDistance !== undefined
        ? `${improved.hammingDistance} vs ${previous.hammingDistance}`
        : "N/A",
      detail: accuracyBetter !== null
        ? accuracyBetter > 0 ? "closer match (improved)" : accuracyBetter < 0 ? "previous closer" : "identical"
        : "Incomplete data",
      trend: accuracyBetter !== null ? (accuracyBetter > 0 ? "up" : accuracyBetter < 0 ? "down" : "neutral") : "neutral",
    },
    {
      icon: Zap,
      label: "Detection",
      value: improvedFound && previousFound ? "Both found" : improvedFound ? "Improved only" : previousFound ? "Previous only" : "Neither",
      detail: improvedFound && !previousFound ? "Improved system detected what previous missed" : improvedFound && previousFound ? "Both systems detected a match" : !improvedFound && previousFound ? "Previous system outperformed" : "No matches found",
      trend: improvedFound && !previousFound ? "up" : !improvedFound && previousFound ? "down" : "neutral",
    },
  ];

  const TrendIcon = ({ trend }: { trend: string }) =>
    trend === "up" ? <TrendingUp className="w-4 h-4 text-success" /> :
    trend === "down" ? <TrendingDown className="w-4 h-4 text-destructive" /> :
    <Minus className="w-4 h-4 text-muted-foreground" />;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.3 }}
      className="rounded-lg border border-border bg-card shadow-card overflow-hidden"
    >
      <div className="px-4 py-3 border-b border-border bg-muted/50">
        <h3 className="font-semibold text-sm text-foreground">📊 Comparison Summary</h3>
      </div>
      <div className="grid sm:grid-cols-3 divide-y sm:divide-y-0 sm:divide-x divide-border">
        {metrics.map((m) => (
          <div key={m.label} className="p-4 flex flex-col items-center text-center gap-1.5">
            <div className="flex items-center gap-2">
              <m.icon className="w-4 h-4 text-muted-foreground" />
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">{m.label}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xl font-bold text-foreground">{m.value}</span>
              <TrendIcon trend={m.trend} />
            </div>
            <span className="text-xs text-muted-foreground">{m.detail}</span>
          </div>
        ))}
      </div>
    </motion.div>
  );
};

export default MetricsSummary;
