import { motion } from "framer-motion";
import { CheckCircle2, XCircle, Clock } from "lucide-react";

export type ResultStatus = "idle" | "loading" | "found" | "not_found" | "error";

interface ResultCardProps {
  title: string;
  status: ResultStatus;
  matchImage?: string;
  matchFilename?: string;
  processingTime?: number;
  hammingDistance?: number;
  accentColor?: string;
}

const ResultCard = ({
  title,
  status,
  matchImage,
  matchFilename,
  processingTime,
  hammingDistance,
}: ResultCardProps) => {
  return (
    <div className="flex-1 min-w-[280px]">
      <div className="rounded-lg border border-border bg-card shadow-card overflow-hidden">
        <div className="px-4 py-3 border-b border-border bg-muted/50">
          <h3 className="font-semibold text-sm text-foreground">{title}</h3>
        </div>

        <div className="p-4 min-h-[260px] flex flex-col items-center justify-center">
          {status === "idle" && (
            <p className="text-sm text-muted-foreground text-center">Upload an image and search to see results.</p>
          )}

          {status === "loading" && (
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
              className="w-10 h-10 border-3 border-primary border-t-transparent rounded-full"
            />
          )}

          {status === "found" && matchImage && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="w-full flex flex-col items-center gap-3"
            >
              <img
                src={matchImage}
                alt="Matched original"
                className="w-full max-h-44 object-contain rounded-md border border-border"
              />
              <div className="flex items-center gap-1.5 text-success">
                <CheckCircle2 className="w-4 h-4" />
                <span className="text-sm font-medium">Match Found</span>
              </div>
              {matchFilename && (
                <p className="text-xs font-mono text-muted-foreground truncate max-w-full">{matchFilename}</p>
              )}
            </motion.div>
          )}

          {status === "not_found" && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center gap-2">
              <XCircle className="w-10 h-10 text-destructive/70" />
              <p className="text-sm text-destructive font-medium">No Match Found</p>
            </motion.div>
          )}

          {status === "error" && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center gap-2">
              <XCircle className="w-10 h-10 text-warning/70" />
              <p className="text-sm text-warning font-medium">Connection Error</p>
              <p className="text-xs text-muted-foreground">Could not reach the server.</p>
            </motion.div>
          )}
        </div>

        {(processingTime !== undefined || hammingDistance !== undefined) && status === "found" && (
          <div className="px-4 py-3 border-t border-border bg-muted/30 flex gap-4 justify-center">
            {processingTime !== undefined && (
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Clock className="w-3.5 h-3.5" />
                <span>{processingTime.toFixed(0)}ms</span>
              </div>
            )}
            {hammingDistance !== undefined && (
              <div className="text-xs text-muted-foreground">
                Hamming: <span className="font-mono font-medium text-foreground">{hammingDistance}</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultCard;
