import { useState } from "react";
import { motion } from "framer-motion";
import { Search, Shield, Zap, GitCompare } from "lucide-react";
import ImageUploader from "@/components/ImageUploader";
import ResultCard, { ResultStatus } from "@/components/ResultCard";
import MetricsSummary from "@/components/MetricsSummary";
import ComparisonCharts from "@/components/ComparisonCharts";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const API_BASE = "http://localhost:5000";

interface SearchResult {
  status: ResultStatus;
  matchImage?: string;
  matchFilename?: string;
  processingTime?: number;
  hammingDistance?: number;
}

const Index = () => {
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<"single" | "compare">("single");
  const [improvedResult, setImprovedResult] = useState<SearchResult>({ status: "idle" });
  const [previousResult, setPreviousResult] = useState<SearchResult>({ status: "idle" });

  const searchSystem = async (
    endpoint: string,
    file: File,
    setter: (r: SearchResult) => void
  ) => {
    setter({ status: "loading" });
    const formData = new FormData();
    formData.append("tampered", file);
    const start = performance.now();

    try {
      const res = await fetch(`${API_BASE}${endpoint}`, { method: "POST", body: formData });
      const elapsed = performance.now() - start;
      if (!res.ok) throw new Error("Server error");
      const data = await res.json();

      if (data.status === "not_found") {
        setter({ status: "not_found", processingTime: elapsed });
      } else {
        setter({
          status: "found",
          matchImage: data.preview,
          matchFilename: data.filename,
          processingTime: data.time_ms ?? elapsed,
          hammingDistance: data.hamming_distance,
        });
      }
    } catch {
      setter({ status: "error" });
    }
  };

  const handleSearch = () => {
    if (!file) return;
    searchSystem("/find-original", file, setImprovedResult);
    if (mode === "compare") {
      searchSystem("/find-original-previous", file, setPreviousResult);
    }
  };

  const handleReset = () => {
    setFile(null);
    setImprovedResult({ status: "idle" });
    setPreviousResult({ status: "idle" });
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Hero */}
      <header className="relative overflow-hidden" style={{ background: "var(--gradient-hero)" }}>
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_50%,rgba(255,255,255,0.1),transparent_60%)]" />
        <div className="container relative py-16 md:py-20 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary-foreground/10 backdrop-blur-sm border border-primary-foreground/20 mb-6">
              <Shield className="w-4 h-4 text-primary-foreground" />
              <span className="text-sm font-medium text-primary-foreground/90">Simhash Forensic Analysis</span>
            </div>
            <h1 className="text-3xl md:text-5xl font-extrabold text-primary-foreground tracking-tight leading-tight">
              Digital Image Forgery Detection
            </h1>
            <p className="mt-4 text-lg text-primary-foreground/80 max-w-xl mx-auto">
              Upload a tampered image to locate its original version using Simhash fingerprinting.
            </p>
          </motion.div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container -mt-8 relative z-10 pb-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.5 }}
          className="bg-card rounded-xl shadow-elevated border border-border p-6 md:p-8"
        >
          {/* Mode Tabs */}
          <Tabs value={mode} onValueChange={(v) => { setMode(v as "single" | "compare"); handleReset(); }}>
            <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
              <TabsList className="bg-muted">
                <TabsTrigger value="single" className="gap-1.5">
                  <Zap className="w-4 h-4" /> Single System
                </TabsTrigger>
                <TabsTrigger value="compare" className="gap-1.5">
                  <GitCompare className="w-4 h-4" /> Compare Systems
                </TabsTrigger>
              </TabsList>

              {mode === "compare" && (
                <p className="text-xs text-muted-foreground">
                  Compare the improved system against the previous implementation side by side.
                </p>
              )}
            </div>

            <TabsContent value="single" className="mt-0">
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <ImageUploader onImageSelect={setFile} label="Tampered Image" />
                  <Button
                    onClick={handleSearch}
                    disabled={!file || improvedResult.status === "loading"}
                    className="w-full gap-2"
                    size="lg"
                  >
                    <Search className="w-4 h-4" />
                    Find Matching Original
                  </Button>
                </div>
                <ResultCard title="Improved System Result" {...improvedResult} />
              </div>
            </TabsContent>

            <TabsContent value="compare" className="mt-0">
              <div className="space-y-6">
                <div className="max-w-sm mx-auto space-y-4">
                  <ImageUploader onImageSelect={setFile} label="Tampered Image" />
                  <Button
                    onClick={handleSearch}
                    disabled={!file || improvedResult.status === "loading"}
                    className="w-full gap-2"
                    size="lg"
                  >
                    <Search className="w-4 h-4" />
                    Run Both Systems
                  </Button>
                </div>

                <div className="grid md:grid-cols-2 gap-6">
                  <ResultCard title="✨ Improved System" {...improvedResult} />
                  <ResultCard title="📦 Previous System" {...previousResult} />
                </div>

                <MetricsSummary improved={improvedResult} previous={previousResult} />
                <ComparisonCharts improved={improvedResult} previous={previousResult} />
              </div>
            </TabsContent>
          </Tabs>
        </motion.div>
      </main>

      {/* Footer */}
      <footer className="text-center py-6 text-sm text-muted-foreground">
        © 2025 Cuchado – Jagonoy Thesis
      </footer>
    </div>
  );
};

export default Index;
