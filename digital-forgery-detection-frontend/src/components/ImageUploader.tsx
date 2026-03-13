import { useCallback, useState } from "react";
import { Upload, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface ImageUploaderProps {
  onImageSelect: (file: File | null) => void;
  label?: string;
}

const ImageUploader = ({ onImageSelect, label = "Upload Image" }: ImageUploaderProps) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFile = useCallback((file: File | null) => {
    if (file) {
      const url = URL.createObjectURL(file);
      setPreview(url);
      onImageSelect(file);
    } else {
      setPreview(null);
      onImageSelect(null);
    }
  }, [onImageSelect]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file?.type.startsWith("image/")) handleFile(file);
  }, [handleFile]);

  const clear = () => {
    setPreview(null);
    onImageSelect(null);
  };

  return (
    <div className="w-full">
      <p className="text-sm font-medium text-foreground mb-2">{label}</p>
      <AnimatePresence mode="wait">
        {!preview ? (
          <motion.label
            key="drop"
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.97 }}
            className={`flex flex-col items-center justify-center w-full h-56 border-2 border-dashed rounded-lg cursor-pointer transition-colors ${
              isDragging ? "border-primary bg-primary/5" : "border-border hover:border-primary/50 bg-card"
            }`}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            <Upload className="w-8 h-8 text-muted-foreground mb-2" />
            <span className="text-sm text-muted-foreground">Drag & drop or click to upload</span>
            <span className="text-xs text-muted-foreground/60 mt-1">PNG, JPG, BMP</span>
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
            />
          </motion.label>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.97 }}
            className="relative w-full h-56 rounded-lg overflow-hidden bg-card shadow-card border border-border"
          >
            <img src={preview} alt="Preview" className="w-full h-full object-contain p-2" />
            <button
              onClick={clear}
              className="absolute top-2 right-2 p-1.5 rounded-full bg-foreground/80 text-background hover:bg-foreground transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ImageUploader;
