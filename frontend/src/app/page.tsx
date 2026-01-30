"use client";

import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import {
  Upload,
  Brain,
  Layers,
  Target,
  Zap,
  BarChart3,
  FileText,
  ChevronDown,
  ExternalLink,
  BookOpen,
} from "lucide-react";
import ImageComparisonSlider from "@/components/ImageComparisonSlider";

const API_URL = "/api";

// ── Class palette ───────────────────────────────────────────────────────────
const CLASSES = [
  { name: "Background", color: [0, 0, 0] },
  { name: "Water", color: [0, 0, 255] },
  { name: "Building No Damage", color: [20, 255, 20] },
  { name: "Building Minor Damage", color: [255, 215, 0] },
  { name: "Building Major Damage", color: [255, 0, 0] },
  { name: "Building Total Destruction", color: [139, 0, 0] },
  { name: "Vehicle", color: [128, 0, 128] },
  { name: "Road-Clear", color: [128, 128, 128] },
  { name: "Road-Blocked", color: [64, 64, 64] },
  { name: "Tree", color: [0, 100, 0] },
  { name: "Pool", color: [0, 128, 255] },
];

// ── Fade-in wrapper ─────────────────────────────────────────────────────────
function FadeIn({
  children,
  delay = 0,
  className = "",
}: {
  children: React.ReactNode;
  delay?: number;
  className?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.6, delay, ease: "easeOut" }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

// ── Bento card ──────────────────────────────────────────────────────────────
function Card({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`bg-[#12141A] border border-[#1E2028] rounded-2xl p-6 ${className}`}
    >
      {children}
    </div>
  );
}

// ── Expandable card ─────────────────────────────────────────────────────────
function Expandable({
  icon: Icon,
  title,
  children,
  defaultOpen = false,
}: {
  icon: React.ElementType;
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <Card>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between text-left"
      >
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-[#2D7DD2]/10 flex items-center justify-center">
            <Icon className="w-5 h-5 text-[#2D7DD2]" />
          </div>
          <h3 className="font-semibold text-white">{title}</h3>
        </div>
        <ChevronDown
          className={`w-5 h-5 text-gray-500 transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>
      {open && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          className="mt-4 text-gray-400 leading-relaxed text-sm space-y-3"
        >
          {children}
        </motion.div>
      )}
    </Card>
  );
}

// ── Metric card ─────────────────────────────────────────────────────────────
function Metric({
  value,
  label,
  delta,
}: {
  value: string;
  label: string;
  delta: string;
}) {
  return (
    <Card className="text-center">
      <p className="text-3xl font-bold text-[#2D7DD2]">{value}</p>
      <p className="text-sm text-gray-500 mt-1">{label}</p>
      <p className="text-sm text-emerald-400 mt-1">&#9650; {delta}</p>
    </Card>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// MAIN PAGE
// ═════════════════════════════════════════════════════════════════════════════
export default function Home() {
  const [preview, setPreview] = useState<string | null>(null);
  const [maskSrc, setMaskSrc] = useState<string | null>(null);
  const [overlaySrc, setOverlaySrc] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback(async (f: File) => {
    setError(null);
    setMaskSrc(null);
    setOverlaySrc(null);

    const objectUrl = URL.createObjectURL(f);
    setPreview(objectUrl);

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", f);
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      setMaskSrc(`data:image/png;base64,${data.mask}`);
      setOverlaySrc(`data:image/png;base64,${data.overlay}`);
    } catch {
      setError(
        "Could not reach the inference server. Make sure the FastAPI backend is running on Jetstream."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    },
    [handleFile]
  );

  // ── Bar chart data ────────────────────────────────────────────────────────
  const barData = [
    { cls: "Background", base: 89.50, ours: 91.20 },
    { cls: "Water", base: 78.30, ours: 80.10 },
    { cls: "No Damage", base: 69.80, ours: 70.30 },
    { cls: "Minor Dmg", base: 58.10, ours: 72.00 },
    { cls: "Major Dmg", base: 59.60, ours: 72.10 },
    { cls: "Total Dest.", base: 59.00, ours: 60.70 },
    { cls: "Road Clear", base: 77.80, ours: 83.50 },
    { cls: "Road Block", base: 55.40, ours: 58.20 },
    { cls: "Tree", base: 85.20, ours: 87.30 },
    { cls: "Pool", base: 76.70, ours: 88.20 },
    { cls: "Vehicle", base: 63.20, ours: 67.10 },
  ];

  return (
    <main className="min-h-screen">
      {/* ════════ HERO ════════ */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-[#2D7DD2]/8 via-transparent to-transparent" />
        <div className="relative max-w-5xl mx-auto px-6 pt-20 pb-16 text-center">
          <FadeIn>
            <div className="flex items-center justify-center gap-3 mb-4">
              <img src="/bina-logo.png" alt="Bina Labs" className="h-16 object-contain" />
              <span className="text-sm font-semibold tracking-wide text-gray-300">Lehigh University</span>
            </div>
          </FadeIn>
          <FadeIn delay={0.1}>
            <h1 className="text-5xl sm:text-6xl font-bold bg-gradient-to-r from-[#2D7DD2] to-[#A0D2FF] bg-clip-text text-transparent leading-tight">
              DA-SegFormer
            </h1>
          </FadeIn>
          <FadeIn delay={0.2}>
            <p className="mt-4 text-lg text-gray-400 max-w-2xl mx-auto leading-relaxed">
              Damage-Aware Semantic Segmentation for Fine-Grained
              Post-Disaster Assessment using a SegFormer-B4 architecture
              with Online Hard Example Mining.
            </p>
          </FadeIn>
          <FadeIn delay={0.3}>
            <p className="mt-3 text-sm text-gray-600">
              Kevin Zhu, William Tang, Raphael Hay Tene, Zesheng Liu, Nhut Le, Dr. Maryam Rahnemoonfar
            </p>
          </FadeIn>
          <FadeIn delay={0.4}>
            <div className="mt-6 flex justify-center gap-4">
              <a
                href="#demo"
                className="px-5 py-2.5 bg-[#2D7DD2] hover:bg-[#2568B5] text-white rounded-xl transition-colors text-sm font-medium"
              >
                Try the Demo
              </a>
              <Link
                href="/learn"
                className="px-5 py-2.5 border border-[#1E2028] hover:border-[#2D7DD2]/40 text-gray-300 hover:text-white rounded-xl transition-colors text-sm font-medium inline-flex items-center gap-2"
              >
                <BookOpen className="w-4 h-4" /> Learn How It Works
              </Link>
            </div>
          </FadeIn>
        </div>
      </section>

      {/* ════════ KEY METRICS ════════ */}
      <section className="max-w-5xl mx-auto px-6 pb-16">
        <FadeIn>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Metric value="74.67%" label="Mean IoU" delta="+4.37% vs baseline" />
            <Metric value="+13.9%" label="Minor Damage" delta="58.1% &rarr; 72.0% IoU" />
            <Metric value="+12.5%" label="Major Damage" delta="59.6% &rarr; 72.1% IoU" />
            <Metric value="+11.5%" label="Pool" delta="76.7% &rarr; 88.2% IoU" />
          </div>
        </FadeIn>
      </section>

      {/* ════════ INTERACTIVE DEMO ════════ */}
      <section id="demo" className="max-w-5xl mx-auto px-6 pb-20 scroll-mt-8">
        <FadeIn>
          <div className="flex items-center gap-3 mb-8">
            <div className="w-10 h-10 rounded-xl bg-[#2D7DD2]/10 flex items-center justify-center">
              <Upload className="w-5 h-5 text-[#2D7DD2]" />
            </div>
            <h2 className="text-2xl font-semibold text-white">Interactive Demo</h2>
          </div>
        </FadeIn>

        <FadeIn delay={0.1}>
          <div className="grid md:grid-cols-[1fr_280px] gap-6">
            {/* Upload zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              className={`relative border-2 border-dashed rounded-2xl p-12 text-center transition-colors cursor-pointer ${
                dragOver
                  ? "border-[#2D7DD2] bg-[#2D7DD2]/5"
                  : "border-[#1E2028] hover:border-[#2D7DD2]/40"
              }`}
              onClick={() => document.getElementById("file-input")?.click()}
            >
              <input
                id="file-input"
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files?.[0]) handleFile(e.target.files[0]);
                }}
              />
              <Upload className="w-10 h-10 text-gray-600 mx-auto mb-4" />
              <p className="text-gray-400">
                Drag &amp; drop a satellite image here, or{" "}
                <span className="text-[#2D7DD2] underline">browse</span>
              </p>
              <p className="text-xs text-gray-600 mt-2">
                JPG, PNG, or TIF &middot; High-resolution post-disaster imagery
              </p>
            </div>

            {/* Legend */}
            <Card>
              <h3 className="font-semibold text-white text-sm mb-3">Class Legend</h3>
              <div className="space-y-2">
                {CLASSES.map((c) => (
                  <div key={c.name} className="flex items-center gap-2.5">
                    <div
                      className="w-3.5 h-3.5 rounded-sm border border-white/10 flex-shrink-0"
                      style={{ backgroundColor: `rgb(${c.color.join(",")})` }}
                    />
                    <span className="text-xs text-gray-400">{c.name}</span>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </FadeIn>

        {/* Loading */}
        {loading && (
          <FadeIn>
            <div className="mt-10 text-center">
              <div className="inline-block w-8 h-8 border-2 border-[#2D7DD2] border-t-transparent rounded-full animate-spin" />
              <p className="text-gray-500 mt-3 text-sm">Running inference on Jetstream GPU...</p>
            </div>
          </FadeIn>
        )}

        {/* Error with preview */}
        {error && preview && (
          <FadeIn>
            <div className="mt-10 space-y-4">
              <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-4 text-sm text-amber-300">
                {error}
              </div>
              <Card>
                <h3 className="text-white font-medium mb-3">Uploaded Image</h3>
                <img src={preview} alt="Uploaded" className="w-full rounded-lg" />
              </Card>
            </div>
          </FadeIn>
        )}

        {/* Comparison slider result */}
        {preview && overlaySrc && !loading && (
          <FadeIn>
            <div className="mt-10 space-y-6">
              <p className="text-sm text-gray-500 text-center">
                Drag the slider to compare the original with the segmentation overlay.
              </p>
              <ImageComparisonSlider
                beforeSrc={preview}
                afterSrc={overlaySrc}
                beforeLabel="Original"
                afterLabel="Segmentation Overlay"
              />
              {maskSrc && (
                <details className="group">
                  <summary className="text-sm text-gray-500 cursor-pointer hover:text-gray-300 transition-colors">
                    View raw segmentation mask
                  </summary>
                  <div className="mt-3">
                    <img
                      src={maskSrc}
                      alt="Segmentation mask"
                      className="w-full max-w-[900px] mx-auto rounded-xl border border-[#1E2028]"
                    />
                  </div>
                </details>
              )}
            </div>
          </FadeIn>
        )}
      </section>

      {/* ════════ HOW IT WORKS ════════ */}
      <section className="max-w-5xl mx-auto px-6 pb-20">
        <FadeIn>
          <div className="flex items-center gap-3 mb-8">
            <div className="w-10 h-10 rounded-xl bg-[#2D7DD2]/10 flex items-center justify-center">
              <Brain className="w-5 h-5 text-[#2D7DD2]" />
            </div>
            <h2 className="text-2xl font-semibold text-white">How It Works</h2>
          </div>
        </FadeIn>

        <div className="grid md:grid-cols-2 gap-4">
          <FadeIn>
            <Expandable icon={Layers} title="Hierarchical MiT-B4 Encoder" defaultOpen>
              <p>
                The <strong className="text-gray-200">Mix Transformer (MiT-B4)</strong> encoder
                from NVIDIA processes input images through 4 hierarchical stages:
              </p>
              <ul className="list-disc list-inside space-y-1 ml-1">
                <li><strong className="text-gray-300">Stage 1</strong>: Fine-grained local features (1/4 resolution)</li>
                <li><strong className="text-gray-300">Stage 2</strong>: Mid-level structural patterns (1/8 resolution)</li>
                <li><strong className="text-gray-300">Stage 3</strong>: Contextual scene understanding (1/16 resolution)</li>
                <li><strong className="text-gray-300">Stage 4</strong>: Global semantic features (1/32 resolution)</li>
              </ul>
              <p>
                These multi-scale features are fused by the lightweight MLP decoder
                to produce the final segmentation map.
              </p>
            </Expandable>
          </FadeIn>

          <FadeIn delay={0.1}>
            <Expandable icon={Target} title="OHEM — Online Hard Example Mining" defaultOpen>
              <p>
                Standard cross-entropy treats every pixel equally. In disaster imagery,
                <strong className="text-gray-200"> 95%+ of pixels are background</strong> — the
                model can reach high accuracy by ignoring rare damage classes.
              </p>
              <p><strong className="text-gray-200">OHEM</strong> fixes this by:</p>
              <ol className="list-decimal list-inside space-y-1 ml-1">
                <li>Computing loss for every pixel in the batch</li>
                <li>Sorting pixels by loss (hardest first)</li>
                <li><strong className="text-gray-200">Keeping only the top k=100,000</strong> hardest pixels for backpropagation</li>
              </ol>
            </Expandable>
          </FadeIn>

          <FadeIn>
            <Expandable icon={Zap} title="High-Resolution Crop Strategy">
              <p>
                RescueNet UAV images are <strong className="text-gray-200">3000x4000 pixels</strong>, captured after
                Hurricane Michael by DJI Mavic Pro drones at 200ft. Naively resizing to 512x512
                destroys the fine texture details needed to distinguish damage levels.
              </p>
              <ul className="list-disc list-inside space-y-1 ml-1">
                <li><strong className="text-gray-300">Training</strong>: Random 1024x1024 center crops preserving high-frequency details</li>
                <li><strong className="text-gray-300">Inference</strong>: Sliding window with 25% overlap and smooth stitching</li>
                <li><strong className="text-gray-300">Result</strong>: +4.37% mIoU gain over resize-based baseline</li>
              </ul>
            </Expandable>
          </FadeIn>

          <FadeIn delay={0.1}>
            <Expandable icon={Target} title="Compound Loss (OHEM + Dice)">
              <p>Two complementary loss functions combined:</p>
              <ul className="list-disc list-inside space-y-1 ml-1">
                <li><strong className="text-gray-300">OHEM Cross-Entropy</strong>: Focuses on hard pixels (top 25%)</li>
                <li><strong className="text-gray-300">Dice Loss</strong>: Optimizes per-class overlap, handles class imbalance</li>
              </ul>
              <p className="mt-2">
                <code className="bg-white/5 px-2 py-1 rounded text-xs text-gray-300">
                  L_total = L_Dice + L_OHEM (equal weighting)
                </code>
              </p>
            </Expandable>
          </FadeIn>
        </div>
      </section>

      {/* ════════ RESULTS ════════ */}
      <section className="max-w-5xl mx-auto px-6 pb-20">
        <FadeIn>
          <div className="flex items-center gap-3 mb-8">
            <div className="w-10 h-10 rounded-xl bg-[#2D7DD2]/10 flex items-center justify-center">
              <BarChart3 className="w-5 h-5 text-[#2D7DD2]" />
            </div>
            <h2 className="text-2xl font-semibold text-white">Results</h2>
          </div>
        </FadeIn>

        <FadeIn>
          <Card>
            <h3 className="text-white font-semibold mb-6">Per-Class IoU Comparison (%)</h3>
            <div className="space-y-3">
              {barData.map((d) => (
                <div key={d.cls} className="grid grid-cols-[100px_1fr] gap-3 items-center">
                  <span className="text-xs text-gray-500 text-right">{d.cls}</span>
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <div
                        className="h-4 rounded-sm bg-gray-700 relative"
                        style={{ width: `${d.base}%` }}
                      >
                        <span className="absolute right-1 top-0 text-[10px] text-gray-400 leading-4">
                          {d.base}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <div
                        className="h-4 rounded-sm bg-[#2D7DD2] relative"
                        style={{ width: `${d.ours}%` }}
                      >
                        <span className="absolute right-1 top-0 text-[10px] text-white leading-4">
                          {d.ours}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="flex items-center gap-6 mt-6 text-xs text-gray-500">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm bg-gray-700" /> Baseline (Phase 1)
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm bg-[#2D7DD2]" /> DA-SegFormer (Ours)
              </div>
            </div>
          </Card>
        </FadeIn>
      </section>

      {/* ════════ PAPER ════════ */}
      <section className="max-w-5xl mx-auto px-6 pb-20">
        <FadeIn>
          <div className="flex items-center gap-3 mb-8">
            <div className="w-10 h-10 rounded-xl bg-[#2D7DD2]/10 flex items-center justify-center">
              <FileText className="w-5 h-5 text-[#2D7DD2]" />
            </div>
            <h2 className="text-2xl font-semibold text-white">Paper</h2>
          </div>
        </FadeIn>

        <FadeIn>
          <Card className="text-center py-12">
            <FileText className="w-12 h-12 text-gray-600 mx-auto mb-4" />
            <h3 className="text-white font-semibold text-lg mb-2">
              DA-SegFormer: Damage-Aware Semantic Segmentation
            </h3>
            <p className="text-gray-500 text-sm mb-6 max-w-md mx-auto">
              Fine-Grained Disaster Assessment for Post-Disaster Imagery
            </p>
            <a
              href="/paper.pdf"
              target="_blank"
              className="inline-flex items-center gap-2 px-6 py-3 bg-[#2D7DD2] hover:bg-[#2568B5] text-white rounded-xl transition-colors text-sm font-medium"
            >
              View Paper <ExternalLink className="w-4 h-4" />
            </a>
            <p className="text-xs text-gray-600 mt-3">
              IGARSS 2026 Submission
            </p>
          </Card>
        </FadeIn>
      </section>

      {/* ════════ FOOTER ════════ */}
      <footer className="border-t border-[#1E2028] py-8">
        <div className="max-w-5xl mx-auto px-6 text-center text-sm text-gray-600">
          <p>DA-SegFormer &middot; Bina Labs &middot; Lehigh University</p>
          <p className="mt-1">Kevin Zhu, William Tang, Raphael Hay Tene, Zesheng Liu, Nhut Le, Dr. Maryam Rahnemoonfar</p>
        </div>
      </footer>
    </main>
  );
}
