"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import {
  ArrowLeft,
  CloudRain,
  Building2,
  Grid3X3,
  Scan,
  Layers,
  Target,
  Zap,
  ChevronDown,
  ImageIcon,
} from "lucide-react";

// ── Reusable components ─────────────────────────────────────────────────────
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

// ── Interactive pixel grid demo ─────────────────────────────────────────────
const GRID_COLORS: Record<string, string> = {
  background: "#1a1a2e",
  water: "#0055ff",
  building_ok: "#14ff14",
  building_minor: "#ffd700",
  building_major: "#ff0000",
  tree: "#006400",
  road: "#808080",
};

const GRID_LABELS: Record<string, string> = {
  background: "Background",
  water: "Water",
  building_ok: "No Damage",
  building_minor: "Minor Damage",
  building_major: "Major Damage",
  tree: "Tree",
  road: "Road",
};

// 10x10 grid representing a simplified aerial view
const PIXEL_GRID = [
  ["tree","tree","tree","background","background","road","road","background","water","water"],
  ["tree","tree","background","building_ok","building_ok","road","road","background","water","water"],
  ["tree","background","building_ok","building_ok","building_ok","road","road","background","water","water"],
  ["background","background","building_ok","building_ok","background","road","road","background","background","water"],
  ["road","road","road","road","road","road","road","road","road","road"],
  ["background","background","building_minor","building_minor","background","background","building_major","building_major","background","background"],
  ["background","building_minor","building_minor","building_minor","background","building_major","building_major","building_major","background","tree"],
  ["background","building_minor","building_minor","background","background","building_major","building_major","background","tree","tree"],
  ["background","background","background","background","road","road","background","background","tree","tree"],
  ["tree","tree","background","background","road","road","background","tree","tree","tree"],
];

function PixelGridDemo() {
  const [hoveredClass, setHoveredClass] = useState<string | null>(null);
  const [showSegmentation, setShowSegmentation] = useState(false);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-gray-400">
          {showSegmentation
            ? "Each pixel is assigned a class label — this is semantic segmentation."
            : "An aerial image is just a grid of pixels. Hover over the grid to see the raw pixels."}
        </p>
        <button
          onClick={() => setShowSegmentation(!showSegmentation)}
          className="px-4 py-2 text-sm rounded-lg bg-[#2D7DD2] hover:bg-[#2568B5] text-white transition-colors"
        >
          {showSegmentation ? "Show Raw Image" : "Segment It"}
        </button>
      </div>

      {/* Grid */}
      <div className="flex justify-center">
        <div className="grid grid-cols-10 gap-0.5 p-3 bg-[#0A0B0F] rounded-xl border border-[#1E2028]">
          {PIXEL_GRID.flat().map((cls, i) => (
            <motion.div
              key={i}
              className="w-8 h-8 sm:w-10 sm:h-10 rounded-sm cursor-pointer transition-all"
              style={{
                backgroundColor: showSegmentation
                  ? GRID_COLORS[cls]
                  : hoveredClass === cls
                    ? GRID_COLORS[cls]
                    : "#2a2d35",
                opacity: hoveredClass && hoveredClass !== cls && !showSegmentation ? 0.3 : 1,
              }}
              whileHover={{ scale: 1.15 }}
              onMouseEnter={() => setHoveredClass(cls)}
              onMouseLeave={() => setHoveredClass(null)}
            />
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap justify-center gap-3">
        {Object.entries(GRID_LABELS).map(([key, label]) => (
          <button
            key={key}
            className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border transition-all ${
              hoveredClass === key
                ? "border-white/30 text-white"
                : "border-[#1E2028] text-gray-500 hover:text-gray-300"
            }`}
            onMouseEnter={() => setHoveredClass(key)}
            onMouseLeave={() => setHoveredClass(null)}
          >
            <div
              className="w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: GRID_COLORS[key] }}
            />
            {label}
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Sliding window patch overlay ────────────────────────────────────────────
function PatchOverlay() {
  // Simulated proportions: image is 4000w x 3000h, patch is 1024, stride is 768
  // As percentages: patch = 25.6% of width, stride = 19.2% of width
  // patch height = 34.13% of height, stride height = 25.6% of height
  const patchW = 25.6;
  const patchH = 34.13;
  const strideW = 19.2;
  const overlapW = patchW - strideW; // 6.4%

  return (
    <div className="relative w-full aspect-[4/3] rounded-xl overflow-hidden border border-[#1E2028]">
      <img
        src="/11028Major.jpg"
        alt="UAV image with patch overlay"
        className="w-full h-full object-cover"
      />
      {/* Patch 1 */}
      <div
        className="absolute border-2 border-[#2D7DD2]/70 rounded-sm"
        style={{ top: 0, left: 0, width: `${patchW}%`, height: `${patchH}%` }}
      />
      {/* Patch 2 (shifted by stride) */}
      <div
        className="absolute border-2 border-[#2D7DD2]/70 rounded-sm"
        style={{ top: 0, left: `${strideW}%`, width: `${patchW}%`, height: `${patchH}%` }}
      />
      {/* Overlap zone highlight */}
      <div
        className="absolute bg-[#2D7DD2]/15"
        style={{ top: 0, left: `${strideW}%`, width: `${overlapW}%`, height: `${patchH}%` }}
      />
      {/* Label */}
      <div
        className="absolute flex items-center justify-center"
        style={{ top: `${patchH / 2 - 3}%`, left: `${strideW}%`, width: `${overlapW}%`, height: "6%" }}
      >
        <span className="bg-black/70 backdrop-blur-sm text-[10px] text-[#A0D2FF] px-2 py-0.5 rounded whitespace-nowrap">
          25% Overlap
        </span>
      </div>
    </div>
  );
}

// ── Feature pyramid visualization ───────────────────────────────────────────
function FeaturePyramid() {
  const stages = [
    { label: "Stage 1", scale: "1/4", size: 256, detail: "Fine textures", example: "Missing shingles, tarp edges" },
    { label: "Stage 2", scale: "1/8", size: 128, detail: "Structural patterns", example: "Roof shapes, debris piles" },
    { label: "Stage 3", scale: "1/16", size: 64, detail: "Scene layout", example: "Building footprints, roads" },
    { label: "Stage 4", scale: "1/32", size: 32, detail: "Global context", example: "Neighborhood-level damage" },
  ];

  return (
    <div className="w-full rounded-xl border border-[#1E2028] bg-[#0A0B0F] p-6">
      {/* Arrow label */}
      <div className="flex items-center justify-between mb-6 px-2">
        <span className="text-[10px] text-[#2D7DD2] font-medium uppercase tracking-wider">High Texture</span>
        <div className="flex-1 mx-4 h-px bg-gradient-to-r from-[#2D7DD2]/60 to-[#2D7DD2]/20 relative">
          <div className="absolute right-0 top-1/2 -translate-y-1/2 w-0 h-0 border-l-4 border-l-[#2D7DD2]/40 border-y-[3px] border-y-transparent" />
        </div>
        <span className="text-[10px] text-[#2D7DD2]/60 font-medium uppercase tracking-wider">High Context</span>
      </div>

      {/* Stages */}
      <div className="flex items-end justify-between gap-4">
        {stages.map((s, i) => {
          // Decreasing sizes: 100%, 70%, 45%, 28%
          const sizes = [100, 70, 45, 28];
          const pct = sizes[i];
          // Progressive blur to simulate abstraction
          const blurs = [0, 2, 5, 10];

          return (
            <div key={i} className="flex-1 flex flex-col items-center gap-2">
              {/* Image square with blur */}
              <div
                className="relative rounded-lg overflow-hidden border border-[#2D7DD2]/30"
                style={{ width: `${pct}%`, aspectRatio: "1" }}
              >
                <img
                  src="/11028Major.jpg"
                  alt={`${s.label} features`}
                  className="w-full h-full object-cover"
                  style={{ filter: `blur(${blurs[i]}px) saturate(${1 - i * 0.15})` }}
                />
                {/* Colored overlay to simulate feature abstraction */}
                <div
                  className="absolute inset-0"
                  style={{ backgroundColor: `rgba(45, 125, 210, ${i * 0.1})` }}
                />
              </div>

              {/* Labels */}
              <div className="text-center mt-1">
                <p className="text-xs font-medium text-white">{s.label}</p>
                <p className="text-[10px] text-[#2D7DD2]">{s.scale} res</p>
                <p className="text-[10px] text-gray-500 mt-0.5">{s.detail}</p>
                <p className="text-[9px] text-gray-600 italic">{s.example}</p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Bottom caption */}
      <p className="text-[10px] text-gray-600 text-center mt-5">
        Each stage halves spatial resolution while doubling channel depth — capturing progressively more abstract features
      </p>
    </div>
  );
}

// ── Step-through explainer ──────────────────────────────────────────────────
function StepExplainer() {
  const [step, setStep] = useState(0);
  const steps = [
    {
      title: "1. Input Image",
      icon: ImageIcon,
      description:
        "A high-resolution UAV image (3000x4000 pixels) captured by a drone flying at 200ft above a disaster zone. Each pixel is just an RGB color value — the computer doesn't know what anything is yet.",
      detail:
        "The RescueNet dataset contains ~2,000 of these images captured after Hurricane Michael using DJI Mavic Pro quadcopters.",
      image: "/11028Major.jpg",
    },
    {
      title: "2. Divide into Patches",
      icon: Grid3X3,
      description:
        "The image is too large to process at once (3000x4000 = 12 million pixels). We divide it into overlapping 1024x1024 patches with a stride of 768, giving 25% overlap between adjacent tiles.",
      detail:
        "This overlap is crucial — it prevents visible grid artifacts at tile boundaries. Predictions in overlapping regions are averaged together.",
      component: "patches",
    },
    {
      title: "3. Encode Features",
      icon: Layers,
      description:
        "Each patch is fed through the MiT-B4 encoder — a hierarchical transformer that extracts features at 4 different scales (1/4, 1/8, 1/16, 1/32 resolution). Lower stages capture fine details like roof textures; higher stages understand scene context.",
      detail:
        "The encoder uses Overlapped Patch Merging (kernel=7, stride=4, padding=3) to preserve building edges and road boundaries that non-overlapping approaches would lose.",
      component: "pyramid",
    },
    {
      title: "4. Decode & Classify",
      icon: Scan,
      description:
        "The lightweight All-MLP decoder takes features from all 4 encoder stages, unifies their channel dimensions, upsamples them to 1/4 resolution, and concatenates them. A final prediction layer assigns one of 10 classes to every pixel.",
      detail:
        "This is where the model decides: is this pixel a damaged building? A road? Water? Each pixel gets a probability distribution across all 10 classes.",
      placeholder: "IMAGE: Show the decoder fusing multi-scale features into a segmentation map",
    },
    {
      title: "5. Stitch & Output",
      icon: Target,
      description:
        "All patch predictions are stitched back together using smooth averaging in the overlap zones. The result is a full-resolution segmentation mask where every pixel is color-coded by class.",
      detail:
        "The final output enables responders to see exactly which buildings are damaged and to what degree — information that would take days to gather manually.",
      placeholder: "IMAGE: Show the final stitched segmentation mask overlaid on the original",
    },
  ];

  const current = steps[step];
  const Icon = current.icon;

  return (
    <Card>
      {/* Step tabs */}
      <div className="flex gap-1 mb-6 overflow-x-auto pb-2">
        {steps.map((s, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-colors ${
              i === step
                ? "bg-[#2D7DD2] text-white"
                : "bg-[#1E2028] text-gray-500 hover:text-gray-300"
            }`}
          >
            {s.title}
          </button>
        ))}
      </div>

      <motion.div
        key={step}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.3 }}
      >
        <div className="flex items-center gap-3 mb-3">
          <div className="w-9 h-9 rounded-lg bg-[#2D7DD2]/10 flex items-center justify-center">
            <Icon className="w-5 h-5 text-[#2D7DD2]" />
          </div>
          <h3 className="text-lg font-semibold text-white">{current.title}</h3>
        </div>

        <p className="text-gray-300 leading-relaxed mb-3">{current.description}</p>
        <p className="text-sm text-gray-500 leading-relaxed mb-4">{current.detail}</p>

        {/* Image, component, or placeholder */}
        {"component" in current && current.component === "patches" ? (
          <PatchOverlay />
        ) : "component" in current && current.component === "pyramid" ? (
          <FeaturePyramid />
        ) : "image" in current && current.image ? (
          <div className="w-full aspect-[16/9] rounded-xl overflow-hidden border border-[#1E2028]">
            <img src={current.image} alt={current.title} className="w-full h-full object-cover" />
          </div>
        ) : (
          <div className="w-full aspect-[16/9] bg-[#0A0B0F] border border-dashed border-[#2a2d35] rounded-xl flex items-center justify-center">
            <p className="text-xs text-gray-600 text-center px-8">{"placeholder" in current ? current.placeholder : ""}</p>
          </div>
        )}
      </motion.div>

      {/* Nav arrows */}
      <div className="flex justify-between mt-6">
        <button
          onClick={() => setStep(Math.max(0, step - 1))}
          disabled={step === 0}
          className="px-4 py-2 text-sm rounded-lg border border-[#1E2028] text-gray-400 hover:text-white disabled:opacity-30 transition-colors"
        >
          Previous
        </button>
        <button
          onClick={() => setStep(Math.min(steps.length - 1, step + 1))}
          disabled={step === steps.length - 1}
          className="px-4 py-2 text-sm rounded-lg bg-[#2D7DD2] hover:bg-[#2568B5] text-white disabled:opacity-30 transition-colors"
        >
          Next
        </button>
      </div>
    </Card>
  );
}

// ── Expandable section ──────────────────────────────────────────────────────
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

// ═════════════════════════════════════════════════════════════════════════════
// LEARN PAGE
// ═════════════════════════════════════════════════════════════════════════════
export default function LearnPage() {
  return (
    <main className="min-h-screen">
      {/* Nav */}
      <div className="max-w-5xl mx-auto px-6 pt-6">
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-sm text-gray-500 hover:text-white transition-colors"
        >
          <ArrowLeft className="w-4 h-4" /> Back to Demo
        </Link>
      </div>

      {/* ════════ HERO ════════ */}
      <section className="max-w-5xl mx-auto px-6 pt-12 pb-16 text-center">
        <FadeIn>
          <p className="text-xs font-semibold tracking-[0.25em] uppercase text-[#4E3629] mb-4">
            Bina Labs &middot; Lehigh University
          </p>
        </FadeIn>
        <FadeIn delay={0.1}>
          <h1 className="text-4xl sm:text-5xl font-bold bg-gradient-to-r from-[#2D7DD2] to-[#A0D2FF] bg-clip-text text-transparent leading-tight">
            Understanding Disaster Segmentation
          </h1>
        </FadeIn>
        <FadeIn delay={0.2}>
          <p className="mt-4 text-lg text-gray-400 max-w-2xl mx-auto leading-relaxed">
            Why automated damage assessment matters, what semantic segmentation is,
            and how DA-SegFormer uses it to save lives.
          </p>
        </FadeIn>
      </section>

      {/* ════════ THE PROBLEM ════════ */}
      <section className="max-w-5xl mx-auto px-6 pb-20">
        <FadeIn>
          <div className="flex items-center gap-3 mb-8">
            <div className="w-10 h-10 rounded-xl bg-[#2D7DD2]/10 flex items-center justify-center">
              <CloudRain className="w-5 h-5 text-[#2D7DD2]" />
            </div>
            <h2 className="text-2xl font-semibold text-white">The Problem</h2>
          </div>
        </FadeIn>

        <FadeIn>
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <h3 className="text-white font-semibold mb-3">The Scale of Disaster</h3>
              <p className="text-gray-400 text-sm leading-relaxed mb-4">
                In 2025 alone, the United States experienced <strong className="text-white">23 natural disasters</strong> costing
                approximately <strong className="text-white">$115 billion</strong>. Hurricanes, floods, and earthquakes
                devastate communities, and the frequency of these events continues to rise.
              </p>
              <p className="text-gray-400 text-sm leading-relaxed">
                After a disaster strikes, the critical first step is <strong className="text-gray-200">rapid damage assessment</strong> —
                understanding which buildings are destroyed, which roads are blocked, and where
                floodwater has spread. This information determines how rescue teams allocate
                resources and which areas receive help first.
              </p>
            </Card>

            <Card>
              <h3 className="text-white font-semibold mb-3">Why Manual Assessment Fails</h3>
              <p className="text-gray-400 text-sm leading-relaxed mb-4">
                Traditional damage assessment relies on field teams physically inspecting
                structures and filing reports. This process is:
              </p>
              <ul className="text-gray-400 text-sm space-y-2 ml-1">
                <li className="flex items-start gap-2">
                  <span className="text-[#2D7DD2] mt-1">&#9679;</span>
                  <span><strong className="text-gray-200">Slow</strong> — taking days or weeks to cover a disaster zone</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#2D7DD2] mt-1">&#9679;</span>
                  <span><strong className="text-gray-200">Dangerous</strong> — inspectors enter unstable, flooded, or debris-filled areas</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#2D7DD2] mt-1">&#9679;</span>
                  <span><strong className="text-gray-200">Incomplete</strong> — some areas are simply inaccessible after major storms</span>
                </li>
              </ul>
              <p className="text-gray-400 text-sm leading-relaxed mt-4">
                UAV (drone) imagery offers a solution — drones can survey vast areas quickly
                and safely. But someone still needs to analyze thousands of high-resolution images.
                <strong className="text-gray-200"> That&apos;s where AI comes in.</strong>
              </p>
            </Card>
          </div>
        </FadeIn>

        {/* Image placeholder */}
        <FadeIn delay={0.1}>
          <div className="mt-6 w-full rounded-xl overflow-hidden border border-[#1E2028]">
            <img
              src="/10781.jpg"
              alt="RescueNet sample: original aerial image, segmentation mask, and colored overlay showing damage classification"
              className="w-full h-auto"
            />
          </div>
          <p className="text-xs text-gray-600 mt-2 text-center">
            RescueNet sample — original UAV image, ground-truth mask, model prediction, and detail crops
          </p>
        </FadeIn>
      </section>

      {/* ════════ THE RESCUENET DATASET ════════ */}
      <section className="max-w-5xl mx-auto px-6 pb-20">
        <FadeIn>
          <div className="flex items-center gap-3 mb-8">
            <div className="w-10 h-10 rounded-xl bg-[#2D7DD2]/10 flex items-center justify-center">
              <Building2 className="w-5 h-5 text-[#2D7DD2]" />
            </div>
            <h2 className="text-2xl font-semibold text-white">The RescueNet Dataset</h2>
          </div>
        </FadeIn>

        <FadeIn>
          <Card>
            <p className="text-gray-400 text-sm leading-relaxed mb-4">
              Our model is trained on <strong className="text-white">RescueNet</strong>, a benchmark dataset created
              by Dr. Rahnemoonfar&apos;s team. It contains <strong className="text-white">1,973 high-resolution UAV images
              (3000x4000 pixels)</strong> captured after Hurricane Michael using DJI Mavic Pro quadcopters
              at 200 feet above ground level.
            </p>
            <p className="text-gray-400 text-sm leading-relaxed mb-6">
              Every pixel in every image has been manually labeled with one of <strong className="text-white">10 semantic classes</strong>.
              Buildings are annotated at four damage levels, allowing the model to learn the subtle
              visual differences between each level of destruction.
            </p>

            {/* Class distribution */}
            <h4 className="text-white font-medium text-sm mb-3">Class Distribution (the imbalance problem)</h4>
            <div className="space-y-2">
              {[
                { cls: "Background", pct: 52.51 },
                { cls: "Tree", pct: 22.05 },
                { cls: "Water", pct: 8.13 },
                { cls: "Minor Damage", pct: 2.63 },
                { cls: "Major Damage", pct: 1.68 },
                { cls: "Road-Blocked", pct: 1.59 },
                { cls: "Total Destruction", pct: 1.44 },
                { cls: "Pool", pct: 0.06 },
              ].map((d) => (
                <div key={d.cls} className="grid grid-cols-[130px_1fr_50px] gap-2 items-center">
                  <span className="text-xs text-gray-500 text-right">{d.cls}</span>
                  <div className="h-3 rounded-sm bg-[#1E2028] overflow-hidden">
                    <div
                      className="h-full rounded-sm bg-[#2D7DD2]"
                      style={{ width: `${Math.max(d.pct, 0.5)}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-500">{d.pct}%</span>
                </div>
              ))}
            </div>
            <p className="text-xs text-gray-600 mt-3">
              Damage classes make up less than 8% of total pixels combined.
              Standard training causes models to ignore these rare but critical classes.
            </p>
          </Card>
        </FadeIn>

        <FadeIn delay={0.1}>
          <div className="mt-6 grid md:grid-cols-4 gap-4">
            <Card className="text-center">
              <p className="text-2xl font-bold text-[#2D7DD2]">1,973</p>
              <p className="text-xs text-gray-500 mt-1">UAV images</p>
            </Card>
            <Card className="text-center">
              <p className="text-2xl font-bold text-[#2D7DD2]">3000x4000</p>
              <p className="text-xs text-gray-500 mt-1">pixels per image</p>
            </Card>
            <Card className="text-center">
              <p className="text-2xl font-bold text-[#2D7DD2]">10</p>
              <p className="text-xs text-gray-500 mt-1">semantic classes</p>
            </Card>
            <Card className="text-center">
              <p className="text-2xl font-bold text-[#2D7DD2]">4</p>
              <p className="text-xs text-gray-500 mt-1">building damage levels</p>
            </Card>
          </div>
        </FadeIn>

        {/* Damage levels */}
        <FadeIn delay={0.2}>
          <div className="mt-6">
            <Card>
              <h4 className="text-white font-medium text-sm mb-4">Building Damage Levels</h4>
              <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  {
                    level: "No Damage",
                    color: "#14ff14",
                    desc: "Building unharmed, no visible structural issues.",
                    image: "/12023Minor.jpg",
                  },
                  {
                    level: "Minor Damage",
                    color: "#ffd700",
                    desc: "Parts damaged but coverable with blue tarp. Roof partially affected.",
                    image: "/10955Medium.jpg",
                  },
                  {
                    level: "Major Damage",
                    color: "#ff0000",
                    desc: "Significant structural damage requiring extensive repairs.",
                    image: "/11683Major.jpg",
                  },
                  {
                    level: "Total Destruction",
                    color: "#8b0000",
                    desc: "Complete failure of two or more major structural components.",
                    image: "/10807Major.jpg",
                  },
                ].map((d) => (
                  <div key={d.level} className="space-y-2">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: d.color }}
                      />
                      <span className="text-sm font-medium text-white">{d.level}</span>
                    </div>
                    {d.image ? (
                      <div className="w-full aspect-square rounded-lg overflow-hidden border border-[#1E2028]">
                        <img src={d.image} alt={d.level} className="w-full h-full object-cover" />
                      </div>
                    ) : (
                      <div className="w-full aspect-square bg-[#0A0B0F] border border-dashed border-[#2a2d35] rounded-lg flex items-center justify-center p-2">
                        <p className="text-[10px] text-gray-600 text-center">Needs image</p>
                      </div>
                    )}
                    <p className="text-xs text-gray-500 leading-relaxed">{d.desc}</p>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </FadeIn>
      </section>

      {/* ════════ WHAT IS SEMANTIC SEGMENTATION ════════ */}
      <section className="max-w-5xl mx-auto px-6 pb-20">
        <FadeIn>
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-xl bg-[#2D7DD2]/10 flex items-center justify-center">
              <Grid3X3 className="w-5 h-5 text-[#2D7DD2]" />
            </div>
            <h2 className="text-2xl font-semibold text-white">What is Semantic Segmentation?</h2>
          </div>
          <p className="text-gray-400 text-sm mb-8 ml-[52px]">
            Interact with the grid below to build an intuition for how it works.
          </p>
        </FadeIn>

        <FadeIn delay={0.1}>
          <Card>
            <PixelGridDemo />
          </Card>
        </FadeIn>

        <FadeIn delay={0.2}>
          <div className="mt-6 grid md:grid-cols-3 gap-4">
            <Card>
              <h4 className="text-white font-medium text-sm mb-2">Image Classification</h4>
              <p className="text-xs text-gray-500 leading-relaxed">
                &ldquo;This image contains a damaged building.&rdquo; One label for the entire image.
                Useful for sorting, but doesn&apos;t tell you <em>where</em> the damage is.
              </p>
            </Card>
            <Card>
              <h4 className="text-white font-medium text-sm mb-2">Object Detection</h4>
              <p className="text-xs text-gray-500 leading-relaxed">
                &ldquo;There&apos;s a damaged building at coordinates (x, y).&rdquo; Draws bounding boxes.
                Better, but boxes are coarse — they don&apos;t follow building boundaries.
              </p>
            </Card>
            <Card className="border-[#2D7DD2]/30">
              <h4 className="text-[#2D7DD2] font-medium text-sm mb-2">
                Semantic Segmentation &#10003;
              </h4>
              <p className="text-xs text-gray-400 leading-relaxed">
                &ldquo;This <em>exact pixel</em> is Major Damage. This pixel is Road. This pixel is Water.&rdquo;
                Every pixel gets a class label — the most precise understanding possible.
              </p>
            </Card>
          </div>
        </FadeIn>
      </section>

      {/* ════════ HOW DA-SEGFORMER WORKS (STEP-THROUGH) ════════ */}
      <section className="max-w-5xl mx-auto px-6 pb-20">
        <FadeIn>
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-xl bg-[#2D7DD2]/10 flex items-center justify-center">
              <Scan className="w-5 h-5 text-[#2D7DD2]" />
            </div>
            <h2 className="text-2xl font-semibold text-white">How DA-SegFormer Works</h2>
          </div>
          <p className="text-gray-400 text-sm mb-8 ml-[52px]">
            Step through the pipeline from raw image to segmentation mask.
          </p>
        </FadeIn>

        <FadeIn delay={0.1}>
          <StepExplainer />
        </FadeIn>
      </section>

      {/* ════════ KEY INNOVATIONS (EXPANDABLE) ════════ */}
      <section className="max-w-5xl mx-auto px-6 pb-20">
        <FadeIn>
          <div className="flex items-center gap-3 mb-8">
            <div className="w-10 h-10 rounded-xl bg-[#2D7DD2]/10 flex items-center justify-center">
              <Zap className="w-5 h-5 text-[#2D7DD2]" />
            </div>
            <h2 className="text-2xl font-semibold text-white">Our Key Innovations</h2>
          </div>
        </FadeIn>

        <div className="space-y-4">
          <FadeIn>
            <Expandable icon={Target} title="Online Hard Example Mining (OHEM)" defaultOpen>
              <p>
                The biggest challenge in disaster imagery: <strong className="text-gray-200">95% of pixels are background or trees</strong>.
                A model can score high accuracy by simply predicting &ldquo;background&rdquo; everywhere and ignoring
                the rare damage pixels that actually matter.
              </p>
              <p>
                OHEM solves this by computing the loss for <em>every</em> pixel, ranking them by difficulty,
                and <strong className="text-gray-200">only backpropagating through the top k=100,000 hardest pixels</strong>.
                This forces the model to focus on the decision boundaries it&apos;s getting wrong —
                typically the edges between Minor and Major damage.
              </p>
              <div className="mt-3 w-full aspect-[16/7] bg-[#0A0B0F] border border-dashed border-[#2a2d35] rounded-xl flex items-center justify-center">
                <p className="text-xs text-gray-600 text-center px-8">
                  IMAGE: Visualization showing pixel-level loss heatmap — bright where the model struggles
                  (damage boundaries), dark where it&apos;s confident (background)
                </p>
              </div>
            </Expandable>
          </FadeIn>

          <FadeIn>
            <Expandable icon={Target} title="Dice Loss for Class Imbalance">
              <p>
                Standard Cross-Entropy treats each pixel equally. With 52.51% of pixels being
                background, the model minimizes loss by getting background right and ignoring
                Minor Damage (2.63%) and Major Damage (1.68%).
              </p>
              <p>
                <strong className="text-gray-200">Dice Loss</strong> computes overlap per class,
                then averages across classes — giving Minor Damage the same weight as Background.
                Combined with OHEM:
              </p>
              <p className="mt-1">
                <code className="bg-white/5 px-3 py-1.5 rounded text-sm text-gray-300">
                  L_total = L_Dice + L_OHEM
                </code>
              </p>
            </Expandable>
          </FadeIn>

          <FadeIn>
            <Expandable icon={Layers} title="Class-Aware Sampling">
              <p>
                During training, random crops from a 3000x4000 image rarely contain damage pixels.
                Most crops are just trees and background — the model never learns damage patterns.
              </p>
              <p>
                Our <strong className="text-gray-200">Class-Aware Sampling</strong> enforces a 50% chance
                that each training crop is centered on a pixel from an underrepresented damage class
                (Minor, Major, or Total Destruction). This guarantees consistent exposure to rare features.
              </p>
            </Expandable>
          </FadeIn>

          <FadeIn>
            <Expandable icon={Zap} title="Resolution-Preserving Inference">
              <p>
                Previous approaches resize 3000x4000 images down to 512x512 for inference —
                a <strong className="text-gray-200">6x reduction</strong> that destroys the subtle texture
                differences between Minor and Major damage (missing shingles, debris patterns, tarp coverage).
              </p>
              <p>
                DA-SegFormer processes images at <strong className="text-gray-200">native resolution</strong> using
                a sliding window (1024x1024 patches, stride 768). The result: +13.9% improvement on
                Minor Damage and +12.5% on Major Damage.
              </p>
            </Expandable>
          </FadeIn>
        </div>
      </section>

      {/* ════════ CTA ════════ */}
      <section className="max-w-5xl mx-auto px-6 pb-20">
        <FadeIn>
          <Card className="text-center py-12">
            <h3 className="text-white font-semibold text-lg mb-2">Try It Yourself</h3>
            <p className="text-gray-500 text-sm mb-6 max-w-md mx-auto">
              Upload a satellite or drone image and see DA-SegFormer classify every pixel in real time.
            </p>
            <Link
              href="/"
              className="inline-flex items-center gap-2 px-6 py-3 bg-[#2D7DD2] hover:bg-[#2568B5] text-white rounded-xl transition-colors text-sm font-medium"
            >
              Go to Interactive Demo
            </Link>
          </Card>
        </FadeIn>
      </section>

      {/* ════════ FOOTER ════════ */}
      <footer className="border-t border-[#1E2028] py-8">
        <div className="max-w-5xl mx-auto px-6 text-center text-sm text-gray-600">
          <p>DA-SegFormer &middot; Bina Labs &middot; Lehigh University</p>
          <p className="mt-1">
            Kevin Zhu, William Tang, Raphael Hay Tene, Zesheng Liu, Nhut Le, Dr. Maryam Rahnemoonfar
          </p>
        </div>
      </footer>
    </main>
  );
}
