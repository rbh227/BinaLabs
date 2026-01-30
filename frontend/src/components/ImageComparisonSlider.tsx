"use client";

import { useState, useRef, useCallback } from "react";

interface Props {
  beforeSrc: string;
  afterSrc: string;
  beforeLabel?: string;
  afterLabel?: string;
}

export default function ImageComparisonSlider({
  beforeSrc,
  afterSrc,
  beforeLabel = "Original",
  afterLabel = "Segmentation",
}: Props) {
  const [position, setPosition] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const updatePosition = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    setPosition((x / rect.width) * 100);
  }, []);

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      dragging.current = true;
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
      updatePosition(e.clientX);
    },
    [updatePosition]
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragging.current) return;
      updatePosition(e.clientX);
    },
    [updatePosition]
  );

  const handlePointerUp = useCallback(() => {
    dragging.current = false;
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative w-full aspect-[4/3] max-w-[900px] mx-auto rounded-xl overflow-hidden cursor-col-resize select-none border border-[#1E2028]"
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    >
      {/* After image (full) */}
      <img
        src={afterSrc}
        alt={afterLabel}
        className="absolute inset-0 w-full h-full object-contain bg-black"
        draggable={false}
      />

      {/* Before image (clipped) */}
      <div
        className="absolute inset-0 overflow-hidden"
        style={{ width: `${position}%` }}
      >
        <img
          src={beforeSrc}
          alt={beforeLabel}
          className="absolute inset-0 w-full h-full object-contain bg-black"
          style={{ width: `${containerRef.current?.offsetWidth ?? 900}px` }}
          draggable={false}
        />
      </div>

      {/* Slider line */}
      <div
        className="absolute top-0 bottom-0 w-0.5 bg-white/80 z-10"
        style={{ left: `${position}%` }}
      >
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-10 h-10 rounded-full bg-white/90 border-2 border-white shadow-lg flex items-center justify-center">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M7 4L3 10L7 16" stroke="#0A0B0F" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M13 4L17 10L13 16" stroke="#0A0B0F" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      </div>

      {/* Labels */}
      <span className="absolute top-4 left-4 bg-black/60 backdrop-blur-sm text-white text-xs px-3 py-1.5 rounded-full z-20">
        {beforeLabel}
      </span>
      <span className="absolute top-4 right-4 bg-black/60 backdrop-blur-sm text-white text-xs px-3 py-1.5 rounded-full z-20">
        {afterLabel}
      </span>
    </div>
  );
}
