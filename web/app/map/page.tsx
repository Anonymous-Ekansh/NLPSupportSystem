"use client";

import React, { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import { useApp } from "@/lib/store";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { ArrowLeft, Star, Filter } from "lucide-react";

interface MapCoord {
  index: number;
  x: number;
  y: number;
  intent: string;
  category: string;
}

interface MapData {
  n_total: number;
  n_saved: number;
  method?: string;
  coordinates: MapCoord[];
}

const CATEGORY_COLORS: Record<string, string> = {
  ACCOUNT: "#3b82f6", // Blue
  CANCEL: "#ef4444", // Red
  DELIVERY: "#10b981", // Emerald
  FEEDBACK: "#f59e0b", // Amber
  INVOICE: "#8b5cf6", // Purple
  ORDER: "#06b6d4", // Cyan
  PAYMENT: "#ec4899", // Pink
  REFUND: "#14b8a6", // Teal
  SUBSCRIPTION: "#f97316", // Orange
  CONTACT: "#6366f1", // Indigo
};

const DEFAULT_COLOR = "#94a3b8";

export default function MapPage() {
  const { currentRun, meta } = useApp();
  const [mapData, setMapData] = useState<MapData | null>(null);
  const [selectedCategories, setSelectedCategories] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);

  const threshold = meta?.a1?.tau ?? 0.65;

  // Load map_points.json from /map_points.json
  useEffect(() => {
    fetch("/map_points.json")
      .then((res) => res.json())
      .then((data: MapData) => {
        setMapData(data);
        // Default: all categories selected
        const cats = new Set(data.coordinates.map((c) => c.category));
        setSelectedCategories(cats);
      })
      .catch((err) => console.error("Failed to load map points:", err))
      .finally(() => setLoading(false));
  }, []);

  const allCategories = useMemo(() => {
    if (!mapData) return [];
    return Array.from(new Set(mapData.coordinates.map((c) => c.category))).sort();
  }, [mapData]);

  const toggleCategory = (cat: string) => {
    setSelectedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) {
        next.delete(cat);
      } else {
        next.add(cat);
      }
      return next;
    });
  };

  // Compute 10 nearest plotted neighbours to currentRun query
  const queryPoint = useMemo(() => {
    return currentRun?.map?.x != null && currentRun?.map?.y != null
      ? {
          x: currentRun.map.x,
          y: currentRun.map.y,
          confidence: currentRun.map.confidence,
          query: currentRun.query,
        }
      : null;
  }, [currentRun]);

  const nearestNeighbourIndices = useMemo(() => {
    if (!queryPoint || !mapData) return new Set<number>();

    const qx = queryPoint.x;
    const qy = queryPoint.y;

    // Euclidean distance
    const dists = mapData.coordinates.map((c) => ({
      index: c.index,
      dist: Math.hypot(c.x - qx, c.y - qy),
    }));

    dists.sort((a, b) => a.dist - b.dist);
    const top10 = dists.slice(0, 10).map((d) => d.index);
    return new Set(top10);
  }, [queryPoint, mapData]);

  // Filter coordinates based on category selection
  const filteredPoints = useMemo(() => {
    if (!mapData) return [];
    return mapData.coordinates.filter((c) => selectedCategories.has(c.category));
  }, [mapData, selectedCategories]);

  const isLowConfidence = queryPoint != null && queryPoint.confidence < threshold;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-4">
      {/* Header Row */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-[var(--card-bg)] border border-[var(--card-border)] rounded-xl p-4">
        <div>
          <div className="flex items-center gap-2">
            <Link
              href="/"
              className="p-1 rounded-lg hover:bg-[var(--card-hover)] text-[var(--muted-text)] hover:text-[var(--foreground)] transition-colors"
              title="Back to Resolve"
            >
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <h1 className="text-2xl font-bold tracking-tight">2D Semantic Embedding Map</h1>
          </div>
          <p className="text-xs text-[var(--muted-text)] mt-1">
            UMAP projection of intent embeddings with interactive category filtering
          </p>
        </div>

        {/* Caption with real numbers */}
        {mapData && (
          <div className="text-xs text-gray-300 font-mono bg-gray-800/80 px-3 py-1.5 rounded-lg border border-gray-700">
            Showing <strong className="text-teal-400">{mapData.n_saved}</strong> of{" "}
            <strong>{mapData.n_total}</strong> points ({mapData.method || "stratified UMAP"})
          </div>
        )}
      </div>

      {/* Main Canvas & Sidebar */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left: Interactive Category Filters & Query Info */}
        <div className="lg:col-span-1 space-y-4">
          {/* Query Projection Info Card */}
          <div className="bg-[var(--card-bg)] border border-[var(--card-border)] rounded-xl p-4 space-y-2">
            <h3 className="text-sm font-bold uppercase tracking-wider text-teal-400 flex items-center gap-1.5">
              <Star className="w-4 h-4 fill-amber-400 text-amber-400" />
              Active Query Projection
            </h3>

            {queryPoint ? (
              <div className="text-xs space-y-2">
                <p className="text-gray-300 font-medium line-clamp-2">
                  &ldquo;{queryPoint.query}&rdquo;
                </p>
                <div className="grid grid-cols-2 gap-1 font-mono text-[11px] bg-gray-900/60 p-2 rounded border border-gray-800">
                  <div>X: {queryPoint.x.toFixed(4)}</div>
                  <div>Y: {queryPoint.y.toFixed(4)}</div>
                  <div className="col-span-2">
                    Confidence: {(queryPoint.confidence * 100).toFixed(1)}%
                  </div>
                </div>

                {isLowConfidence ? (
                  <div className="p-2 rounded bg-amber-950/40 border border-amber-600/50 text-amber-300 text-xs">
                    ⭐ Hollow star: far from every known cluster (confidence &lt;{" "}
                    {Math.round(threshold * 100)}%)
                  </div>
                ) : (
                  <div className="p-2 rounded bg-emerald-950/40 border border-emerald-600/50 text-emerald-300 text-xs">
                    ★ Plotted near dense cluster (confidence &ge;{" "}
                    {Math.round(threshold * 100)}%)
                  </div>
                )}

                <p className="text-[11px] text-[var(--muted-text)] italic">
                  10 nearest plotted neighbours highlighted by Euclidean distance.
                </p>
              </div>
            ) : (
              <div className="text-xs text-[var(--muted-text)] space-y-2 py-2">
                <p>No resolution run yet.</p>
                <Link
                  href="/"
                  className="inline-block px-3 py-1.5 rounded-lg btn-teal text-xs font-semibold"
                >
                  Run a query in Resolve →
                </Link>
              </div>
            )}
          </div>

          {/* Category Click-to-filter Legend */}
          <div className="bg-[var(--card-bg)] border border-[var(--card-border)] rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-gray-400 flex items-center gap-1">
                <Filter className="w-3.5 h-3.5" /> Filter Categories
              </h3>
              <button
                onClick={() => {
                  if (selectedCategories.size === allCategories.length) {
                    setSelectedCategories(new Set());
                  } else {
                    setSelectedCategories(new Set(allCategories));
                  }
                }}
                className="text-[11px] text-teal-400 hover:underline cursor-pointer"
              >
                {selectedCategories.size === allCategories.length ? "Deselect all" : "Select all"}
              </button>
            </div>

            <div className="space-y-1.5 max-h-[360px] overflow-y-auto pr-1">
              {allCategories.map((cat) => {
                const isSelected = selectedCategories.has(cat);
                const color = CATEGORY_COLORS[cat] || DEFAULT_COLOR;

                return (
                  <button
                    key={cat}
                    onClick={() => toggleCategory(cat)}
                    className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-all cursor-pointer ${
                      isSelected
                        ? "bg-gray-800/80 border-gray-700 text-white"
                        : "bg-gray-950/40 border-gray-900 text-gray-500 opacity-50"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: color }}
                      />
                      <span>{cat}</span>
                    </div>
                    <span className="font-mono text-[10px] text-gray-400">
                      {mapData?.coordinates.filter((c) => c.category === cat).length}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right: ScatterChart Visualization */}
        <div className="lg:col-span-3 bg-[var(--card-bg)] border border-[var(--card-border)] rounded-xl p-4 flex flex-col h-[650px] shadow-sm">
          {loading ? (
            <div className="flex-1 flex items-center justify-center text-sm text-[var(--muted-text)]">
              Loading 3,000 UMAP coordinates...
            </div>
          ) : (
            <div className="flex-1 w-full h-full relative">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <XAxis
                    type="number"
                    dataKey="x"
                    name="UMAP 1"
                    stroke="#475569"
                    fontSize={11}
                    tickLine={false}
                    axisLine={{ stroke: "#334155" }}
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    name="UMAP 2"
                    stroke="#475569"
                    fontSize={11}
                    tickLine={false}
                    axisLine={{ stroke: "#334155" }}
                  />
                  <ZAxis range={[25, 25]} />
                  <Tooltip
                    cursor={{ strokeDasharray: "3 3" }}
                    content={({ payload }) => {
                      if (!payload || payload.length === 0) return null;
                      const pt = payload[0].payload as MapCoord;
                      return (
                        <div className="p-2.5 rounded-lg bg-gray-900 border border-gray-700 text-xs font-mono shadow-lg text-white space-y-1">
                          <div className="font-bold text-teal-300">{pt.intent}</div>
                          <div className="text-indigo-300 font-semibold">{pt.category}</div>
                          <div className="text-gray-400 text-[10px]">
                            ({pt.x.toFixed(2)}, {pt.y.toFixed(2)})
                          </div>
                        </div>
                      );
                    }}
                  />

                  {/* Scatter layer for dataset points */}
                  <Scatter name="Training points" data={filteredPoints}>
                    {filteredPoints.map((entry, index) => {
                      const isNeighbour = nearestNeighbourIndices.has(entry.index);
                      const baseColor = CATEGORY_COLORS[entry.category] || DEFAULT_COLOR;

                      return (
                        <Cell
                          key={`cell-${index}`}
                          fill={isNeighbour ? "#fbbf24" : baseColor}
                          stroke={isNeighbour ? "#ffffff" : "transparent"}
                          strokeWidth={isNeighbour ? 2 : 0}
                          opacity={isNeighbour ? 1.0 : 0.65}
                        />
                      );
                    })}
                  </Scatter>

                  {/* Scatter layer for Active Query Star */}
                  {queryPoint && (
                    <Scatter
                      name="Active Query"
                      data={[{ x: queryPoint.x, y: queryPoint.y, query: queryPoint.query }]}
                      shape={
                        isLowConfidence
                          ? ({ cx = 0, cy = 0 }: { cx?: number; cy?: number }) => (
                              <svg x={cx - 10} y={cy - 10} width={20} height={20} viewBox="0 0 24 24">
                                <polygon
                                  points="12,2 15,9 22,9 17,14 19,21 12,17 5,21 7,14 2,9 9,9"
                                  fill="none"
                                  stroke="#f59e0b"
                                  strokeWidth="2.5"
                                />
                              </svg>
                            )
                          : ({ cx = 0, cy = 0 }: { cx?: number; cy?: number }) => (
                              <svg x={cx - 12} y={cy - 12} width={24} height={24} viewBox="0 0 24 24">
                                <polygon
                                  points="12,2 15,9 22,9 17,14 19,21 12,17 5,21 7,14 2,9 9,9"
                                  fill="#14b8a6"
                                  stroke="#ffffff"
                                  strokeWidth="2"
                                />
                              </svg>
                            )
                      }
                    />
                  )}
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
