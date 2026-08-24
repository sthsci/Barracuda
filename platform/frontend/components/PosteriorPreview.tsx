"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";
import type { Config, Data, Layout } from "plotly.js";

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => <div className="plotLoading" aria-label="Loading posterior plot" />,
});

const gaussian = (x: number, mean: number, sd: number) =>
  Math.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));

export function PosteriorPreview({ compact = false }: { compact?: boolean }) {
  const { data, layout, config } = useMemo(() => {
    const x = Array.from({ length: 121 }, (_, index) => index * 0.055);
    const series = [
      { name: "Control", mean: 2.8, sd: 0.52, color: "#486857" },
      { name: "Rituximab", mean: 3.9, sd: 0.58, color: "#D9A825" },
      { name: "Bispecific", mean: 4.8, sd: 0.43, color: "#B8512B" },
    ];
    const traces: Data[] = series.map((item) => ({
      type: "scatter",
      mode: "lines",
      name: item.name,
      x,
      y: x.map((value) => gaussian(value, item.mean, item.sd)),
      line: { color: item.color, width: compact ? 2 : 2.5, shape: "spline" },
      fill: "tozeroy",
      fillcolor: `${item.color}1F`,
      hovertemplate: `${item.name}<br>Mean event rate: %{x:.2f}<br>Posterior density: %{y:.3f}<extra></extra>`,
    }));
    const chartLayout: Partial<Layout> = {
      autosize: true,
      height: compact ? 285 : 370,
      margin: compact ? { l: 50, r: 18, t: 26, b: 54 } : { l: 64, r: 26, t: 34, b: 64 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "#F5F1E8",
      font: { family: "Avenir Next, Segoe UI, sans-serif", color: "#5B554A", size: compact ? 10 : 12 },
      hovermode: "x unified",
      showlegend: true,
      legend: { orientation: "h", x: 0, y: 1.12, font: { size: compact ? 9 : 11 } },
      xaxis: {
        title: { text: "Mean event rate, μλ" },
        gridcolor: "#D8D0C2",
        zeroline: false,
        fixedrange: false,
      },
      yaxis: {
        title: { text: compact ? "Density" : "Posterior density" },
        gridcolor: "#D8D0C2",
        zeroline: false,
        rangemode: "tozero",
      },
    };
    const chartConfig: Partial<Config> = {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
      toImageButtonOptions: { format: "png", filename: "barracuda_posterior", scale: 2 },
    };
    return { data: traces, layout: chartLayout, config: chartConfig };
  }, [compact]);

  return (
    <div className="plotFrame" role="img" aria-label="Example posterior distributions for three experimental conditions">
      <Plot
        data={data}
        layout={layout}
        config={config}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
