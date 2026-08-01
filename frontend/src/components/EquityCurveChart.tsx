import { useMemo, useRef, useState } from "react";
import type { EquityPoint } from "../types";

const WIDTH = 640;
const HEIGHT = 220;
const PAD = { top: 12, right: 12, bottom: 24, left: 56 };

export default function EquityCurveChart({ points, startingBankroll }: {
  points: EquityPoint[];
  startingBankroll: number;
}) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const { linePath, xOf, yOf, minY, maxY } = useMemo(() => {
    const values = points.map((p) => p.bankroll);
    const minY = Math.min(startingBankroll, ...values);
    const maxY = Math.max(startingBankroll, ...values);
    const span = maxY - minY || 1;
    const innerW = WIDTH - PAD.left - PAD.right;
    const innerH = HEIGHT - PAD.top - PAD.bottom;

    const xOf = (i: number) => PAD.left + (points.length <= 1 ? 0 : (i / (points.length - 1)) * innerW);
    const yOf = (v: number) => PAD.top + innerH - ((v - minY) / span) * innerH;

    const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${xOf(i)} ${yOf(p.bankroll)}`).join(" ");
    return { linePath, xOf, yOf, minY, maxY };
  }, [points, startingBankroll]);

  if (points.length === 0) {
    return <p className="text-sm text-neutral-500">No bets were placed at this edge threshold.</p>;
  }

  function handleMove(e: React.MouseEvent<SVGSVGElement>) {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const scaleX = WIDTH / rect.width;
    const x = (e.clientX - rect.left) * scaleX;
    const innerW = WIDTH - PAD.left - PAD.right;
    const frac = Math.min(1, Math.max(0, (x - PAD.left) / innerW));
    const idx = Math.round(frac * (points.length - 1));
    setHoverIdx(Math.min(points.length - 1, Math.max(0, idx)));
  }

  const hovered = hoverIdx !== null ? points[hoverIdx] : null;

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      className="w-full text-neutral-400"
      onMouseMove={handleMove}
      onMouseLeave={() => setHoverIdx(null)}
    >
      {/* hairline gridlines */}
      {[minY, (minY + maxY) / 2, maxY].map((v, i) => (
        <g key={i}>
          <line
            x1={PAD.left} x2={WIDTH - PAD.right} y1={yOf(v)} y2={yOf(v)}
            stroke="currentColor" strokeOpacity={0.15} strokeWidth={1}
          />
          <text x={PAD.left - 8} y={yOf(v)} textAnchor="end" dominantBaseline="middle" fontSize={10} fill="currentColor">
            ${Math.round(v).toLocaleString()}
          </text>
        </g>
      ))}

      {/* starting bankroll reference line */}
      <line
        x1={PAD.left} x2={WIDTH - PAD.right} y1={yOf(startingBankroll)} y2={yOf(startingBankroll)}
        stroke="currentColor" strokeOpacity={0.3} strokeDasharray="2 3" strokeWidth={1}
      />

      <path d={linePath} fill="none" className="stroke-series1 dark:stroke-series1-dark" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />

      {hovered && hoverIdx !== null && (
        <>
          <line
            x1={xOf(hoverIdx)} x2={xOf(hoverIdx)} y1={PAD.top} y2={HEIGHT - PAD.bottom}
            stroke="currentColor" strokeOpacity={0.3} strokeWidth={1}
          />
          <circle cx={xOf(hoverIdx)} cy={yOf(hovered.bankroll)} r={4} className="fill-series1 dark:fill-series1-dark" stroke="white" strokeWidth={2} />
          <text
            x={Math.min(xOf(hoverIdx) + 8, WIDTH - 110)}
            y={PAD.top + 12}
            fontSize={11}
            className="fill-neutral-700 dark:fill-neutral-200"
          >
            {hovered.date} · ${hovered.bankroll.toLocaleString()}
          </text>
        </>
      )}
    </svg>
  );
}
