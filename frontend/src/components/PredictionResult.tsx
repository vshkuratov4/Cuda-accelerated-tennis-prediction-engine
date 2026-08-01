import type { PredictResponse } from "../types";

function Dot({ className }: { className: string }) {
  return <span className={`inline-block h-2.5 w-2.5 rounded-full ${className}`} />;
}

const CONFIDENCE_GLYPH: Record<string, string> = { High: "●", Medium: "◐", Low: "○" };

export default function PredictionResult({ result }: { result: PredictResponse }) {
  const { player1, player2, winner, prob1, prob2 } = result;
  const pct1 = prob1 * 100;
  const pct2 = prob2 * 100;
  const hasOdds = result.edge1 !== undefined && result.edge2 !== undefined;

  const bestEdge =
    hasOdds && result.edge1! > result.edge2! && result.edge1! > 0
      ? { side: player1, edge: result.edge1!, kelly: result.kelly_stake1! }
      : hasOdds && result.edge2! > 0
      ? { side: player2, edge: result.edge2!, kelly: result.kelly_stake2! }
      : null;

  return (
    <div className="space-y-6 rounded-xl border border-neutral-200 dark:border-neutral-700 p-5">
      <div className="flex items-center gap-2 text-sm font-medium text-good">
        <span aria-hidden>●</span>
        <span className="text-neutral-800 dark:text-neutral-100">
          {winner} is favored to win
        </span>
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-sm text-neutral-600 dark:text-neutral-300">
          <span className="flex items-center gap-1.5">
            <Dot className="bg-series1 dark:bg-series1-dark" />
            {player1}
          </span>
          <span className="flex items-center gap-1.5">
            {player2}
            <Dot className="bg-series2 dark:bg-series2-dark" />
          </span>
        </div>

        <div className="relative h-6 w-full overflow-visible">
          <div className="flex h-6 w-full overflow-hidden rounded-full bg-neutral-100 dark:bg-neutral-800">
            <div
              className="bg-series1 dark:bg-series1-dark"
              style={{ width: `${pct1}%`, marginRight: pct1 > 0 && pct2 > 0 ? "2px" : 0 }}
            />
            <div className="bg-series2 dark:bg-series2-dark" style={{ width: `${pct2}%` }} />
          </div>
          {/* 95% confidence band ticks for player1's win probability */}
          <div
            className="pointer-events-none absolute top-[-2px] h-[28px] w-px bg-neutral-500/60 dark:bg-neutral-300/60"
            style={{ left: `${result.ci_low1 * 100}%` }}
          />
          <div
            className="pointer-events-none absolute top-[-2px] h-[28px] w-px bg-neutral-500/60 dark:bg-neutral-300/60"
            style={{ left: `${result.ci_high1 * 100}%` }}
          />
        </div>

        <div className="flex justify-between text-base font-semibold tabular-nums">
          <span>{pct1.toFixed(1)}%</span>
          <span>{pct2.toFixed(1)}%</span>
        </div>

        <p className="text-xs text-neutral-500" title="Spread across the calibration model's 3 internal sub-models, gated by how many matches each player has on record">
          <span aria-hidden>{CONFIDENCE_GLYPH[result.confidence]}</span>{" "}
          Confidence: {result.confidence} (±{((result.ci_high1 - result.prob1) * 100).toFixed(1)} pts) —{" "}
          {result.player1_matches.toLocaleString()} vs {result.player2_matches.toLocaleString()} matches on record
        </p>
      </div>

      {hasOdds && (
        <div className="space-y-2 border-t border-neutral-200 dark:border-neutral-700 pt-4 text-sm">
          <p className="font-medium text-neutral-700 dark:text-neutral-200">Edge &amp; staking</p>
          <div className="grid grid-cols-2 gap-3 tabular-nums text-neutral-600 dark:text-neutral-300">
            <div>
              Implied ({player1}): {(result.implied_prob1! * 100).toFixed(1)}%
            </div>
            <div>
              Implied ({player2}): {(result.implied_prob2! * 100).toFixed(1)}%
            </div>
            <div>Edge ({player1}): {(result.edge1! * 100).toFixed(1)}%</div>
            <div>Edge ({player2}): {(result.edge2! * 100).toFixed(1)}%</div>
          </div>
          {bestEdge ? (
            <p className="text-neutral-800 dark:text-neutral-100">
              Suggested half-Kelly stake on <strong>{bestEdge.side}</strong>:{" "}
              {(bestEdge.kelly * 100).toFixed(1)}% of bankroll.
            </p>
          ) : (
            <p className="text-neutral-500">No positive-edge bet detected at these odds.</p>
          )}
        </div>
      )}
    </div>
  );
}
