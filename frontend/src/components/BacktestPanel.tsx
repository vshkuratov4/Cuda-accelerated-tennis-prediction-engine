import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { BacktestStatus } from "../types";
import EquityCurveChart from "./EquityCurveChart";

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-neutral-400">{label}</p>
      <p className="text-lg font-semibold tabular-nums text-neutral-800 dark:text-neutral-100">{value}</p>
    </div>
  );
}

export default function BacktestPanel() {
  const [edgePct, setEdgePct] = useState("10");
  const [kellyFraction, setKellyFraction] = useState("0.5");
  const [status, setStatus] = useState<BacktestStatus | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    api.backtestStatus().then(setStatus).catch(() => {});
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, []);

  function startPolling() {
    if (pollRef.current) return;
    pollRef.current = window.setInterval(async () => {
      const s = await api.backtestStatus();
      setStatus(s);
      if (s.status === "done" || s.status === "error") {
        window.clearInterval(pollRef.current!);
        pollRef.current = null;
      }
    }, 2000);
  }

  async function run() {
    await api.startBacktest({
      edge_threshold: Number(edgePct) / 100,
      kelly_fraction: Number(kellyFraction),
    });
    setStatus({ status: "running", logs: [], error: null, result: null });
    startPolling();
  }

  const isRunning = status?.status === "running";
  const result = status?.result;

  return (
    <div className="space-y-6">
      <p className="text-sm text-neutral-500">
        Walk-forward simulation: for each recent season, a model is trained only on prior
        seasons, then bets are placed on that season's matches whenever the model's edge over
        the bookmaker's implied probability clears your threshold. Bankroll compounds across
        seasons in chronological order (stakes are capped per bet as a risk-management
        safeguard against oversized, overconfident bets).
      </p>
      <p className="text-xs text-neutral-500">
        This is a methodology demonstration of leakage-free walk-forward evaluation, not a
        profitability claim — real bookmaker markets are efficient, genuine edges are rare, and
        results at low edge thresholds are usually calibration noise rather than real signal.
        Expect ROI to often be flat or negative. Do not use this for real wagering decisions.
      </p>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div>
          <label className="mb-1 block text-sm font-medium text-neutral-600 dark:text-neutral-300">
            Minimum edge (%)
          </label>
          <input
            type="number" min="0" max="100" step="0.5" value={edgePct}
            onChange={(e) => setEdgePct(e.target.value)}
            className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-surface
                       dark:bg-surface-dark px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-series1"
          />
        </div>
        <div>
          <label className="mb-1 block text-sm font-medium text-neutral-600 dark:text-neutral-300">
            Kelly fraction (0-1, 0.5 = half-Kelly)
          </label>
          <input
            type="number" min="0.01" max="1" step="0.05" value={kellyFraction}
            onChange={(e) => setKellyFraction(e.target.value)}
            className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-surface
                       dark:bg-surface-dark px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-series1"
          />
        </div>
      </div>

      <button
        onClick={run}
        disabled={isRunning}
        className="rounded-lg bg-series1 dark:bg-series1-dark px-4 py-2 text-sm font-semibold text-white
                   transition hover:opacity-90 disabled:opacity-50"
      >
        {isRunning ? "Running backtest..." : "Run backtest"}
      </button>

      {isRunning && (
        <div className="max-h-32 overflow-auto rounded-lg border border-neutral-200 dark:border-neutral-700
                        p-3 font-mono text-xs text-neutral-500">
          {status?.logs.map((line, i) => (
            <div key={i}>{line}</div>
          ))}
        </div>
      )}

      {status?.status === "error" && (
        <p className="text-sm text-red-600 dark:text-red-400">Backtest failed: {status.error}</p>
      )}

      {result && (
        <div className="space-y-5 border-t border-neutral-200 dark:border-neutral-700 pt-5">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Seasons tested" value={result.seasons_tested.join(", ")} />
            <Stat label="Bets placed" value={result.total_bets.toLocaleString()} />
            <Stat label="Win rate" value={`${(result.win_rate * 100).toFixed(1)}%`} />
            <Stat label="ROI" value={`${result.roi_pct >= 0 ? "+" : ""}${result.roi_pct.toFixed(1)}%`} />
          </div>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-2">
            <Stat label="Starting bankroll" value={`$${result.starting_bankroll.toLocaleString()}`} />
            <Stat label="Final bankroll" value={`$${result.final_bankroll.toLocaleString()}`} />
          </div>

          <div>
            <p className="mb-2 text-sm font-medium text-neutral-700 dark:text-neutral-200">Equity curve</p>
            <EquityCurveChart points={result.equity_curve} startingBankroll={result.starting_bankroll} />
          </div>

          <div>
            <p className="mb-2 text-sm font-medium text-neutral-700 dark:text-neutral-200">Season breakdown</p>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="text-neutral-400">
                    <th className="pb-1 pr-4">Season</th>
                    <th className="pb-1 pr-4">Bets</th>
                    <th className="pb-1 pr-4">Win rate</th>
                    <th className="pb-1">ROI</th>
                  </tr>
                </thead>
                <tbody className="tabular-nums">
                  {result.season_breakdown.map((s) => (
                    <tr key={s.year} className="border-t border-neutral-100 dark:border-neutral-800">
                      <td className="py-1 pr-4">{s.year}</td>
                      <td className="py-1 pr-4">{s.bets}</td>
                      <td className="py-1 pr-4">{(s.win_rate * 100).toFixed(1)}%</td>
                      <td className="py-1">{s.roi_pct >= 0 ? "+" : ""}{s.roi_pct.toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
