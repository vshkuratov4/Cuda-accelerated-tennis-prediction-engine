import { useState } from "react";
import type { PredictRequest } from "../types";
import Combobox from "./Combobox";

const ROUND_LABELS: Record<string, string> = {
  R128: "Round of 128",
  R64: "Round of 64",
  R32: "Round of 32",
  R16: "Round of 16",
  QF: "Quarterfinals",
  SF: "Semifinals",
  F: "Final",
};

interface PredictFormProps {
  players: string[];
  surfaces: string[];
  rounds: string[];
  onSubmit: (req: PredictRequest) => void;
  loading: boolean;
}

export default function PredictForm({ players, surfaces, rounds, onSubmit, loading }: PredictFormProps) {
  const [player1, setPlayer1] = useState("");
  const [player2, setPlayer2] = useState("");
  const [surface, setSurface] = useState(surfaces[0] ?? "Hard");
  const [round, setRound] = useState(rounds[0] ?? "");
  const [oddsOpen, setOddsOpen] = useState(false);
  const [odds1, setOdds1] = useState("");
  const [odds2, setOdds2] = useState("");
  const [error, setError] = useState<string | null>(null);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!player1 || !player2) {
      setError("Choose both players.");
      return;
    }
    if (player1 === player2) {
      setError("Choose two different players.");
      return;
    }
    setError(null);
    onSubmit({
      player1,
      player2,
      surface,
      round: round || undefined,
      odds1: odds1 ? Number(odds1) : undefined,
      odds2: odds2 ? Number(odds2) : undefined,
    });
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Combobox label="Player 1" options={players} value={player1} onChange={setPlayer1} />
        <Combobox label="Player 2" options={players} value={player2} onChange={setPlayer2} />
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div>
          <label className="mb-1 block text-sm font-medium text-neutral-600 dark:text-neutral-300">
            Surface
          </label>
          <select
            value={surface}
            onChange={(e) => setSurface(e.target.value)}
            className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-surface
                       dark:bg-surface-dark px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-series1"
          >
            {surfaces.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="mb-1 block text-sm font-medium text-neutral-600 dark:text-neutral-300">
            Round
          </label>
          <select
            value={round}
            onChange={(e) => setRound(e.target.value)}
            className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-surface
                       dark:bg-surface-dark px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-series1"
          >
            {rounds.map((r) => (
              <option key={r} value={r}>
                {ROUND_LABELS[r] ?? r}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="rounded-lg border border-neutral-200 dark:border-neutral-700">
        <button
          type="button"
          onClick={() => setOddsOpen((v) => !v)}
          className="flex w-full items-center justify-between px-4 py-2.5 text-sm font-medium text-neutral-600
                     dark:text-neutral-300"
        >
          Betting odds (optional)
          <span className="text-neutral-400">{oddsOpen ? "−" : "+"}</span>
        </button>
        {oddsOpen && (
          <div className="grid grid-cols-1 gap-4 border-t border-neutral-200 dark:border-neutral-700 p-4 sm:grid-cols-2">
            <div>
              <label className="mb-1 block text-sm text-neutral-500">Player 1 decimal odds</label>
              <input
                type="number"
                step="0.01"
                min="1.01"
                value={odds1}
                onChange={(e) => setOdds1(e.target.value)}
                placeholder="e.g. 1.85"
                className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-surface
                           dark:bg-surface-dark px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-series1"
              />
            </div>
            <div>
              <label className="mb-1 block text-sm text-neutral-500">Player 2 decimal odds</label>
              <input
                type="number"
                step="0.01"
                min="1.01"
                value={odds2}
                onChange={(e) => setOdds2(e.target.value)}
                placeholder="e.g. 2.10"
                className="w-full rounded-lg border border-neutral-300 dark:border-neutral-700 bg-surface
                           dark:bg-surface-dark px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-series1"
              />
            </div>
          </div>
        )}
      </div>

      {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}

      <button
        type="submit"
        disabled={loading}
        className="w-full rounded-lg bg-series1 dark:bg-series1-dark px-4 py-2.5 text-sm font-semibold text-white
                   transition hover:opacity-90 disabled:opacity-50"
      >
        {loading ? "Predicting..." : "Predict winner"}
      </button>
    </form>
  );
}
