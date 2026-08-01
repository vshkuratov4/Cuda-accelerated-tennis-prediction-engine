import { useEffect, useState } from "react";
import { api } from "./api/client";
import PredictForm from "./components/PredictForm";
import PredictionResult from "./components/PredictionResult";
import ModelInfoPanel from "./components/ModelInfoPanel";
import BacktestPanel from "./components/BacktestPanel";
import type { PredictRequest, PredictResponse } from "./types";

type Tab = "predict" | "model" | "backtest";

export default function App() {
  const [tab, setTab] = useState<Tab>("predict");
  const [players, setPlayers] = useState<string[]>([]);
  const [surfaces, setSurfaces] = useState<string[]>([]);
  const [rounds, setRounds] = useState<string[]>([]);
  const [modelReady, setModelReady] = useState(true);

  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.health().then((h) => setModelReady(h.model_ready)).catch(() => setModelReady(false));
    api.players().then((r) => setPlayers(r.players)).catch(() => {});
    api.meta().then((m) => {
      setSurfaces(m.surfaces);
      setRounds(m.rounds);
    }).catch(() => {});
  }, []);

  async function handlePredict(req: PredictRequest) {
    setLoading(true);
    setError(null);
    try {
      setResult(await api.predict(req));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto min-h-screen max-w-3xl px-4 py-10">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold text-neutral-900 dark:text-neutral-50">
          Tennis Match Predictor
        </h1>
        <p className="mt-1 text-sm text-neutral-500">
          Pick two players to get a win-probability prediction.
        </p>
      </header>

      <nav className="mb-6 flex gap-1 rounded-lg bg-neutral-100 dark:bg-neutral-800 p-1 text-sm font-medium">
        {(["predict", "model", "backtest"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`flex-1 rounded-md py-1.5 transition ${
              tab === t
                ? "bg-surface dark:bg-surface-dark text-neutral-900 dark:text-neutral-50 shadow-sm"
                : "text-neutral-500"
            }`}
          >
            {t === "predict" ? "Predict" : t === "model" ? "Model & Data" : "Backtest"}
          </button>
        ))}
      </nav>

      {!modelReady && (
        <p className="mb-4 rounded-lg bg-amber-50 dark:bg-amber-950 px-3 py-2 text-sm text-amber-800 dark:text-amber-200">
          No trained model is available yet. Check the terminal that launched the app, or train one from
          the "Model & Data" tab once data has been processed.
        </p>
      )}

      {tab === "predict" && (
        <div className="space-y-6">
          <PredictForm
            players={players}
            surfaces={surfaces}
            rounds={rounds}
            onSubmit={handlePredict}
            loading={loading}
          />
          {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}
          {result && <PredictionResult result={result} />}
        </div>
      )}

      {tab === "model" && <ModelInfoPanel />}
      {tab === "backtest" && <BacktestPanel />}
    </div>
  );
}
