import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { ModelInfoResponse, SyncInfo, TimeEstimate, TrainStatus } from "../types";

function FeatureImportanceChart({ importances }: { importances: Record<string, number> }) {
  const entries = Object.entries(importances)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10);
  const max = Math.max(...entries.map(([, v]) => v), 1e-9);

  return (
    <div className="space-y-2">
      {entries.map(([name, value]) => (
        <div key={name} className="grid grid-cols-[9rem_1fr_3rem] items-center gap-2 text-xs">
          <span className="truncate text-neutral-500" title={name}>
            {name}
          </span>
          <span className="h-3 rounded-r bg-neutral-100 dark:bg-neutral-800">
            <span
              className="block h-3 rounded-r bg-series1 dark:bg-series1-dark"
              style={{ width: `${(value / max) * 100}%` }}
            />
          </span>
          <span className="tabular-nums text-neutral-500">{(value * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

export default function ModelInfoPanel() {
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);
  const [syncInfo, setSyncInfo] = useState<SyncInfo | null>(null);
  const [estimate, setEstimate] = useState<TimeEstimate | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [status, setStatus] = useState<TrainStatus | null>(null);
  const pollRef = useRef<number | null>(null);

  function refreshInfo() {
    api.modelInfo().then(setInfo).catch(() => setInfo(null));
  }

  useEffect(() => {
    refreshInfo();
    api.syncInfo().then(setSyncInfo).catch(() => {});
  }, []);

  function startPolling() {
    if (pollRef.current) return;
    pollRef.current = window.setInterval(async () => {
      const s = await api.trainStatus();
      setStatus(s);
      if (s.status === "done" || s.status === "error") {
        window.clearInterval(pollRef.current!);
        pollRef.current = null;
        refreshInfo();
      }
    }, 1500);
  }

  useEffect(() => () => {
    if (pollRef.current) window.clearInterval(pollRef.current);
  }, []);

  async function openRetrainDialog() {
    setConfirming(true);
    setEstimate(null);
    try {
      setEstimate(await api.modelEstimate());
    } catch {
      setEstimate(null);
    }
  }

  async function confirmRetrain(mode: "fast" | "tuned") {
    setConfirming(false);
    await api.retrain(mode);
    setStatus({ status: "running", mode, logs: [], error: null });
    startPolling();
  }

  const active = info?.active;
  const isRunning = status?.status === "running";

  return (
    <div className="space-y-6">
      {active ? (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="Mode" value={active.mode} />
          <Stat label="Hardware" value={active.device_label} />
          <Stat label="Test accuracy" value={`${(active.test_accuracy * 100).toFixed(1)}%`} />
          <Stat label="Players" value={active.num_players.toLocaleString()} />
          <Stat label="Train rows" value={active.train_rows.toLocaleString()} />
          <Stat label="Test rows" value={active.test_rows.toLocaleString()} />
          <Stat label="Data range" value={`${active.date_range[0]}–${active.date_range[1]}`} />
          <Stat label="Trained" value={new Date(active.trained_at).toLocaleString()} />
        </div>
      ) : (
        <p className="text-sm text-neutral-500">No trained model yet.</p>
      )}

      {info?.has_new_data && (
        <p className="rounded-lg bg-amber-50 dark:bg-amber-950 px-3 py-2 text-sm text-amber-800 dark:text-amber-200">
          New data is available since this model was trained.
        </p>
      )}

      <p className="text-xs text-neutral-500">
        {syncInfo?.last_checked
          ? `Data last synced: ${new Date(syncInfo.last_checked).toLocaleString()}${
              syncInfo.changed ? ` (updated seasons: ${syncInfo.updated_years.join(", ")})` : " (no changes found)"
            }`
          : "Data sync: never run. See README.md to schedule sync_data.py."}
      </p>

      {active && Object.keys(active.feature_importances).length > 0 && (
        <div>
          <p className="mb-2 text-sm font-medium text-neutral-700 dark:text-neutral-200">
            Top feature importances
          </p>
          <FeatureImportanceChart importances={active.feature_importances} />
        </div>
      )}

      <div className="border-t border-neutral-200 dark:border-neutral-700 pt-4">
        {!confirming && !isRunning && (
          <button
            onClick={openRetrainDialog}
            className="rounded-lg border border-neutral-300 dark:border-neutral-700 px-4 py-2 text-sm font-medium
                       hover:bg-neutral-50 dark:hover:bg-neutral-800"
          >
            Retrain model
          </button>
        )}

        {confirming && (
          <div className="space-y-3 rounded-lg border border-neutral-200 dark:border-neutral-700 p-4 text-sm">
            {estimate ? (
              <>
                <p>Hardware: {estimate.hardware}</p>
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={() => confirmRetrain("fast")}
                    className="rounded-lg bg-series1 dark:bg-series1-dark px-3 py-1.5 font-medium text-white"
                  >
                    Fast — {estimate.fast_label}
                  </button>
                  <button
                    onClick={() => confirmRetrain("tuned")}
                    className="rounded-lg border border-neutral-300 dark:border-neutral-700 px-3 py-1.5 font-medium"
                  >
                    Tuned — {estimate.tuned_label}
                  </button>
                  <button
                    onClick={() => setConfirming(false)}
                    className="rounded-lg px-3 py-1.5 text-neutral-500"
                  >
                    Cancel
                  </button>
                </div>
              </>
            ) : (
              <p className="text-neutral-500">Estimating training time...</p>
            )}
          </div>
        )}

        {isRunning && (
          <div className="space-y-2 rounded-lg border border-neutral-200 dark:border-neutral-700 p-4 text-sm">
            <p className="font-medium">Training ({status?.mode}) in progress…</p>
            <div className="max-h-32 overflow-auto font-mono text-xs text-neutral-500">
              {status?.logs.map((line, i) => (
                <div key={i}>{line}</div>
              ))}
            </div>
          </div>
        )}

        {status?.status === "done" && (
          <p className="mt-2 text-sm text-good">Training complete — model updated.</p>
        )}
        {status?.status === "error" && (
          <p className="mt-2 text-sm text-red-600 dark:text-red-400">Training failed: {status.error}</p>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-neutral-400">{label}</p>
      <p className="text-sm font-semibold tabular-nums text-neutral-800 dark:text-neutral-100">{value}</p>
    </div>
  );
}
