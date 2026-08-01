import type {
  BacktestRequest,
  BacktestStatus,
  HealthResponse,
  MetaResponse,
  ModelInfoResponse,
  PredictRequest,
  PredictResponse,
  SyncInfo,
  TimeEstimate,
  TrainStatus,
} from "../types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail ?? `Request failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),
  players: () => request<{ players: string[] }>("/api/players"),
  meta: () => request<MetaResponse>("/api/meta"),
  predict: (body: PredictRequest) =>
    request<PredictResponse>("/api/predict", { method: "POST", body: JSON.stringify(body) }),
  modelInfo: () => request<ModelInfoResponse>("/api/model/info"),
  modelEstimate: () => request<TimeEstimate>("/api/model/estimate"),
  retrain: (mode: "fast" | "tuned") =>
    request<{ status: string; mode: string }>("/api/model/retrain", {
      method: "POST",
      body: JSON.stringify({ mode }),
    }),
  trainStatus: () => request<TrainStatus>("/api/model/train-status"),
  startBacktest: (body: BacktestRequest) =>
    request<{ status: string }>("/api/backtest", { method: "POST", body: JSON.stringify(body) }),
  backtestStatus: () => request<BacktestStatus>("/api/backtest/status"),
  syncInfo: () => request<SyncInfo>("/api/data/sync-info"),
};
