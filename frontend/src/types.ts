export interface PredictRequest {
  player1: string;
  player2: string;
  surface: string;
  round?: string;
  odds1?: number;
  odds2?: number;
}

export interface PredictResponse {
  player1: string;
  player2: string;
  winner: string;
  prob1: number;
  prob2: number;
  confidence: "High" | "Medium" | "Low";
  prob1_std: number;
  ci_low1: number;
  ci_high1: number;
  ci_low2: number;
  ci_high2: number;
  player1_matches: number;
  player2_matches: number;
  implied_prob1?: number;
  implied_prob2?: number;
  edge1?: number;
  edge2?: number;
  kelly_stake1?: number;
  kelly_stake2?: number;
}

export interface MetaResponse {
  surfaces: string[];
  rounds: string[];
}

export interface HealthResponse {
  status: string;
  device: string;
  device_label: string;
  model_ready: boolean;
}

export interface ModelMeta {
  version_id: string;
  mode: "fast" | "tuned";
  device: string;
  device_label: string;
  trained_at: string;
  data_hash: string | null;
  train_rows: number;
  test_rows: number;
  train_accuracy: number;
  test_accuracy: number;
  date_range: [number, number];
  num_players: number;
  feature_importances: Record<string, number>;
  selected_features: string[];
}

export interface ModelInfoResponse {
  active: ModelMeta | null;
  versions: ModelMeta[];
  has_new_data: boolean;
  current_data_hash: string | null;
}

export interface TrainStatus {
  status: "idle" | "running" | "done" | "error";
  mode: string | null;
  logs: string[];
  error: string | null;
}

export interface TimeEstimate {
  fast_seconds: number;
  tuned_seconds: number;
  fast_label: string;
  tuned_label: string;
  hardware: string;
}

export interface BacktestRequest {
  edge_threshold: number;
  kelly_fraction: number;
  max_seasons?: number;
}

export interface EquityPoint {
  date: string;
  bankroll: number;
}

export interface SeasonBreakdown {
  year: number;
  bets: number;
  win_rate: number;
  roi_pct: number;
}

export interface BacktestResult {
  seasons_tested: number[];
  starting_bankroll: number;
  final_bankroll: number;
  roi_pct: number;
  total_bets: number;
  win_rate: number;
  equity_curve: EquityPoint[];
  season_breakdown: SeasonBreakdown[];
  params: { edge_threshold: number; kelly_fraction: number; max_seasons: number };
}

export interface BacktestStatus {
  status: "idle" | "running" | "done" | "error";
  logs: string[];
  error: string | null;
  result: BacktestResult | null;
}

export interface SyncInfo {
  last_checked: string | null;
  changed: boolean | null;
  updated_years: number[];
  retrained_version: string | null;
}
