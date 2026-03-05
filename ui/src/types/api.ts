// ── Enums ──────────────────────────────────────────────────────

export type SimulationMode = 'mvp' | 'extended' | 'abm' | 'integrated';
export type SimulationStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
export type ExportFormat = 'csv' | 'png' | 'svg' | 'pdf';
export type AnalysisType = 'sensitivity' | 'estimation';

// ── Simulation ────────────────────────────────────────────────

export interface SimulationRequest {
  mode: SimulationMode;
  // Initial conditions
  P0: number;
  Ne0: number;
  M1_0: number;
  M2_0: number;
  F0: number;
  Mf0: number;
  E0: number;
  S0: number;
  C_TNF0: number;
  C_IL10_0: number;
  D0: number;
  O2_0: number;
  // Time
  t_max_hours: number;
  dt: number;
  // Therapy
  prp_enabled: boolean;
  pemf_enabled: boolean;
  prp_intensity: number;
  pemf_frequency: number;
  pemf_intensity: number;
  // RNG
  random_seed: number | null;
  // Monte Carlo
  n_trajectories: number;
  // Upload
  upload_id: string | null;
}

export const DEFAULT_SIMULATION_PARAMS: SimulationRequest = {
  mode: 'extended',
  P0: 500,
  Ne0: 200,
  M1_0: 100,
  M2_0: 10,
  F0: 50,
  Mf0: 0,
  E0: 20,
  S0: 40,
  C_TNF0: 10,
  C_IL10_0: 0.5,
  D0: 5,
  O2_0: 80,
  t_max_hours: 720,
  dt: 0.1,
  prp_enabled: false,
  pemf_enabled: false,
  prp_intensity: 1.0,
  pemf_frequency: 50.0,
  pemf_intensity: 1.0,
  random_seed: 42,
  n_trajectories: 1,
  upload_id: null,
};

export interface SimulationResponse {
  simulation_id: string;
  status: SimulationStatus;
  created_at: string;
  mode: SimulationMode;
}

export interface SimulationStatusResponse {
  simulation_id: string;
  status: SimulationStatus;
  progress: number;
  message: string | null;
  created_at: string;
  completed_at: string | null;
}

// ── Upload ────────────────────────────────────────────────────

export interface UploadResponse {
  upload_id: string;
  filename: string;
  status: string;
  created_at: string;
  metadata: Record<string, unknown> | null;
}

// ── Results & Export ──────────────────────────────────────────

export interface ResultsResponse {
  simulation_id: string;
  mode: SimulationMode;
  times: number[];
  variables: Record<string, number[]>;
  metadata: Record<string, string | number>;
}

export interface ExportRequest {
  format: ExportFormat;
  include_populations: boolean;
  include_cytokines: boolean;
  include_ecm: boolean;
  include_phases: boolean;
}

// ── Analysis ──────────────────────────────────────────────────

export interface SensitivityRequest {
  simulation_params: SimulationRequest;
  parameters: string[];
  method: 'sobol' | 'morris';
  n_samples: number;
}

export interface EstimationRequest {
  upload_id: string;
  target_variable: string;
  method: 'mcmc' | 'optimization';
  n_samples: number;
}

export interface AnalysisResponse {
  analysis_id: string;
  analysis_type: AnalysisType;
  status: SimulationStatus;
  created_at: string;
  progress: number;
  result: Record<string, unknown> | null;
}

// ── Health ────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  version: string;
  uptime_seconds: number;
}

// ── Errors ────────────────────────────────────────────────────

export interface ErrorResponse {
  error: string;
  detail: string | null;
  code: string | null;
}

// ── Visualization request types ──────────────────────────────

export interface SimulationParams {
  P0: number;
  Ne0: number;
  M1_0: number;
  M2_0: number;
  F0: number;
  Mf0: number;
  E0: number;
  S0: number;
  C_TNF0: number;
  C_IL10_0: number;
  D0: number;
  O2_0: number;
  t_max_hours: number;
  dt: number;
  prp_enabled: boolean;
  pemf_enabled: boolean;
  prp_intensity: number;
  pemf_frequency: number;
  pemf_intensity: number;
  random_seed: number | null;
}

export interface PopulationsRequest {
  simulation: SimulationParams;
  variables?: string[] | null;
  height?: number;
}

export interface CytokinesRequest {
  simulation: SimulationParams;
  variables?: string[] | null;
  layout?: 'overlay' | 'subplots';
  height?: number;
}

export interface ECMRequest {
  simulation: SimulationParams;
  height?: number;
}

export interface PhasesRequest {
  simulation: SimulationParams;
  height?: number;
}

export interface ComparisonRequest {
  simulation: SimulationParams;
  variable?: string;
  show_all_populations?: boolean;
  height?: number;
}

export interface VizExportRequest {
  simulation: SimulationParams;
  include_populations?: boolean;
  include_cytokines?: boolean;
  include_ecm?: boolean;
  include_phases?: boolean;
}

// ── Plotly types ──────────────────────────────────────────────

export interface PlotlyFigure {
  data: Record<string, unknown>[];
  layout: Record<string, unknown>;
}

// ── Utility: extract SimulationParams from SimulationRequest ─

export function toSimulationParams(req: SimulationRequest): SimulationParams {
  return {
    P0: req.P0,
    Ne0: req.Ne0,
    M1_0: req.M1_0,
    M2_0: req.M2_0,
    F0: req.F0,
    Mf0: req.Mf0,
    E0: req.E0,
    S0: req.S0,
    C_TNF0: req.C_TNF0,
    C_IL10_0: req.C_IL10_0,
    D0: req.D0,
    O2_0: req.O2_0,
    t_max_hours: req.t_max_hours,
    dt: req.dt,
    prp_enabled: req.prp_enabled,
    pemf_enabled: req.pemf_enabled,
    prp_intensity: req.prp_intensity,
    pemf_frequency: req.pemf_frequency,
    pemf_intensity: req.pemf_intensity,
    random_seed: req.random_seed,
  };
}
