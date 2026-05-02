export type SimulationMode = 'mvp' | 'extended' | 'abm' | 'integrated';
export type SimulationStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
export type ExportFormat = 'csv' | 'png' | 'svg' | 'pdf';
export type AnalysisType = 'sensitivity' | 'estimation';

export interface SimulationRequest {
  mode: SimulationMode;
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
  C_PDGF0: number;
  C_VEGF0: number;
  C_TGFb0: number;
  C_MCP1_0: number;
  C_IL8_0: number;
  rho_collagen0: number;
  C_MMP0: number;
  rho_fibrin0: number;
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
  n_trajectories: number;
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
  C_PDGF0: 5.0,
  C_VEGF0: 2.0,
  C_TGFb0: 3.0,
  C_MCP1_0: 5.0,
  C_IL8_0: 8.0,
  rho_collagen0: 0.1,
  C_MMP0: 1.0,
  rho_fibrin0: 5.0,
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
  error_message: string | null;
  created_at: string;
  completed_at: string | null;
  params_json?: SimulationRequest | null;
}

export interface UploadMetadata {
  n_events?: number;
  n_channels?: number;
  channels?: string[];
  cytometer?: string | null;
  fcs_version?: string | null;
  parameter_source?: string;
  initial_conditions?: Partial<SimulationRequest>;
  supported_exports?: ExportFormat[];
  [key: string]: unknown;
}

export interface UploadResponse {
  upload_id: string;
  filename: string;
  status: string;
  created_at: string;
  metadata: UploadMetadata | null;
}

export interface ResultsResponse {
  simulation_id: string;
  mode: SimulationMode;
  times: number[];
  variables: Record<string, number[]>;
  metadata: Record<string, unknown>;
}

export interface ExportRequest {
  format: ExportFormat;
  include_populations: boolean;
  include_cytokines: boolean;
  include_ecm: boolean;
  include_phases: boolean;
}

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

export interface SensitivityResult {
  method: 'sobol' | 'morris';
  parameters: string[];
  S1?: number[];
  ST?: number[];
  S1_conf?: number[];
  ST_conf?: number[];
  mu_star?: number[];
  sigma?: number[];
  mu_star_conf?: number[];
  n_samples: number;
  n_runs?: number;
  warning?: string;
  error?: string;
}

export interface AnalysisResponse {
  analysis_id: string;
  analysis_type: AnalysisType;
  status: SimulationStatus;
  created_at: string;
  progress: number;
  result: Record<string, unknown> | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  uptime_seconds: number;
}

export interface ErrorResponse {
  error: string;
  detail: string | null;
  code: string | null;
}

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
  C_PDGF0: number;
  C_VEGF0: number;
  C_TGFb0: number;
  C_MCP1_0: number;
  C_IL8_0: number;
  rho_collagen0: number;
  C_MMP0: number;
  rho_fibrin0: number;
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

export interface PlotlyFigure {
  data: Record<string, unknown>[];
  layout: Record<string, unknown>;
}

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
    C_PDGF0: req.C_PDGF0,
    C_VEGF0: req.C_VEGF0,
    C_TGFb0: req.C_TGFb0,
    C_MCP1_0: req.C_MCP1_0,
    C_IL8_0: req.C_IL8_0,
    rho_collagen0: req.rho_collagen0,
    C_MMP0: req.C_MMP0,
    rho_fibrin0: req.rho_fibrin0,
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

// --- Analysis Visualization Requests ---

export interface SobolVizRequest {
  analysis_id: string;
  metric?: 'S1' | 'ST' | 'both';
  top_n?: number | null;
  show_confidence?: boolean;
  height?: number;
}

export interface MorrisVizRequest {
  analysis_id: string;
  highlight_influential?: boolean;
  threshold_ratio?: number;
  show_labels?: boolean;
  show_wedge?: boolean;
  height?: number;
}

export interface PosteriorVizRequest {
  analysis_id: string;
  parameters?: string[] | null;
  layout?: 'marginals' | 'corner';
  show_ci?: boolean;
  show_point_estimate?: boolean;
  n_bins?: number;
  height?: number;
}

export interface ConvergenceVizRequest {
  analysis_id: string;
  metrics?: string[] | null;
  show_rhat_threshold?: boolean;
  height?: number;
}

export interface EstimationResult {
  method: string;
  target_variable: string;
  upload_id: string;
  point_estimates: Record<string, number>;
  ci_lower: Record<string, number>;
  ci_upper: Record<string, number>;
  log_likelihood: number | null;
  aic: number | null;
  bic: number | null;
  n_observations: number;
  n_estimated_params: number;
  elapsed_seconds: number;
  n_samples: number;
  n_chains: number;
  diagnostics: {
    converged: boolean;
    rhat: Record<string, number>;
    ess_bulk: Record<string, number>;
    ess_tail: Record<string, number>;
    warnings: string[];
  } | null;
  posterior_samples?: Record<string, number[]>;
}

// --- Parameter Bounds ---

export interface ParameterBoundItem {
  name: string;
  lower: number;
  upper: number;
  nominal: number;
  group: string;
}

export interface ParameterBoundsResponse {
  bounds: ParameterBoundItem[];
  total: number;
}
