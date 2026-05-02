import { useMutation, useQuery } from '@tanstack/react-query';
import { apiClient, API_V1, API_VIZ } from '../lib/api';
import type {
  SensitivityRequest,
  EstimationRequest,
  AnalysisResponse,
  ParameterBoundsResponse,
  PlotlyFigure,
  SobolVizRequest,
  MorrisVizRequest,
  PosteriorVizRequest,
  ConvergenceVizRequest,
} from '../types/api';

export function useRunSensitivity() {
  return useMutation({
    mutationFn: async (req: SensitivityRequest) => {
      const { data } = await apiClient.post<AnalysisResponse>(
        `${API_V1}/analysis/sensitivity`,
        req,
      );
      return data;
    },
  });
}

export function useRunEstimation() {
  return useMutation({
    mutationFn: async (req: EstimationRequest) => {
      const { data } = await apiClient.post<AnalysisResponse>(
        `${API_V1}/analysis/estimation`,
        req,
      );
      return data;
    },
  });
}

export function useParameterBounds() {
  return useQuery({
    queryKey: ['parameter-bounds'],
    queryFn: async () => {
      const { data } = await apiClient.get<ParameterBoundsResponse>(
        `${API_V1}/parameters/bounds`,
      );
      return data;
    },
    staleTime: Infinity,
  });
}

export function useAnalysisStatus(id: string | null) {
  return useQuery({
    queryKey: ['analysis', id],
    queryFn: async () => {
      const { data } = await apiClient.get<AnalysisResponse>(
        `${API_V1}/analysis/${id}`,
      );
      return data;
    },
    enabled: !!id,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === 'completed' || status === 'failed' || status === 'cancelled') return false;
      return 5000;
    },
  });
}

export function useCancelAnalysis() {
  return useMutation({
    mutationFn: async (id: string) => {
      const { data } = await apiClient.post<AnalysisResponse>(
        `${API_V1}/analysis/${id}/cancel`,
      );
      return data;
    },
  });
}


// ── Analysis Visualization Hooks ───────────────────────────────

export function useSobolViz(request: SobolVizRequest | null) {
  return useQuery({
    queryKey: ['analysis-viz', 'sobol', request?.analysis_id, request],
    queryFn: async () => {
      const { data } = await apiClient.post<PlotlyFigure>(
        `${API_VIZ}/analysis/sobol`,
        request,
      );
      return data;
    },
    enabled: !!request,
    staleTime: Infinity,
  });
}

export function useMorrisViz(request: MorrisVizRequest | null) {
  return useQuery({
    queryKey: ['analysis-viz', 'morris', request?.analysis_id, request],
    queryFn: async () => {
      const { data } = await apiClient.post<PlotlyFigure>(
        `${API_VIZ}/analysis/morris`,
        request,
      );
      return data;
    },
    enabled: !!request,
    staleTime: Infinity,
  });
}

export function usePosteriorViz(request: PosteriorVizRequest | null) {
  return useQuery({
    queryKey: ['analysis-viz', 'posterior', request?.analysis_id, request],
    queryFn: async () => {
      const { data } = await apiClient.post<PlotlyFigure>(
        `${API_VIZ}/analysis/posterior`,
        request,
      );
      return data;
    },
    enabled: !!request,
    staleTime: Infinity,
  });
}

export function useConvergenceViz(request: ConvergenceVizRequest | null) {
  return useQuery({
    queryKey: ['analysis-viz', 'convergence', request?.analysis_id, request],
    queryFn: async () => {
      const { data } = await apiClient.post<PlotlyFigure>(
        `${API_VIZ}/analysis/convergence`,
        request,
      );
      return data;
    },
    enabled: !!request,
    staleTime: Infinity,
  });
}
