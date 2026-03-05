import { useMutation, useQuery } from '@tanstack/react-query';
import { apiClient, API_V1 } from '../lib/api';
import type {
  SensitivityRequest,
  EstimationRequest,
  AnalysisResponse,
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
      if (status === 'completed' || status === 'failed') return false;
      return 2000;
    },
  });
}
