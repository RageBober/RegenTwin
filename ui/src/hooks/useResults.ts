import { useQuery } from '@tanstack/react-query';
import { apiClient, API_V1 } from '../lib/api';
import type { ResultsResponse, SimulationStatusResponse } from '../types/api';

export function useResults(simulationId: string | undefined) {
  return useQuery({
    queryKey: ['results', simulationId],
    queryFn: async () => {
      const { data } = await apiClient.get<ResultsResponse>(
        `${API_V1}/results/${simulationId}`,
      );
      return data;
    },
    enabled: !!simulationId,
    staleTime: Infinity,
  });
}

export function useSimulationMeta(simulationId: string | undefined) {
  return useQuery({
    queryKey: ['simulation-meta', simulationId],
    queryFn: async () => {
      const { data } = await apiClient.get<SimulationStatusResponse>(
        `${API_V1}/simulate/${simulationId}`,
      );
      return data;
    },
    enabled: !!simulationId,
    staleTime: Infinity,
  });
}
