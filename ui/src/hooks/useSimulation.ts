import { useMutation, useQuery } from '@tanstack/react-query';
import { apiClient, API_V1 } from '../lib/api';
import type {
  SimulationRequest,
  SimulationResponse,
  SimulationStatusResponse,
} from '../types/api';

export function useStartSimulation() {
  return useMutation({
    mutationFn: async (params: SimulationRequest) => {
      const { data } = await apiClient.post<SimulationResponse>(
        `${API_V1}/simulate`,
        params,
      );
      return data;
    },
  });
}

export function useSimulationStatus(id: string | null) {
  return useQuery({
    queryKey: ['simulation', id],
    queryFn: async () => {
      const { data } = await apiClient.get<SimulationStatusResponse>(
        `${API_V1}/simulate/${id}`,
      );
      return data;
    },
    enabled: !!id,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === 'completed' || status === 'failed' || status === 'cancelled') {
        return false;
      }
      return 2000;
    },
  });
}

export function useCancelSimulation() {
  return useMutation({
    mutationFn: async (id: string) => {
      const { data } = await apiClient.post(`${API_V1}/simulate/${id}/cancel`);
      return data;
    },
  });
}

export function useSimulationsList(status?: string) {
  return useQuery({
    queryKey: ['simulations', status],
    queryFn: async () => {
      const params: Record<string, string> = {};
      if (status && status !== 'all') params.status = status;
      const { data } = await apiClient.get<SimulationStatusResponse[]>(
        `${API_V1}/simulations`,
        { params },
      );
      return data;
    },
  });
}
