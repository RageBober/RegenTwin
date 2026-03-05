import { useQuery } from '@tanstack/react-query';
import { apiClient, API_VIZ } from '../lib/api';
import type { PlotlyFigure } from '../types/api';

export interface SpatialRequest {
  n_stem?: number;
  n_macro?: number;
  n_fibro?: number;
  n_neutrophil?: number;
  n_endothelial?: number;
  t_max_hours?: number;
  dt?: number;
  timestep?: number;
  bin_size?: number;
  agent_types?: string[] | null;
  color_by?: 'type' | 'energy' | 'age';
  height?: number;
  domain_size?: number;
  random_seed?: number | null;
}

const DEFAULT_SPATIAL: SpatialRequest = {
  n_stem: 20,
  n_macro: 30,
  n_fibro: 15,
  n_neutrophil: 40,
  n_endothelial: 10,
  t_max_hours: 48,
  dt: 1.0,
  timestep: -1,
  bin_size: 10,
  color_by: 'type',
  height: 500,
  domain_size: 100,
  random_seed: 42,
};

function useSpatialQuery(endpoint: string, params: SpatialRequest) {
  const merged = { ...DEFAULT_SPATIAL, ...params };
  return useQuery({
    queryKey: ['spatial', endpoint, merged],
    queryFn: async () => {
      const { data } = await apiClient.post<PlotlyFigure>(
        `${API_VIZ}/spatial/${endpoint}`,
        merged,
      );
      return data;
    },
    staleTime: Infinity,
  });
}

export function useSpatialHeatmap(params: SpatialRequest) {
  return useSpatialQuery('heatmap', params);
}

export function useSpatialScatter(params: SpatialRequest) {
  return useSpatialQuery('scatter', params);
}

export function useSpatialInflammation(params: SpatialRequest) {
  return useSpatialQuery('inflammation', params);
}
