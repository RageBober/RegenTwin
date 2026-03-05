import { useQuery } from '@tanstack/react-query';
import { apiClient, API_VIZ } from '../lib/api';
import type {
  SimulationParams,
  PlotlyFigure,
} from '../types/api';

/**
 * POST-based viz query: runs a new simulation on-the-fly.
 * Used when no simulation_id is available (e.g. quick preview).
 */
function useVizQuery(
  endpoint: string,
  params: SimulationParams | null,
  extraBody?: Record<string, unknown>,
) {
  return useQuery({
    queryKey: ['viz', endpoint, params, extraBody],
    queryFn: async () => {
      const { data } = await apiClient.post<PlotlyFigure>(
        `${API_VIZ}/${endpoint}`,
        { simulation: params, ...extraBody },
      );
      return data;
    },
    enabled: !!params,
    staleTime: Infinity,
  });
}

/**
 * GET-based viz query: loads chart from cached simulation results.
 * Much faster — no re-simulation needed.
 */
function useCachedVizQuery(
  chartType: string,
  simulationId: string | undefined,
  queryParams?: Record<string, string>,
) {
  return useQuery({
    queryKey: ['viz-cached', chartType, simulationId, queryParams],
    queryFn: async () => {
      const { data } = await apiClient.get<PlotlyFigure>(
        `${API_VIZ}/from-result/${simulationId}/${chartType}`,
        { params: queryParams },
      );
      return data;
    },
    enabled: !!simulationId,
    staleTime: Infinity,
  });
}

export function usePopulations(
  params: SimulationParams | null,
  variables?: string[],
  simulationId?: string,
) {
  const cached = useCachedVizQuery(
    'populations',
    simulationId,
    variables ? { variables: variables.join(',') } : undefined,
  );
  const live = useVizQuery(
    'populations',
    simulationId ? null : params,
    variables ? { variables } : undefined,
  );
  return simulationId ? cached : live;
}

export function useCytokines(
  params: SimulationParams | null,
  layout?: 'overlay' | 'subplots',
  simulationId?: string,
) {
  const cached = useCachedVizQuery(
    'cytokines',
    simulationId,
    layout ? { layout } : undefined,
  );
  const live = useVizQuery(
    'cytokines',
    simulationId ? null : params,
    layout ? { layout } : undefined,
  );
  return simulationId ? cached : live;
}

export function useECM(params: SimulationParams | null, simulationId?: string) {
  const cached = useCachedVizQuery('ecm', simulationId);
  const live = useVizQuery('ecm', simulationId ? null : params);
  return simulationId ? cached : live;
}

export function usePhases(params: SimulationParams | null, simulationId?: string) {
  const cached = useCachedVizQuery('phases', simulationId);
  const live = useVizQuery('phases', simulationId ? null : params);
  return simulationId ? cached : live;
}

export function useComparison(
  params: SimulationParams | null,
  variable?: string,
  showAll?: boolean,
) {
  // Comparison always runs live (4 scenarios, no single cached result)
  return useVizQuery('comparison', params, {
    ...(variable ? { variable } : {}),
    ...(showAll ? { show_all_populations: true } : {}),
  });
}
