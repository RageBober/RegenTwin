import { usePopulations } from '../../hooks/useVisualization';
import PlotlyChart from './PlotlyChart';
import type { SimulationParams } from '../../types/api';

interface Props {
  params: SimulationParams;
  simulationId?: string;
}

export default function PopulationCharts({ params, simulationId }: Props) {
  const { data, isLoading, error } = usePopulations(params, undefined, simulationId);

  return (
    <PlotlyChart
      figure={data}
      loading={isLoading}
      error={error ? String(error) : null}
    />
  );
}
