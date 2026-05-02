import { useSpatialInflammation } from '../../hooks/useSpatialData';
import PlotlyChart from './PlotlyChart';

interface Props {
  simulationId?: string;
}

export default function InflammationMap({ simulationId }: Props) {
  const { data, isLoading, error } = useSpatialInflammation({ simulation_id: simulationId });

  return (
    <div data-testid="inflammation-map">
      <PlotlyChart figure={data} loading={isLoading} error={error ? String(error) : null} />
    </div>
  );
}
