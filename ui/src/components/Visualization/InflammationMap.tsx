import { useSpatialInflammation } from '../../hooks/useSpatialData';
import PlotlyChart from './PlotlyChart';

export default function InflammationMap() {
  const { data, isLoading, error } = useSpatialInflammation({});

  return (
    <PlotlyChart
      figure={data}
      loading={isLoading}
      error={error ? String(error) : null}
    />
  );
}
