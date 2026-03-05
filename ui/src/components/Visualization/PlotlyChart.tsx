import Plot from 'react-plotly.js';
import { useTranslation } from 'react-i18next';
import type { PlotlyFigure } from '../../types/api';

interface PlotlyChartProps {
  figure: PlotlyFigure | undefined;
  loading?: boolean;
  error?: string | null;
}

export default function PlotlyChart({ figure, loading, error }: PlotlyChartProps) {
  const { t } = useTranslation();

  if (loading) {
    return (
      <div className="flex h-96 items-center justify-center rounded-lg bg-slate-50 dark:bg-slate-800/50">
        <div className="flex flex-col items-center gap-2">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary-200 border-t-primary-600" />
          <span className="text-sm text-slate-400">{t('common.loading')}</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-96 items-center justify-center rounded-lg bg-red-50 dark:bg-red-900/10">
        <span className="text-sm text-red-500">{error}</span>
      </div>
    );
  }

  if (!figure) return null;

  return (
    <Plot
      data={figure.data as unknown as Plotly.PlotData[]}
      layout={{
        ...(figure.layout as Partial<Plotly.Layout>),
        autosize: true,
        margin: { l: 60, r: 30, t: 40, b: 50 },
      }}
      useResizeHandler
      style={{ width: '100%', height: '100%', minHeight: '400px' }}
      config={{
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
      }}
    />
  );
}
