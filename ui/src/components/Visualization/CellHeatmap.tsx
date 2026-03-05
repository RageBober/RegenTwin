import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useSpatialHeatmap } from '../../hooks/useSpatialData';
import PlotlyChart from './PlotlyChart';

const AGENT_TYPES = ['stem', 'macro', 'fibro', 'neutrophil', 'endothelial', 'myofibroblast'];

export default function CellHeatmap() {
  const { t } = useTranslation();
  const [binSize, setBinSize] = useState(10);
  const [selectedTypes, setSelectedTypes] = useState<string[] | null>(null);

  const { data, isLoading, error } = useSpatialHeatmap({
    bin_size: binSize,
    agent_types: selectedTypes,
  });

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-500 dark:text-slate-400">Bin size:</label>
          <input
            type="range"
            min={5}
            max={25}
            value={binSize}
            onChange={(e) => setBinSize(Number(e.target.value))}
            className="w-24"
          />
          <span className="text-xs text-slate-600 dark:text-slate-300">{binSize}</span>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-500 dark:text-slate-400">
            {t('dashboard.params.populations')}:
          </label>
          <select
            value={selectedTypes ? selectedTypes.join(',') : 'all'}
            onChange={(e) =>
              setSelectedTypes(e.target.value === 'all' ? null : [e.target.value])
            }
            className="rounded border border-slate-300 bg-white px-2 py-1 text-xs dark:border-slate-600 dark:bg-slate-700 dark:text-slate-200"
          >
            <option value="all">{t('history.all')}</option>
            {AGENT_TYPES.map((type) => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
        </div>
      </div>
      <PlotlyChart
        figure={data}
        loading={isLoading}
        error={error ? String(error) : null}
      />
    </div>
  );
}
