import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useComparison } from '../../hooks/useVisualization';
import PlotlyChart from './PlotlyChart';
import type { SimulationParams } from '../../types/api';

const VARIABLES = ['F', 'P', 'Ne', 'M1', 'M2', 'Mf', 'E', 'S'];

interface Props {
  params: SimulationParams;
}

export default function TherapyComparison({ params }: Props) {
  const { t } = useTranslation();
  const [variable, setVariable] = useState('F');
  const [showAll, setShowAll] = useState(false);

  const { data, isLoading, error } = useComparison(params, variable, showAll);

  return (
    <div>
      <div className="mb-3 flex items-center gap-4">
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-500 dark:text-slate-400">
            {t('dashboard.variables.F').split(' ')[0]}:
          </label>
          <select
            value={variable}
            onChange={(e) => setVariable(e.target.value)}
            className="rounded border border-slate-300 bg-white px-2 py-1 text-xs dark:border-slate-600 dark:bg-slate-700 dark:text-slate-200"
          >
            {VARIABLES.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>
        <label className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
          <input
            type="checkbox"
            checked={showAll}
            onChange={(e) => setShowAll(e.target.checked)}
            className="accent-primary-600"
          />
          All populations
        </label>
      </div>
      <PlotlyChart
        figure={data}
        loading={isLoading}
        error={error ? String(error) : null}
      />
    </div>
  );
}
