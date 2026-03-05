import { useTranslation } from 'react-i18next';
import { useSimulationStore } from '../../stores/simulationStore';
import type { SimulationMode } from '../../types/api';

const MODES: { value: SimulationMode; labelKey: string }[] = [
  { value: 'mvp', labelKey: 'dashboard.model.mvp' },
  { value: 'extended', labelKey: 'dashboard.model.extended' },
  { value: 'abm', labelKey: 'dashboard.model.abm' },
  { value: 'integrated', labelKey: 'dashboard.model.integrated' },
];

export default function ModelSelector() {
  const { t } = useTranslation();
  const mode = useSimulationStore((s) => s.params.mode);
  const setParam = useSimulationStore((s) => s.setParam);

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-800">
      <h3 className="mb-3 text-sm font-semibold uppercase text-slate-500 dark:text-slate-400">
        {t('dashboard.model.title')}
      </h3>
      <div className="space-y-2">
        {MODES.map(({ value, labelKey }) => (
          <label
            key={value}
            className={`flex cursor-pointer items-center gap-3 rounded-lg border px-3 py-2 text-sm transition-colors ${
              mode === value
                ? 'border-primary-400 bg-primary-50 text-primary-700 dark:border-primary-600 dark:bg-primary-900/20 dark:text-primary-300'
                : 'border-slate-200 text-slate-600 hover:border-slate-300 dark:border-slate-600 dark:text-slate-300'
            }`}
          >
            <input
              type="radio"
              name="mode"
              value={value}
              checked={mode === value}
              onChange={() => setParam('mode', value)}
              className="accent-primary-600"
            />
            <span>{t(labelKey)}</span>
          </label>
        ))}
      </div>
    </div>
  );
}
