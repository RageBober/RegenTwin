import { useTranslation } from 'react-i18next';
import { useSimulationStore } from '../../stores/simulationStore';
import NumberInput from '../common/NumberInput';

export default function TherapyConfig() {
  const { t } = useTranslation();
  const params = useSimulationStore((s) => s.params);
  const setParam = useSimulationStore((s) => s.setParam);

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-800">
      <h3 className="mb-3 text-sm font-semibold uppercase text-slate-500 dark:text-slate-400">
        {t('dashboard.params.title')}
      </h3>

      <div className="space-y-4">
        {/* Initial Conditions — Populations */}
        <details open>
          <summary className="cursor-pointer text-xs font-medium text-slate-600 dark:text-slate-300">
            {t('dashboard.params.populations')}
          </summary>
          <div className="mt-2 space-y-1.5">
            <NumberInput label={t('dashboard.variables.P')} value={params.P0} onChange={(v) => setParam('P0', v)} min={0} step={10} />
            <NumberInput label={t('dashboard.variables.Ne')} value={params.Ne0} onChange={(v) => setParam('Ne0', v)} min={0} step={10} />
            <NumberInput label={t('dashboard.variables.M1')} value={params.M1_0} onChange={(v) => setParam('M1_0', v)} min={0} step={10} />
            <NumberInput label={t('dashboard.variables.M2')} value={params.M2_0} onChange={(v) => setParam('M2_0', v)} min={0} step={1} />
            <NumberInput label={t('dashboard.variables.F')} value={params.F0} onChange={(v) => setParam('F0', v)} min={0} step={10} />
            <NumberInput label={t('dashboard.variables.Mf')} value={params.Mf0} onChange={(v) => setParam('Mf0', v)} min={0} step={1} />
            <NumberInput label={t('dashboard.variables.E')} value={params.E0} onChange={(v) => setParam('E0', v)} min={0} step={5} />
            <NumberInput label={t('dashboard.variables.S')} value={params.S0} onChange={(v) => setParam('S0', v)} min={0} step={5} />
          </div>
        </details>

        {/* Cytokines & Auxiliary */}
        <details>
          <summary className="cursor-pointer text-xs font-medium text-slate-600 dark:text-slate-300">
            {t('dashboard.params.cytokines')} / {t('dashboard.params.auxiliary')}
          </summary>
          <div className="mt-2 space-y-1.5">
            <NumberInput label={t('dashboard.variables.C_TNF')} value={params.C_TNF0} onChange={(v) => setParam('C_TNF0', v)} min={0} step={0.5} />
            <NumberInput label={t('dashboard.variables.C_IL10')} value={params.C_IL10_0} onChange={(v) => setParam('C_IL10_0', v)} min={0} step={0.1} />
            <NumberInput label={t('dashboard.variables.D')} value={params.D0} onChange={(v) => setParam('D0', v)} min={0} step={1} />
            <NumberInput label={t('dashboard.variables.O2')} value={params.O2_0} onChange={(v) => setParam('O2_0', v)} min={0} step={5} />
          </div>
        </details>

        {/* Time */}
        <details open>
          <summary className="cursor-pointer text-xs font-medium text-slate-600 dark:text-slate-300">
            {t('dashboard.params.timeSettings')}
          </summary>
          <div className="mt-2 space-y-1.5">
            <NumberInput label={t('dashboard.params.tMax')} value={params.t_max_hours} onChange={(v) => setParam('t_max_hours', v)} min={24} max={1440} step={24} />
            <div className="flex items-center justify-between gap-2">
              <label className="text-xs text-slate-600 dark:text-slate-300">{t('dashboard.params.dt')}</label>
              <select
                value={params.dt}
                onChange={(e) => setParam('dt', Number(e.target.value))}
                className="w-24 rounded border border-slate-300 bg-white px-2 py-1 text-xs dark:border-slate-600 dark:bg-slate-700 dark:text-slate-200"
              >
                <option value={0.01}>0.01</option>
                <option value={0.05}>0.05</option>
                <option value={0.1}>0.1</option>
                <option value={0.5}>0.5</option>
              </select>
            </div>
          </div>
        </details>

        {/* Therapy */}
        <details open>
          <summary className="cursor-pointer text-xs font-medium text-slate-600 dark:text-slate-300">
            {t('dashboard.params.therapy')}
          </summary>
          <div className="mt-2 space-y-2">
            {/* PRP */}
            <label className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300">
              <input
                type="checkbox"
                checked={params.prp_enabled}
                onChange={(e) => setParam('prp_enabled', e.target.checked)}
                className="accent-primary-600"
              />
              {t('dashboard.params.prpEnabled')}
            </label>
            {params.prp_enabled && (
              <NumberInput
                label={t('dashboard.params.prpIntensity')}
                value={params.prp_intensity}
                onChange={(v) => setParam('prp_intensity', v)}
                min={0}
                max={2}
                step={0.1}
              />
            )}

            {/* PEMF */}
            <label className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300">
              <input
                type="checkbox"
                checked={params.pemf_enabled}
                onChange={(e) => setParam('pemf_enabled', e.target.checked)}
                className="accent-primary-600"
              />
              {t('dashboard.params.pemfEnabled')}
            </label>
            {params.pemf_enabled && (
              <>
                <NumberInput
                  label={t('dashboard.params.pemfFrequency')}
                  value={params.pemf_frequency}
                  onChange={(v) => setParam('pemf_frequency', v)}
                  min={1}
                  max={100}
                  step={1}
                />
                <NumberInput
                  label={t('dashboard.params.pemfIntensity')}
                  value={params.pemf_intensity}
                  onChange={(v) => setParam('pemf_intensity', v)}
                  min={0}
                  max={2}
                  step={0.1}
                />
              </>
            )}
          </div>
        </details>

        {/* Monte Carlo */}
        <details>
          <summary className="cursor-pointer text-xs font-medium text-slate-600 dark:text-slate-300">
            {t('dashboard.params.monteCarlo')}
          </summary>
          <div className="mt-2 space-y-1.5">
            <NumberInput label={t('dashboard.params.nTrajectories')} value={params.n_trajectories} onChange={(v) => setParam('n_trajectories', v)} min={1} max={1000} step={1} />
            <NumberInput label={t('dashboard.params.randomSeed')} value={params.random_seed ?? 42} onChange={(v) => setParam('random_seed', v)} min={0} step={1} />
          </div>
        </details>
      </div>
    </div>
  );
}
