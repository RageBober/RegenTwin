import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import { useRunValidation } from '../../hooks/useAnalysis';
import type { ValidationDatasetId, ValidationResponse } from '../../types/api';
import MethodInfo from './MethodInfo';

const DATASETS: ValidationDatasetId[] = [
  'literature-xue2009',
  'literature-flegg2010',
  'HPA-skin-baseline',
  'GSE28914',
];

function scoreColor(score: number): { bg: string; text: string; label: 'High' | 'Mid' | 'Low' } {
  if (score >= 0.8) return { bg: 'bg-emerald-500/15', text: 'text-emerald-600 dark:text-emerald-400', label: 'High' };
  if (score >= 0.5) return { bg: 'bg-accent-500/15', text: 'text-accent-600 dark:text-accent-400', label: 'Mid' };
  return { bg: 'bg-red-500/15', text: 'text-red-600 dark:text-red-400', label: 'Low' };
}

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return '-';
  if (typeof v === 'number') return Number.isFinite(v) ? v.toFixed(3) : String(v);
  if (typeof v === 'string') return v;
  if (typeof v === 'boolean') return v ? '✓' : '✗';
  return JSON.stringify(v);
}

export default function ValidationView() {
  const { t } = useTranslation();
  const [datasetId, setDatasetId] = useState<ValidationDatasetId>('literature-xue2009');
  const [tMax, setTMax] = useState(720);
  const [dt, setDt] = useState(0.1);
  const [result, setResult] = useState<ValidationResponse | null>(null);

  const runMutation = useRunValidation();

  const handleRun = () => {
    runMutation.mutate(
      { dataset_id: datasetId, t_max: tMax, dt },
      {
        onSuccess: (data) => setResult(data),
      },
    );
  };

  const score = result ? scoreColor(result.overall_score) : null;
  const scoreLabelKey = score ? `analysis.validation.score${score.label}` : '';

  return (
    <div className="space-y-5">
      {/* Intro card */}
      <div className="card p-4 border-primary-500/15 bg-primary-500/5">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1">
            <h3 className="text-xs font-semibold uppercase tracking-wider
                           text-primary-700 dark:text-primary-300 mb-1.5">
              {t('analysis.validation.title')}
            </h3>
            <p className="text-xs leading-relaxed text-primary-700/80 dark:text-primary-300/70">
              {t('analysis.validation.intro')}
            </p>
          </div>
          <MethodInfo kind="validation" />
        </div>
      </div>

      {/* Config: dataset + t_max + dt */}
      <div className="grid grid-cols-3 gap-3">
        <div className="card p-4">
          <label className="block text-xs font-semibold uppercase tracking-wider
                            text-primary-500/60 dark:text-primary-400/50 mb-2">
            {t('analysis.validation.dataset')}
          </label>
          <select
            value={datasetId}
            onChange={(e) => setDatasetId(e.target.value as ValidationDatasetId)}
            className="w-full rounded-lg border border-border bg-surface-1 px-3 py-2
                       text-sm font-mono text-primary-800 dark:text-primary-200
                       focus:outline-none focus:ring-1 focus:ring-primary-500/30"
          >
            {DATASETS.map((d) => (
              <option key={d} value={d}>{t(`analysis.datasets.${d}.name`)}</option>
            ))}
          </select>
          <p className="mt-2 text-2xs text-primary-500/60 dark:text-primary-400/50 leading-relaxed">
            {t(`analysis.datasets.${datasetId}.desc`)}
          </p>
        </div>

        <div className="card p-4">
          <label className="block text-xs font-semibold uppercase tracking-wider
                            text-primary-500/60 dark:text-primary-400/50 mb-2">
            {t('analysis.validation.tMax')}
          </label>
          <input
            type="number"
            value={tMax}
            onChange={(e) => setTMax(Number(e.target.value))}
            min={1}
            max={2000}
            step={1}
            className="w-full rounded-lg border border-border bg-surface-1 px-3 py-2
                       text-sm font-mono text-primary-800 dark:text-primary-200
                       focus:outline-none focus:ring-1 focus:ring-primary-500/30"
          />
        </div>

        <div className="card p-4">
          <label className="block text-xs font-semibold uppercase tracking-wider
                            text-primary-500/60 dark:text-primary-400/50 mb-2">
            {t('analysis.validation.dt')}
          </label>
          <input
            type="number"
            value={dt}
            onChange={(e) => setDt(Number(e.target.value))}
            min={0.01}
            max={10}
            step={0.01}
            className="w-full rounded-lg border border-border bg-surface-1 px-3 py-2
                       text-sm font-mono text-primary-800 dark:text-primary-200
                       focus:outline-none focus:ring-1 focus:ring-primary-500/30"
          />
        </div>
      </div>

      {/* Run button */}
      <button
        onClick={handleRun}
        disabled={runMutation.isPending}
        className="rounded-xl bg-primary-500 px-5 py-2.5 text-sm font-medium text-white
                   hover:bg-primary-600 shadow-glow-sm hover:shadow-glow
                   transition-all duration-200 disabled:opacity-40 disabled:shadow-none"
      >
        {runMutation.isPending ? '...' : t('analysis.validation.run')}
      </button>

      {/* Loading spinner */}
      {runMutation.isPending && (
        <div className="flex items-center gap-3 text-xs text-primary-500/70">
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary-500 border-t-transparent" />
          <span>{t('common.loading')}</span>
        </div>
      )}

      {/* Error */}
      {runMutation.isError && (
        <div className="card p-4 border-red-500/20 bg-red-500/5 text-sm text-red-600 dark:text-red-400">
          {(runMutation.error as Error)?.message ?? t('common.error')}
        </div>
      )}

      {/* Result */}
      {result && score && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4"
        >
          {/* Overall score */}
          <div className="card p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider
                          text-primary-500/60 dark:text-primary-400/50 mb-3">
              {t('analysis.validation.results')}
            </h3>
            <div className="flex items-baseline gap-4">
              <div className={`px-3 py-2 rounded-lg ${score.bg}`}>
                <div className={`text-2xl font-bold font-mono ${score.text}`}>
                  {result.overall_score.toFixed(3)}
                </div>
                <div className={`text-2xs font-medium uppercase tracking-wider ${score.text}`}>
                  {t(scoreLabelKey)}
                </div>
              </div>
              <div className="flex flex-col gap-1 text-xs text-primary-500/60">
                <span>{t('analysis.validation.overallScore')}</span>
                <span className="font-mono">
                  {t('analysis.validation.elapsed')}: {result.elapsed_seconds.toFixed(2)}s
                </span>
              </div>
            </div>
            {result.errors && result.errors.length > 0 && (
              <div className="mt-3 rounded-lg border border-red-500/20 bg-red-500/5 p-3">
                <div className="text-2xs font-semibold uppercase tracking-wider text-red-600 dark:text-red-400 mb-1">
                  {t('analysis.validation.errors')}
                </div>
                <ul className="list-disc list-inside text-xs text-red-600/90 dark:text-red-400/90 space-y-0.5">
                  {result.errors.map((err, i) => (
                    <li key={i} className="font-mono">{err}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Initial conditions */}
          {Object.keys(result.initial_conditions).length > 0 && (
            <div className="card p-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider
                            text-primary-500/60 dark:text-primary-400/50 mb-3">
                {t('analysis.validation.initialConditions')}
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="py-2 pr-4 text-left text-primary-500/60">Variable</th>
                      <th className="py-2 text-right text-primary-500/60">Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(result.initial_conditions).map(([k, v]) => (
                      <tr key={k} className="border-b border-border/50">
                        <td className="py-1.5 pr-4 text-primary-800 dark:text-primary-200">{k}</td>
                        <td className="py-1.5 text-right text-primary-700 dark:text-primary-300">
                          {formatValue(v)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Metrics */}
          {result.validation && Object.keys(result.validation).length > 0 && (
            <div className="card p-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider
                            text-primary-500/60 dark:text-primary-400/50 mb-3">
                {t('analysis.validation.metrics')}
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono">
                  <tbody>
                    {Object.entries(result.validation).map(([k, v]) => (
                      <tr key={k} className="border-b border-border/50">
                        <td className="py-1.5 pr-4 text-primary-800 dark:text-primary-200">{k}</td>
                        <td className="py-1.5 text-right text-primary-700 dark:text-primary-300">
                          {formatValue(v)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </motion.div>
      )}

      {!result && !runMutation.isPending && !runMutation.isError && (
        <p className="text-xs italic text-primary-500/50 dark:text-primary-400/40">
          {t('analysis.validation.noResult')}
        </p>
      )}
    </div>
  );
}
