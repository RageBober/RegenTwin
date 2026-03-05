import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import Plot from 'react-plotly.js';
import { useRunSensitivity, useAnalysisStatus } from '../../hooks/useAnalysis';
import { DEFAULT_SIMULATION_PARAMS } from '../../types/api';

const AVAILABLE_PARAMS = ['r', 'K', 'alpha_prp', 'beta_pemf', 'delta_Ne', 'delta_M', 'sigma'];

export default function SensitivityView() {
  const { t } = useTranslation();
  const [method, setMethod] = useState<'sobol' | 'morris'>('sobol');
  const [nSamples, setNSamples] = useState(256);
  const [selectedParams, setSelectedParams] = useState(['r', 'K', 'alpha_prp', 'beta_pemf']);
  const [analysisId, setAnalysisId] = useState<string | null>(null);

  const runMutation = useRunSensitivity();
  const { data: analysisData } = useAnalysisStatus(analysisId);

  const handleRun = () => {
    runMutation.mutate(
      {
        simulation_params: DEFAULT_SIMULATION_PARAMS,
        parameters: selectedParams,
        method,
        n_samples: nSamples,
      },
      {
        onSuccess: (data) => setAnalysisId(data.analysis_id),
      },
    );
  };

  const isComplete = analysisData?.status === 'completed';
  const result = analysisData?.result as Record<string, Record<string, number>> | null;

  return (
    <div className="space-y-6">
      {/* Configuration */}
      <div className="grid grid-cols-3 gap-4">
        <div>
          <label className="mb-1 block text-xs font-medium text-slate-600 dark:text-slate-300">
            {t('analysis.sensitivity.method')}
          </label>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value as 'sobol' | 'morris')}
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-600 dark:bg-slate-700 dark:text-slate-200"
          >
            <option value="sobol">Sobol</option>
            <option value="morris">Morris</option>
          </select>
        </div>

        <div>
          <label className="mb-1 block text-xs font-medium text-slate-600 dark:text-slate-300">
            {t('analysis.sensitivity.nSamples')}
          </label>
          <input
            type="number"
            value={nSamples}
            onChange={(e) => setNSamples(Number(e.target.value))}
            min={64}
            max={4096}
            step={64}
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-600 dark:bg-slate-700 dark:text-slate-200"
          />
        </div>

        <div className="flex items-end">
          <button
            onClick={handleRun}
            disabled={runMutation.isPending || (analysisData?.status === 'running')}
            className="w-full rounded-lg bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700 disabled:opacity-50"
          >
            {runMutation.isPending ? '...' : t('analysis.sensitivity.run')}
          </button>
        </div>
      </div>

      {/* Parameter selection */}
      <div>
        <label className="mb-2 block text-xs font-medium text-slate-600 dark:text-slate-300">
          {t('analysis.sensitivity.parameters')}
        </label>
        <div className="flex flex-wrap gap-2">
          {AVAILABLE_PARAMS.map((p) => (
            <label
              key={p}
              className={`flex cursor-pointer items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs ${
                selectedParams.includes(p)
                  ? 'border-primary-400 bg-primary-50 text-primary-700 dark:border-primary-600 dark:bg-primary-900/20 dark:text-primary-300'
                  : 'border-slate-200 text-slate-500 dark:border-slate-600 dark:text-slate-400'
              }`}
            >
              <input
                type="checkbox"
                checked={selectedParams.includes(p)}
                onChange={(e) => {
                  if (e.target.checked) {
                    setSelectedParams([...selectedParams, p]);
                  } else {
                    setSelectedParams(selectedParams.filter((x) => x !== p));
                  }
                }}
                className="hidden"
              />
              {p}
            </label>
          ))}
        </div>
      </div>

      {/* Progress */}
      {analysisData?.status === 'running' && (
        <div className="space-y-2">
          <div className="flex justify-between text-xs text-slate-500">
            <span>{t('simulation.progress')}</span>
            <span>{Math.round(analysisData.progress)}%</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
            <div
              className="h-full rounded-full bg-primary-600 transition-all"
              style={{ width: `${analysisData.progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Results */}
      {isComplete && result && (
        <div className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-800">
          <h3 className="mb-3 text-sm font-semibold text-slate-700 dark:text-slate-200">
            {t('analysis.sensitivity.results')}
          </h3>
          <Plot
            data={[
              {
                type: 'bar' as const,
                x: Object.keys(result),
                y: Object.values(result).map((v) => v.S1 ?? v.mu_star ?? 0),
                name: method === 'sobol' ? 'S1' : 'μ*',
                marker: { color: '#3b82f6' },
              },
              ...(method === 'sobol'
                ? [
                    {
                      type: 'bar' as const,
                      x: Object.keys(result),
                      y: Object.values(result).map((v) => v.ST ?? 0),
                      name: 'ST',
                      marker: { color: '#93c5fd' },
                    },
                  ]
                : []),
            ]}
            layout={{
              autosize: true,
              barmode: 'group',
              xaxis: { title: { text: t('analysis.sensitivity.parameters') } },
              yaxis: { title: { text: method === 'sobol' ? t('analysis.sensitivity.s1') : 'μ*' } },
              margin: { l: 60, r: 20, t: 20, b: 60 },
            }}
            useResizeHandler
            style={{ width: '100%', height: '350px' }}
            config={{ displayModeBar: false }}
          />
        </div>
      )}
    </div>
  );
}
