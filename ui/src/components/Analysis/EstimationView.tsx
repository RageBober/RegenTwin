import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import Plot from 'react-plotly.js';
import { useRunEstimation, useAnalysisStatus, usePosteriorViz, useConvergenceViz } from '../../hooks/useAnalysis';
import { useSimulationStore } from '../../stores/simulationStore';
import type { EstimationResult } from '../../types/api';
import UploadFCS from '../Upload/UploadFCS';
import MethodInfo from './MethodInfo';

const TARGET_VARIABLES = ['F', 'Ne', 'M1', 'M2', 'P', 'E'] as const;
const METHODS = ['mcmc', 'optimization'] as const;

export default function EstimationView() {
  const { t } = useTranslation();
  const uploadId = useSimulationStore((s) => s.uploadId);
  const [method, setMethod] = useState<'mcmc' | 'optimization'>('mcmc');
  const [targetVariable, setTargetVariable] = useState('F');
  const [nSamples, setNSamples] = useState(500);
  const [analysisId, setAnalysisId] = useState<string | null>(null);

  const runMutation = useRunEstimation();
  const { data: analysisData } = useAnalysisStatus(analysisId);

  const handleRun = () => {
    if (!uploadId) return;
    runMutation.mutate(
      {
        upload_id: uploadId,
        target_variable: targetVariable,
        method,
        n_samples: nSamples,
      },
      {
        onSuccess: (data) => setAnalysisId(data.analysis_id),
      },
    );
  };

  const isComplete = analysisData?.status === 'completed' && !!analysisData.result;
  const isFailed = analysisData?.status === 'failed';
  const result = isComplete ? (analysisData.result as unknown as EstimationResult) : null;
  const failureMessage = isFailed
    ? (analysisData.result as { error?: string } | null)?.error ?? null
    : null;

  // Серверные графики: posterior и convergence
  const hasPosterior = !!result?.posterior_samples;
  const hasDiagnostics = !!result?.diagnostics;

  const posteriorVizRequest = isComplete && hasPosterior && analysisId
    ? { analysis_id: analysisId } : null;
  const convergenceVizRequest = isComplete && hasDiagnostics && analysisId
    ? { analysis_id: analysisId } : null;

  const { data: posteriorFig, isLoading: posteriorLoading } = usePosteriorViz(posteriorVizRequest);
  const { data: convergenceFig, isLoading: convergenceLoading } = useConvergenceViz(convergenceVizRequest);

  return (
    <div className="space-y-5">
      {/* Intro card */}
      <div className="card p-4 border-primary-500/15 bg-primary-500/5">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1">
            <h3 className="text-xs font-semibold uppercase tracking-wider
                           text-primary-700 dark:text-primary-300 mb-1.5">
              {t('analysis.estimation.title')}
            </h3>
            <p className="text-xs leading-relaxed text-primary-700/80 dark:text-primary-300/70">
              {t('analysis.estimation.intro')}
            </p>
          </div>
          <MethodInfo kind="estimation" />
        </div>
        <p className="mt-2 text-2xs text-primary-500/60 dark:text-primary-400/50">
          {t('analysis.estimation.dataFormatHint')}
        </p>
      </div>

      {/* No upload: inline upload widget */}
      {!uploadId && (
        <div className="card p-4 border-accent-500/20 bg-accent-500/5">
          <p className="mb-3 text-sm text-accent-700 dark:text-accent-400">
            {t('analysis.estimation.needUpload')}
          </p>
          <UploadFCS />
        </div>
      )}

      {/* Config */}
      <div className="grid grid-cols-3 gap-3">
        <div className="card p-4">
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-semibold uppercase tracking-wider
                              text-primary-500/60 dark:text-primary-400/50">
              {t('analysis.estimation.method')}
            </label>
            <MethodInfo kind={method} />
          </div>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value as typeof method)}
            className="w-full rounded-lg border border-border bg-surface-1 px-3 py-2
                       text-sm font-mono text-primary-800 dark:text-primary-200
                       focus:outline-none focus:ring-1 focus:ring-primary-500/30"
          >
            {METHODS.map((m) => (
              <option key={m} value={m}>{m.toUpperCase()}</option>
            ))}
          </select>
        </div>

        <div className="card p-4">
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-semibold uppercase tracking-wider
                              text-primary-500/60 dark:text-primary-400/50">
              {t('analysis.estimation.targetVariable')}
            </label>
            <MethodInfo kind="targetVariable" />
          </div>
          <select
            value={targetVariable}
            onChange={(e) => setTargetVariable(e.target.value)}
            className="w-full rounded-lg border border-border bg-surface-1 px-3 py-2
                       text-sm font-mono text-primary-800 dark:text-primary-200
                       focus:outline-none focus:ring-1 focus:ring-primary-500/30"
          >
            {TARGET_VARIABLES.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>

        <div className="card p-4">
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-semibold uppercase tracking-wider
                              text-primary-500/60 dark:text-primary-400/50">
              {t('analysis.estimation.nSamples')}
            </label>
            <MethodInfo kind="nSamples" />
          </div>
          <input
            type="number"
            value={nSamples}
            onChange={(e) => setNSamples(Number(e.target.value))}
            min={100}
            max={10000}
            step={100}
            className="w-full rounded-lg border border-border bg-surface-1 px-3 py-2
                       text-sm font-mono text-primary-800 dark:text-primary-200
                       focus:outline-none focus:ring-1 focus:ring-primary-500/30"
          />
        </div>
      </div>

      {/* Run button */}
      <button
        onClick={handleRun}
        disabled={!uploadId || runMutation.isPending || analysisData?.status === 'running'}
        className="rounded-xl bg-primary-500 px-5 py-2.5 text-sm font-medium text-white
                   hover:bg-primary-600 shadow-glow-sm hover:shadow-glow
                   transition-all duration-200 disabled:opacity-40 disabled:shadow-none"
      >
        {runMutation.isPending ? '...' : t('analysis.estimation.run')}
      </button>

      {/* Progress */}
      {analysisData?.status === 'running' && (
        <div className="space-y-2">
          <div className="flex justify-between text-xs">
            <span className="text-primary-700 dark:text-primary-300">{t('simulation.progress')}</span>
            <span className="font-mono text-primary-500">{Math.round(analysisData.progress)}%</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-surface-2">
            <motion.div
              className="h-full rounded-full bg-gradient-to-r from-primary-500 to-primary-400"
              initial={{ width: 0 }}
              animate={{ width: `${analysisData.progress}%` }}
              transition={{ duration: 0.4 }}
            />
          </div>
        </div>
      )}

      {/* Failed */}
      {isFailed && (
        <div className="card p-4 border-red-500/20 bg-red-500/5 text-sm text-red-600 dark:text-red-400">
          <div className="font-medium">{t('simulation.failed')}</div>
          {failureMessage && (
            <div className="mt-1 font-mono text-xs opacity-80">{failureMessage}</div>
          )}
        </div>
      )}

      {/* Results */}
      {isComplete && result && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4"
        >
          {/* Point estimates table */}
          <div className="card p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider
                          text-primary-500/60 dark:text-primary-400/50 mb-3">
              {t('analysis.estimation.pointEstimates')}
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="border-b border-border">
                    <th className="py-2 pr-4 text-left text-primary-500/60">Parameter</th>
                    <th className="py-2 pr-4 text-right text-primary-500/60">Estimate</th>
                    <th className="py-2 pr-4 text-right text-primary-500/60">CI Lower</th>
                    <th className="py-2 text-right text-primary-500/60">CI Upper</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.point_estimates).map(([param, value]) => (
                    <tr key={param} className="border-b border-border/50">
                      <td className="py-1.5 pr-4 text-primary-800 dark:text-primary-200">{param}</td>
                      <td className="py-1.5 pr-4 text-right">{typeof value === 'number' ? value.toFixed(4) : value}</td>
                      <td className="py-1.5 pr-4 text-right text-primary-500/60">
                        {result.ci_lower[param]?.toFixed(4) ?? '-'}
                      </td>
                      <td className="py-1.5 text-right text-primary-500/60">
                        {result.ci_upper[param]?.toFixed(4) ?? '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {/* Fit statistics */}
            <div className="mt-3 flex gap-4 text-xs text-primary-500/60">
              {result.log_likelihood != null && (
                <span>LL: {result.log_likelihood.toFixed(2)}</span>
              )}
              {result.aic != null && <span>AIC: {result.aic.toFixed(2)}</span>}
              {result.bic != null && <span>BIC: {result.bic.toFixed(2)}</span>}
              <span>{result.elapsed_seconds.toFixed(1)}s</span>
            </div>
          </div>

          {/* Posterior distributions */}
          {hasPosterior && (
            <div className="card p-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider
                            text-primary-500/60 dark:text-primary-400/50 mb-3">
                Posterior Distributions
              </h3>
              {posteriorLoading ? (
                <div className="flex items-center justify-center h-[350px]">
                  <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary-500 border-t-transparent" />
                </div>
              ) : posteriorFig ? (
                <Plot
                  data={posteriorFig.data as Plotly.Data[]}
                  layout={{
                    ...posteriorFig.layout as Partial<Plotly.Layout>,
                    autosize: true,
                  }}
                  useResizeHandler
                  style={{ width: '100%', height: '500px' }}
                  config={{ displayModeBar: false }}
                />
              ) : null}
            </div>
          )}

          {/* Convergence diagnostics */}
          {hasDiagnostics && (
            <div className="card p-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider
                            text-primary-500/60 dark:text-primary-400/50 mb-3">
                Convergence Diagnostics
              </h3>
              {result.diagnostics && !result.diagnostics.converged && (
                <p className="mb-2 text-xs text-accent-600 dark:text-accent-400">
                  {result.diagnostics.warnings?.join('; ')}
                </p>
              )}
              {convergenceLoading ? (
                <div className="flex items-center justify-center h-[350px]">
                  <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary-500 border-t-transparent" />
                </div>
              ) : convergenceFig ? (
                <Plot
                  data={convergenceFig.data as Plotly.Data[]}
                  layout={{
                    ...convergenceFig.layout as Partial<Plotly.Layout>,
                    autosize: true,
                  }}
                  useResizeHandler
                  style={{ width: '100%', height: '500px' }}
                  config={{ displayModeBar: false }}
                />
              ) : null}
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}
