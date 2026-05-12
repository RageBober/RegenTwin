import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import Plot from 'react-plotly.js';
import {
  useAnalysisStatus,
  useCancelAnalysis,
  useRunSensitivity,
  useSobolViz,
  useMorrisViz,
} from '../../hooks/useAnalysis';
import { useSimulationStore } from '../../stores/simulationStore';
import { useSimulationMeta } from '../../hooks/useResults';
import { DEFAULT_SIMULATION_PARAMS } from '../../types/api';
import type { SensitivityResult, SimulationRequest } from '../../types/api';
import MethodInfo from './MethodInfo';
import ParameterPicker from './ParameterPicker';

interface Props {
  prefilledSimulationId?: string;
}

export default function SensitivityView({ prefilledSimulationId }: Props) {
  const { t } = useTranslation();
  const globalParams = useSimulationStore((s) => s.params);
  const [method, setMethod] = useState<'sobol' | 'morris'>('sobol');
  const [nSamples, setNSamples] = useState(256);
  const [selectedParams, setSelectedParams] = useState<string[]>([]);
  const [analysisId, setAnalysisId] = useState<string | null>(null);

  // Локальный snapshot параметров: либо из готовой симуляции, либо из глобального стора
  const [localParams, setLocalParams] = useState<SimulationRequest>(globalParams);
  const { data: simMeta } = useSimulationMeta(prefilledSimulationId);

  useEffect(() => {
    if (prefilledSimulationId && simMeta?.params_json) {
      setLocalParams(simMeta.params_json);
    } else if (!prefilledSimulationId) {
      setLocalParams(globalParams);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- следим за источником данных, не за самим объектом
  }, [prefilledSimulationId, simMeta?.params_json]);

  const runMutation = useRunSensitivity();
  const cancelMutation = useCancelAnalysis();
  const { data: analysisData } = useAnalysisStatus(analysisId);

  const handleRun = () => {
    runMutation.mutate(
      {
        simulation_params: localParams ?? DEFAULT_SIMULATION_PARAMS,
        parameters: selectedParams,
        method,
        n_samples: nSamples,
      },
      {
        onSuccess: (data) => setAnalysisId(data.analysis_id),
      },
    );
  };

  const result = analysisData?.result as SensitivityResult | null;
  const isComplete = analysisData?.status === 'completed' && !!result;

  const sobolVizRequest = isComplete && result?.method === 'sobol' && analysisId
    ? { analysis_id: analysisId } : null;
  const morrisVizRequest = isComplete && result?.method === 'morris' && analysisId
    ? { analysis_id: analysisId } : null;

  const { data: sobolFig, isLoading: sobolLoading } = useSobolViz(sobolVizRequest);
  const { data: morrisFig, isLoading: morrisLoading } = useMorrisViz(morrisVizRequest);

  const vizFig = method === 'sobol' ? sobolFig : morrisFig;
  const vizLoading = method === 'sobol' ? sobolLoading : morrisLoading;

  return (
    <div className="space-y-5">
      {/* Intro card */}
      <div className="card p-4 border-primary-500/15 bg-primary-500/5">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1">
            <h3 className="text-xs font-semibold uppercase tracking-wider
                           text-primary-700 dark:text-primary-300 mb-1.5">
              {t('analysis.sensitivity.title')}
            </h3>
            <p className="text-xs leading-relaxed text-primary-700/80 dark:text-primary-300/70">
              {t('analysis.sensitivity.intro')}
            </p>
          </div>
          <MethodInfo kind="sensitivity" />
        </div>
        <p className="mt-2 text-2xs text-primary-500/60 dark:text-primary-400/50">
          {t('analysis.sensitivity.hint')}
        </p>
      </div>

      {/* Config: method + n_samples */}
      <div className="grid grid-cols-2 gap-3">
        <div className="card p-4">
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-semibold uppercase tracking-wider
                              text-primary-500/60 dark:text-primary-400/50">
              {t('analysis.sensitivity.method')}
            </label>
            <MethodInfo kind={method} />
          </div>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value as 'sobol' | 'morris')}
            className="w-full rounded-lg border border-border bg-surface-1 px-3 py-2
                       text-sm font-mono text-primary-800 dark:text-primary-200
                       focus:outline-none focus:ring-1 focus:ring-primary-500/30"
          >
            <option value="sobol">Sobol</option>
            <option value="morris">Morris</option>
          </select>
        </div>

        <div className="card p-4">
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-semibold uppercase tracking-wider
                              text-primary-500/60 dark:text-primary-400/50">
              {t('analysis.sensitivity.nSamples')}
            </label>
            <MethodInfo kind="nSamples" />
          </div>
          <input
            type="number"
            value={nSamples}
            onChange={(e) => setNSamples(Number(e.target.value))}
            min={64}
            max={4096}
            step={64}
            className="w-full rounded-lg border border-border bg-surface-1 px-3 py-2
                       text-sm font-mono text-primary-800 dark:text-primary-200
                       focus:outline-none focus:ring-1 focus:ring-primary-500/30"
          />
        </div>
      </div>

      {/* Parameters picker (grouped + bulk + presets) */}
      <ParameterPicker value={selectedParams} onChange={setSelectedParams} />

      {/* Run button */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleRun}
          disabled={selectedParams.length === 0 || runMutation.isPending || analysisData?.status === 'running'}
          className="rounded-xl bg-primary-500 px-5 py-2.5 text-sm font-medium text-white
                     hover:bg-primary-600 shadow-glow-sm hover:shadow-glow
                     transition-all duration-200 disabled:opacity-40 disabled:shadow-none"
        >
          {runMutation.isPending ? '...' : t('analysis.sensitivity.run')}
        </button>
        {analysisData?.status === 'running' && analysisId && (
          <button
            onClick={() => cancelMutation.mutate(analysisId)}
            disabled={cancelMutation.isPending}
            className="rounded-xl border border-red-400/20 px-4 py-2 text-sm font-medium text-red-500 hover:bg-red-500/5 transition-colors disabled:opacity-40"
          >
            {cancelMutation.isPending ? '...' : t('common.cancel')}
          </button>
        )}
      </div>

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

      {/* Results */}
      {isComplete && result && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="card p-4"
        >
          <h3 className="text-xs font-semibold uppercase tracking-wider
                        text-primary-500/60 dark:text-primary-400/50 mb-3">
            {t('analysis.sensitivity.results')}
          </h3>
          {result.warning && (
            <p className="mb-3 text-xs text-accent-600 dark:text-accent-400">{result.warning}</p>
          )}
          {result.error ? (
            <p className="text-sm text-red-500">{result.error}</p>
          ) : vizLoading ? (
            <div className="flex items-center justify-center h-[350px]">
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary-500 border-t-transparent" />
            </div>
          ) : vizFig ? (
            <Plot
              data={vizFig.data as Plotly.Data[]}
              layout={{
                ...vizFig.layout as Partial<Plotly.Layout>,
                autosize: true,
              }}
              useResizeHandler
              style={{ width: '100%', height: '400px' }}
              config={{ displayModeBar: false }}
            />
          ) : null}
        </motion.div>
      )}
    </div>
  );
}
