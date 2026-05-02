import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import Plot from 'react-plotly.js';
import { useAnalysisStatus, useCancelAnalysis, useParameterBounds, useRunSensitivity, useSobolViz, useMorrisViz } from '../../hooks/useAnalysis';
import { useSimulationStore } from '../../stores/simulationStore';
import type { SensitivityResult } from '../../types/api';

export default function SensitivityView() {
  const { t } = useTranslation();
  const simulationParams = useSimulationStore((s) => s.params);
  const [method, setMethod] = useState<'sobol' | 'morris'>('sobol');
  const [nSamples, setNSamples] = useState(256);
  const [selectedParams, setSelectedParams] = useState<string[]>([]);
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [search, setSearch] = useState('');

  const { data: boundsData, isLoading: boundsLoading } = useParameterBounds();
  const runMutation = useRunSensitivity();
  const cancelMutation = useCancelAnalysis();
  const { data: analysisData } = useAnalysisStatus(analysisId);

  const availableParams = useMemo(
    () => boundsData?.bounds.map((b) => b.name) ?? [],
    [boundsData],
  );
  const filteredParams = useMemo(
    () => availableParams.filter((param) => param.toLowerCase().includes(search.trim().toLowerCase())),
    [availableParams, search],
  );

  // Дефолтный выбор первых 4 параметров при загрузке
  useEffect(() => {
    if (availableParams.length > 0 && selectedParams.length === 0) {
      setSelectedParams(availableParams.slice(0, 4));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- инициализация только при первой загрузке bounds
  }, [availableParams.length]);

  const handleRun = () => {
    runMutation.mutate(
      {
        simulation_params: simulationParams,
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

  // Серверные графики: запрашиваем Plotly-фигуру с бэкенда
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
      {/* Hint — анализ не требует файла */}
      <p className="text-xs text-primary-500/60 dark:text-primary-400/50">
        {t('analysis.sensitivity.hint')}
      </p>

      {/* Config */}
      <div className="grid grid-cols-2 gap-3">
        <div className="card p-4">
          <label className="block text-xs font-semibold uppercase tracking-wider
                           text-primary-500/60 dark:text-primary-400/50 mb-2">
            {t('analysis.sensitivity.method')}
          </label>
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
          <label className="block text-xs font-semibold uppercase tracking-wider
                           text-primary-500/60 dark:text-primary-400/50 mb-2">
            {t('analysis.sensitivity.nSamples')}
          </label>
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

      {/* Parameters */}
      <div className="card p-4">
        <p className="mb-3 text-xs text-primary-500/50 dark:text-primary-400/45">
          Это параметры математической модели. Выберите только коэффициенты, влияние которых нужно оценить, и при необходимости отфильтруйте список по имени.
        </p>
        <input
          type="text"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Поиск параметра..."
          className="mb-3 w-full rounded-lg border border-border bg-surface-1 px-3 py-2 text-sm font-mono text-left text-primary-800 dark:text-primary-200"
        />
        <label className="block text-xs font-semibold uppercase tracking-wider
                         text-primary-500/60 dark:text-primary-400/50 mb-3">
          {t('analysis.sensitivity.parameters')}
          {boundsData && (
            <span className="ml-2 text-primary-400/40 font-normal normal-case">
              ({selectedParams.length} из {availableParams.length})
            </span>
          )}
        </label>
        {boundsLoading ? (
          <div className="flex gap-2">
            {[...Array(7)].map((_, i) => (
              <div key={i} className="h-8 w-16 animate-pulse rounded-lg bg-surface-2" />
            ))}
          </div>
        ) : (
        <div className="flex flex-wrap gap-2 max-h-48 overflow-y-auto">
          {filteredParams.map((param) => {
            const selected = selectedParams.includes(param);
            return (
              <button
                key={param}
                onClick={() => {
                  if (selected) setSelectedParams(selectedParams.filter((v) => v !== param));
                  else setSelectedParams([...selectedParams, param]);
                }}
                className={`rounded-lg border px-3 py-1.5 text-xs font-mono font-medium
                           transition-all duration-150
                           ${selected
                             ? 'border-primary-500/25 bg-primary-500/8 text-primary-600 dark:text-primary-400 shadow-inner-glow'
                             : 'border-border text-primary-900/30 dark:text-primary-100/20 hover:bg-surface-2'
                           }`}
              >
                {param}
              </button>
            );
          })}
        </div>
        )}
      </div>

      {/* Run button */}
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
          className="ml-3 rounded-xl border border-red-400/20 px-4 py-2 text-sm font-medium text-red-500 hover:bg-red-500/5 transition-colors disabled:opacity-40"
        >
          {cancelMutation.isPending ? '...' : t('common.cancel')}
        </button>
      )}

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
