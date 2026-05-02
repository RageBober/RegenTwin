import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowPathIcon, EyeIcon, PlayIcon, StopIcon } from '@heroicons/react/24/solid';
import { useSimulationStore } from '../../stores/simulationStore';
import { useCancelSimulation, useSimulationStatus, useStartSimulation } from '../../hooks/useSimulation';
import { useSimulationWS } from '../../hooks/useSimulationWS';
import { usePopulations } from '../../hooks/useVisualization';
import PlotlyChart from '../Visualization/PlotlyChart';

const fadeVariant = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.3 } },
  exit: { opacity: 0, y: -8, transition: { duration: 0.2 } },
};

export default function SimulationRunner() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const params = useSimulationStore((s) => s.params);
  const activeId = useSimulationStore((s) => s.activeSimulationId);
  const setActiveId = useSimulationStore((s) => s.setActiveSimulationId);

  const startMutation = useStartSimulation();
  const cancelMutation = useCancelSimulation();
  const { data: statusData } = useSimulationStatus(activeId);
  const ws = useSimulationWS(activeId && statusData?.status === 'running' ? activeId : null);

  const { data: previewFigure, isLoading: previewLoading } = usePopulations(null, undefined, activeId && statusData?.status === 'completed' ? activeId : undefined);

  const isComplete = statusData?.status === 'completed' || ws.status === 'complete';
  const isRunning = statusData?.status === 'running' && !isComplete;
  const isFailed = statusData?.status === 'failed' || ws.status === 'failed' || ws.status === 'not_found';
  const isCancelled = statusData?.status === 'cancelled' || ws.status === 'cancelled';
  const isIdle = !activeId || isFailed || isCancelled;

  const progress = isRunning ? ws.progress : isComplete ? 100 : statusData?.progress ?? 0;
  const message = isRunning ? ws.message : statusData?.message ?? ws.message;

  const handleStart = useCallback(() => {
    startMutation.mutate(params, {
      onSuccess: (data) => {
        setActiveId(data.simulation_id);
      },
    });
  }, [params, setActiveId, startMutation]);

  const handleCancel = useCallback(() => {
    if (activeId) cancelMutation.mutate(activeId);
  }, [activeId, cancelMutation]);

  const handleRetry = useCallback(() => {
    setActiveId(null);
    handleStart();
  }, [handleStart, setActiveId]);

  const handleViewResults = useCallback(() => {
    if (activeId) navigate(`/results/${activeId}`);
  }, [activeId, navigate]);

  return (
    <div className="card p-6">
      <AnimatePresence mode="wait">
        {/* ── Idle / Failed / Cancelled ── */}
        {isIdle && (
          <motion.div key="idle" {...fadeVariant} className="flex flex-col items-center gap-4">
            {isFailed && (
              <div className="w-full card bg-red-500/5 border-red-500/15 p-3 text-sm text-red-600 dark:text-red-400">
                {message || t('simulation.failed')}
              </div>
            )}
            {isCancelled && (
              <div className="w-full card bg-accent-500/5 border-accent-500/15 p-3 text-sm text-accent-700 dark:text-accent-400">
                {message || t('simulation.cancelled')}
              </div>
            )}
            <button
              onClick={isFailed || isCancelled ? handleRetry : handleStart}
              disabled={startMutation.isPending}
              data-testid="simulation-run-button"
              className="flex items-center gap-2 rounded-xl
                         bg-primary-500 hover:bg-primary-600
                         px-6 py-3 text-sm font-medium text-white
                         shadow-glow-sm hover:shadow-glow
                         transition-all duration-200
                         disabled:opacity-50 disabled:shadow-none"
            >
              {isFailed || isCancelled
                ? <ArrowPathIcon className="h-4 w-4" />
                : <PlayIcon className="h-4 w-4" />
              }
              {isFailed || isCancelled ? t('common.retry') : t('simulation.run')}
            </button>
          </motion.div>
        )}

        {/* ── Running ── */}
        {isRunning && (
          <motion.div key="running" {...fadeVariant} className="space-y-4" data-testid="simulation-running">

            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-primary-700 dark:text-primary-300">
                {t('simulation.running')}
              </span>
              <span className="font-mono text-xs text-primary-500">{Math.round(progress)}%</span>
            </div>

            {/* Progress bar */}
            <div className="h-2 w-full overflow-hidden rounded-full bg-surface-2">
              <motion.div
                className="h-full rounded-full bg-gradient-to-r from-primary-500 to-primary-400"
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.4, ease: 'easeOut' }}
              />
            </div>

            {message && (
              <p className="text-xs text-primary-900/40 dark:text-primary-100/30">{message}</p>
            )}

            <button
              onClick={handleCancel}
              disabled={cancelMutation.isPending}
              data-testid="simulation-cancel-button"
              className="flex items-center gap-1.5 rounded-lg border border-red-400/20
                         px-3 py-1.5 text-xs font-medium text-red-500
                         hover:bg-red-500/5 transition-colors"
            >
              <StopIcon className="h-3.5 w-3.5" />
              {t('common.cancel')}
            </button>
          </motion.div>
        )}

        {/* ── Complete ── */}
        {isComplete && (
          <motion.div key="complete" {...fadeVariant} className="flex flex-col items-center gap-4">
            <div className="w-full card bg-emerald-500/5 border-emerald-500/15 p-3 text-center
                           text-sm text-emerald-600 dark:text-emerald-400">
              {t('simulation.complete')}
            </div>

            {/* Quick preview chart */}
            {(previewFigure || previewLoading) && (
              <div className="w-full">
                <PlotlyChart figure={previewFigure} loading={previewLoading} error={null} />
              </div>
            )}

            <div className="flex gap-3">
              <button
                onClick={handleViewResults}
                data-testid="simulation-view-results-button"
                className="flex items-center gap-2 rounded-xl
                           bg-primary-500 hover:bg-primary-600
                           px-5 py-2.5 text-sm font-medium text-white
                           shadow-glow-sm hover:shadow-glow transition-all"
              >
                <EyeIcon className="h-4 w-4" />
                {t('simulation.viewResults')}
              </button>
              <button
                onClick={() => setActiveId(null)}
                className="rounded-xl border border-border px-4 py-2.5
                           text-sm font-medium text-primary-700 dark:text-primary-300
                           hover:bg-surface-2 transition-colors"
              >
                {t('simulation.run')}
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
