import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { PlayIcon, StopIcon, ArrowPathIcon, EyeIcon } from '@heroicons/react/24/solid';
import { useSimulationStore } from '../../stores/simulationStore';
import { useStartSimulation, useCancelSimulation, useSimulationStatus } from '../../hooks/useSimulation';
import { useSimulationWS } from '../../hooks/useSimulationWS';

export default function SimulationRunner() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const params = useSimulationStore((s) => s.params);
  const activeId = useSimulationStore((s) => s.activeSimulationId);
  const setActiveId = useSimulationStore((s) => s.setActiveSimulationId);

  const startMutation = useStartSimulation();
  const cancelMutation = useCancelSimulation();
  const { data: statusData } = useSimulationStatus(activeId);
  const ws = useSimulationWS(
    activeId && statusData?.status === 'running' ? activeId : null,
  );

  const isRunning = statusData?.status === 'running';
  const isComplete = statusData?.status === 'completed';
  const isFailed = statusData?.status === 'failed';
  const isCancelled = statusData?.status === 'cancelled';
  const isIdle = !activeId || isFailed || isCancelled;

  const progress = isRunning ? ws.progress : (isComplete ? 100 : 0);
  const message = isRunning ? ws.message : '';

  const handleStart = useCallback(() => {
    startMutation.mutate(params, {
      onSuccess: (data) => {
        setActiveId(data.simulation_id);
      },
    });
  }, [params, startMutation, setActiveId]);

  const handleCancel = useCallback(() => {
    if (activeId) {
      cancelMutation.mutate(activeId);
    }
  }, [activeId, cancelMutation]);

  const handleRetry = useCallback(() => {
    setActiveId(null);
    handleStart();
  }, [setActiveId, handleStart]);

  const handleViewResults = useCallback(() => {
    if (activeId) {
      navigate(`/results/${activeId}`);
    }
  }, [activeId, navigate]);

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-6 dark:border-slate-700 dark:bg-slate-800">
      {/* Idle state */}
      {isIdle && (
        <div className="flex flex-col items-center gap-4">
          {isFailed && (
            <div className="w-full rounded-lg bg-red-50 p-3 text-sm text-red-700 dark:bg-red-900/20 dark:text-red-300">
              {statusData?.message || t('simulation.failed')}
            </div>
          )}
          {isCancelled && (
            <div className="w-full rounded-lg bg-yellow-50 p-3 text-sm text-yellow-700 dark:bg-yellow-900/20 dark:text-yellow-300">
              {t('simulation.cancelled')}
            </div>
          )}
          <button
            onClick={isFailed || isCancelled ? handleRetry : handleStart}
            disabled={startMutation.isPending}
            className="flex items-center gap-2 rounded-lg bg-primary-600 px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-primary-700 disabled:opacity-50"
          >
            {isFailed || isCancelled ? (
              <ArrowPathIcon className="h-5 w-5" />
            ) : (
              <PlayIcon className="h-5 w-5" />
            )}
            {isFailed || isCancelled ? t('common.retry') : t('simulation.run')}
          </button>
        </div>
      )}

      {/* Running state */}
      {isRunning && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-slate-700 dark:text-slate-200">
              {t('simulation.running')}
            </span>
            <span className="text-sm text-slate-500 dark:text-slate-400">
              {Math.round(progress)}%
            </span>
          </div>

          {/* Progress bar */}
          <div className="h-3 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
            <div
              className="h-full rounded-full bg-primary-600 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>

          {message && (
            <p className="text-xs text-slate-500 dark:text-slate-400">{message}</p>
          )}

          <button
            onClick={handleCancel}
            disabled={cancelMutation.isPending}
            className="flex items-center gap-2 rounded-lg border border-red-300 px-4 py-2 text-sm font-medium text-red-600 transition-colors hover:bg-red-50 dark:border-red-700 dark:text-red-400 dark:hover:bg-red-900/20"
          >
            <StopIcon className="h-4 w-4" />
            {t('common.cancel')}
          </button>
        </div>
      )}

      {/* Complete state */}
      {isComplete && (
        <div className="flex flex-col items-center gap-4">
          <div className="w-full rounded-lg bg-green-50 p-3 text-center text-sm text-green-700 dark:bg-green-900/20 dark:text-green-300">
            {t('simulation.complete')}
          </div>
          <div className="flex gap-3">
            <button
              onClick={handleViewResults}
              className="flex items-center gap-2 rounded-lg bg-primary-600 px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-primary-700"
            >
              <EyeIcon className="h-4 w-4" />
              {t('simulation.viewResults')}
            </button>
            <button
              onClick={() => setActiveId(null)}
              className="rounded-lg border border-slate-300 px-4 py-2.5 text-sm font-medium text-slate-600 transition-colors hover:bg-slate-50 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-700"
            >
              {t('simulation.run')}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
