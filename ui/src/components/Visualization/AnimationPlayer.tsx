import { useEffect, useRef, useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PauseIcon, PlayIcon } from '@heroicons/react/24/solid';
import { useSpatialScatter } from '../../hooks/useSpatialData';
import PlotlyChart from './PlotlyChart';

interface Props {
  simulationId?: string;
  timepoints?: number[];
}

export default function AnimationPlayer({ simulationId, timepoints }: Props) {
  const { t } = useTranslation();
  const maxSteps = Math.max(timepoints?.length ?? 48, 1);
  const [currentStep, setCurrentStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(500);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clampedStep = Math.min(currentStep, maxSteps - 1);
  const { data, isLoading, error } = useSpatialScatter({
    simulation_id: simulationId,
    timestep: clampedStep,
    t_max_hours: maxSteps,
  });

  const togglePlay = useCallback(() => {
    setPlaying((p) => !p);
  }, []);

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(() => {
        setCurrentStep((step) => {
          if (step >= maxSteps - 1) {
            setPlaying(false);
            return 0;
          }
          return step + 1;
        });
      }, speed);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [maxSteps, playing, speed]);

  const displayedTime = timepoints?.[clampedStep] ?? clampedStep;
  const displayedMaxTime = timepoints?.[maxSteps - 1] ?? maxSteps;

  return (
    <div data-testid="animation-player">
      <div className="mb-3 flex items-center gap-4">
        <button
          onClick={togglePlay}
          data-testid={playing ? 'animation-pause' : 'animation-play'}
          className="flex items-center gap-1.5 rounded-lg bg-primary-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-primary-700"
        >
          {playing ? <PauseIcon className="h-4 w-4" /> : <PlayIcon className="h-4 w-4" />}
          {playing ? t('common.cancel') : 'Play'}
        </button>

        <input
          type="range"
          min={0}
          max={maxSteps - 1}
          value={clampedStep}
          onChange={(e) => setCurrentStep(Number(e.target.value))}
          data-testid="animation-timestep"
          className="flex-1"
        />
        <span className="text-xs text-slate-500 dark:text-slate-400">
          t = {displayedTime}h / {displayedMaxTime}h
        </span>

        <div className="flex items-center gap-1">
          <label className="text-xs text-slate-500 dark:text-slate-400">{t('viz.speed')}</label>
          <select
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="rounded border border-slate-300 bg-white px-1 py-0.5 text-xs dark:border-slate-600 dark:bg-slate-700 dark:text-slate-200"
          >
            <option value={1000}>0.5x</option>
            <option value={500}>1x</option>
            <option value={250}>2x</option>
            <option value={100}>5x</option>
          </select>
        </div>
      </div>

      <PlotlyChart figure={data} loading={isLoading} error={error ? String(error) : null} />
    </div>
  );
}
