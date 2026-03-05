import { useState, useEffect, useRef, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PlayIcon, PauseIcon } from '@heroicons/react/24/solid';
import { useSpatialScatter } from '../../hooks/useSpatialData';
import PlotlyChart from './PlotlyChart';

const TOTAL_STEPS = 48;

export default function AnimationPlayer() {
  const { t } = useTranslation();
  const [currentStep, setCurrentStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(500); // ms per frame
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const { data, isLoading, error } = useSpatialScatter({
    timestep: currentStep,
    t_max_hours: TOTAL_STEPS,
  });

  const togglePlay = useCallback(() => {
    setPlaying((p) => !p);
  }, []);

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(() => {
        setCurrentStep((s) => {
          if (s >= TOTAL_STEPS - 1) {
            setPlaying(false);
            return 0;
          }
          return s + 1;
        });
      }, speed);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [playing, speed]);

  return (
    <div>
      <div className="mb-3 flex items-center gap-4">
        <button
          onClick={togglePlay}
          className="flex items-center gap-1.5 rounded-lg bg-primary-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-primary-700"
        >
          {playing ? <PauseIcon className="h-4 w-4" /> : <PlayIcon className="h-4 w-4" />}
          {playing ? t('common.cancel') : 'Play'}
        </button>

        <input
          type="range"
          min={0}
          max={TOTAL_STEPS - 1}
          value={currentStep}
          onChange={(e) => setCurrentStep(Number(e.target.value))}
          className="flex-1"
        />
        <span className="text-xs text-slate-500 dark:text-slate-400">
          t = {currentStep}h / {TOTAL_STEPS}h
        </span>

        <div className="flex items-center gap-1">
          <label className="text-xs text-slate-500 dark:text-slate-400">Speed:</label>
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

      <PlotlyChart
        figure={data}
        loading={isLoading}
        error={error ? String(error) : null}
      />
    </div>
  );
}
