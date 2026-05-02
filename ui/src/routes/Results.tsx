import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useParams } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  UsersIcon,
  BeakerIcon,
  CubeIcon,
  ClockIcon,
  ArrowsRightLeftIcon,
  MapIcon,
  FireIcon,
  PlayIcon,
  ViewfinderCircleIcon,
} from '@heroicons/react/24/outline';
import { useResults, useSimulationMeta } from '../hooks/useResults';
import { toSimulationParams } from '../types/api';
import type { ExportFormat, SimulationMode, SimulationRequest, SimulationStatusResponse } from '../types/api';
import PopulationCharts from '../components/Visualization/PopulationCharts';
import CytokineCharts from '../components/Visualization/CytokineCharts';
import ECMCharts from '../components/Visualization/ECMCharts';
import PhaseTimeline from '../components/Visualization/PhaseTimeline';
import TherapyComparison from '../components/Visualization/TherapyComparison';
import CellHeatmap from '../components/Visualization/CellHeatmap';
import InflammationMap from '../components/Visualization/InflammationMap';
import AnimationPlayer from '../components/Visualization/AnimationPlayer';
import SpatialView3D from '../components/Visualization/SpatialView3D';
import ExportPanel from '../components/Results/ExportPanel';

const TAB_CONFIG: Record<SimulationMode, readonly string[]> = {
  mvp: ['populations', 'cytokines'],
  extended: ['populations', 'cytokines', 'ecm', 'phases', 'comparison'],
  abm: ['populations', 'heatmap', 'inflammation', 'animation', 'spatial3d'],
  integrated: ['populations', 'cytokines', 'ecm', 'phases', 'comparison'],
};

const TAB_ICONS: Record<string, React.ComponentType<React.SVGProps<SVGSVGElement>>> = {
  populations: UsersIcon,
  cytokines: BeakerIcon,
  ecm: CubeIcon,
  phases: ClockIcon,
  comparison: ArrowsRightLeftIcon,
  heatmap: MapIcon,
  inflammation: FireIcon,
  animation: PlayIcon,
  spatial3d: ViewfinderCircleIcon,
};

type Tab = (typeof TAB_CONFIG)[SimulationMode][number];

const STATUS_STYLE: Record<string, string> = {
  completed: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400',
  running:   'bg-accent-500/10 text-accent-600 dark:text-accent-400',
  failed:    'bg-red-500/10 text-red-600 dark:text-red-400',
  pending:   'bg-primary-500/10 text-primary-600 dark:text-primary-400',
  cancelled: 'bg-primary-900/5 text-primary-900/40 dark:text-primary-100/30',
};

function getErrorMessage(error: unknown): string {
  if (typeof error === 'string') return error;
  if (error && typeof error === 'object') {
    const response = (error as { response?: { data?: { detail?: unknown } } }).response;
    if (typeof response?.data?.detail === 'string' && response.data.detail.length > 0)
      return response.data.detail;
    const message = (error as { message?: unknown }).message;
    if (typeof message === 'string' && message.length > 0) return message;
  }
  return 'Unknown error';
}

export default function Results() {
  const { t } = useTranslation();
  const { id } = useParams<{ id: string }>();
  const [activeTab, setActiveTab] = useState<Tab>('populations');

  const {
    data: results,
    isLoading: resultsLoading,
    isError: resultsError,
    error: resultsFailure,
    refetch: refetchResults,
  } = useResults(id);
  const {
    data: meta,
    isLoading: metaLoading,
    isError: metaError,
    error: metaFailure,
    refetch: refetchMeta,
  } = useSimulationMeta(id);

  const simulationRequest = (meta as SimulationStatusResponse | undefined)?.params_json as
    | SimulationRequest
    | undefined;
  const vizParams = useMemo(
    () => (simulationRequest ? toSimulationParams(simulationRequest) : null),
    [simulationRequest],
  );
  const mode = results?.mode ?? simulationRequest?.mode ?? 'extended';
  const isMonteCarlo = (results?.metadata?.n_trajectories as number | undefined) != null
    && (results?.metadata?.n_trajectories as number) > 1;
  const isExtendedMC = isMonteCarlo && (results?.metadata?.extended_mc as boolean | undefined) === true;
  const mcUnsupportedTabs = isExtendedMC ? ['comparison'] : ['ecm', 'phases', 'comparison'];
  const availableTabs = isMonteCarlo
    ? TAB_CONFIG[mode].filter((tab) => !mcUnsupportedTabs.includes(tab))
    : TAB_CONFIG[mode];
  const resolvedActiveTab = availableTabs.includes(activeTab) ? activeTab : availableTabs[0];
  const createdAtLabel = meta?.created_at ? new Date(meta.created_at).toLocaleString() : 'n/a';
  const errorMessage = resultsError
    ? getErrorMessage(resultsFailure)
    : metaError
      ? getErrorMessage(metaFailure)
      : null;

  const handleRetry = () => {
    void refetchResults();
    void refetchMeta();
  };

  if (metaLoading || resultsLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary-200 border-t-primary-500" />
      </div>
    );
  }

  if (errorMessage) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <div className="card max-w-md p-6 text-center">
          <h2 className="text-base font-semibold text-primary-800 dark:text-primary-200">
            {t('results.states.loadFailedTitle')}
          </h2>
          <p className="mt-2 text-sm text-primary-900/50 dark:text-primary-100/40">
            {errorMessage}
          </p>
          <button
            className="mt-4 rounded-xl bg-primary-500 px-4 py-2 text-sm font-medium text-white
                       hover:bg-primary-600 transition-colors"
            onClick={handleRetry}
          >
            {t('common.retry')}
          </button>
        </div>
      </div>
    );
  }

  if (!id || !vizParams || !results) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <div className="card max-w-md p-6 text-center">
          <h2 className="text-base font-semibold text-primary-800 dark:text-primary-200">
            {t('results.states.unavailableTitle')}
          </h2>
          <p className="mt-2 text-sm text-primary-900/50 dark:text-primary-100/40">
            {t('results.states.unavailableDescription')}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col p-6 max-w-7xl mx-auto">
      {/* ── Header card ── */}
      <div className="card p-4 mb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div>
              <h1 className="font-display text-lg font-bold text-primary-800 dark:text-primary-200">
                {t('results.title')}
              </h1>
              <div className="flex items-center gap-3 mt-1 text-xs">
                <span className="font-mono text-primary-500/60">
                  {id.slice(0, 8)}
                </span>
                <span className={`badge-status ${STATUS_STYLE[meta?.status ?? 'pending']}`}>
                  {meta?.status}
                </span>
                <span className="text-primary-900/30 dark:text-primary-100/20">
                  {mode}
                </span>
                <span className="text-primary-900/30 dark:text-primary-100/20">
                  {createdAtLabel}
                </span>
              </div>
            </div>
          </div>

          {/* Export buttons in header */}
          <ExportPanel
            simulationId={id}
            params={vizParams}
            mode={mode}
            supportedExports={results?.metadata?.supported_exports as ExportFormat[] | undefined}
          />
        </div>
      </div>

      {/* ── Tab navigation ── */}
      <div className="flex items-center gap-1 mb-4 overflow-x-auto pb-1">
        {availableTabs.map((tab) => {
          const Icon = TAB_ICONS[tab] ?? UsersIcon;
          const isActive = resolvedActiveTab === tab;
          return (
            <button
              key={tab}
              onClick={() => setActiveTab(tab as Tab)}
              className={`relative flex items-center gap-1.5 px-3 py-2 rounded-lg
                         text-xs font-medium whitespace-nowrap transition-all duration-150
                         ${isActive
                           ? 'text-primary-700 dark:text-primary-300'
                           : 'text-primary-900/35 dark:text-primary-100/25 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-surface-2'
                         }`}
            >
              {isActive && (
                <motion.div
                  layoutId="results-tab"
                  className="absolute inset-0 rounded-lg bg-primary-500/8 dark:bg-primary-400/8
                             border border-primary-500/15 dark:border-primary-400/10"
                  transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                />
              )}
              <Icon className="h-3.5 w-3.5 relative z-10" />
              <span className="relative z-10">{t(`results.tabs.${tab}` as const)}</span>
            </button>
          );
        })}
      </div>

      {/* ── Tab content ── */}
      <div className="flex-1 min-h-0">
        <AnimatePresence mode="wait">
          <motion.div
            key={resolvedActiveTab}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.2 }}
            className="h-full"
          >
            {resolvedActiveTab === 'populations' && <PopulationCharts params={vizParams} simulationId={id} />}
            {resolvedActiveTab === 'cytokines' && <CytokineCharts params={vizParams} simulationId={id} />}
            {resolvedActiveTab === 'ecm' && <ECMCharts params={vizParams} simulationId={id} />}
            {resolvedActiveTab === 'phases' && <PhaseTimeline params={vizParams} simulationId={id} />}
            {resolvedActiveTab === 'comparison' && <TherapyComparison params={vizParams} simulationId={id} />}
            {resolvedActiveTab === 'heatmap' && <CellHeatmap simulationId={id} />}
            {resolvedActiveTab === 'inflammation' && <InflammationMap simulationId={id} />}
            {resolvedActiveTab === 'animation' && <AnimationPlayer simulationId={id} timepoints={results.times} />}
            {resolvedActiveTab === 'spatial3d' && <SpatialView3D simulationId={id} />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
