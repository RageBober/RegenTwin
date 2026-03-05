import { useState, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useParams } from 'react-router-dom';
import { useSimulationMeta } from '../hooks/useResults';
import { toSimulationParams } from '../types/api';
import type { SimulationParams, SimulationRequest } from '../types/api';
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
import { useQuery } from '@tanstack/react-query';
import { apiClient, API_V1 } from '../lib/api';

const TABS = ['populations', 'cytokines', 'ecm', 'phases', 'comparison', 'heatmap', 'inflammation', 'animation', 'spatial3d'] as const;
type Tab = typeof TABS[number];

export default function Results() {
  const { t } = useTranslation();
  const { id } = useParams<{ id: string }>();
  const [activeTab, setActiveTab] = useState<Tab>('populations');

  const { data: meta, isLoading: metaLoading } = useSimulationMeta(id);

  // Get simulation params from the simulation record
  const { data: simRecord } = useQuery({
    queryKey: ['sim-record', id],
    queryFn: async () => {
      const { data } = await apiClient.get(`${API_V1}/simulate/${id}`);
      return data;
    },
    enabled: !!id,
    staleTime: Infinity,
  });

  // Extract SimulationParams from the stored record
  const vizParams: SimulationParams | null = useMemo(() => {
    if (!simRecord?.params_json) return null;
    // simRecord.params_json contains the original SimulationRequest
    return toSimulationParams(simRecord.params_json as SimulationRequest);
  }, [simRecord]);

  if (metaLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary-200 border-t-primary-600" />
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col p-6">
      {/* Header */}
      <div className="mb-4">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
          {t('results.title')}
        </h1>
        <div className="mt-1 flex gap-4 text-xs text-slate-500 dark:text-slate-400">
          <span>{t('results.metadata.simulationId')}: <span className="font-mono">{id?.slice(0, 8)}...</span></span>
          {meta && (
            <>
              <span>{t('results.metadata.status')}: {meta.status}</span>
              <span>{t('results.metadata.created')}: {new Date(meta.created_at).toLocaleString()}</span>
            </>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-200 dark:border-slate-700">
        {TABS.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2.5 text-sm font-medium transition-colors ${
              activeTab === tab
                ? 'border-b-2 border-primary-600 text-primary-600 dark:text-primary-400'
                : 'text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200'
            }`}
          >
            {t(`results.tabs.${tab}`)}
          </button>
        ))}
      </div>

      {/* Chart area */}
      <div className="mt-4 flex-1">
        {vizParams ? (
          <>
            {activeTab === 'populations' && <PopulationCharts params={vizParams} simulationId={id} />}
            {activeTab === 'cytokines' && <CytokineCharts params={vizParams} simulationId={id} />}
            {activeTab === 'ecm' && <ECMCharts params={vizParams} simulationId={id} />}
            {activeTab === 'phases' && <PhaseTimeline params={vizParams} simulationId={id} />}
            {activeTab === 'comparison' && <TherapyComparison params={vizParams} />}
            {activeTab === 'heatmap' && <CellHeatmap />}
            {activeTab === 'inflammation' && <InflammationMap />}
            {activeTab === 'animation' && <AnimationPlayer />}
            {activeTab === 'spatial3d' && <SpatialView3D />}
          </>
        ) : (
          <div className="flex h-96 items-center justify-center">
            <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary-200 border-t-primary-600" />
          </div>
        )}
      </div>

      {/* Export */}
      <div className="mt-4">
        <ExportPanel simulationId={id} params={vizParams ?? undefined} />
      </div>
    </div>
  );
}
