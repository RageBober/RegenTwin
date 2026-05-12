import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import { XMarkIcon } from '@heroicons/react/24/outline';
import SensitivityView from '../components/Analysis/SensitivityView';
import EstimationView from '../components/Analysis/EstimationView';
import ValidationView from '../components/Analysis/ValidationView';

const TABS = ['sensitivity', 'estimation', 'validation'] as const;
type Tab = (typeof TABS)[number];

export default function Analysis() {
  const { t } = useTranslation();
  const { id } = useParams<{ id?: string }>();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<Tab>('sensitivity');

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="min-h-full p-6 lg:p-8 max-w-5xl mx-auto"
    >
      <h1 className="font-display text-xl font-bold tracking-tight
                     text-primary-800 dark:text-primary-200 mb-4">
        {t('analysis.title')}
      </h1>

      {/* Loaded simulation banner */}
      {id && (
        <motion.div
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 flex items-start justify-between gap-3 rounded-xl
                     border border-accent-500/20 bg-accent-500/5 px-4 py-2.5"
        >
          <div className="flex-1">
            <p className="text-xs font-medium text-accent-700 dark:text-accent-300">
              {t('analysis.loadedSimulation', { id: id.slice(0, 8) })}
            </p>
            <p className="text-2xs text-accent-600/80 dark:text-accent-400/70 mt-0.5">
              {t('analysis.loadedSimulationHint')}
            </p>
          </div>
          <button
            type="button"
            onClick={() => navigate('/analysis')}
            aria-label={t('analysis.resetSimulation')}
            className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-2xs font-medium
                       text-accent-700 dark:text-accent-300 hover:bg-accent-500/10 transition-colors"
          >
            <XMarkIcon className="h-3 w-3" />
            {t('analysis.resetSimulation')}
          </button>
        </motion.div>
      )}

      {/* Tab navigation */}
      <div className="flex items-center gap-1 mb-6">
        {TABS.map((tab) => {
          const isActive = activeTab === tab;
          return (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`relative px-4 py-2 rounded-lg text-sm font-medium transition-all duration-150
                         ${isActive
                           ? 'text-primary-700 dark:text-primary-300'
                           : 'text-primary-900/35 dark:text-primary-100/25 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-surface-2'
                         }`}
            >
              {isActive && (
                <motion.div
                  layoutId="analysis-tab"
                  className="absolute inset-0 rounded-lg bg-primary-500/8 dark:bg-primary-400/8
                             border border-primary-500/15 dark:border-primary-400/10"
                  transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                />
              )}
              <span className="relative z-10">{t(`analysis.${tab}.title`)}</span>
            </button>
          );
        })}
      </div>

      {activeTab === 'sensitivity' && <SensitivityView prefilledSimulationId={id} />}
      {activeTab === 'estimation' && <EstimationView />}
      {activeTab === 'validation' && <ValidationView />}
    </motion.div>
  );
}
