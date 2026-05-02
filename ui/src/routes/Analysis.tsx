import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import SensitivityView from '../components/Analysis/SensitivityView';
import EstimationView from '../components/Analysis/EstimationView';

const TABS = ['sensitivity', 'estimation'] as const;
type Tab = (typeof TABS)[number];

export default function Analysis() {
  const { t } = useTranslation();
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

      {activeTab === 'sensitivity' && <SensitivityView />}
      {activeTab === 'estimation' && <EstimationView />}
    </motion.div>
  );
}
