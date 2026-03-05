import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import SensitivityView from '../components/Analysis/SensitivityView';

type AnalysisTab = 'sensitivity' | 'estimation';

export default function Analysis() {
  const { t } = useTranslation();
  const [activeTab, setActiveTab] = useState<AnalysisTab>('sensitivity');

  return (
    <div className="p-6">
      <h1 className="mb-6 text-2xl font-bold text-slate-900 dark:text-white">
        {t('analysis.title')}
      </h1>

      <div className="flex border-b border-slate-200 dark:border-slate-700">
        {(['sensitivity', 'estimation'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2.5 text-sm font-medium transition-colors ${
              activeTab === tab
                ? 'border-b-2 border-primary-600 text-primary-600 dark:text-primary-400'
                : 'text-slate-500 hover:text-slate-700 dark:text-slate-400'
            }`}
          >
            {t(`analysis.${tab}.title`)}
          </button>
        ))}
      </div>

      <div className="mt-6">
        {activeTab === 'sensitivity' && <SensitivityView />}
        {activeTab === 'estimation' && (
          <div className="flex h-64 items-center justify-center rounded-lg border-2 border-dashed border-slate-300 dark:border-slate-600">
            <p className="text-sm text-slate-400">{t('analysis.estimation.title')} — coming soon</p>
          </div>
        )}
      </div>
    </div>
  );
}
