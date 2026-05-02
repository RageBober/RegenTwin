import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import { SunIcon, MoonIcon } from '@heroicons/react/24/outline';
import { useUIStore } from '../stores/uiStore';
import { useSimulationStore } from '../stores/simulationStore';
import { resetApiBaseUrl, updateApiBaseUrl } from '../lib/api';

export default function Settings() {
  const { t, i18n } = useTranslation();
  const { theme, setTheme } = useUIStore();
  const resetParams = useSimulationStore((s) => s.resetParams);
  const [apiUrl, setApiUrl] = useState(
    () => localStorage.getItem('regentwin-api-url') || '',
  );

  const handleSaveApiUrl = () => {
    updateApiBaseUrl(apiUrl);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="min-h-full p-6 lg:p-8 max-w-lg mx-auto"
    >
      <h1 className="font-display text-xl font-bold tracking-tight
                     text-primary-800 dark:text-primary-200 mb-6">
        {t('settings.title')}
      </h1>

      <div className="space-y-4">
        {/* Language */}
        <div className="card p-4">
          <label className="block text-xs font-semibold uppercase tracking-wider
                           text-primary-500/60 dark:text-primary-400/50 mb-2">
            {t('settings.language')}
          </label>
          <select
            value={i18n.language}
            onChange={(e) => {
              i18n.changeLanguage(e.target.value);
              localStorage.setItem('regentwin-language', e.target.value);
            }}
            className="w-full rounded-lg border border-border bg-surface-1 px-3 py-2
                       text-sm text-primary-800 dark:text-primary-200
                       focus:outline-none focus:ring-1 focus:ring-primary-500/30"
          >
            <option value="ru">Русский</option>
            <option value="en">English</option>
          </select>
        </div>

        {/* Theme */}
        <div className="card p-4">
          <label className="block text-xs font-semibold uppercase tracking-wider
                           text-primary-500/60 dark:text-primary-400/50 mb-2">
            {t('settings.theme')}
          </label>
          <div className="flex gap-2">
            <button
              onClick={() => setTheme('light')}
              className={`flex items-center gap-2 rounded-lg border px-4 py-2.5 text-sm
                         font-medium transition-all duration-150
                         ${theme === 'light'
                           ? 'border-primary-500/20 bg-primary-500/5 text-primary-700 dark:text-primary-300 shadow-glow-sm'
                           : 'border-border text-primary-900/40 dark:text-primary-100/30 hover:bg-surface-2'
                         }`}
            >
              <SunIcon className="h-4 w-4" />
              {t('settings.themeLight')}
            </button>
            <button
              onClick={() => setTheme('dark')}
              className={`flex items-center gap-2 rounded-lg border px-4 py-2.5 text-sm
                         font-medium transition-all duration-150
                         ${theme === 'dark'
                           ? 'border-primary-500/20 bg-primary-500/5 text-primary-700 dark:text-primary-300 shadow-glow-sm'
                           : 'border-border text-primary-900/40 dark:text-primary-100/30 hover:bg-surface-2'
                         }`}
            >
              <MoonIcon className="h-4 w-4" />
              {t('settings.themeDark')}
            </button>
          </div>
        </div>

        {/* API URL */}
        <div className="card p-4">
          <label className="block text-xs font-semibold uppercase tracking-wider
                           text-primary-500/60 dark:text-primary-400/50 mb-2">
            {t('settings.apiUrl')}
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="http://localhost:8000"
              className="flex-1 rounded-lg border border-border bg-surface-1 px-3 py-2
                         text-sm font-mono text-primary-800 dark:text-primary-200
                         placeholder:text-primary-400/40
                         focus:outline-none focus:ring-1 focus:ring-primary-500/30"
            />
            <button
              onClick={handleSaveApiUrl}
              className="rounded-lg bg-primary-500 px-4 py-2 text-sm font-medium text-white
                         hover:bg-primary-600 transition-colors"
            >
              {t('common.save')}
            </button>
            <button
              onClick={() => { resetApiBaseUrl(); setApiUrl(''); }}
              className="rounded-lg border border-border px-4 py-2 text-sm font-medium
                         text-primary-600 dark:text-primary-400
                         hover:bg-surface-2 transition-colors"
            >
              {t('settings.resetApiUrl')}
            </button>
          </div>
          <p className="mt-1.5 text-xs text-primary-500/40">
            {t('settings.apiUrlHint')}
          </p>
        </div>

        {/* Reset */}
        <div className="card p-4">
          <button
            onClick={resetParams}
            className="rounded-lg border border-red-400/20 px-4 py-2
                       text-sm font-medium text-red-500
                       hover:bg-red-500/5 transition-colors"
          >
            {t('settings.resetDefaults')}
          </button>
        </div>
      </div>
    </motion.div>
  );
}
