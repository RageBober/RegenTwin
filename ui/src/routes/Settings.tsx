import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useUIStore } from '../stores/uiStore';
import { useSimulationStore } from '../stores/simulationStore';
import { updateApiBaseUrl } from '../lib/api';

export default function Settings() {
  const { t, i18n } = useTranslation();
  const { theme, setTheme } = useUIStore();
  const resetParams = useSimulationStore((s) => s.resetParams);
  const [apiUrl, setApiUrl] = useState(
    () => localStorage.getItem('regentwin-api-url') || 'http://localhost:8000',
  );

  const handleSaveApiUrl = () => {
    updateApiBaseUrl(apiUrl);
  };

  return (
    <div className="p-6">
      <h1 className="mb-6 text-2xl font-bold text-slate-900 dark:text-white">
        {t('settings.title')}
      </h1>

      <div className="max-w-lg space-y-6">
        {/* Language */}
        <div className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-800">
          <label className="mb-2 block text-sm font-medium text-slate-700 dark:text-slate-200">
            {t('settings.language')}
          </label>
          <select
            value={i18n.language}
            onChange={(e) => {
              i18n.changeLanguage(e.target.value);
              localStorage.setItem('regentwin-language', e.target.value);
            }}
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-600 dark:bg-slate-700 dark:text-slate-200"
          >
            <option value="ru">Русский</option>
            <option value="en">English</option>
          </select>
        </div>

        {/* Theme */}
        <div className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-800">
          <label className="mb-2 block text-sm font-medium text-slate-700 dark:text-slate-200">
            {t('settings.theme')}
          </label>
          <div className="flex gap-2">
            <button
              onClick={() => setTheme('light')}
              className={`rounded-lg border px-4 py-2 text-sm transition-colors ${
                theme === 'light'
                  ? 'border-primary-400 bg-primary-50 text-primary-700 dark:border-primary-600 dark:bg-primary-900/20'
                  : 'border-slate-300 text-slate-600 dark:border-slate-600 dark:text-slate-300'
              }`}
            >
              {t('settings.themeLight')}
            </button>
            <button
              onClick={() => setTheme('dark')}
              className={`rounded-lg border px-4 py-2 text-sm transition-colors ${
                theme === 'dark'
                  ? 'border-primary-400 bg-primary-50 text-primary-700 dark:border-primary-600 dark:bg-primary-900/20'
                  : 'border-slate-300 text-slate-600 dark:border-slate-600 dark:text-slate-300'
              }`}
            >
              {t('settings.themeDark')}
            </button>
          </div>
        </div>

        {/* API URL */}
        <div className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-800">
          <label className="mb-2 block text-sm font-medium text-slate-700 dark:text-slate-200">
            {t('settings.apiUrl')}
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              className="flex-1 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-600 dark:bg-slate-700 dark:text-slate-200"
            />
            <button
              onClick={handleSaveApiUrl}
              className="rounded-lg bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700"
            >
              {t('common.save')}
            </button>
          </div>
        </div>

        {/* Reset defaults */}
        <div className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-800">
          <button
            onClick={resetParams}
            className="rounded-lg border border-red-300 px-4 py-2 text-sm font-medium text-red-600 hover:bg-red-50 dark:border-red-700 dark:text-red-400 dark:hover:bg-red-900/20"
          >
            {t('settings.resetDefaults')}
          </button>
        </div>
      </div>
    </div>
  );
}
