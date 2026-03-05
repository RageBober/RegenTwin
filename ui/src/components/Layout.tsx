import { useState, useEffect } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import {
  HomeIcon,
  Cog6ToothIcon,
  ClockIcon,
  AdjustmentsHorizontalIcon,
  BeakerIcon,
  Bars3Icon,
} from '@heroicons/react/24/outline';
import { apiClient, API_V1 } from '../lib/api';

const navItems = [
  { path: '/', icon: HomeIcon, labelKey: 'nav.home' },
  { path: '/dashboard', icon: AdjustmentsHorizontalIcon, labelKey: 'nav.dashboard' },
  { path: '/analysis', icon: BeakerIcon, labelKey: 'nav.analysis' },
  { path: '/history', icon: ClockIcon, labelKey: 'nav.history' },
  { path: '/settings', icon: Cog6ToothIcon, labelKey: 'nav.settings' },
] as const;

export default function Layout() {
  const { t, i18n } = useTranslation();
  const [collapsed, setCollapsed] = useState(false);
  const [backendOnline, setBackendOnline] = useState(false);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await apiClient.get(`${API_V1}/health`);
        setBackendOnline(true);
      } catch {
        setBackendOnline(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const toggleLanguage = () => {
    const newLang = i18n.language === 'ru' ? 'en' : 'ru';
    i18n.changeLanguage(newLang);
    localStorage.setItem('regentwin-language', newLang);
  };

  return (
    <div className="flex h-screen w-full bg-slate-50 dark:bg-slate-900">
      {/* Sidebar */}
      <aside
        className={`flex flex-col border-r border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-800 transition-all duration-200 ${
          collapsed ? 'w-16' : 'w-56'
        }`}
      >
        {/* Header */}
        <div className="flex items-center gap-2 border-b border-slate-200 px-3 py-3 dark:border-slate-700">
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="rounded p-1.5 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-700"
            aria-label="Toggle sidebar"
          >
            <Bars3Icon className="h-5 w-5" />
          </button>
          {!collapsed && (
            <span className="text-lg font-bold text-primary-600 dark:text-primary-400">
              RegenTwin
            </span>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 px-2 py-3">
          {navItems.map(({ path, icon: Icon, labelKey }) => (
            <NavLink
              key={path}
              to={path}
              end={path === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-primary-50 text-primary-700 dark:bg-primary-900/30 dark:text-primary-300'
                    : 'text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-700'
                }`
              }
            >
              <Icon className="h-5 w-5 flex-shrink-0" />
              {!collapsed && <span>{t(labelKey)}</span>}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="border-t border-slate-200 px-3 py-3 dark:border-slate-700">
          {/* Language toggle */}
          <button
            onClick={toggleLanguage}
            className="mb-2 flex w-full items-center justify-center gap-1 rounded-lg px-2 py-1.5 text-xs font-medium text-slate-500 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-700"
          >
            {collapsed ? (i18n.language === 'ru' ? 'RU' : 'EN') : (i18n.language === 'ru' ? 'RU / EN' : 'EN / RU')}
          </button>

          {/* Health indicator */}
          <div className="flex items-center gap-2">
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                backendOnline ? 'bg-green-500' : 'bg-red-500'
              }`}
            />
            {!collapsed && (
              <span className="text-xs text-slate-500 dark:text-slate-400">
                {backendOnline ? t('common.backendOnline') : t('common.backendOffline')}
              </span>
            )}
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}
