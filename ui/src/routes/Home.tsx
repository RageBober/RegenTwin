import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';
import {
  ArrowUpTrayIcon,
  PlayIcon,
  ClockIcon,
} from '@heroicons/react/24/outline';

export default function Home() {
  const { t } = useTranslation();

  const quickStartItems = [
    { to: '/dashboard', icon: ArrowUpTrayIcon, labelKey: 'home.uploadData' },
    { to: '/dashboard', icon: PlayIcon, labelKey: 'home.runSimulation' },
    { to: '/history', icon: ClockIcon, labelKey: 'home.viewHistory' },
  ];

  return (
    <div className="flex min-h-full flex-col items-center justify-center p-8">
      <div className="max-w-2xl text-center">
        <h1 className="mb-2 text-4xl font-bold text-slate-900 dark:text-white">
          {t('home.title')}
        </h1>
        <p className="mb-2 text-xl text-primary-600 dark:text-primary-400">
          {t('home.subtitle')}
        </p>
        <p className="mb-10 text-slate-500 dark:text-slate-400">
          {t('home.description')}
        </p>

        <h2 className="mb-4 text-lg font-semibold text-slate-700 dark:text-slate-200">
          {t('home.quickStart')}
        </h2>
        <div className="grid grid-cols-3 gap-4">
          {quickStartItems.map(({ to, icon: Icon, labelKey }) => (
            <Link
              key={labelKey}
              to={to}
              className="flex flex-col items-center gap-3 rounded-xl border border-slate-200 bg-white p-6 shadow-sm transition-all hover:border-primary-300 hover:shadow-md dark:border-slate-700 dark:bg-slate-800 dark:hover:border-primary-600"
            >
              <Icon className="h-8 w-8 text-primary-500" />
              <span className="text-sm font-medium text-slate-700 dark:text-slate-200">
                {t(labelKey)}
              </span>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
