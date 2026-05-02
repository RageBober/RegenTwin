import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  ArrowUpTrayIcon,
  CpuChipIcon,
  PlayIcon,
  ChartBarIcon,
  ClockIcon,
  ArrowRightIcon,
} from '@heroicons/react/24/outline';
import { useSimulationsList } from '../hooks/useSimulation';
import logoSvg from '../assets/logo.svg';

const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08, delayChildren: 0.15 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.25, 0.46, 0.45, 0.94] as [number, number, number, number] } },
};

const STATUS_COLOR: Record<string, string> = {
  completed: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-400',
  running:   'bg-accent-500/15 text-accent-700 dark:text-accent-400',
  failed:    'bg-red-500/15 text-red-600 dark:text-red-400',
  pending:   'bg-primary-500/10 text-primary-600 dark:text-primary-400',
  cancelled: 'bg-slate-500/10 text-slate-500 dark:text-slate-400',
};

export default function Home() {
  const { t } = useTranslation();
  const { data: simulations } = useSimulationsList();

  const recentSims = (simulations ?? []).slice(0, 3);

  const workflowSteps = [
    {
      to: '/dashboard',
      icon: ArrowUpTrayIcon,
      step: '01',
      titleKey: 'home.uploadData',
      color: 'from-primary-500/20 to-primary-600/10',
      iconColor: 'text-primary-500',
    },
    {
      to: '/dashboard',
      icon: CpuChipIcon,
      step: '02',
      titleKey: 'home.configureModel',
      color: 'from-primary-400/20 to-primary-500/10',
      iconColor: 'text-primary-500',
    },
    {
      to: '/dashboard',
      icon: PlayIcon,
      step: '03',
      titleKey: 'home.runSimulation',
      color: 'from-accent-500/20 to-accent-600/10',
      iconColor: 'text-accent-500',
    },
    {
      to: '/history',
      icon: ChartBarIcon,
      step: '04',
      titleKey: 'home.analyzeResults',
      color: 'from-emerald-500/20 to-emerald-600/10',
      iconColor: 'text-emerald-600 dark:text-emerald-400',
    },
  ];

  return (
    <motion.div
      variants={stagger}
      initial="hidden"
      animate="show"
      className="min-h-full p-6 lg:p-10 max-w-5xl mx-auto"
    >
      {/* ── Hero ── */}
      <motion.div variants={fadeUp} className="mb-10">
        <div className="flex items-center gap-3 mb-3">
          <img src={logoSvg} alt="" className="h-10 w-10 text-primary-500" />
          <h1 className="font-display text-3xl font-bold tracking-tight
                         text-primary-800 dark:text-primary-200">
            {t('home.title')}
          </h1>
        </div>
        <p className="text-lg font-medium text-primary-600 dark:text-primary-400 mb-1.5">
          {t('home.subtitle')}
        </p>
        <p className="text-sm text-primary-900/50 dark:text-primary-100/40 max-w-xl leading-relaxed">
          {t('home.description')}
        </p>
      </motion.div>

      {/* ── Workflow Steps ── */}
      <motion.div variants={fadeUp} className="mb-10">
        <h2 className="text-xs font-semibold uppercase tracking-wider
                       text-primary-500/60 dark:text-primary-400/50 mb-4">
          {t('home.quickStart')}
        </h2>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {workflowSteps.map(({ to, icon: Icon, step, titleKey, color, iconColor }) => (
            <Link
              key={step}
              to={to}
              className={`group card p-4 relative overflow-hidden
                         hover:shadow-glow-sm transition-all duration-200`}
            >
              {/* Gradient bg */}
              <div className={`absolute inset-0 bg-gradient-to-br ${color} opacity-0
                              group-hover:opacity-100 transition-opacity duration-300`} />
              <div className="relative">
                <span className="font-mono text-2xs font-semibold text-primary-400/40
                                 dark:text-primary-500/40 mb-2 block">
                  {step}
                </span>
                <Icon className={`h-6 w-6 mb-2.5 ${iconColor}`} />
                <span className="text-sm font-medium text-primary-800 dark:text-primary-200
                                 group-hover:text-primary-700 dark:group-hover:text-primary-100">
                  {t(titleKey)}
                </span>
              </div>
            </Link>
          ))}
        </div>
      </motion.div>

      {/* ── Recent Simulations ── */}
      <motion.div variants={fadeUp}>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xs font-semibold uppercase tracking-wider
                         text-primary-500/60 dark:text-primary-400/50">
            {t('home.recentSimulations')}
          </h2>
          {recentSims.length > 0 && (
            <Link
              to="/history"
              className="flex items-center gap-1 text-xs font-medium
                         text-primary-500 hover:text-primary-600
                         dark:text-primary-400 dark:hover:text-primary-300
                         transition-colors"
            >
              {t('home.viewHistory')}
              <ArrowRightIcon className="h-3 w-3" />
            </Link>
          )}
        </div>

        {recentSims.length > 0 ? (
          <div className="space-y-2">
            {recentSims.map((sim) => (
              <Link
                key={sim.simulation_id}
                to={`/results/${sim.simulation_id}`}
                className="card flex items-center justify-between px-4 py-3
                           hover:shadow-glow-sm transition-all group"
              >
                <div className="flex items-center gap-3">
                  <span className="font-mono text-xs text-primary-600/60 dark:text-primary-400/50">
                    {sim.simulation_id.substring(0, 8)}
                  </span>
                  <span className={`badge-status ${STATUS_COLOR[sim.status] ?? STATUS_COLOR.pending}`}>
                    {sim.status}
                  </span>
                  <span className="text-xs text-primary-900/40 dark:text-primary-100/30">
                    {sim.params_json?.mode ?? '—'}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-primary-900/30 dark:text-primary-100/20">
                    {sim.created_at ? new Date(sim.created_at).toLocaleDateString() : ''}
                  </span>
                  <ArrowRightIcon className="h-3.5 w-3.5 text-primary-400/40
                                            group-hover:text-primary-500 transition-colors" />
                </div>
              </Link>
            ))}
          </div>
        ) : (
          <div className="card p-8 text-center">
            <ClockIcon className="h-8 w-8 mx-auto mb-2 text-primary-400/30" />
            <p className="text-sm text-primary-900/40 dark:text-primary-100/30">
              {t('history.noSimulations')}
            </p>
            <Link
              to="/dashboard"
              className="inline-flex items-center gap-1 mt-3 text-sm font-medium
                         text-primary-500 hover:text-primary-600
                         dark:text-primary-400 dark:hover:text-primary-300"
            >
              {t('home.runSimulation')}
              <ArrowRightIcon className="h-3 w-3" />
            </Link>
          </div>
        )}
      </motion.div>
    </motion.div>
  );
}
