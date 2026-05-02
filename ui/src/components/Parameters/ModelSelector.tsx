import { useTranslation } from 'react-i18next';
import { useSimulationStore } from '../../stores/simulationStore';
import type { SimulationMode } from '../../types/api';
import { motion } from 'framer-motion';

/* Simple inline icons for each model type */
const MvpIcon = (props: React.SVGProps<SVGSVGElement>) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" {...props}>
    <path d="M3 17l4-8 4 6 4-10 6 12" />
  </svg>
);

const SdeIcon = (props: React.SVGProps<SVGSVGElement>) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" {...props}>
    <path d="M3 17c2-3 3-8 5-8s2 5 4 5 2-6 4-6 3 4 5 4" />
    <path d="M3 12c2-2 3-5 5-5s2 3 4 3 2-4 4-4 3 3 5 3" opacity={0.4} />
  </svg>
);

const AbmIcon = (props: React.SVGProps<SVGSVGElement>) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} {...props}>
    <circle cx="8" cy="8" r="2" /><circle cx="16" cy="8" r="2" /><circle cx="12" cy="16" r="2" />
    <circle cx="5" cy="15" r="1.5" /><circle cx="19" cy="14" r="1.5" />
    <path d="M9.5 9.5l2 5M14.5 9.5l-2 5" strokeWidth={1} opacity={0.3} />
  </svg>
);

const MODES: { value: Exclude<SimulationMode, 'integrated'>; labelKey: string; descKey: string; icon: React.FC<React.SVGProps<SVGSVGElement>> }[] = [
  { value: 'mvp',      labelKey: 'dashboard.model.mvp',      descKey: 'dashboard.model.mvpDesc',      icon: MvpIcon },
  { value: 'extended',  labelKey: 'dashboard.model.extended',  descKey: 'dashboard.model.extendedDesc',  icon: SdeIcon },
  { value: 'abm',      labelKey: 'dashboard.model.abm',      descKey: 'dashboard.model.abmDesc',      icon: AbmIcon },
];

export default function ModelSelector() {
  const { t } = useTranslation();
  const mode = useSimulationStore((s) => s.params.mode);
  const setParam = useSimulationStore((s) => s.setParam);

  return (
    <div>
      <h3 className="text-xs font-semibold uppercase tracking-wider
                     text-primary-500/60 dark:text-primary-400/50 mb-3">
        {t('dashboard.model.title')}
      </h3>
      <div className="grid grid-cols-3 gap-3">
        {MODES.map(({ value, labelKey, descKey, icon: Icon }) => {
          const isActive = mode === value;
          return (
            <button
              key={value}
              onClick={() => setParam('mode', value)}
              className={`relative card p-4 text-left transition-all duration-200
                         ${isActive
                           ? 'border-primary-500/30 dark:border-primary-400/20 shadow-glow-sm'
                           : 'hover:border-primary-300/20'
                         }`}
            >
              {isActive && (
                <motion.div
                  layoutId="model-active"
                  className="absolute inset-0 rounded-xl bg-primary-500/5 dark:bg-primary-400/5"
                  transition={{ type: 'spring', stiffness: 350, damping: 30 }}
                />
              )}
              <div className="relative">
                <Icon className={`h-8 w-8 mb-2 ${
                  isActive
                    ? 'text-primary-500 dark:text-primary-400'
                    : 'text-primary-400/40 dark:text-primary-500/30'
                }`} />
                <div className={`text-sm font-medium mb-0.5 ${
                  isActive
                    ? 'text-primary-700 dark:text-primary-300'
                    : 'text-primary-900/60 dark:text-primary-100/40'
                }`}>
                  {t(labelKey)}
                </div>
                <div className="text-2xs text-primary-900/35 dark:text-primary-100/25 leading-relaxed">
                  {t(descKey)}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
