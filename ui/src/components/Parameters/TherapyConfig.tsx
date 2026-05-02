import { useTranslation } from 'react-i18next';
import { Disclosure, DisclosureButton, DisclosurePanel } from '@headlessui/react';
import { ChevronRightIcon } from '@heroicons/react/24/outline';
import { motion, AnimatePresence } from 'framer-motion';
import { useSimulationStore } from '../../stores/simulationStore';
import NumberInput from '../common/NumberInput';

function Section({
  title,
  icon,
  defaultOpen = false,
  children,
}: {
  title: string;
  icon: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  return (
    <Disclosure defaultOpen={defaultOpen}>
      {({ open }) => (
        <div className="card overflow-hidden">
          <DisclosureButton
            className="flex w-full items-center gap-2.5 px-4 py-3 text-left
                       hover:bg-surface-2/50 transition-colors"
          >
            <span className="text-base">{icon}</span>
            <span className="flex-1 text-xs font-semibold uppercase tracking-wider
                           text-primary-700/70 dark:text-primary-300/60">
              {title}
            </span>
            <ChevronRightIcon
              className={`h-3.5 w-3.5 text-primary-400/40 transition-transform duration-200
                         ${open ? 'rotate-90' : ''}`}
            />
          </DisclosureButton>
          <AnimatePresence initial={false}>
            {open && (
              <DisclosurePanel static>
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2, ease: 'easeInOut' }}
                  className="overflow-hidden"
                >
                  <div className="px-4 pb-3 pt-1 space-y-2 border-t border-border">
                    {children}
                  </div>
                </motion.div>
              </DisclosurePanel>
            )}
          </AnimatePresence>
        </div>
      )}
    </Disclosure>
  );
}

export default function TherapyConfig() {
  const { t } = useTranslation();
  const params = useSimulationStore((s) => s.params);
  const setParam = useSimulationStore((s) => s.setParam);

  return (
    <div>
      <h3 className="text-xs font-semibold uppercase tracking-wider
                     text-primary-500/60 dark:text-primary-400/50 mb-3">
        {t('dashboard.params.title')}
      </h3>

      <div className="space-y-2">
        {/* Cell Populations */}
        <Section title={t('dashboard.params.populations')} icon="🔬" defaultOpen>
          <NumberInput testId="param-P0"  label={t('dashboard.variables.P')}  value={params.P0}   onChange={(v) => setParam('P0', v)}   min={0} step={10} />
          <NumberInput testId="param-Ne0" label={t('dashboard.variables.Ne')} value={params.Ne0}  onChange={(v) => setParam('Ne0', v)}  min={0} step={10} />
          <NumberInput testId="param-M10" label={t('dashboard.variables.M1')} value={params.M1_0} onChange={(v) => setParam('M1_0', v)} min={0} step={10} />
          <NumberInput testId="param-M20" label={t('dashboard.variables.M2')} value={params.M2_0} onChange={(v) => setParam('M2_0', v)} min={0} step={1} />
          <NumberInput testId="param-F0"  label={t('dashboard.variables.F')}  value={params.F0}   onChange={(v) => setParam('F0', v)}   min={0} step={10} />
          <NumberInput testId="param-Mf0" label={t('dashboard.variables.Mf')} value={params.Mf0}  onChange={(v) => setParam('Mf0', v)}  min={0} step={1} />
          <NumberInput testId="param-E0"  label={t('dashboard.variables.E')}  value={params.E0}   onChange={(v) => setParam('E0', v)}   min={0} step={5} />
          <NumberInput testId="param-S0"  label={t('dashboard.variables.S')}  value={params.S0}   onChange={(v) => setParam('S0', v)}   min={0} step={5} />
        </Section>

        {/* Cytokines */}
        <Section title={t('dashboard.params.cytokines')} icon="🧬">
          <NumberInput label={t('dashboard.variables.C_TNF')}  value={params.C_TNF0}   onChange={(v) => setParam('C_TNF0', v)}   min={0} step={0.5} />
          <NumberInput label={t('dashboard.variables.C_IL10')} value={params.C_IL10_0}  onChange={(v) => setParam('C_IL10_0', v)} min={0} step={0.1} />
          <NumberInput label={t('dashboard.variables.D')}      value={params.D0}        onChange={(v) => setParam('D0', v)}       min={0} step={1} />
          <NumberInput label={t('dashboard.variables.O2')}     value={params.O2_0}      onChange={(v) => setParam('O2_0', v)}     min={0} step={5} />
        </Section>

        {/* Time Settings */}
        <Section title={t('dashboard.params.timeSettings')} icon="⏱" defaultOpen>
          <NumberInput testId="param-t-max" label={t('dashboard.params.tMax')} value={params.t_max_hours} onChange={(v) => setParam('t_max_hours', v)} min={24} max={1440} step={24} />
          <div className="flex items-center justify-between gap-2">
            <label className="text-xs text-primary-900/50 dark:text-primary-100/40 truncate flex-1">
              {t('dashboard.params.dt')}
            </label>
            <select
              value={params.dt}
              onChange={(e) => setParam('dt', Number(e.target.value))}
              className="w-24 rounded-lg border border-border bg-surface-1 px-2 py-1.5
                         text-xs font-mono text-right
                         text-primary-800 dark:text-primary-200
                         focus:outline-none focus:ring-1 focus:ring-primary-500/30"
            >
              <option value={0.01}>0.01</option>
              <option value={0.05}>0.05</option>
              <option value={0.1}>0.1</option>
              <option value={0.5}>0.5</option>
            </select>
          </div>
        </Section>

        {/* Therapy */}
        <Section title={t('dashboard.params.therapy')} icon="💊" defaultOpen>
          {/* PRP */}
          <label className="flex items-center gap-2 text-xs text-primary-800/60 dark:text-primary-200/50 cursor-pointer">
            <input
              type="checkbox"
              checked={params.prp_enabled}
              onChange={(e) => setParam('prp_enabled', e.target.checked)}
              className="rounded border-border text-primary-500 focus:ring-primary-500/30"
            />
            {t('dashboard.params.prpEnabled')}
          </label>
          {params.prp_enabled && (
            <NumberInput
              label={t('dashboard.params.prpIntensity')}
              value={params.prp_intensity}
              onChange={(v) => setParam('prp_intensity', v)}
              min={0} max={2} step={0.1}
            />
          )}

          {/* PEMF */}
          <label className="flex items-center gap-2 text-xs text-primary-800/60 dark:text-primary-200/50 cursor-pointer">
            <input
              type="checkbox"
              checked={params.pemf_enabled}
              onChange={(e) => setParam('pemf_enabled', e.target.checked)}
              className="rounded border-border text-primary-500 focus:ring-primary-500/30"
            />
            {t('dashboard.params.pemfEnabled')}
          </label>
          {params.pemf_enabled && (
            <>
              <NumberInput
                label={t('dashboard.params.pemfFrequency')}
                value={params.pemf_frequency}
                onChange={(v) => setParam('pemf_frequency', v)}
                min={1} max={100} step={1}
              />
              <NumberInput
                label={t('dashboard.params.pemfIntensity')}
                value={params.pemf_intensity}
                onChange={(v) => setParam('pemf_intensity', v)}
                min={0} max={2} step={0.1}
              />
            </>
          )}
        </Section>

        {/* Monte Carlo */}
        <Section title={t('dashboard.params.monteCarlo')} icon="🎲">
          <NumberInput testId="param-n-trajectories" label={t('dashboard.params.nTrajectories')} value={params.n_trajectories} onChange={(v) => setParam('n_trajectories', v)} min={1} max={1000} step={1} />
          <NumberInput testId="param-random-seed"    label={t('dashboard.params.randomSeed')}    value={params.random_seed ?? 42} onChange={(v) => setParam('random_seed', v)} min={0} step={1} />
        </Section>
      </div>
    </div>
  );
}
