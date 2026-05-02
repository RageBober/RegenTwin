import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import logoSvg from '../assets/logo.svg';

const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06, delayChildren: 0.1 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] as [number, number, number, number] } },
};

const agents = [
  { key: 'stem',     marker: 'CD34+' },
  { key: 'macro',    marker: 'CD14+/CD68+' },
  { key: 'fibro',    marker: '—' },
  { key: 'neutro',   marker: 'CD66b+' },
  { key: 'endo',     marker: 'CD31+' },
  { key: 'myofibro', marker: '\u03B1-SMA+' },
] as const;

export default function About() {
  const { t } = useTranslation();

  return (
    <motion.div
      variants={stagger}
      initial="hidden"
      animate="show"
      className="min-h-full p-6 lg:p-8 max-w-4xl mx-auto"
    >
      {/* Header */}
      <motion.div variants={fadeUp} className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <img src={logoSvg} alt="" className="h-8 w-8 text-primary-500" />
          <h1 className="font-display text-2xl font-bold tracking-tight
                         text-primary-800 dark:text-primary-200">
            {t('about.title')}
          </h1>
        </div>
        <p className="text-sm text-primary-900/50 dark:text-primary-100/40 max-w-2xl leading-relaxed">
          {t('about.subtitle')}
        </p>
      </motion.div>

      {/* Mathematical Model */}
      <motion.section variants={fadeUp} className="card p-5 mb-4">
        <h2 className="text-sm font-semibold text-primary-700 dark:text-primary-300 mb-3">
          {t('about.mathModel')}
        </h2>
        <p className="text-xs text-primary-900/50 dark:text-primary-100/40 mb-3 leading-relaxed">
          {t('about.mathModelDesc')}
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="rounded-lg bg-surface-2 p-3">
            <h3 className="text-xs font-medium text-primary-600 dark:text-primary-400 mb-1">
              {t('about.macroLevel')}
            </h3>
            <p className="text-2xs text-primary-900/40 dark:text-primary-100/30 leading-relaxed mb-2">
              {t('about.macroLevelDesc')}
            </p>
            <div className="font-mono text-2xs text-primary-700 dark:text-primary-300 bg-surface-1 rounded p-2 overflow-x-auto">
              dN_t = [rN_t(1 - N_t/K) + &alpha;f(PRP) + &beta;g(PEMF) - &delta;N_t]dt + &sigma;N_t dW_t
            </div>
          </div>
          <div className="rounded-lg bg-surface-2 p-3">
            <h3 className="text-xs font-medium text-primary-600 dark:text-primary-400 mb-1">
              {t('about.microLevel')}
            </h3>
            <p className="text-2xs text-primary-900/40 dark:text-primary-100/30 leading-relaxed">
              {t('about.microLevelDesc')}
            </p>
          </div>
        </div>
      </motion.section>

      {/* Agent Types */}
      <motion.section variants={fadeUp} className="card p-5 mb-4">
        <h2 className="text-sm font-semibold text-primary-700 dark:text-primary-300 mb-3">
          {t('about.agentTypes')}
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-border">
                <th className="pb-2 pr-4 text-2xs font-semibold uppercase tracking-wider text-primary-500/50">
                  {t('about.agentType')}
                </th>
                <th className="pb-2 pr-4 text-2xs font-semibold uppercase tracking-wider text-primary-500/50">
                  {t('about.marker')}
                </th>
                <th className="pb-2 text-2xs font-semibold uppercase tracking-wider text-primary-500/50">
                  {t('about.role')}
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {agents.map(({ key, marker }) => (
                <tr key={key}>
                  <td className="py-2 pr-4 text-xs text-primary-800 dark:text-primary-200">
                    {t(`about.agents.${key}`)}
                  </td>
                  <td className="py-2 pr-4 font-mono text-2xs text-primary-500/70">
                    {marker}
                  </td>
                  <td className="py-2 text-xs text-primary-900/50 dark:text-primary-100/40">
                    {t(`about.agents.${key}Role`)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.section>

      {/* Therapy Models */}
      <motion.section variants={fadeUp} className="card p-5 mb-4">
        <h2 className="text-sm font-semibold text-primary-700 dark:text-primary-300 mb-3">
          {t('about.therapyModels')}
        </h2>
        <div className="space-y-3">
          {[
            { title: 'PRP (Platelet-Rich Plasma)', desc: t('about.prpDesc') },
            { title: 'PEMF (Pulsed Electromagnetic Field)', desc: t('about.pemfDesc') },
            { title: t('about.synergy'), desc: t('about.synergyDesc') },
          ].map(({ title, desc }) => (
            <div key={title} className="rounded-lg bg-surface-2 p-3">
              <h3 className="text-xs font-medium text-primary-600 dark:text-primary-400 mb-0.5">
                {title}
              </h3>
              <p className="text-2xs text-primary-900/40 dark:text-primary-100/30 leading-relaxed">
                {desc}
              </p>
            </div>
          ))}
        </div>
      </motion.section>

      {/* Tech Stack */}
      <motion.section variants={fadeUp} className="card p-5 mb-4">
        <h2 className="text-sm font-semibold text-primary-700 dark:text-primary-300 mb-3">
          {t('about.techStack')}
        </h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {[
            { label: t('about.backend'),       value: 'Python 3.11+ / FastAPI / SQLAlchemy' },
            { label: t('about.frontend'),      value: 'React / TypeScript / Tailwind CSS' },
            { label: t('about.desktop'),       value: 'Tauri / Rust' },
            { label: t('about.scientific'),    value: 'NumPy / SciPy / PyMC / SALib' },
            { label: t('about.visualization'), value: 'Plotly / Three.js' },
            { label: t('about.methods'),       value: 'Euler-Maruyama / Monte Carlo / cKDTree' },
          ].map(({ label, value }) => (
            <div key={label}>
              <h4 className="text-xs font-medium text-primary-700 dark:text-primary-300">{label}</h4>
              <p className="text-2xs text-primary-900/40 dark:text-primary-100/30">{value}</p>
            </div>
          ))}
        </div>
      </motion.section>

      {/* Footer */}
      <motion.p variants={fadeUp} className="text-center text-2xs text-primary-900/30 dark:text-primary-100/20 py-4">
        {t('about.license')}
      </motion.p>
    </motion.div>
  );
}
