import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence } from 'framer-motion';
import { InformationCircleIcon, XMarkIcon } from '@heroicons/react/24/outline';
import type { MethodKind } from '../../types/api';

interface Props {
  kind: MethodKind;
  align?: 'left' | 'right';
}

export default function MethodInfo({ kind, align = 'right' }: Props) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const title = t(`analysis.methods.${kind}.title`);
  const body = t(`analysis.methods.${kind}.body`);

  return (
    <div ref={ref} className="relative inline-flex">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-label={t('analysis.methods.infoLabel')}
        className="inline-flex items-center justify-center rounded-full p-1
                   text-primary-500/50 hover:text-primary-600 dark:hover:text-primary-300
                   hover:bg-primary-500/8 transition-colors"
      >
        <InformationCircleIcon className="h-4 w-4" />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.97 }}
            transition={{ duration: 0.12 }}
            className={`absolute top-full mt-2 z-30 w-72
                        ${align === 'right' ? 'right-0' : 'left-0'}
                        rounded-xl border border-border bg-surface-1 shadow-xl
                        p-3.5 text-left`}
          >
            <div className="flex items-start justify-between gap-2 mb-1.5">
              <h4 className="text-xs font-semibold uppercase tracking-wider
                             text-primary-700 dark:text-primary-200">
                {title}
              </h4>
              <button
                type="button"
                onClick={() => setOpen(false)}
                aria-label={t('analysis.methods.close')}
                className="text-primary-400/40 hover:text-primary-600 dark:hover:text-primary-300"
              >
                <XMarkIcon className="h-3.5 w-3.5" />
              </button>
            </div>
            <p className="text-xs leading-relaxed text-primary-700/80 dark:text-primary-300/80">
              {body}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
