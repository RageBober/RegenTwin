import { useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import {
  EyeIcon,
  MagnifyingGlassIcon,
  ClockIcon,
  ArrowRightIcon,
  TrashIcon,
} from '@heroicons/react/24/outline';
import { useQueryClient } from '@tanstack/react-query';
import { apiClient, API_V1 } from '../lib/api';
import { useSimulationsList } from '../hooks/useSimulation';

const STATUS_STYLE: Record<string, string> = {
  completed: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400',
  running:   'bg-accent-500/10 text-accent-600 dark:text-accent-400',
  failed:    'bg-red-500/10 text-red-600 dark:text-red-400',
  pending:   'bg-primary-500/10 text-primary-600 dark:text-primary-400',
  cancelled: 'bg-primary-900/5 text-primary-900/40 dark:text-primary-100/30',
};

const STATUSES = ['all', 'completed', 'running', 'failed', 'pending'] as const;
const PAGE_SIZE = 20;

export default function History() {
  const { t } = useTranslation();
  const [statusFilter, setStatusFilter] = useState('all');
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(0);
  const [clearing, setClearing] = useState(false);
  const queryClient = useQueryClient();
  const { data: simulations, isLoading } = useSimulationsList(statusFilter);

  const handleClear = async () => {
    if (!window.confirm(t('history.clearConfirm'))) return;
    setClearing(true);
    try {
      await apiClient.delete(`${API_V1}/simulations`);
      await queryClient.invalidateQueries({ queryKey: ['simulations'] });
      setPage(0);
    } finally {
      setClearing(false);
    }
  };

  const filtered = useMemo(() => {
    if (!simulations) return [];
    if (!search.trim()) return simulations;
    const q = search.toLowerCase();
    return simulations.filter((s) =>
      s.simulation_id.toLowerCase().includes(q) ||
      s.status.toLowerCase().includes(q) ||
      (s.params_json?.mode ?? '').toLowerCase().includes(q)
    );
  }, [simulations, search]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageItems = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="min-h-full p-6 lg:p-8 max-w-5xl mx-auto"
    >
      {/* Header */}
      <div className="mb-6 flex items-center justify-between">
        <h1 className="font-display text-xl font-bold tracking-tight
                       text-primary-800 dark:text-primary-200">
          {t('history.title')}
        </h1>
        {simulations && simulations.length > 0 && (
          <button
            onClick={handleClear}
            disabled={clearing}
            className="inline-flex items-center gap-1.5 rounded-lg border border-red-300/50
                       px-3 py-1.5 text-xs font-medium text-red-500 dark:text-red-400
                       hover:bg-red-500/10 transition-colors disabled:opacity-40"
          >
            <TrashIcon className="h-3.5 w-3.5" />
            {clearing ? '...' : t('history.clearAll')}
          </button>
        )}
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3 mb-4">
        {/* Search */}
        <div className="relative flex-1 max-w-xs">
          <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2
                                         h-3.5 w-3.5 text-primary-400/40" />
          <input
            type="text"
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(0); }}
            placeholder="Search ID..."
            className="w-full pl-8 pr-3 py-2 rounded-lg border border-border bg-surface-1
                       text-xs text-primary-800 dark:text-primary-200
                       placeholder:text-primary-400/30
                       focus:outline-none focus:ring-1 focus:ring-primary-500/30"
          />
        </div>

        {/* Status filter pills */}
        <div className="flex items-center gap-1">
          {STATUSES.map((status) => (
            <button
              key={status}
              onClick={() => { setStatusFilter(status); setPage(0); }}
              className={`px-2.5 py-1.5 rounded-lg text-2xs font-medium uppercase tracking-wide
                         transition-all duration-150
                         ${statusFilter === status
                           ? 'bg-primary-500/10 text-primary-600 dark:text-primary-400'
                           : 'text-primary-900/30 dark:text-primary-100/20 hover:text-primary-600 hover:bg-surface-2'
                         }`}
            >
              {status === 'all' ? t('history.all') : status}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      {isLoading ? (
        <div className="flex items-center justify-center py-16">
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-primary-200 border-t-primary-500" />
        </div>
      ) : pageItems.length === 0 ? (
        <div className="card p-12 text-center">
          <ClockIcon className="h-10 w-10 mx-auto mb-3 text-primary-400/20" />
          <p className="text-sm text-primary-900/40 dark:text-primary-100/30 mb-3">
            {t('history.noSimulations')}
          </p>
          <Link
            to="/dashboard"
            className="inline-flex items-center gap-1 text-sm font-medium
                       text-primary-500 hover:text-primary-600
                       dark:text-primary-400 dark:hover:text-primary-300"
          >
            {t('home.runSimulation')}
            <ArrowRightIcon className="h-3 w-3" />
          </Link>
        </div>
      ) : (
        <>
          {/* Table */}
          <div className="card overflow-hidden">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-border">
                  {['id', 'mode', 'status', 'created', 'actions'].map((col) => (
                    <th
                      key={col}
                      className="px-4 py-2.5 text-2xs font-semibold uppercase tracking-wider
                                 text-primary-500/50 dark:text-primary-400/40"
                    >
                      {t(`history.columns.${col}`)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {pageItems.map((sim) => (
                  <tr
                    key={sim.simulation_id}
                    className="hover:bg-surface-2/50 transition-colors"
                  >
                    <td className="px-4 py-3">
                      <span className="font-mono text-xs text-primary-600/70 dark:text-primary-400/60">
                        {sim.simulation_id.slice(0, 8)}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span className="text-xs text-primary-900/50 dark:text-primary-100/40">
                        {sim.params_json?.mode ?? '—'}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`badge-status ${STATUS_STYLE[sim.status] ?? ''}`}>
                        {sim.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-xs text-primary-900/40 dark:text-primary-100/30">
                      {new Date(sim.created_at).toLocaleString()}
                    </td>
                    <td className="px-4 py-3">
                      {sim.status === 'completed' && (
                        <Link
                          to={`/results/${sim.simulation_id}`}
                          className="inline-flex items-center gap-1 text-xs font-medium
                                     text-primary-500 hover:text-primary-600
                                     dark:text-primary-400 dark:hover:text-primary-300 transition-colors"
                        >
                          <EyeIcon className="h-3.5 w-3.5" />
                          {t('history.view')}
                        </Link>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between mt-3">
              <span className="text-2xs text-primary-900/30 dark:text-primary-100/20">
                {filtered.length} total
              </span>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setPage(Math.max(0, page - 1))}
                  disabled={page === 0}
                  className="px-2.5 py-1 rounded text-xs font-medium
                             text-primary-500 hover:bg-surface-2
                             disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  Prev
                </button>
                <span className="px-2 text-2xs text-primary-900/40 dark:text-primary-100/30">
                  {page + 1} / {totalPages}
                </span>
                <button
                  onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
                  disabled={page >= totalPages - 1}
                  className="px-2.5 py-1 rounded text-xs font-medium
                             text-primary-500 hover:bg-surface-2
                             disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </motion.div>
  );
}
