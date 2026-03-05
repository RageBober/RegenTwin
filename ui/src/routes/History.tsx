import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { EyeIcon } from '@heroicons/react/24/outline';
import { useSimulationsList } from '../hooks/useSimulation';

const STATUS_BADGES: Record<string, string> = {
  pending: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300',
  running: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300',
  completed: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300',
  failed: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300',
  cancelled: 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300',
};

export default function History() {
  const { t } = useTranslation();
  const [statusFilter, setStatusFilter] = useState('all');
  const { data: simulations, isLoading } = useSimulationsList(statusFilter);

  return (
    <div className="p-6">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
          {t('history.title')}
        </h1>
        <div className="flex items-center gap-2">
          <label className="text-sm text-slate-500 dark:text-slate-400">
            {t('history.filterByStatus')}:
          </label>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-700 dark:text-slate-200"
          >
            <option value="all">{t('history.all')}</option>
            <option value="completed">Completed</option>
            <option value="running">Running</option>
            <option value="failed">Failed</option>
            <option value="pending">Pending</option>
          </select>
        </div>
      </div>

      <div className="overflow-hidden rounded-lg border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-800">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-slate-200 bg-slate-50 dark:border-slate-700 dark:bg-slate-900">
            <tr>
              {['id', 'status', 'created', 'actions'].map((col) => (
                <th
                  key={col}
                  className="px-4 py-3 font-medium text-slate-500 dark:text-slate-400"
                >
                  {t(`history.columns.${col}`)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-slate-700">
            {isLoading && (
              <tr>
                <td colSpan={4} className="px-4 py-8 text-center text-slate-400">
                  {t('common.loading')}
                </td>
              </tr>
            )}
            {!isLoading && (!simulations || simulations.length === 0) && (
              <tr>
                <td colSpan={4} className="px-4 py-8 text-center text-slate-400">
                  {t('history.noSimulations')}
                </td>
              </tr>
            )}
            {simulations?.map((sim) => (
              <tr key={sim.simulation_id} className="hover:bg-slate-50 dark:hover:bg-slate-700/50">
                <td className="px-4 py-3 font-mono text-xs text-slate-600 dark:text-slate-300">
                  {sim.simulation_id.slice(0, 8)}...
                </td>
                <td className="px-4 py-3">
                  <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${STATUS_BADGES[sim.status] || ''}`}>
                    {sim.status}
                  </span>
                </td>
                <td className="px-4 py-3 text-xs text-slate-500 dark:text-slate-400">
                  {new Date(sim.created_at).toLocaleString()}
                </td>
                <td className="px-4 py-3">
                  {sim.status === 'completed' && (
                    <Link
                      to={`/results/${sim.simulation_id}`}
                      className="inline-flex items-center gap-1 text-xs font-medium text-primary-600 hover:text-primary-700 dark:text-primary-400"
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
    </div>
  );
}
