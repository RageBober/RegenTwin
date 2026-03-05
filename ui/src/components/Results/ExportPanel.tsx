import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ArrowDownTrayIcon } from '@heroicons/react/24/outline';
import { apiClient, API_V1, API_VIZ } from '../../lib/api';
import type { SimulationParams } from '../../types/api';

interface Props {
  simulationId?: string;
  params?: SimulationParams;
}

export default function ExportPanel({ simulationId, params }: Props) {
  const { t } = useTranslation();
  const [exporting, setExporting] = useState<string | null>(null);

  const downloadFile = async (format: 'csv' | 'png' | 'pdf') => {
    setExporting(format);
    try {
      let response;

      if (simulationId) {
        // Use results-based export
        response = await apiClient.post(
          `${API_V1}/export/${simulationId}`,
          { format },
          { responseType: 'blob' },
        );
      } else if (params) {
        // Use viz-based export
        response = await apiClient.post(
          `${API_VIZ}/export/${format}`,
          { simulation: params },
          { responseType: 'blob' },
        );
      } else {
        return;
      }

      const blob = new Blob([response.data]);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `regentwin_export.${format}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export failed:', err);
    } finally {
      setExporting(null);
    }
  };

  const formats = [
    { key: 'csv' as const, label: t('results.export.csv') },
    { key: 'png' as const, label: t('results.export.png') },
    { key: 'pdf' as const, label: t('results.export.pdf') },
  ];

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-800">
      <h3 className="mb-3 text-sm font-semibold uppercase text-slate-500 dark:text-slate-400">
        {t('results.export.title')}
      </h3>
      <div className="flex flex-wrap gap-2">
        {formats.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => downloadFile(key)}
            disabled={exporting !== null}
            className="flex items-center gap-1.5 rounded-lg border border-slate-300 px-3 py-2 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-50 disabled:opacity-50 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-700"
          >
            <ArrowDownTrayIcon className="h-3.5 w-3.5" />
            {exporting === key ? '...' : label}
          </button>
        ))}
      </div>
    </div>
  );
}
