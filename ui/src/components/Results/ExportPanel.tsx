import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ArrowDownTrayIcon } from '@heroicons/react/24/outline';
import { apiClient, API_V1, API_VIZ } from '../../lib/api';
import type { ExportFormat, SimulationMode, SimulationParams } from '../../types/api';

interface Props {
  simulationId?: string;
  params?: SimulationParams;
  mode?: SimulationMode;
  supportedExports?: ExportFormat[];
}

const FORMATS_BY_MODE: Record<SimulationMode, ExportFormat[]> = {
  mvp: ['csv'],
  extended: ['csv', 'png', 'pdf'],
  abm: ['csv'],
  integrated: ['csv', 'png', 'pdf'],
};

async function getErrorMessage(error: unknown): Promise<string> {
  if (typeof error === 'string') return error;
  if (error && typeof error === 'object') {
    const response = (error as { response?: { data?: unknown } }).response;
    if (response?.data instanceof Blob) {
      try {
        const text = await response.data.text();
        const json = JSON.parse(text);
        if (typeof json.detail === 'string' && json.detail.length > 0) return json.detail;
      } catch {
        // not JSON or unreadable, fall through
      }
    } else if (response?.data && typeof (response.data as { detail?: unknown }).detail === 'string') {
      return (response.data as { detail: string }).detail;
    }
    const message = (error as { message?: unknown }).message;
    if (typeof message === 'string' && message.length > 0) return message;
  }
  return 'Unknown error';
}

export default function ExportPanel({ simulationId, params, mode = 'extended', supportedExports }: Props) {
  const { t } = useTranslation();
  const [exporting, setExporting] = useState<ExportFormat | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const downloadFile = async (format: ExportFormat) => {
    setExporting(format);
    setErrorMessage(null);
    try {
      let response;
      if (simulationId) {
        response = await apiClient.post(`${API_V1}/export/${simulationId}`, { format }, { responseType: 'blob' });
      } else if (params) {
        response = await apiClient.post(`${API_VIZ}/export/${format}`, { simulation: params }, { responseType: 'blob' });
      } else {
        setErrorMessage(t('results.export.unavailable'));
        return;
      }
      const disposition: string = response.headers['content-disposition'] ?? '';
      const match = disposition.match(/filename[^;=\n]*=\s*"?([^";\n]+)"?/i);
      const filename = match?.[1] ?? `regentwin_export.${format}`;
      const blob = new Blob([response.data]);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      link.click();
      URL.revokeObjectURL(url);
      setErrorMessage(null);
    } catch (err) {
      console.error('Export failed:', err);
      const msg = await getErrorMessage(err);
      setErrorMessage(msg);
    } finally {
      setExporting(null);
    }
  };

  const liveFormats: ExportFormat[] = ['csv', 'png', 'pdf'];
  const formats = (simulationId ? (supportedExports ?? FORMATS_BY_MODE[mode]) : liveFormats).map((key) => ({
    key,
    label: t(`results.export.${key}`),
  }));

  return (
    <div className="flex items-center gap-2">
      {formats.map(({ key, label }) => (
        <button
          key={key}
          onClick={() => downloadFile(key)}
          disabled={exporting !== null}
          className="flex items-center gap-1.5 rounded-lg border border-border
                     px-3 py-1.5 text-xs font-medium
                     text-primary-700 dark:text-primary-300
                     hover:bg-surface-2 transition-colors
                     disabled:opacity-40"
        >
          <ArrowDownTrayIcon className="h-3.5 w-3.5" />
          {exporting === key ? '...' : label}
        </button>
      ))}
      {errorMessage && (
        <p className="text-xs text-red-500 ml-2" role="alert">
          {t('results.export.failed')}: {errorMessage}
        </p>
      )}
    </div>
  );
}
