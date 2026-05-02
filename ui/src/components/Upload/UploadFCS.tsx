import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useMutation } from '@tanstack/react-query';
import { ArrowUpTrayIcon, CheckCircleIcon, XCircleIcon } from '@heroicons/react/24/outline';
import { apiClient, API_V1 } from '../../lib/api';
import { useSimulationStore } from '../../stores/simulationStore';
import type { SimulationRequest, UploadResponse } from '../../types/api';

export default function UploadFCS() {
  const { t } = useTranslation();
  const applyUpload = useSimulationStore((s) => s.applyUpload);
  const [dragOver, setDragOver] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      const { data } = await apiClient.post<UploadResponse>(`${API_V1}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return data;
    },
    onSuccess: (data) => {
      applyUpload(data.upload_id, data.metadata?.initial_conditions as Partial<SimulationRequest> | undefined);
      setUploadResult(data);
    },
  });

  const handleFile = useCallback(
    (file: File) => {
      const lower = file.name.toLowerCase();
      const isAccepted = lower.endsWith('.fcs') || lower.endsWith('.csv');
      if (!isAccepted) return;
      uploadMutation.mutate(file);
    },
    [uploadMutation],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  return (
    <div>
      <h3 className="text-xs font-semibold uppercase tracking-wider
                     text-primary-500/60 dark:text-primary-400/50 mb-3">
        {t('dashboard.upload.title')}
      </h3>

      <label
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`card flex cursor-pointer flex-col items-center gap-2.5 rounded-xl
                   border-2 border-dashed p-8 transition-all duration-200
                   ${dragOver
                     ? 'border-primary-400 bg-primary-500/5 shadow-glow-sm dark:border-primary-500'
                     : 'border-border hover:border-primary-400/30 hover:shadow-glow-sm'
                   }`}
      >
        <div className={`rounded-full p-3 transition-colors
                        ${dragOver
                          ? 'bg-primary-500/10 text-primary-500'
                          : 'bg-surface-2 text-primary-400/40'
                        }`}>
          <ArrowUpTrayIcon className="h-6 w-6" />
        </div>
        <span className="text-center text-xs text-primary-900/40 dark:text-primary-100/30">
          {t('dashboard.upload.dropzone')}
        </span>
        <input
          type="file"
          accept=".fcs,.csv"
          onChange={handleInputChange}
          className="hidden"
        />
      </label>

      {uploadMutation.isPending && (
        <p className="mt-2 text-xs text-primary-500 animate-pulse">{t('dashboard.upload.uploading')}</p>
      )}

      {uploadMutation.isError && (
        <div className="mt-2 flex items-center gap-1.5 text-xs text-red-500">
          <XCircleIcon className="h-4 w-4" />
          {t('dashboard.upload.error')}
        </div>
      )}

      {uploadResult && (
        <div className="mt-2 flex flex-col gap-1 text-xs text-emerald-600 dark:text-emerald-400">
          <div className="flex items-center gap-1.5">
            <CheckCircleIcon className="h-4 w-4" />
            <span className="font-mono">{uploadResult.filename}</span>
          </div>
          {uploadResult.metadata?.parameter_source && (
            <span className="text-primary-900/30 dark:text-primary-100/20">
              source: {String(uploadResult.metadata.parameter_source)}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
