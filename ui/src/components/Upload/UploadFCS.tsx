import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useMutation } from '@tanstack/react-query';
import { ArrowUpTrayIcon, CheckCircleIcon, XCircleIcon } from '@heroicons/react/24/outline';
import { apiClient, API_V1 } from '../../lib/api';
import { useSimulationStore } from '../../stores/simulationStore';
import type { UploadResponse } from '../../types/api';

export default function UploadFCS() {
  const { t } = useTranslation();
  const setUploadId = useSimulationStore((s) => s.setUploadId);
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
      setUploadId(data.upload_id);
      setUploadResult(data);
    },
  });

  const handleFile = useCallback(
    (file: File) => {
      if (!file.name.endsWith('.fcs')) return;
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
    <div className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-800">
      <h3 className="mb-3 text-sm font-semibold uppercase text-slate-500 dark:text-slate-400">
        {t('dashboard.upload.title')}
      </h3>

      <label
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`flex cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed p-6 transition-colors ${
          dragOver
            ? 'border-primary-400 bg-primary-50 dark:bg-primary-900/20'
            : 'border-slate-300 hover:border-primary-300 dark:border-slate-600'
        }`}
      >
        <ArrowUpTrayIcon className="h-8 w-8 text-slate-400" />
        <span className="text-center text-xs text-slate-500 dark:text-slate-400">
          {t('dashboard.upload.dropzone')}
        </span>
        <input
          type="file"
          accept=".fcs"
          onChange={handleInputChange}
          className="hidden"
        />
      </label>

      {uploadMutation.isPending && (
        <p className="mt-2 text-xs text-primary-600">{t('dashboard.upload.uploading')}</p>
      )}

      {uploadMutation.isError && (
        <div className="mt-2 flex items-center gap-1 text-xs text-red-600">
          <XCircleIcon className="h-4 w-4" />
          {t('dashboard.upload.error')}
        </div>
      )}

      {uploadResult && (
        <div className="mt-2 flex items-center gap-1 text-xs text-green-600">
          <CheckCircleIcon className="h-4 w-4" />
          <span>{uploadResult.filename}</span>
        </div>
      )}
    </div>
  );
}
