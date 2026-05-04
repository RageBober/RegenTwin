import { useEffect, useMemo, useRef, useState } from 'react';
import type { ChangeEvent, ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient, API_V1 } from '../lib/api';
import { useSimulationStore } from '../stores/simulationStore';
import { useCancelSimulation, useSimulationsList, useSimulationStatus, useStartSimulation } from '../hooks/useSimulation';
import { useSimulationWS } from '../hooks/useSimulationWS';
import { useResults } from '../hooks/useResults';
import {
  buildAreaPath,
  buildPath,
  buildYTicks,
  formatTick,
  maxOfSeries,
  niceMax,
  pickSeries,
  type SeriesScale,
} from '../lib/chartPath';
import { DEFAULT_PRESET, PRESETS, PRESET_LIST, type PresetName } from '../data/presets';
import type { SimulationMode, SimulationRequest, UploadResponse } from '../types/api';

type DashboardState = 'idle' | 'running' | 'complete' | 'error';
type SectionKey = 'populations' | 'cytokines' | 'time' | 'therapy' | 'monteCarlo';

type UploadedFileInfo = {
  name: string;
  size: number;
};

const INITIAL_SECTIONS: Record<SectionKey, boolean> = {
  populations: true,
  cytokines: false,
  time: true,
  therapy: true,
  monteCarlo: false,
};

function formatFileSize(size?: number | null): string {
  if (!size || size <= 0) return '—';
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function getModeLabel(mode?: SimulationMode | null): string {
  switch (mode) {
    case 'abm':
      return 'ABM';
    case 'extended':
      return 'SDE-20';
    case 'mvp':
      return 'MVP';
    case 'integrated':
      return 'Integrated';
    default:
      return '—';
  }
}

function getTherapyLabel(params?: Partial<SimulationRequest> | null): string {
  if (!params) return 'Control';
  if (params.prp_enabled && params.pemf_enabled) return 'PRP+PEMF';
  if (params.prp_enabled) return 'PRP only';
  if (params.pemf_enabled) return 'PEMF only';
  return 'Control';
}

function getStateFromStatus(status?: string | null): DashboardState {
  if (status === 'running' || status === 'pending' || status === 'cancelling') return 'running';
  if (status === 'completed') return 'complete';
  if (status === 'failed') return 'error';
  return 'idle';
}

function getStatusBadge(status?: string | null) {
  switch (status) {
    case 'running':
    case 'pending':
      return { dotColor: 'var(--warning)', badgeClass: 'badge badge-warning', label: 'running' };
    case 'cancelling':
      return { dotColor: 'var(--warning)', badgeClass: 'badge badge-warning', label: 'cancelling' };
    case 'cancelled':
      return { dotColor: 'var(--text-muted)', badgeClass: 'badge badge-info', label: 'cancelled' };
    case 'failed':
      return { dotColor: 'var(--danger)', badgeClass: 'badge badge-warning', label: 'failed' };
    case 'completed':
      return { dotColor: 'var(--success)', badgeClass: 'badge badge-success', label: 'completed' };
    default:
      return { dotColor: 'var(--info)', badgeClass: 'badge badge-info', label: 'waiting' };
  }
}

function formatClock(value?: string | null): string {
  if (!value) return '--:--';
  return new Date(value).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' });
}

function shortId(id?: string | null): string {
  return id ? id.slice(0, 8) : 'new-run';
}

function NumericField({
  label,
  value,
  onChange,
  step = 1,
  min = 0,
  max,
}: {
  label: ReactNode;
  value: number;
  onChange: (value: number) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  return (
    <div className="flex items-center gap-2">
      <label className="text-[11px] flex-1" style={{ color: 'var(--text-secondary)' }}>
        {label}
      </label>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
        style={{ width: 96 }}
      />
    </div>
  );
}

function SliderField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
          {label}
        </span>
        <span className="text-[10px] font-mono" style={{ color: 'var(--accent-text)' }}>
          {value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        className="range-slim"
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  );
}

function AccordionSection({
  open,
  onToggle,
  title,
  icon,
  badge,
  children,
}: {
  open: boolean;
  onToggle: () => void;
  title: string;
  icon: ReactNode;
  badge?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className={`accordion${open ? ' open' : ''}`}>
      <button className="acc-head" onClick={onToggle} type="button">
        {icon}
        <span className="flex-1 text-[11px] font-semibold uppercase tracking-wider">{title}</span>
        {badge}
        <svg
          className="acc-icon h-3 w-3"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          viewBox="0 0 24 24"
          style={{
            color: 'var(--text-muted)',
            transform: open ? 'rotate(90deg)' : 'rotate(0deg)',
            transition: 'transform 0.2s ease',
            transformOrigin: 'center',
          }}
          aria-hidden="true"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="m8.25 4.5 7.5 7.5-7.5 7.5" />
        </svg>
      </button>
      {open ? <div className="acc-body pl-2 pr-1 mt-1.5 space-y-1.5">{children}</div> : null}
    </div>
  );
}

function IdleStateView({ onUpload }: { onUpload: () => void }) {
  return (
    <div className="card flex flex-col items-center justify-center" style={{ minHeight: 600, padding: 40 }}>
      <div className="empty-zone" style={{ width: '100%', maxWidth: 520 }}>
        <div className="mx-auto mb-4 h-12 w-12 rounded-xl flex items-center justify-center" style={{ background: 'var(--accent-soft)' }}>
          <svg className="h-6 w-6" fill="none" stroke="var(--accent)" strokeWidth="1.5" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M7.5 7.5h-.75A2.25 2.25 0 0 0 4.5 9.75v7.5a2.25 2.25 0 0 0 2.25 2.25h7.5a2.25 2.25 0 0 0 2.25-2.25v-7.5a2.25 2.25 0 0 0-2.25-2.25h-.75m0-3-3-3m0 0-3 3m3-3v11.25"
            />
          </svg>
        </div>
        <div className="font-display text-base font-semibold mb-1.5" style={{ color: 'var(--text-primary)' }}>
          Загрузите .fcs файл для начала
        </div>
        <div className="text-xs mb-5" style={{ color: 'var(--text-muted)' }}>
          Или выберите один из пресетов в тулбаре выше — Chronic wound, Healthy control, Diabetic
        </div>
        <div className="flex items-center gap-2 justify-center">
          <button className="btn-primary" onClick={onUpload} type="button">
            <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M7.5 12 12 7.5m0 0 4.5 4.5M12 7.5V21" />
            </svg>
            Загрузить .fcs
          </button>
          <button className="btn-outline" type="button">
            Использовать пресет
          </button>
        </div>
      </div>
      <div className="mt-6 text-[11px] flex items-center gap-4" style={{ color: 'var(--text-muted)' }}>
        <span className="flex items-center gap-1.5">
          <svg className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
          </svg>
          Формат: .fcs (flow cytometry)
        </span>
        <span>·</span>
        <span>Макс. размер: 50 MB</span>
      </div>
    </div>
  );
}

function RunningStateView(_: {
  progress: number;
  message: string;
  onCancel: () => void;
  cancelling: boolean;
  modeLabel: string;
  tMaxHours: number;
}) {
  const { progress, message, onCancel, cancelling, modeLabel, tMaxHours } = _;
  const currentHours = Math.round((tMaxHours * progress) / 100);

  return (
    <div className="card p-6 flex flex-col items-center gap-5" style={{ minHeight: 600, justifyContent: 'center' }}>
      <div style={{ width: '100%', maxWidth: 540 }}>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div
              className="w-2 h-2 rounded-full"
              style={{ background: 'var(--accent)', boxShadow: '0 0 8px var(--accent)', animation: 'pulse 1.5s infinite' }}
            />
            <span className="font-display text-base font-semibold" style={{ color: 'var(--text-primary)' }}>
              Симуляция выполняется
            </span>
          </div>
          <span className="font-mono text-sm" style={{ color: 'var(--accent-text)' }}>
            {Math.round(progress)}%
          </span>
        </div>

        <div className="progress-track mb-3">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>

        <div className="flex items-center gap-3 mb-5 text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
          <svg className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="1.6" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6l4 2M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20Z" />
          </svg>
          <span>{message || `Running ${modeLabel} simulation — t=${currentHours}/${tMaxHours}h`}</span>
          <div className="flex-1" />
          <span>{currentHours}/{tMaxHours} ч</span>
        </div>

        <button
          className="btn-outline"
          style={{ borderColor: 'rgba(248,81,73,0.3)', color: '#f85149', opacity: cancelling ? 0.6 : 1 }}
          onClick={onCancel}
          type="button"
          disabled={cancelling}
        >
          <svg className="h-3 w-3" fill="currentColor" viewBox="0 0 24 24">
            <rect x="5" y="5" width="14" height="14" rx="1" />
          </svg>
          {cancelling ? 'Отмена…' : 'Отмена'}
        </button>
      </div>

      <div className="grid grid-cols-3 gap-3 w-full max-w-[540px] mt-4 pt-4 border-t" style={{ borderColor: 'var(--border-default)' }}>
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Популяции</span>
          </div>
          <svg viewBox="0 0 100 24" className="w-full" style={{ height: 24 }}>
            <polyline points="0,20 20,18 40,15 60,10 80,7 100,5" fill="none" stroke="var(--accent)" strokeWidth="1.3" />
          </svg>
        </div>
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>TGF-β</span>
          </div>
          <svg viewBox="0 0 100 24" className="w-full" style={{ height: 24 }}>
            <polyline points="0,22 30,21 60,18 80,13 100,8" fill="none" stroke="#b392f0" strokeWidth="1.3" />
          </svg>
        </div>
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Коллаген</span>
          </div>
          <svg viewBox="0 0 100 24" className="w-full" style={{ height: 24 }}>
            <polyline points="0,22 40,21 70,18 100,14" fill="none" stroke="#58a6ff" strokeWidth="1.3" />
          </svg>
        </div>
      </div>
    </div>
  );
}

function ErrorStateView(_: { message: string; onRetry: () => void }) {
  const { message, onRetry } = _;
  return (
    <div className="card flex flex-col items-center justify-center gap-4" style={{ minHeight: 600, padding: 40 }}>
      <div style={{ width: '100%', maxWidth: 520 }}>
        <div className="card p-5" style={{ background: 'rgba(248,81,73,0.06)', borderColor: 'rgba(248,81,73,0.25)' }}>
          <div className="flex items-start gap-3">
            <div className="h-10 w-10 rounded-lg shrink-0 flex items-center justify-center" style={{ background: 'rgba(248,81,73,0.12)' }}>
              <svg className="h-5 w-5" fill="none" stroke="#f85149" strokeWidth="1.6" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z" />
              </svg>
            </div>
            <div className="flex-1">
              <div className="font-display text-sm font-semibold mb-1" style={{ color: '#ff7b72' }}>
                Ошибка симуляции
              </div>
              <div className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                {message || 'Симуляция завершилась с ошибкой. Проверьте логи сервера.'}
              </div>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2 mt-4">
          <button className="btn-primary" onClick={onRetry} type="button">
            <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99" />
            </svg>
            Повторить
          </button>
          <button className="btn-outline" type="button">
            Открыть логи
          </button>
          <button className="btn-ghost" type="button">
            Сообщить о баге
          </button>
        </div>
      </div>
    </div>
  );
}

type ResultsData = {
  times: number[];
  variables: Record<string, number[]>;
  metadata?: Record<string, unknown>;
};

const POPULATION_SERIES: { label: string; color: string; aliases: string[]; strokeWidth?: number }[] = [
  { label: 'P',  color: '#d29922', aliases: ['P'] },
  { label: 'Ne', color: '#f85149', aliases: ['Ne'] },
  { label: 'M1', color: '#e5534b', aliases: ['M1', 'mean_M1', 'macro'] },
  { label: 'M2', color: '#56d364', aliases: ['M2', 'mean_M2'] },
  { label: 'F',  color: '#58a6ff', aliases: ['F', 'mean_F', 'fibro'], strokeWidth: 1.3 },
  { label: 'Mf', color: '#b392f0', aliases: ['Mf', 'mean_Mf'], strokeWidth: 1.3 },
  { label: 'E',  color: '#6ac1b4', aliases: ['E', 'mean_E'] },
  { label: 'S',  color: '#d4843e', aliases: ['S', 'mean_S', 'stem'] },
];

const CYTOKINE_SERIES: { label: string; color: string; aliases: string[]; strokeWidth?: number; opacity?: number }[] = [
  { label: 'TGF-β', color: '#b392f0', aliases: ['C_TGFb', 'mean_C_TGFb'], strokeWidth: 1.3 },
  { label: 'VEGF',  color: '#6ac1b4', aliases: ['C_VEGF', 'mean_C_VEGF'] },
  { label: 'PDGF',  color: '#58a6ff', aliases: ['C_PDGF', 'mean_C_PDGF'] },
  { label: 'IL-10', color: '#56d364', aliases: ['C_IL10', 'mean_C_IL10'] },
  { label: 'TNF-α', color: '#f85149', aliases: ['C_TNF', 'mean_C_TNF'], opacity: 0.85 },
  { label: 'MCP-1', color: '#d29922', aliases: ['C_MCP1', 'mean_C_MCP1'], opacity: 0.75 },
];

const X_LABEL_POSITIONS = [40, 127, 215, 303, 380] as const;

function formatXTicks(tMax: number): string[] {
  if (!Number.isFinite(tMax) || tMax <= 0) return ['0', '180', '360', '540', '720'];
  return X_LABEL_POSITIONS.map((_, i) => {
    const t = (tMax / 4) * i;
    return t >= 100 ? Math.round(t).toString() : t.toFixed(1);
  });
}

type CardKind = 'populations' | 'cytokines' | 'ecm' | 'phases';

function ExpandButton({ onClick }: { onClick?: () => void }) {
  if (!onClick) return null;
  return (
    <button
      className="btn-ghost p-1"
      style={{ padding: 3 }}
      type="button"
      onClick={onClick}
      title="Развернуть"
    >
      <svg className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" />
      </svg>
    </button>
  );
}

function ChartModalShell({
  title,
  subtitle,
  icon,
  onClose,
  children,
}: {
  title: string;
  subtitle?: string;
  icon: ReactNode;
  onClose: () => void;
  children: ReactNode;
}) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(10, 13, 18, 0.72)',
        backdropFilter: 'blur(6px)',
        zIndex: 50,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
      }}
      onClick={onClose}
    >
      <div
        className="card"
        style={{
          width: 'min(1200px, 95vw)',
          height: 'min(820px, 92vh)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
        onClick={(event) => event.stopPropagation()}
      >
        <div
          className="flex items-center gap-2 px-4 py-3 border-b"
          style={{ borderColor: 'var(--border-default)' }}
        >
          {icon}
          <h3 className="font-display text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{title}</h3>
          {subtitle ? (
            <span className="text-[11px] font-mono" style={{ color: 'var(--text-muted)' }}>{subtitle}</span>
          ) : null}
          <div className="flex-1" />
          <button className="btn-ghost" onClick={onClose} type="button" title="Закрыть (Esc)">
            <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', padding: 16, gap: 12 }}>
          {children}
        </div>
      </div>
    </div>
  );
}

function sampleIndexes(total: number, maxRows: number): number[] {
  if (total <= 0) return [];
  if (total <= maxRows) return Array.from({ length: total }, (_, i) => i);
  const step = (total - 1) / (maxRows - 1);
  const idxs: number[] = [];
  for (let i = 0; i < maxRows; i += 1) idxs.push(Math.round(i * step));
  return idxs;
}

function formatCell(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return '—';
  const abs = Math.abs(value);
  if (abs === 0) return '0';
  if (abs >= 10_000) return value.toFixed(0);
  if (abs >= 100) return value.toFixed(1);
  if (abs >= 1) return value.toFixed(2);
  if (abs >= 0.01) return value.toFixed(3);
  return value.toExponential(2);
}

function ChartDataTable({
  times,
  series,
  totalLabel,
}: {
  times: number[];
  series: { label: string; color: string; values: number[] | undefined }[];
  totalLabel?: string;
}) {
  const rows = useMemo(() => {
    const idxs = sampleIndexes(times.length, 200);
    return idxs.map((i) => ({
      idx: i,
      t: times[i] ?? 0,
      values: series.map((s) => s.values?.[i]),
    }));
  }, [times, series]);

  return (
    <div
      style={{
        flex: 1,
        minHeight: 0,
        overflow: 'auto',
        border: '1px solid var(--border-default)',
        borderRadius: 8,
        background: 'var(--surface-2)',
      }}
    >
      <div
        className="flex items-center justify-between px-3 py-2 border-b"
        style={{
          borderColor: 'var(--border-default)',
          position: 'sticky',
          top: 0,
          background: 'var(--surface-1)',
          zIndex: 1,
        }}
      >
        <span className="section-label">Таблица данных</span>
        <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
          {totalLabel ?? `${times.length} точек · показано ${rows.length}`}
        </span>
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }} className="font-mono">
        <thead>
          <tr style={{ background: 'var(--surface-1)' }}>
            <th
              style={{
                padding: '6px 10px',
                textAlign: 'left',
                color: 'var(--text-muted)',
                fontWeight: 500,
                borderBottom: '1px solid var(--border-default)',
                position: 'sticky',
                top: 37,
                background: 'var(--surface-1)',
              }}
            >
              t, ч
            </th>
            {series.map((s) => (
              <th
                key={s.label}
                style={{
                  padding: '6px 10px',
                  textAlign: 'right',
                  color: s.color,
                  fontWeight: 500,
                  borderBottom: '1px solid var(--border-default)',
                  position: 'sticky',
                  top: 37,
                  background: 'var(--surface-1)',
                }}
              >
                {s.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.idx} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
              <td style={{ padding: '4px 10px', color: 'var(--text-secondary)' }}>{formatCell(row.t)}</td>
              {row.values.map((v, i) => (
                <td key={i} style={{ padding: '4px 10px', textAlign: 'right', color: 'var(--text-primary)' }}>
                  {formatCell(v)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PopulationCard({ simulationId, onExpand }: { simulationId: string | null; onExpand?: () => void }) {
  const { data, isLoading } = useResults(simulationId ?? undefined);
  const results = data as ResultsData | undefined;

  const { paths, yTicks, xTicks, hasData } = useMemo(() => {
    if (!results) return { paths: [], yTicks: buildYTicks(1), xTicks: formatXTicks(720), hasData: false };
    const times = results.times ?? [];
    const tMax = times.length ? (times[times.length - 1] ?? 720) : 720;
    const keys = POPULATION_SERIES.flatMap((s) => s.aliases);
    const rawMax = maxOfSeries(results.variables, keys);
    const yMax = niceMax(rawMax);
    const scale: SeriesScale = { xMin: 0, xMax: tMax, yMin: 0, yMax, plotLeft: 40, plotRight: 390, plotTop: 20, plotBottom: 180 };
    const paths = POPULATION_SERIES.map((series) => ({
      color: series.color,
      strokeWidth: series.strokeWidth ?? 1,
      d: buildPath(times, pickSeries(results.variables, series.aliases), scale),
    }));
    return { paths, yTicks: buildYTicks(rawMax), xTicks: formatXTicks(tMax), hasData: rawMax > 0 };
  }, [results]);

  return (
    <div className="card p-4 flex flex-col">
      <div className="flex items-center gap-2 mb-2">
        <svg className="h-3.5 w-3.5" fill="none" stroke="var(--accent-text)" strokeWidth="1.6" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M15 19.1a9.4 9.4 0 0 0 2.6.4 9.3 9.3 0 0 0 4.1-1 4.1 4.1 0 0 0-7.5-2.5M15 19.1v-.1c0-1.1-.3-2.1-.8-3.1M15 19.1v.1a12.3 12.3 0 0 1-6.4 1.8c-2.3 0-4.5-.6-6.4-1.8v-.1a6.4 6.4 0 0 1 12-3.1M12 6.4a3.4 3.4 0 1 1-6.75 0 3.4 3.4 0 0 1 6.75 0Zm8.25 2.3a2.6 2.6 0 1 1-5.25 0 2.6 2.6 0 0 1 5.25 0Z" />
        </svg>
        <h3 className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>Клеточные популяции</h3>
        <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>клеток/мкл</span>
        <div className="flex-1" />
        <ExpandButton onClick={onExpand} />
      </div>

      <div className="flex-1 relative">
        <svg viewBox="0 0 400 220" className="w-full h-full" preserveAspectRatio="none">
          <g className="chart-grid">
            <line x1="40" y1="20" x2="390" y2="20" />
            <line x1="40" y1="60" x2="390" y2="60" />
            <line x1="40" y1="100" x2="390" y2="100" />
            <line x1="40" y1="140" x2="390" y2="140" />
            <line x1="40" y1="180" x2="390" y2="180" />
          </g>
          {yTicks.map((value, idx) => (
            <text key={idx} x={35} y={23 + idx * 40} className="chart-axis" textAnchor="end">{formatTick(value)}</text>
          ))}
          {xTicks.map((label, idx) => (
            <text key={idx} x={X_LABEL_POSITIONS[idx]} y={200} className="chart-axis">{label}</text>
          ))}

          {paths.map((p, idx) => (
            p.d ? <path key={idx} className="plot-line" stroke={p.color} strokeWidth={p.strokeWidth} d={p.d} /> : null
          ))}

          {!hasData && !isLoading ? (
            <text x="215" y="110" textAnchor="middle" className="chart-axis" opacity="0.5">нет данных для выбранного прогона</text>
          ) : null}
          {isLoading && !hasData ? (
            <text x="215" y="110" textAnchor="middle" className="chart-axis" opacity="0.5">загрузка данных…</text>
          ) : null}
        </svg>
      </div>

      <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2 text-[10px]">
        {POPULATION_SERIES.map((s) => (
          <span key={s.label} className="flex items-center gap-1"><span className="w-2.5 h-0.5" style={{ background: s.color }} /><span style={{ color: 'var(--text-secondary)' }}>{s.label}</span></span>
        ))}
      </div>
    </div>
  );
}

function CytokinesCard({ simulationId, onExpand }: { simulationId: string | null; onExpand?: () => void }) {
  const { data, isLoading } = useResults(simulationId ?? undefined);
  const results = data as ResultsData | undefined;

  const { paths, yTicks, xTicks, hasData } = useMemo(() => {
    if (!results) return { paths: [], yTicks: buildYTicks(1), xTicks: formatXTicks(720), hasData: false };
    const times = results.times ?? [];
    const tMax = times.length ? (times[times.length - 1] ?? 720) : 720;
    const keys = CYTOKINE_SERIES.flatMap((s) => s.aliases);
    const rawMax = maxOfSeries(results.variables, keys);
    const yMax = niceMax(rawMax);
    // Шкала: 5 горизонтальных делений (30, 75, 120, 165, ~208 baseline)
    const scale: SeriesScale = { xMin: 0, xMax: tMax, yMin: 0, yMax, plotLeft: 40, plotRight: 390, plotTop: 30, plotBottom: 180 };
    const paths = CYTOKINE_SERIES.map((series) => ({
      color: series.color,
      strokeWidth: series.strokeWidth ?? 1,
      opacity: series.opacity ?? 1,
      d: buildPath(times, pickSeries(results.variables, series.aliases), scale),
    }));
    return { paths, yTicks: buildYTicks(rawMax), xTicks: formatXTicks(tMax), hasData: rawMax > 0 };
  }, [results]);

  return (
    <div className="card p-4 flex flex-col">
      <div className="flex items-center gap-2 mb-2">
        <svg className="h-3.5 w-3.5" fill="none" stroke="#79b8ff" strokeWidth="1.6" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.1v5.7c0 .6-.24 1.17-.66 1.6L5 14.5M14.25 3.1v5.7c0 .6.24 1.17.66 1.6L19.8 15.3M5 14.5h14.8" />
        </svg>
        <h3 className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>Цитокины</h3>
        <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>нг/мл</span>
        <div className="flex-1" />
        <ExpandButton onClick={onExpand} />
      </div>
      <div className="flex-1 relative">
        <svg viewBox="0 0 400 220" className="w-full h-full" preserveAspectRatio="none">
          <g className="chart-grid">
            <line x1="40" y1="30" x2="390" y2="30" />
            <line x1="40" y1="75" x2="390" y2="75" />
            <line x1="40" y1="120" x2="390" y2="120" />
            <line x1="40" y1="165" x2="390" y2="165" />
          </g>
          {yTicks.map((value, idx) => (
            <text key={idx} x={35} y={33 + idx * 45} className="chart-axis" textAnchor="end">{formatTick(value)}</text>
          ))}
          <text x={40} y={200} className="chart-axis">{xTicks[0]}</text>
          <text x={215} y={200} className="chart-axis">{xTicks[2]}</text>
          <text x={380} y={200} className="chart-axis">{xTicks[4]}</text>

          {paths.map((p, idx) => (
            p.d ? <path key={idx} className="plot-line" stroke={p.color} strokeWidth={p.strokeWidth} opacity={p.opacity} d={p.d} /> : null
          ))}

          {!hasData && !isLoading ? (
            <text x="215" y="110" textAnchor="middle" className="chart-axis" opacity="0.5">нет данных для выбранного прогона</text>
          ) : null}
          {isLoading && !hasData ? (
            <text x="215" y="110" textAnchor="middle" className="chart-axis" opacity="0.5">загрузка данных…</text>
          ) : null}
        </svg>
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2 text-[10px]">
        {CYTOKINE_SERIES.map((s) => (
          <span key={s.label} className="flex items-center gap-1"><span className="w-2.5 h-0.5" style={{ background: s.color }} /><span style={{ color: 'var(--text-secondary)' }}>{s.label}</span></span>
        ))}
      </div>
    </div>
  );
}

function EcmCard({ simulationId, onExpand }: { simulationId: string | null; onExpand?: () => void }) {
  const { data, isLoading } = useResults(simulationId ?? undefined);
  const results = data as ResultsData | undefined;

  const { collagenArea, fibrinLine, mmpLine, leftTicks, rightTicks, xTicks, hasData } = useMemo(() => {
    if (!results) return { collagenArea: '', fibrinLine: '', mmpLine: '', leftTicks: buildYTicks(60), rightTicks: buildYTicks(3000), xTicks: formatXTicks(720), hasData: false };
    const times = results.times ?? [];
    const tMax = times.length ? (times[times.length - 1] ?? 720) : 720;
    const leftRaw = maxOfSeries(results.variables, ['rho_collagen', 'rho_fibrin', 'mean_rho_collagen', 'mean_rho_fibrin']);
    const rightRaw = maxOfSeries(results.variables, ['C_MMP', 'mean_C_MMP']);
    const yMaxLeft = niceMax(leftRaw);
    const yMaxRight = niceMax(rightRaw);
    const scaleLeft: SeriesScale = { xMin: 0, xMax: tMax, yMin: 0, yMax: yMaxLeft, plotLeft: 40, plotRight: 390, plotTop: 30, plotBottom: 180 };
    const scaleRight: SeriesScale = { ...scaleLeft, yMax: yMaxRight };
    const collagenArea = buildAreaPath(times, pickSeries(results.variables, ['rho_collagen', 'mean_rho_collagen']), scaleLeft);
    const fibrinLine = buildPath(times, pickSeries(results.variables, ['rho_fibrin', 'mean_rho_fibrin']), scaleLeft);
    const mmpLine = buildPath(times, pickSeries(results.variables, ['C_MMP', 'mean_C_MMP']), scaleRight);
    return { collagenArea, fibrinLine, mmpLine, leftTicks: buildYTicks(leftRaw), rightTicks: buildYTicks(rightRaw), xTicks: formatXTicks(tMax), hasData: leftRaw + rightRaw > 0 };
  }, [results]);

  return (
    <div className="card p-4 flex flex-col">
      <div className="flex items-center gap-2 mb-2">
        <svg className="h-3.5 w-3.5" fill="none" stroke="var(--accent-text)" strokeWidth="1.6" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M21 7.5 12 2.25 3 7.5m18 0-9 5.25m9-5.25v9L12 21.75M3 7.5 12 12.75M3 7.5v9L12 21.75m0-9v9" />
        </svg>
        <h3 className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>Внеклеточный матрикс</h3>
        <div className="flex-1" />
        <ExpandButton onClick={onExpand} />
      </div>
      <div className="flex-1 relative">
        <svg viewBox="0 0 400 220" className="w-full h-full" preserveAspectRatio="none">
          <g className="chart-grid">
            <line x1="40" y1="30" x2="390" y2="30" />
            <line x1="40" y1="75" x2="390" y2="75" />
            <line x1="40" y1="120" x2="390" y2="120" />
            <line x1="40" y1="165" x2="390" y2="165" />
          </g>
          {leftTicks.map((value, idx) => (
            <text key={idx} x={35} y={33 + idx * 45} className="chart-axis" textAnchor="end">{formatTick(value)}</text>
          ))}
          {rightTicks.map((value, idx) => (
            <text key={idx} x={395} y={33 + idx * 45} className="chart-axis" textAnchor="start">{formatTick(value)}</text>
          ))}
          <text x={40} y={200} className="chart-axis">{xTicks[0]}</text>
          <text x={215} y={200} className="chart-axis">{xTicks[2]}</text>
          <text x={380} y={200} className="chart-axis">{xTicks[4]}</text>

          {collagenArea ? <path d={collagenArea} fill="#58a6ff" opacity="0.35" stroke="#58a6ff" strokeWidth="0.8" /> : null}
          {fibrinLine ? <path className="plot-line" stroke="#d29922" strokeWidth="1.1" d={fibrinLine} /> : null}
          {mmpLine ? <path className="plot-line" stroke="#f85149" strokeWidth="1.3" strokeDasharray="4,3" d={mmpLine} /> : null}

          {!hasData && !isLoading ? (
            <text x="215" y="110" textAnchor="middle" className="chart-axis" opacity="0.5">нет данных для выбранного прогона</text>
          ) : null}
          {isLoading && !hasData ? (
            <text x="215" y="110" textAnchor="middle" className="chart-axis" opacity="0.5">загрузка данных…</text>
          ) : null}
        </svg>
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2 text-[10px]">
        <span className="flex items-center gap-1"><span className="w-2.5 h-2 rounded-sm opacity-40" style={{ background: '#58a6ff' }} /><span style={{ color: 'var(--text-secondary)' }}>Коллаген (ρ_c)</span></span>
        <span className="flex items-center gap-1"><span className="w-2.5 h-0.5" style={{ background: '#d29922' }} /><span style={{ color: 'var(--text-secondary)' }}>Фибрин (ρ_f)</span></span>
        <span className="flex items-center gap-1"><span className="w-2.5 h-0.5" style={{ background: '#f85149', borderTop: '1px dashed #f85149' }} /><span style={{ color: 'var(--text-secondary)' }}>MMP <span style={{ color: 'var(--text-muted)' }}>(правая ось)</span></span></span>
      </div>
    </div>
  );
}

const PHASE_FALLBACK = { hemostasis: 3, inflammation: 15, proliferation: 55, remodeling: 27 };

function readPhaseShares(metadata: Record<string, unknown> | undefined) {
  const raw = metadata?.['phase_shares'];
  if (raw && typeof raw === 'object') {
    const obj = raw as Record<string, unknown>;
    const h = Number(obj.hemostasis ?? 0);
    const i = Number(obj.inflammation ?? 0);
    const p = Number(obj.proliferation ?? 0);
    const r = Number(obj.remodeling ?? 0);
    const total = h + i + p + r;
    if (total > 0 && [h, i, p, r].every(Number.isFinite)) {
      return { hemostasis: h, inflammation: i, proliferation: p, remodeling: r };
    }
  }
  return PHASE_FALLBACK;
}

function PhasesCard({ simulationId, onExpand }: { simulationId: string | null; onExpand?: () => void }) {
  const { data, isLoading } = useResults(simulationId ?? undefined);
  const results = data as ResultsData | undefined;

  const { fLine, mLine, tnfLine, yTicks, xTicks, phases, hasData } = useMemo(() => {
    const fallback = { fLine: '', mLine: '', tnfLine: '', yTicks: buildYTicks(60000), xTicks: formatXTicks(720), phases: PHASE_FALLBACK, hasData: false };
    if (!results) return fallback;
    const times = results.times ?? [];
    const tMax = times.length ? (times[times.length - 1] ?? 720) : 720;
    const rawMax = maxOfSeries(results.variables, ['F', 'Mf', 'mean_F', 'mean_Mf', 'fibro']);
    const yMax = niceMax(rawMax);
    const scale: SeriesScale = { xMin: 0, xMax: tMax, yMin: 0, yMax, plotLeft: 40, plotRight: 390, plotTop: 30, plotBottom: 170 };
    const fLine = buildPath(times, pickSeries(results.variables, ['F', 'mean_F', 'fibro']), scale);
    const mLine = buildPath(times, pickSeries(results.variables, ['Mf', 'mean_Mf']), scale);
    const tnfLine = buildPath(times, pickSeries(results.variables, ['C_TNF', 'mean_C_TNF']), { ...scale, yMax: niceMax(maxOfSeries(results.variables, ['C_TNF', 'mean_C_TNF'])) });
    return { fLine, mLine, tnfLine, yTicks: buildYTicks(rawMax), xTicks: formatXTicks(tMax), phases: readPhaseShares(results.metadata), hasData: rawMax > 0 };
  }, [results]);

  return (
    <div className="card p-4 flex flex-col">
      <div className="flex items-center gap-2 mb-2">
        <svg className="h-3.5 w-3.5" fill="none" stroke="#e3b341" strokeWidth="1.6" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="9" />
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 7v5l3 2" />
        </svg>
        <h3 className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>Фазы заживления</h3>
        <div className="flex-1" />
        <ExpandButton onClick={onExpand} />
      </div>
      <div className="flex h-3 rounded overflow-hidden mb-2">
        <div style={{ flex: phases.hemostasis, background: '#e5534b' }} title={`Hemostasis ${phases.hemostasis.toFixed(0)}%`} />
        <div style={{ flex: phases.inflammation, background: '#d29922' }} title={`Inflammation ${phases.inflammation.toFixed(0)}%`} />
        <div style={{ flex: phases.proliferation, background: '#56d364' }} title={`Proliferation ${phases.proliferation.toFixed(0)}%`} />
        <div style={{ flex: phases.remodeling, background: '#6ac1b4' }} title={`Remodeling ${phases.remodeling.toFixed(0)}%`} />
      </div>
      <div className="flex-1 relative">
        <svg viewBox="0 0 400 200" className="w-full h-full" preserveAspectRatio="none">
          <g className="chart-grid">
            <line x1="40" y1="30" x2="390" y2="30" />
            <line x1="40" y1="75" x2="390" y2="75" />
            <line x1="40" y1="120" x2="390" y2="120" />
            <line x1="40" y1="165" x2="390" y2="165" />
          </g>
          {yTicks.map((value, idx) => (
            <text key={idx} x={35} y={33 + idx * 45} className="chart-axis" textAnchor="end">{formatTick(value)}</text>
          ))}
          <text x={40} y={185} className="chart-axis">{xTicks[0]}</text>
          <text x={215} y={185} className="chart-axis">{xTicks[2]}</text>
          <text x={380} y={185} className="chart-axis">{xTicks[4]}</text>

          {fLine ? <path className="plot-line" stroke="#58a6ff" strokeWidth="1.3" d={fLine} /> : null}
          {mLine ? <path className="plot-line" stroke="#b392f0" strokeWidth="1" opacity="0.8" d={mLine} /> : null}
          {tnfLine ? <path className="plot-line" stroke="#f85149" strokeWidth="1" opacity="0.7" d={tnfLine} /> : null}

          {!hasData && !isLoading ? (
            <text x="215" y="105" textAnchor="middle" className="chart-axis" opacity="0.5">нет данных для выбранного прогона</text>
          ) : null}
          {isLoading && !hasData ? (
            <text x="215" y="105" textAnchor="middle" className="chart-axis" opacity="0.5">загрузка данных…</text>
          ) : null}
        </svg>
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2 text-[10px]">
        <span className="flex items-center gap-1"><span className="w-2.5 h-2 rounded-sm" style={{ background: '#e5534b' }} /><span style={{ color: 'var(--text-secondary)' }}>Hemostasis</span></span>
        <span className="flex items-center gap-1"><span className="w-2.5 h-2 rounded-sm" style={{ background: '#d29922' }} /><span style={{ color: 'var(--text-secondary)' }}>Inflammation</span></span>
        <span className="flex items-center gap-1"><span className="w-2.5 h-2 rounded-sm" style={{ background: '#56d364' }} /><span style={{ color: 'var(--text-secondary)' }}>Proliferation</span></span>
        <span className="flex items-center gap-1"><span className="w-2.5 h-2 rounded-sm" style={{ background: '#6ac1b4' }} /><span style={{ color: 'var(--text-secondary)' }}>Remodeling</span></span>
        <span className="flex items-center gap-1"><span className="w-2.5 h-0.5" style={{ background: '#58a6ff' }} /><span style={{ color: 'var(--text-secondary)' }}>F</span></span>
      </div>
    </div>
  );
}

function CompleteStateView({
  simulationId,
  onExpand,
}: {
  simulationId: string | null;
  onExpand: (kind: CardKind) => void;
}) {
  return (
    <div className="grid grid-cols-2 grid-rows-2 gap-4" style={{ minHeight: 600 }}>
      <PopulationCard simulationId={simulationId} onExpand={() => onExpand('populations')} />
      <CytokinesCard simulationId={simulationId} onExpand={() => onExpand('cytokines')} />
      <EcmCard simulationId={simulationId} onExpand={() => onExpand('ecm')} />
      <PhasesCard simulationId={simulationId} onExpand={() => onExpand('phases')} />
    </div>
  );
}

const POPULATION_ICON = (
  <svg className="h-4 w-4" fill="none" stroke="var(--accent-text)" strokeWidth="1.6" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" d="M15 19.1a9.4 9.4 0 0 0 2.6.4 9.3 9.3 0 0 0 4.1-1 4.1 4.1 0 0 0-7.5-2.5M15 19.1v-.1c0-1.1-.3-2.1-.8-3.1M15 19.1v.1a12.3 12.3 0 0 1-6.4 1.8c-2.3 0-4.5-.6-6.4-1.8v-.1a6.4 6.4 0 0 1 12-3.1M12 6.4a3.4 3.4 0 1 1-6.75 0 3.4 3.4 0 0 1 6.75 0Zm8.25 2.3a2.6 2.6 0 1 1-5.25 0 2.6 2.6 0 0 1 5.25 0Z" />
  </svg>
);
const CYTOKINES_ICON = (
  <svg className="h-4 w-4" fill="none" stroke="#79b8ff" strokeWidth="1.6" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.1v5.7c0 .6-.24 1.17-.66 1.6L5 14.5M14.25 3.1v5.7c0 .6.24 1.17.66 1.6L19.8 15.3M5 14.5h14.8" />
  </svg>
);
const ECM_ICON = (
  <svg className="h-4 w-4" fill="none" stroke="var(--accent-text)" strokeWidth="1.6" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" d="M21 7.5 12 2.25 3 7.5m18 0-9 5.25m9-5.25v9L12 21.75M3 7.5 12 12.75M3 7.5v9L12 21.75m0-9v9" />
  </svg>
);
const PHASES_ICON = (
  <svg className="h-4 w-4" fill="none" stroke="#e3b341" strokeWidth="1.6" viewBox="0 0 24 24">
    <circle cx="12" cy="12" r="9" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 7v5l3 2" />
  </svg>
);

function ExpandedChartFrame({ children, svgAspect }: { children: ReactNode; svgAspect: string }) {
  return (
    <div
      style={{
        border: '1px solid var(--border-default)',
        borderRadius: 8,
        background: 'var(--surface-2)',
        padding: 12,
        aspectRatio: svgAspect,
        minHeight: 300,
        maxHeight: '55%',
      }}
    >
      {children}
    </div>
  );
}

function PopulationExpanded({ simulationId, onClose }: { simulationId: string | null; onClose: () => void }) {
  const { data, isLoading } = useResults(simulationId ?? undefined);
  const results = data as ResultsData | undefined;

  const { paths, yTicks, xTicks, hasData, times, tableSeries } = useMemo(() => {
    const empty = { paths: [], yTicks: buildYTicks(1), xTicks: formatXTicks(720), hasData: false, times: [] as number[], tableSeries: [] as { label: string; color: string; values: number[] | undefined }[] };
    if (!results) return empty;
    const t = results.times ?? [];
    const tMax = t.length ? (t[t.length - 1] ?? 720) : 720;
    const keys = POPULATION_SERIES.flatMap((s) => s.aliases);
    const rawMax = maxOfSeries(results.variables, keys);
    const yMax = niceMax(rawMax);
    const scale: SeriesScale = { xMin: 0, xMax: tMax, yMin: 0, yMax, plotLeft: 40, plotRight: 390, plotTop: 20, plotBottom: 180 };
    const paths = POPULATION_SERIES.map((series) => ({
      color: series.color,
      strokeWidth: series.strokeWidth ?? 1,
      d: buildPath(t, pickSeries(results.variables, series.aliases), scale),
    }));
    const tableSeries = POPULATION_SERIES.map((series) => ({
      label: series.label,
      color: series.color,
      values: pickSeries(results.variables, series.aliases),
    }));
    return { paths, yTicks: buildYTicks(rawMax), xTicks: formatXTicks(tMax), hasData: rawMax > 0, times: t, tableSeries };
  }, [results]);

  return (
    <ChartModalShell
      title="Клеточные популяции"
      subtitle="клеток/мкл"
      icon={POPULATION_ICON}
      onClose={onClose}
    >
      <ExpandedChartFrame svgAspect="400 / 220">
        <svg viewBox="0 0 400 220" className="w-full h-full" preserveAspectRatio="none">
          <g className="chart-grid">
            <line x1="40" y1="20" x2="390" y2="20" />
            <line x1="40" y1="60" x2="390" y2="60" />
            <line x1="40" y1="100" x2="390" y2="100" />
            <line x1="40" y1="140" x2="390" y2="140" />
            <line x1="40" y1="180" x2="390" y2="180" />
          </g>
          {yTicks.map((value, idx) => (
            <text key={idx} x={35} y={23 + idx * 40} className="chart-axis" textAnchor="end">{formatTick(value)}</text>
          ))}
          {xTicks.map((label, idx) => (
            <text key={idx} x={X_LABEL_POSITIONS[idx]} y={200} className="chart-axis">{label}</text>
          ))}
          {paths.map((p, idx) => (
            p.d ? <path key={idx} className="plot-line" stroke={p.color} strokeWidth={p.strokeWidth} d={p.d} /> : null
          ))}
          {!hasData && !isLoading ? (
            <text x="215" y="110" textAnchor="middle" className="chart-axis" opacity="0.5">нет данных для выбранного прогона</text>
          ) : null}
          {isLoading && !hasData ? (
            <text x="215" y="110" textAnchor="middle" className="chart-axis" opacity="0.5">загрузка данных…</text>
          ) : null}
        </svg>
      </ExpandedChartFrame>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
        {POPULATION_SERIES.map((s) => (
          <span key={s.label} className="flex items-center gap-1.5"><span className="w-3 h-0.5" style={{ background: s.color }} /><span style={{ color: 'var(--text-secondary)' }}>{s.label}</span></span>
        ))}
      </div>
      <ChartDataTable times={times} series={tableSeries} />
    </ChartModalShell>
  );
}

function CytokinesExpanded({ simulationId, onClose }: { simulationId: string | null; onClose: () => void }) {
  const { data, isLoading } = useResults(simulationId ?? undefined);
  const results = data as ResultsData | undefined;

  const { paths, yTicks, xTicks, hasData, times, tableSeries } = useMemo(() => {
    const empty = { paths: [], yTicks: buildYTicks(1), xTicks: formatXTicks(720), hasData: false, times: [] as number[], tableSeries: [] as { label: string; color: string; values: number[] | undefined }[] };
    if (!results) return empty;
    const t = results.times ?? [];
    const tMax = t.length ? (t[t.length - 1] ?? 720) : 720;
    const keys = CYTOKINE_SERIES.flatMap((s) => s.aliases);
    const rawMax = maxOfSeries(results.variables, keys);
    const yMax = niceMax(rawMax);
    const scale: SeriesScale = { xMin: 0, xMax: tMax, yMin: 0, yMax, plotLeft: 40, plotRight: 390, plotTop: 30, plotBottom: 180 };
    const paths = CYTOKINE_SERIES.map((series) => ({
      color: series.color,
      strokeWidth: series.strokeWidth ?? 1,
      opacity: series.opacity ?? 1,
      d: buildPath(t, pickSeries(results.variables, series.aliases), scale),
    }));
    const tableSeries = CYTOKINE_SERIES.map((series) => ({
      label: series.label,
      color: series.color,
      values: pickSeries(results.variables, series.aliases),
    }));
    return { paths, yTicks: buildYTicks(rawMax), xTicks: formatXTicks(tMax), hasData: rawMax > 0, times: t, tableSeries };
  }, [results]);

  return (
    <ChartModalShell title="Цитокины" subtitle="нг/мл" icon={CYTOKINES_ICON} onClose={onClose}>
      <ExpandedChartFrame svgAspect="400 / 220">
        <svg viewBox="0 0 400 220" className="w-full h-full" preserveAspectRatio="none">
          <g className="chart-grid">
            <line x1="40" y1="30" x2="390" y2="30" />
            <line x1="40" y1="75" x2="390" y2="75" />
            <line x1="40" y1="120" x2="390" y2="120" />
            <line x1="40" y1="165" x2="390" y2="165" />
          </g>
          {yTicks.map((value, idx) => (
            <text key={idx} x={35} y={33 + idx * 45} className="chart-axis" textAnchor="end">{formatTick(value)}</text>
          ))}
          <text x={40} y={200} className="chart-axis">{xTicks[0]}</text>
          <text x={215} y={200} className="chart-axis">{xTicks[2]}</text>
          <text x={380} y={200} className="chart-axis">{xTicks[4]}</text>
          {paths.map((p, idx) => (
            p.d ? <path key={idx} className="plot-line" stroke={p.color} strokeWidth={p.strokeWidth} opacity={p.opacity} d={p.d} /> : null
          ))}
          {!hasData && !isLoading ? (
            <text x="215" y="110" textAnchor="middle" className="chart-axis" opacity="0.5">нет данных для выбранного прогона</text>
          ) : null}
          {isLoading && !hasData ? (
            <text x="215" y="110" textAnchor="middle" className="chart-axis" opacity="0.5">загрузка данных…</text>
          ) : null}
        </svg>
      </ExpandedChartFrame>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
        {CYTOKINE_SERIES.map((s) => (
          <span key={s.label} className="flex items-center gap-1.5"><span className="w-3 h-0.5" style={{ background: s.color }} /><span style={{ color: 'var(--text-secondary)' }}>{s.label}</span></span>
        ))}
      </div>
      <ChartDataTable times={times} series={tableSeries} />
    </ChartModalShell>
  );
}

function EcmExpanded({ simulationId, onClose }: { simulationId: string | null; onClose: () => void }) {
  const { data, isLoading } = useResults(simulationId ?? undefined);
  const results = data as ResultsData | undefined;

  const { collagenArea, fibrinLine, mmpLine, leftTicks, rightTicks, xTicks, hasData, times, tableSeries } = useMemo(() => {
    const empty = { collagenArea: '', fibrinLine: '', mmpLine: '', leftTicks: buildYTicks(60), rightTicks: buildYTicks(3000), xTicks: formatXTicks(720), hasData: false, times: [] as number[], tableSeries: [] as { label: string; color: string; values: number[] | undefined }[] };
    if (!results) return empty;
    const t = results.times ?? [];
    const tMax = t.length ? (t[t.length - 1] ?? 720) : 720;
    const leftRaw = maxOfSeries(results.variables, ['rho_collagen', 'rho_fibrin', 'mean_rho_collagen', 'mean_rho_fibrin']);
    const rightRaw = maxOfSeries(results.variables, ['C_MMP', 'mean_C_MMP']);
    const yMaxLeft = niceMax(leftRaw);
    const yMaxRight = niceMax(rightRaw);
    const scaleLeft: SeriesScale = { xMin: 0, xMax: tMax, yMin: 0, yMax: yMaxLeft, plotLeft: 40, plotRight: 390, plotTop: 30, plotBottom: 180 };
    const scaleRight: SeriesScale = { ...scaleLeft, yMax: yMaxRight };
    const collagen = pickSeries(results.variables, ['rho_collagen', 'mean_rho_collagen']);
    const fibrin = pickSeries(results.variables, ['rho_fibrin', 'mean_rho_fibrin']);
    const mmp = pickSeries(results.variables, ['C_MMP', 'mean_C_MMP']);
    return {
      collagenArea: buildAreaPath(t, collagen, scaleLeft),
      fibrinLine: buildPath(t, fibrin, scaleLeft),
      mmpLine: buildPath(t, mmp, scaleRight),
      leftTicks: buildYTicks(leftRaw),
      rightTicks: buildYTicks(rightRaw),
      xTicks: formatXTicks(tMax),
      hasData: leftRaw + rightRaw > 0,
      times: t,
      tableSeries: [
        { label: 'Коллаген', color: '#58a6ff', values: collagen },
        { label: 'Фибрин', color: '#d29922', values: fibrin },
        { label: 'MMP', color: '#f85149', values: mmp },
      ],
    };
  }, [results]);

  return (
    <ChartModalShell title="Внеклеточный матрикс" icon={ECM_ICON} onClose={onClose}>
      <ExpandedChartFrame svgAspect="400 / 220">
        <svg viewBox="0 0 400 220" className="w-full h-full" preserveAspectRatio="none">
          <g className="chart-grid">
            <line x1="40" y1="30" x2="390" y2="30" />
            <line x1="40" y1="75" x2="390" y2="75" />
            <line x1="40" y1="120" x2="390" y2="120" />
            <line x1="40" y1="165" x2="390" y2="165" />
          </g>
          {leftTicks.map((value, idx) => (
            <text key={idx} x={35} y={33 + idx * 45} className="chart-axis" textAnchor="end">{formatTick(value)}</text>
          ))}
          {rightTicks.map((value, idx) => (
            <text key={idx} x={395} y={33 + idx * 45} className="chart-axis" textAnchor="start">{formatTick(value)}</text>
          ))}
          <text x={40} y={200} className="chart-axis">{xTicks[0]}</text>
          <text x={215} y={200} className="chart-axis">{xTicks[2]}</text>
          <text x={380} y={200} className="chart-axis">{xTicks[4]}</text>
          {collagenArea ? <path d={collagenArea} fill="#58a6ff" opacity="0.35" stroke="#58a6ff" strokeWidth="0.8" /> : null}
          {fibrinLine ? <path className="plot-line" stroke="#d29922" strokeWidth="1.1" d={fibrinLine} /> : null}
          {mmpLine ? <path className="plot-line" stroke="#f85149" strokeWidth="1.3" strokeDasharray="4,3" d={mmpLine} /> : null}
          {!hasData && !isLoading ? (
            <text x="215" y="110" textAnchor="middle" className="chart-axis" opacity="0.5">нет данных для выбранного прогона</text>
          ) : null}
          {isLoading && !hasData ? (
            <text x="215" y="110" textAnchor="middle" className="chart-axis" opacity="0.5">загрузка данных…</text>
          ) : null}
        </svg>
      </ExpandedChartFrame>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
        <span className="flex items-center gap-1.5"><span className="w-3 h-2 rounded-sm opacity-40" style={{ background: '#58a6ff' }} /><span style={{ color: 'var(--text-secondary)' }}>Коллаген (ρ_c)</span></span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-0.5" style={{ background: '#d29922' }} /><span style={{ color: 'var(--text-secondary)' }}>Фибрин (ρ_f)</span></span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-0.5" style={{ background: '#f85149' }} /><span style={{ color: 'var(--text-secondary)' }}>MMP (правая ось)</span></span>
      </div>
      <ChartDataTable times={times} series={tableSeries} />
    </ChartModalShell>
  );
}

function PhasesExpanded({ simulationId, onClose }: { simulationId: string | null; onClose: () => void }) {
  const { data, isLoading } = useResults(simulationId ?? undefined);
  const results = data as ResultsData | undefined;

  const { fLine, mLine, tnfLine, yTicks, xTicks, phases, hasData, times, tableSeries } = useMemo(() => {
    const fallback = { fLine: '', mLine: '', tnfLine: '', yTicks: buildYTicks(60000), xTicks: formatXTicks(720), phases: PHASE_FALLBACK, hasData: false, times: [] as number[], tableSeries: [] as { label: string; color: string; values: number[] | undefined }[] };
    if (!results) return fallback;
    const t = results.times ?? [];
    const tMax = t.length ? (t[t.length - 1] ?? 720) : 720;
    const rawMax = maxOfSeries(results.variables, ['F', 'Mf', 'mean_F', 'mean_Mf', 'fibro']);
    const yMax = niceMax(rawMax);
    const scale: SeriesScale = { xMin: 0, xMax: tMax, yMin: 0, yMax, plotLeft: 40, plotRight: 390, plotTop: 30, plotBottom: 170 };
    const fValues = pickSeries(results.variables, ['F', 'mean_F', 'fibro']);
    const mfValues = pickSeries(results.variables, ['Mf', 'mean_Mf']);
    const tnfValues = pickSeries(results.variables, ['C_TNF', 'mean_C_TNF']);
    return {
      fLine: buildPath(t, fValues, scale),
      mLine: buildPath(t, mfValues, scale),
      tnfLine: buildPath(t, tnfValues, { ...scale, yMax: niceMax(maxOfSeries(results.variables, ['C_TNF', 'mean_C_TNF'])) }),
      yTicks: buildYTicks(rawMax),
      xTicks: formatXTicks(tMax),
      phases: readPhaseShares(results.metadata),
      hasData: rawMax > 0,
      times: t,
      tableSeries: [
        { label: 'F', color: '#58a6ff', values: fValues },
        { label: 'Mf', color: '#b392f0', values: mfValues },
        { label: 'TNF-α', color: '#f85149', values: tnfValues },
      ],
    };
  }, [results]);

  return (
    <ChartModalShell title="Фазы заживления" icon={PHASES_ICON} onClose={onClose}>
      <div className="flex h-3 rounded overflow-hidden">
        <div style={{ flex: phases.hemostasis, background: '#e5534b' }} title={`Hemostasis ${phases.hemostasis.toFixed(0)}%`} />
        <div style={{ flex: phases.inflammation, background: '#d29922' }} title={`Inflammation ${phases.inflammation.toFixed(0)}%`} />
        <div style={{ flex: phases.proliferation, background: '#56d364' }} title={`Proliferation ${phases.proliferation.toFixed(0)}%`} />
        <div style={{ flex: phases.remodeling, background: '#6ac1b4' }} title={`Remodeling ${phases.remodeling.toFixed(0)}%`} />
      </div>
      <ExpandedChartFrame svgAspect="400 / 200">
        <svg viewBox="0 0 400 200" className="w-full h-full" preserveAspectRatio="none">
          <g className="chart-grid">
            <line x1="40" y1="30" x2="390" y2="30" />
            <line x1="40" y1="75" x2="390" y2="75" />
            <line x1="40" y1="120" x2="390" y2="120" />
            <line x1="40" y1="165" x2="390" y2="165" />
          </g>
          {yTicks.map((value, idx) => (
            <text key={idx} x={35} y={33 + idx * 45} className="chart-axis" textAnchor="end">{formatTick(value)}</text>
          ))}
          <text x={40} y={185} className="chart-axis">{xTicks[0]}</text>
          <text x={215} y={185} className="chart-axis">{xTicks[2]}</text>
          <text x={380} y={185} className="chart-axis">{xTicks[4]}</text>
          {fLine ? <path className="plot-line" stroke="#58a6ff" strokeWidth="1.3" d={fLine} /> : null}
          {mLine ? <path className="plot-line" stroke="#b392f0" strokeWidth="1" opacity="0.8" d={mLine} /> : null}
          {tnfLine ? <path className="plot-line" stroke="#f85149" strokeWidth="1" opacity="0.7" d={tnfLine} /> : null}
          {!hasData && !isLoading ? (
            <text x="215" y="105" textAnchor="middle" className="chart-axis" opacity="0.5">нет данных для выбранного прогона</text>
          ) : null}
          {isLoading && !hasData ? (
            <text x="215" y="105" textAnchor="middle" className="chart-axis" opacity="0.5">загрузка данных…</text>
          ) : null}
        </svg>
      </ExpandedChartFrame>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
        <span className="flex items-center gap-1.5"><span className="w-3 h-2 rounded-sm" style={{ background: '#e5534b' }} /><span style={{ color: 'var(--text-secondary)' }}>Hemostasis</span></span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-2 rounded-sm" style={{ background: '#d29922' }} /><span style={{ color: 'var(--text-secondary)' }}>Inflammation</span></span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-2 rounded-sm" style={{ background: '#56d364' }} /><span style={{ color: 'var(--text-secondary)' }}>Proliferation</span></span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-2 rounded-sm" style={{ background: '#6ac1b4' }} /><span style={{ color: 'var(--text-secondary)' }}>Remodeling</span></span>
      </div>
      <ChartDataTable times={times} series={tableSeries} />
    </ChartModalShell>
  );
}

function ExpandedChartModal({
  kind,
  simulationId,
  onClose,
}: {
  kind: CardKind;
  simulationId: string | null;
  onClose: () => void;
}) {
  if (kind === 'populations') return <PopulationExpanded simulationId={simulationId} onClose={onClose} />;
  if (kind === 'cytokines') return <CytokinesExpanded simulationId={simulationId} onClose={onClose} />;
  if (kind === 'ecm') return <EcmExpanded simulationId={simulationId} onClose={onClose} />;
  return <PhasesExpanded simulationId={simulationId} onClose={onClose} />;
}

export default function Dashboard() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const presetMenuRef = useRef<HTMLDivElement | null>(null);
  const [preset, setPreset] = useState<PresetName>(DEFAULT_PRESET);
  const [showPresetMenu, setShowPresetMenu] = useState(false);
  const [compareEnabled, setCompareEnabled] = useState(false);
  const [openSections, setOpenSections] = useState<Record<SectionKey, boolean>>(INITIAL_SECTIONS);
  const [expandedCard, setExpandedCard] = useState<CardKind | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<UploadedFileInfo | null>(null);
  const [isLaunching, setIsLaunching] = useState(false);

  const params = useSimulationStore((state) => state.params);
  const setParam = useSimulationStore((state) => state.setParam);
  const setParams = useSimulationStore((state) => state.setParams);
  const resetParams = useSimulationStore((state) => state.resetParams);
  const applyUpload = useSimulationStore((state) => state.applyUpload);
  const activeSimulationId = useSimulationStore((state) => state.activeSimulationId);
  const setActiveSimulationId = useSimulationStore((state) => state.setActiveSimulationId);

  const { data: simulations } = useSimulationsList();
  const allRuns = simulations ?? [];
  const fallbackRunId = activeSimulationId ?? allRuns[0]?.simulation_id ?? null;
  const inspectedRunId = selectedRunId ?? fallbackRunId;
  const { data: selectedStatusData } = useSimulationStatus(inspectedRunId);
  const cancelMutation = useCancelSimulation();
  const startMutation = useStartSimulation();
  const ws = useSimulationWS(
    inspectedRunId && selectedStatusData?.status === 'running' ? inspectedRunId : null,
  );

  useEffect(() => {
    const handlePointerDown = (event: MouseEvent) => {
      if (!presetMenuRef.current?.contains(event.target as Node)) {
        setShowPresetMenu(false);
      }
    };

    document.addEventListener('mousedown', handlePointerDown);
    return () => document.removeEventListener('mousedown', handlePointerDown);
  }, []);

  const selectedRun = useMemo(
    () => allRuns.find((item) => item.simulation_id === inspectedRunId) ?? null,
    [allRuns, inspectedRunId],
  );

  const resolvedStatus = selectedStatusData?.status ?? selectedRun?.status ?? null;
  const resolvedState = isLaunching
    ? 'running'
    : getStateFromStatus(resolvedStatus) === 'idle' && allRuns.some((item) => item.status === 'completed')
      ? 'complete'
      : getStateFromStatus(resolvedStatus);

  const selectedParams = selectedStatusData?.params_json ?? selectedRun?.params_json ?? null;
  const statusMeta = getStatusBadge(resolvedStatus);
  const progress = isLaunching
    ? 8
    : ws.status === 'complete'
      ? 100
      : resolvedState === 'running'
        ? ws.progress || selectedStatusData?.progress || 12
        : selectedStatusData?.progress || 0;
  const message = isLaunching
    ? 'Подготовка новой симуляции...'
    : ws.message || selectedStatusData?.message || '';
  const activeHours = selectedParams?.t_max_hours ?? params.t_max_hours;
  const activeModeLabel = getModeLabel(selectedParams?.mode ?? params.mode);
  const recentRuns = allRuns.slice(0, 5);
  const canOpenResults = inspectedRunId && resolvedStatus === 'completed';
  const errorMessage = selectedStatusData?.error_message || selectedStatusData?.message || '';
  const therapyCount = Number(params.prp_enabled) + Number(params.pemf_enabled);

  const uploadMutation = useMutation({
    mutationFn: async ({ file }: { file: File }) => {
      const formData = new FormData();
      formData.append('file', file);
      const { data } = await apiClient.post<UploadResponse>(`${API_V1}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return { response: data, file };
    },
    onSuccess: ({ response, file }) => {
      applyUpload(response.upload_id, response.metadata?.initial_conditions as Partial<SimulationRequest> | undefined);
      setUploadedFile({ name: file.name, size: file.size });
    },
  });

  const triggerUpload = () => fileInputRef.current?.click();

  const handleFileSelection = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const lower = file.name.toLowerCase();
    if (!lower.endsWith('.fcs') && !lower.endsWith('.csv')) return;
    uploadMutation.mutate({ file });
    event.target.value = '';
  };

  const toggleSection = (section: SectionKey) => {
    setOpenSections((current) => ({ ...current, [section]: !current[section] }));
  };

  const applyPreset = (name: PresetName) => {
    setPreset(name);
    resetParams();
    setParams(PRESETS[name]);
    setShowPresetMenu(false);
  };

  const handleStart = () => {
    if (isLaunching || resolvedState === 'running') return;
    setIsLaunching(true);
    setSelectedRunId(null);
    setActiveSimulationId(null);
    startMutation.mutate(params, {
      onSuccess: (response) => {
        setActiveSimulationId(response.simulation_id);
        setSelectedRunId(response.simulation_id);
        setIsLaunching(false);
        void queryClient.invalidateQueries({ queryKey: ['simulations'] });
      },
      onError: () => {
        setIsLaunching(false);
      },
    });
  };

  const handleCancel = () => {
    if (!inspectedRunId) return;
    const runId = inspectedRunId;
    cancelMutation.mutate(runId, {
      onSuccess: () => {
        setIsLaunching(false);
        void queryClient.invalidateQueries({ queryKey: ['simulations'] });
        void queryClient.invalidateQueries({ queryKey: ['simulation', runId] });
        void queryClient.refetchQueries({ queryKey: ['simulation', runId] });
      },
      onError: (error) => {
        console.error('Не удалось отменить симуляцию', error);
        void queryClient.invalidateQueries({ queryKey: ['simulations'] });
        void queryClient.invalidateQueries({ queryKey: ['simulation', runId] });
      },
    });
  };

  const handleRetry = () => {
    setActiveSimulationId(null);
    setSelectedRunId(null);
    handleStart();
  };

  const handleReset = () => {
    resetParams();
    setUploadedFile(null);
  };

  return (
    <div className="min-h-full p-5">
      <h2 className="sr-only">{t('dashboard.upload.title')}</h2>
      <h2 className="sr-only">{t('dashboard.model.title')}</h2>

      <input ref={fileInputRef} type="file" accept=".fcs,.csv" className="hidden" onChange={handleFileSelection} />

      <div className="flex items-center gap-3 mb-4">
        <h1 className="font-display text-[22px] font-bold tracking-tight" style={{ color: 'var(--text-primary)' }}>
          Панель управления
        </h1>
        <div className="flex-1" />

        <div className="flex items-center gap-2.5 px-3 py-1.5 rounded-lg text-xs card">
          <div className="w-1.5 h-1.5 rounded-full" style={{ background: statusMeta.dotColor }} />
          <span style={{ color: 'var(--text-muted)' }}>
            {resolvedStatus === 'running' || resolvedStatus === 'pending' ? 'Активная:' : 'Просмотр:'}
          </span>
          <span className="font-mono" style={{ color: 'var(--text-primary)' }}>{shortId(inspectedRunId)}</span>
          <span className={statusMeta.badgeClass}>{statusMeta.label}</span>
          <span className="font-mono" style={{ color: 'var(--text-muted)' }}>{activeHours} ч</span>
        </div>
      </div>

      <div className="card p-2 mb-4 flex items-center gap-2 flex-wrap">
        <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-md" style={{ background: 'var(--surface-2)', border: '1px solid var(--border-default)' }}>
          <svg className="h-3.5 w-3.5" fill="none" stroke={uploadedFile || params.upload_id ? '#56d364' : '#6e7681'} strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
          </svg>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Данные:</span>
          <span className="text-xs font-mono" style={{ color: 'var(--text-primary)' }}>
            {uploadedFile?.name ?? (params.upload_id ? 'uploaded-session' : 'не выбраны')}
          </span>
          <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
            {uploadMutation.isPending ? 'загрузка...' : formatFileSize(uploadedFile?.size)}
          </span>
          <button className="ml-1 opacity-50 hover:opacity-100" title="Заменить файл" onClick={triggerUpload} type="button">
            <svg className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99" />
            </svg>
          </button>
        </div>

        <div className="divider-v" style={{ height: 28 }} />

        <div className="flex items-center gap-1.5">
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Модель:</span>
          <div className="seg">
            <button className={params.mode === 'abm' ? 'on' : ''} onClick={() => setParam('mode', 'abm')} type="button">ABM</button>
            <button className={params.mode === 'extended' ? 'on' : ''} onClick={() => setParam('mode', 'extended')} type="button">SDE-20</button>
            <button className={params.mode === 'mvp' ? 'on' : ''} onClick={() => setParam('mode', 'mvp')} type="button">MVP</button>
          </div>
        </div>

        <div className="divider-v" style={{ height: 28 }} />

        <div className="flex items-center gap-1.5 relative" ref={presetMenuRef}>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Пресет:</span>
          <button className="btn-outline" onClick={() => setShowPresetMenu((current) => !current)} type="button">
            {preset}
            <svg className="h-3 w-3 opacity-60" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25 12 15.75 4.5 8.25" />
            </svg>
          </button>
          {showPresetMenu ? (
            <div className="card absolute top-[calc(100%+8px)] left-[56px] z-20 min-w-[220px] p-1" style={{ background: 'rgba(22,27,34,0.98)', backdropFilter: 'blur(8px)' }}>
              {PRESET_LIST.map(({ name, description }) => (
                <button
                  key={name}
                  className="w-full text-left px-2.5 py-1.5 rounded-md text-xs hover:bg-surface-2"
                  style={{ color: name === preset ? 'var(--accent-text)' : 'var(--text-primary)' }}
                  onClick={() => applyPreset(name)}
                  type="button"
                  title={description}
                >
                  {name}
                </button>
              ))}
            </div>
          ) : null}
        </div>

        <div className="flex-1" />

        <label className="btn-ghost cursor-pointer">
          <input type="checkbox" checked={compareEnabled} onChange={(event) => setCompareEnabled(event.target.checked)} />
          <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.6" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 21 3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
          </svg>
          Сравнение
        </label>

        <button className="btn-ghost" onClick={() => canOpenResults && navigate(`/results/${inspectedRunId}`)} type="button" disabled={!canOpenResults}>
          <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.6" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
          </svg>
          Экспорт
        </button>

        <button className="btn-primary" onClick={handleStart} type="button" disabled={isLaunching || resolvedState === 'running'}>
          <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z" /></svg>
          {isLaunching ? 'Запуск...' : 'Запустить'}
        </button>
      </div>

      <div className="grid gap-4" style={{ gridTemplateColumns: '320px 1fr' }}>
        <div className="card p-3 space-y-1.5 overflow-y-auto" style={{ maxHeight: 'calc(100vh - 235px)' }}>
          <div className="flex items-center justify-between px-1 py-1 mb-1">
            <span className="section-label">Параметры</span>
            <button className="btn-ghost text-[10px]" style={{ padding: '3px 6px' }} onClick={handleReset} type="button">Сброс</button>
          </div>

          <AccordionSection open={openSections.populations} onToggle={() => toggleSection('populations')} title="Клеточные популяции" icon={<svg className="h-3.5 w-3.5" fill="none" stroke="var(--accent-text)" strokeWidth="1.6" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3" /><circle cx="5" cy="5" r="2" /><circle cx="19" cy="5" r="2" /><circle cx="5" cy="19" r="2" /><circle cx="19" cy="19" r="2" /></svg>}>
            <NumericField label={<>Тромбоциты <span style={{ color: 'var(--text-muted)' }}>(P)</span></>} value={params.P0} onChange={(value) => setParam('P0', value)} />
            <NumericField label={<>Нейтрофилы <span style={{ color: 'var(--text-muted)' }}>(Ne)</span></>} value={params.Ne0} onChange={(value) => setParam('Ne0', value)} />
            <NumericField label="Макрофаги M1" value={params.M1_0} step={0.001} onChange={(value) => setParam('M1_0', value)} />
            <NumericField label="Макрофаги M2" value={params.M2_0} step={0.001} onChange={(value) => setParam('M2_0', value)} />
            <NumericField label={<>Фибробласты <span style={{ color: 'var(--text-muted)' }}>(F)</span></>} value={params.F0} onChange={(value) => setParam('F0', value)} />
            <NumericField label={<>Миофибробласты <span style={{ color: 'var(--text-muted)' }}>(Mf)</span></>} value={params.Mf0} onChange={(value) => setParam('Mf0', value)} />
            <NumericField label={<>Эндотелиальные <span style={{ color: 'var(--text-muted)' }}>(E)</span></>} value={params.E0} onChange={(value) => setParam('E0', value)} />
            <NumericField label={<>Стволовые <span style={{ color: 'var(--text-muted)' }}>(S)</span></>} value={params.S0} step={0.001} onChange={(value) => setParam('S0', value)} />
          </AccordionSection>

          <AccordionSection open={openSections.cytokines} onToggle={() => toggleSection('cytokines')} title="Цитокины" icon={<svg className="h-3.5 w-3.5" fill="none" stroke="#79b8ff" strokeWidth="1.6" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.1v5.7c0 .6-.24 1.17-.66 1.6L5 14.5M14.25 3.1v5.7c0 .6.24 1.17.66 1.6L19.8 15.3M5 14.5h14.8" /></svg>}>
            <NumericField label="TNF-α" value={params.C_TNF0} step={0.001} onChange={(value) => setParam('C_TNF0', value)} />
            <NumericField label="IL-10" value={params.C_IL10_0} step={0.0001} onChange={(value) => setParam('C_IL10_0', value)} />
            <NumericField label="Сигнал повреждения (D)" value={params.D0} step={0.01} onChange={(value) => setParam('D0', value)} />
            <NumericField label="Кислород (O₂)" value={params.O2_0} step={0.01} onChange={(value) => setParam('O2_0', value)} />
          </AccordionSection>

          <AccordionSection open={openSections.time} onToggle={() => toggleSection('time')} title="Настройки времени" icon={<svg className="h-3.5 w-3.5" fill="none" stroke="#e3b341" strokeWidth="1.6" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" /><path strokeLinecap="round" strokeLinejoin="round" d="M12 7v5l3 2" /></svg>}>
            <NumericField label="Время симуляции (ч)" value={params.t_max_hours} min={24} step={24} onChange={(value) => setParam('t_max_hours', value)} />
            <div className="flex items-center gap-2">
              <label className="text-[11px] flex-1" style={{ color: 'var(--text-secondary)' }}>Шаг интегрирования</label>
              <select value={params.dt} style={{ width: 96 }} onChange={(event) => setParam('dt', Number(event.target.value))}>
                <option value={0.01}>0.01</option>
                <option value={0.05}>0.05</option>
                <option value={0.1}>0.1</option>
              </select>
            </div>
          </AccordionSection>

          <AccordionSection open={openSections.therapy} onToggle={() => toggleSection('therapy')} title="Терапия" icon={<svg className="h-3.5 w-3.5" fill="none" stroke="var(--accent-text)" strokeWidth="1.6" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="m9.4 2.6 12 12a3 3 0 0 1-4.2 4.2l-12-12a3 3 0 0 1 4.2-4.2ZM8 8l8 8" /></svg>} badge={<span className="badge" style={{ background: 'rgba(59,152,137,0.2)', color: 'var(--accent-text)', padding: '1px 6px' }}>{therapyCount} активны</span>}>
            <div>
              <div className="flex items-center gap-2 mb-1">
                <label className="flex items-center gap-2 text-[11px] cursor-pointer" style={{ color: 'var(--text-primary)' }}>
                  <input type="checkbox" checked={params.prp_enabled} onChange={(event) => setParam('prp_enabled', event.target.checked)} />
                  <span className="font-medium">PRP терапия</span>
                </label>
                <div className="flex-1" />
                <span className="text-[10px] font-mono" style={{ color: 'var(--accent-text)' }}>{params.prp_enabled ? 'active' : 'off'}</span>
              </div>
              {params.prp_enabled ? <div className="pl-5"><SliderField label="Интенсивность" value={params.prp_intensity} min={0} max={2} step={0.1} onChange={(value) => setParam('prp_intensity', value)} /></div> : null}
            </div>
            <div>
              <div className="flex items-center gap-2 mb-1">
                <label className="flex items-center gap-2 text-[11px] cursor-pointer" style={{ color: 'var(--text-primary)' }}>
                  <input type="checkbox" checked={params.pemf_enabled} onChange={(event) => setParam('pemf_enabled', event.target.checked)} />
                  <span className="font-medium">PEMF терапия</span>
                </label>
                <div className="flex-1" />
                <span className="text-[10px] font-mono" style={{ color: 'var(--accent-text)' }}>{params.pemf_enabled ? 'active' : 'off'}</span>
              </div>
              {params.pemf_enabled ? <div className="pl-5 space-y-1.5"><SliderField label="Частота, Гц" value={params.pemf_frequency} min={1} max={100} step={1} onChange={(value) => setParam('pemf_frequency', value)} /><SliderField label="Интенсивность" value={params.pemf_intensity} min={0} max={2} step={0.1} onChange={(value) => setParam('pemf_intensity', value)} /></div> : null}
            </div>
          </AccordionSection>

          <AccordionSection open={openSections.monteCarlo} onToggle={() => toggleSection('monteCarlo')} title="Монте-Карло" icon={<svg className="h-3.5 w-3.5" fill="none" stroke="#b392f0" strokeWidth="1.6" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="3" /><circle cx="8" cy="8" r="1.3" fill="#b392f0" /><circle cx="16" cy="16" r="1.3" fill="#b392f0" /><circle cx="16" cy="8" r="1.3" fill="#b392f0" /><circle cx="8" cy="16" r="1.3" fill="#b392f0" /></svg>}>
            <NumericField label="Число траекторий" value={params.n_trajectories} min={1} step={1} onChange={(value) => setParam('n_trajectories', value)} />
            <NumericField label="Зерно ГСЧ" value={params.random_seed ?? 4} min={0} step={1} onChange={(value) => setParam('random_seed', value)} />
          </AccordionSection>
        </div>

        <div>
          {resolvedState === 'running' ? <RunningStateView progress={progress} message={message} onCancel={handleCancel} cancelling={cancelMutation.isPending} modeLabel={activeModeLabel} tMaxHours={activeHours} /> : resolvedState === 'error' ? <ErrorStateView message={errorMessage} onRetry={handleRetry} /> : resolvedState === 'complete' ? <CompleteStateView simulationId={inspectedRunId} onExpand={setExpandedCard} /> : <IdleStateView onUpload={triggerUpload} />}
        </div>
      </div>

      <div className="card p-2.5 mt-4 flex items-center gap-3">
        <span className="section-label px-1 shrink-0">Последние прогоны</span>
        <div className="flex items-center gap-2 flex-1 overflow-x-auto">
          {recentRuns.map((run, index) => (
            <button key={run.simulation_id} className={`chip${(inspectedRunId ?? run.simulation_id) === run.simulation_id || (!inspectedRunId && index === 0) ? ' active' : ''}`} onClick={() => setSelectedRunId(run.simulation_id)} type="button">
              <div className="dot" style={{ background: run.status === 'completed' ? 'var(--success)' : run.status === 'running' ? 'var(--warning)' : 'var(--text-muted)', opacity: run.status === 'completed' || run.status === 'running' ? 1 : 0.5 }} />
              <span className="font-mono" style={{ color: (inspectedRunId ?? recentRuns[0]?.simulation_id) === run.simulation_id ? 'var(--text-primary)' : 'var(--text-secondary)' }}>{shortId(run.simulation_id)}</span>
              <span style={{ color: 'var(--text-secondary)' }}>{getModeLabel(run.params_json?.mode)} · {getTherapyLabel(run.params_json)}</span>
              <span className="font-mono" style={{ color: 'var(--text-muted)' }}>{formatClock(run.created_at)}</span>
            </button>
          ))}
        </div>
        <button className="btn-ghost text-xs shrink-0" style={{ color: 'var(--accent-text)' }} onClick={() => navigate('/history')} type="button">Открыть историю →</button>
      </div>

      {expandedCard ? (
        <ExpandedChartModal
          kind={expandedCard}
          simulationId={inspectedRunId}
          onClose={() => setExpandedCard(null)}
        />
      ) : null}
    </div>
  );
}
