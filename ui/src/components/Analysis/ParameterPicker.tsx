import { useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ChevronDownIcon, MagnifyingGlassIcon } from '@heroicons/react/24/outline';
import { useParameterBounds } from '../../hooks/useAnalysis';
import type { ParameterBoundItem } from '../../types/api';

interface Props {
  value: string[];
  onChange: (value: string[]) => void;
}

const CYTOKINE_GROUPS = ['cytokine_secretion', 'cytokine_degradation'];
const ECM_GROUPS = ['ecm'];

export default function ParameterPicker({ value, onChange }: Props) {
  const { t } = useTranslation();
  const { data: boundsData, isLoading } = useParameterBounds();
  const [search, setSearch] = useState('');
  const [openGroups, setOpenGroups] = useState<Set<string>>(new Set());
  const [presetsOpen, setPresetsOpen] = useState(false);
  const presetsRef = useRef<HTMLDivElement>(null);

  // Группировка параметров
  const grouped = useMemo(() => {
    const map = new Map<string, ParameterBoundItem[]>();
    for (const b of boundsData?.bounds ?? []) {
      const arr = map.get(b.group) ?? [];
      arr.push(b);
      map.set(b.group, arr);
    }
    return map;
  }, [boundsData]);

  const groupNames = useMemo(() => Array.from(grouped.keys()).sort(), [grouped]);

  // Открыть первую группу при первой загрузке
  useEffect(() => {
    if (groupNames.length > 0 && openGroups.size === 0) {
      setOpenGroups(new Set([groupNames[0]]));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- инициализация только при первой загрузке
  }, [groupNames.length]);

  // Click-outside для presets dropdown
  useEffect(() => {
    if (!presetsOpen) return;
    const handler = (e: MouseEvent) => {
      if (presetsRef.current && !presetsRef.current.contains(e.target as Node)) {
        setPresetsOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [presetsOpen]);

  const search_lc = search.trim().toLowerCase();

  // Фильтр по имени параметра ИЛИ имени группы (локализованному)
  const filteredByGroup = useMemo(() => {
    const out = new Map<string, ParameterBoundItem[]>();
    for (const [group, items] of grouped.entries()) {
      if (!search_lc) {
        out.set(group, items);
        continue;
      }
      const groupLabel = t(`analysis.parameters.groups.${group}`, group).toLowerCase();
      const groupMatches = groupLabel.includes(search_lc);
      const filtered = groupMatches
        ? items
        : items.filter((p) => p.name.toLowerCase().includes(search_lc));
      if (filtered.length > 0) out.set(group, filtered);
    }
    return out;
  }, [grouped, search_lc, t]);

  const visibleParams = useMemo(
    () => Array.from(filteredByGroup.values()).flat().map((b) => b.name),
    [filteredByGroup],
  );

  const valueSet = useMemo(() => new Set(value), [value]);

  const togglePar = (name: string) => {
    if (valueSet.has(name)) onChange(value.filter((v) => v !== name));
    else onChange([...value, name]);
  };

  const toggleGroup = (group: string) => {
    const next = new Set(openGroups);
    if (next.has(group)) next.delete(group);
    else next.add(group);
    setOpenGroups(next);
  };

  const groupItems = (group: string): ParameterBoundItem[] =>
    filteredByGroup.get(group) ?? [];

  const selectGroup = (group: string) => {
    const names = groupItems(group).map((b) => b.name);
    const allSelected = names.every((n) => valueSet.has(n));
    if (allSelected) {
      // Снять выбор со всех в группе
      onChange(value.filter((v) => !names.includes(v)));
    } else {
      // Добавить недостающие
      const merged = new Set(value);
      names.forEach((n) => merged.add(n));
      onChange(Array.from(merged));
    }
  };

  const selectVisible = () => {
    const merged = new Set(value);
    visibleParams.forEach((n) => merged.add(n));
    onChange(Array.from(merged));
  };

  const clearAll = () => onChange([]);

  const presetCytokines = () => {
    const names = (boundsData?.bounds ?? [])
      .filter((b) => CYTOKINE_GROUPS.includes(b.group))
      .map((b) => b.name);
    onChange(names);
    setPresetsOpen(false);
    // Раскрыть соответствующие группы
    setOpenGroups((prev) => {
      const next = new Set(prev);
      CYTOKINE_GROUPS.forEach((g) => next.add(g));
      return next;
    });
  };

  const presetEcm = () => {
    const names = (boundsData?.bounds ?? [])
      .filter((b) => ECM_GROUPS.includes(b.group))
      .map((b) => b.name);
    onChange(names);
    setPresetsOpen(false);
    setOpenGroups((prev) => {
      const next = new Set(prev);
      ECM_GROUPS.forEach((g) => next.add(g));
      return next;
    });
  };

  const presetDefault = () => {
    const first4 = (boundsData?.bounds ?? []).slice(0, 4).map((b) => b.name);
    onChange(first4);
    setPresetsOpen(false);
  };

  if (isLoading) {
    return (
      <div className="card p-4">
        <div className="flex flex-wrap gap-2">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="h-8 w-20 animate-pulse rounded-lg bg-surface-2" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="card p-4 space-y-3">
      {/* Header: search + counter */}
      <div className="flex items-center justify-between gap-3">
        <label className="text-xs font-semibold uppercase tracking-wider
                          text-primary-500/60 dark:text-primary-400/50">
          {t('analysis.sensitivity.parameters')}
          {boundsData && (
            <span className="ml-2 text-primary-400/40 font-normal normal-case">
              ({value.length} / {boundsData.total})
            </span>
          )}
        </label>
      </div>

      {/* Search */}
      <div className="relative">
        <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2
                                       h-3.5 w-3.5 text-primary-400/40" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder={t('analysis.parameters.actions.search')}
          className="w-full pl-9 pr-3 py-2 rounded-lg border border-border bg-surface-1
                     text-xs font-mono text-primary-800 dark:text-primary-200
                     placeholder:text-primary-400/30
                     focus:outline-none focus:ring-1 focus:ring-primary-500/30"
        />
      </div>

      {/* Bulk actions */}
      <div className="flex items-center gap-2 flex-wrap">
        <button
          type="button"
          onClick={selectVisible}
          className="rounded-lg border border-border px-2.5 py-1 text-2xs font-medium
                     text-primary-600 dark:text-primary-400 hover:bg-surface-2 transition-colors"
        >
          {t('analysis.parameters.actions.selectVisible')}
        </button>
        <button
          type="button"
          onClick={clearAll}
          disabled={value.length === 0}
          className="rounded-lg border border-border px-2.5 py-1 text-2xs font-medium
                     text-primary-600 dark:text-primary-400 hover:bg-surface-2 transition-colors
                     disabled:opacity-30"
        >
          {t('analysis.parameters.actions.clearAll')}
        </button>

        <div ref={presetsRef} className="relative">
          <button
            type="button"
            onClick={() => setPresetsOpen((v) => !v)}
            className="inline-flex items-center gap-1 rounded-lg border border-border px-2.5 py-1
                       text-2xs font-medium text-primary-600 dark:text-primary-400
                       hover:bg-surface-2 transition-colors"
          >
            {t('analysis.parameters.actions.presets')}
            <ChevronDownIcon className="h-3 w-3" />
          </button>
          {presetsOpen && (
            <div className="absolute left-0 top-full mt-1 z-20 min-w-[160px]
                            rounded-lg border border-border bg-surface-1 shadow-lg py-1">
              <button
                type="button"
                onClick={presetCytokines}
                className="block w-full text-left px-3 py-1.5 text-xs
                           text-primary-700 dark:text-primary-300 hover:bg-surface-2"
              >
                {t('analysis.parameters.actions.presetCytokines')}
              </button>
              <button
                type="button"
                onClick={presetEcm}
                className="block w-full text-left px-3 py-1.5 text-xs
                           text-primary-700 dark:text-primary-300 hover:bg-surface-2"
              >
                {t('analysis.parameters.actions.presetEcm')}
              </button>
              <button
                type="button"
                onClick={presetDefault}
                className="block w-full text-left px-3 py-1.5 text-xs
                           text-primary-700 dark:text-primary-300 hover:bg-surface-2"
              >
                {t('analysis.parameters.actions.presetDefault')}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Groups */}
      <div className="space-y-1.5 max-h-[420px] overflow-y-auto pr-1">
        {Array.from(filteredByGroup.entries()).map(([group, items]) => {
          const isOpen = openGroups.has(group);
          const selectedInGroup = items.filter((b) => valueSet.has(b.name)).length;
          const allSelected = selectedInGroup === items.length && items.length > 0;
          const someSelected = selectedInGroup > 0 && !allSelected;
          return (
            <div key={group} className="rounded-lg border border-border/60 bg-surface-1/40">
              <div className="flex items-center justify-between px-2.5 py-1.5">
                <button
                  type="button"
                  onClick={() => toggleGroup(group)}
                  className="flex items-center gap-1.5 flex-1 text-left"
                >
                  <ChevronDownIcon
                    className={`h-3 w-3 text-primary-500/50 transition-transform
                                ${isOpen ? '' : '-rotate-90'}`}
                  />
                  <span className="text-xs font-medium text-primary-700 dark:text-primary-300">
                    {t(`analysis.parameters.groups.${group}`, group)}
                  </span>
                  <span className="text-2xs text-primary-400/50 font-mono">
                    ({selectedInGroup}/{items.length})
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => selectGroup(group)}
                  title={t('analysis.parameters.actions.selectGroup')}
                  className={`text-2xs font-medium px-2 py-0.5 rounded
                              ${allSelected
                                ? 'text-primary-600 dark:text-primary-400 bg-primary-500/10'
                                : someSelected
                                ? 'text-primary-500/70 bg-primary-500/5'
                                : 'text-primary-500/40 hover:bg-surface-2'
                              }`}
                >
                  {allSelected ? '✓' : someSelected ? '−' : '+'}
                </button>
              </div>
              {isOpen && (
                <div className="flex flex-wrap gap-1.5 px-2.5 pb-2.5">
                  {items.map((b) => {
                    const selected = valueSet.has(b.name);
                    return (
                      <button
                        key={b.name}
                        type="button"
                        onClick={() => togglePar(b.name)}
                        title={`${b.name}: ${b.lower} … ${b.nominal} … ${b.upper}`}
                        className={`rounded-lg border px-2.5 py-1 text-2xs font-mono font-medium
                                   transition-all duration-150
                                   ${selected
                                     ? 'border-primary-500/30 bg-primary-500/10 text-primary-600 dark:text-primary-400 shadow-inner-glow'
                                     : 'border-border text-primary-900/40 dark:text-primary-100/30 hover:bg-surface-2'
                                   }`}
                      >
                        {b.name}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
        {filteredByGroup.size === 0 && (
          <p className="text-xs text-primary-500/50 italic px-2 py-3">
            {t('analysis.parameters.actions.noParameters')}
          </p>
        )}
      </div>

      {value.length === 0 && (
        <p className="text-2xs text-accent-600/80 dark:text-accent-400/70">
          {t('analysis.parameters.actions.selectAtLeastOne')}
        </p>
      )}
    </div>
  );
}
