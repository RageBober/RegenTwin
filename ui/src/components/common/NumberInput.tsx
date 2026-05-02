interface NumberInputProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  testId?: string;
}

export default function NumberInput({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  disabled = false,
  testId,
}: NumberInputProps) {
  return (
    <div className="flex items-center justify-between gap-2">
      <label className="text-xs text-primary-900/50 dark:text-primary-100/40 truncate flex-1">
        {label}
      </label>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        data-testid={testId}
        className="w-24 rounded-lg border border-border bg-surface-1 px-2.5 py-1.5
                   text-xs font-mono text-right
                   text-primary-800 dark:text-primary-200
                   focus:outline-none focus:ring-1 focus:ring-primary-500/30
                   disabled:opacity-40 transition-colors"
      />
    </div>
  );
}
