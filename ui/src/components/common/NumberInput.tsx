interface NumberInputProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}

export default function NumberInput({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  disabled = false,
}: NumberInputProps) {
  return (
    <div className="flex items-center justify-between gap-2">
      <label className="text-xs text-slate-600 dark:text-slate-300 truncate flex-1">
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
        className="w-24 rounded border border-slate-300 bg-white px-2 py-1 text-xs text-right dark:border-slate-600 dark:bg-slate-700 dark:text-slate-200 disabled:opacity-50"
      />
    </div>
  );
}
