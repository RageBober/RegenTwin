export type SeriesScale = {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  plotLeft: number;
  plotRight: number;
  plotTop: number;
  plotBottom: number;
};

export function buildPath(
  times: number[] | undefined,
  values: number[] | undefined,
  scale: SeriesScale,
): string {
  if (!times || !values || !times.length || !values.length) return '';
  const { xMin, xMax, yMin, yMax, plotLeft, plotRight, plotTop, plotBottom } = scale;
  const xSpan = xMax - xMin || 1;
  const ySpan = yMax - yMin || 1;
  const pxW = plotRight - plotLeft;
  const pxH = plotBottom - plotTop;
  const len = Math.min(times.length, values.length);
  const step = Math.max(1, Math.floor(len / 240));
  const pts: string[] = [];
  for (let i = 0; i < len; i += step) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    const x = plotLeft + ((times[i] - xMin) / xSpan) * pxW;
    const yClamped = Math.max(yMin, Math.min(yMax, v));
    const y = plotBottom - ((yClamped - yMin) / ySpan) * pxH;
    pts.push(`${pts.length === 0 ? 'M' : 'L'} ${x.toFixed(1)},${y.toFixed(1)}`);
  }
  const lastIdx = len - 1;
  if (lastIdx >= 0 && lastIdx % step !== 0) {
    const v = values[lastIdx];
    if (Number.isFinite(v)) {
      const x = plotLeft + ((times[lastIdx] - xMin) / xSpan) * pxW;
      const yClamped = Math.max(yMin, Math.min(yMax, v));
      const y = plotBottom - ((yClamped - yMin) / ySpan) * pxH;
      pts.push(`L ${x.toFixed(1)},${y.toFixed(1)}`);
    }
  }
  return pts.join(' ');
}

export function buildAreaPath(
  times: number[] | undefined,
  values: number[] | undefined,
  scale: SeriesScale,
): string {
  const line = buildPath(times, values, scale);
  if (!line) return '';
  const { plotLeft, plotRight, plotBottom } = scale;
  return `${line} L ${plotRight.toFixed(1)},${plotBottom.toFixed(1)} L ${plotLeft.toFixed(1)},${plotBottom.toFixed(1)} Z`;
}

export function maxOfSeries(
  variables: Record<string, number[]> | undefined,
  keys: string[],
): number {
  if (!variables) return 0;
  let hi = 0;
  for (const key of keys) {
    const arr = variables[key];
    if (!arr) continue;
    for (const v of arr) {
      if (Number.isFinite(v) && v > hi) hi = v;
    }
  }
  return hi;
}

export function pickSeries(
  variables: Record<string, number[]> | undefined,
  aliases: string[],
): number[] | undefined {
  if (!variables) return undefined;
  for (const name of aliases) {
    const arr = variables[name];
    if (arr && arr.length) return arr;
  }
  return undefined;
}

export function formatTick(value: number): string {
  const abs = Math.abs(value);
  if (abs === 0) return '0';
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(abs >= 10_000_000 ? 0 : 1)}M`;
  if (abs >= 1_000) return `${(value / 1_000).toFixed(abs >= 10_000 ? 0 : 1)}k`;
  if (abs >= 10) return value.toFixed(0);
  if (abs >= 1) return value.toFixed(1);
  if (abs >= 0.01) return value.toFixed(2);
  return value.toExponential(1);
}

export function niceMax(rawMax: number): number {
  if (!Number.isFinite(rawMax) || rawMax <= 0) return 1;
  const padded = rawMax * 1.08;
  const pow = Math.pow(10, Math.floor(Math.log10(padded)));
  const m = padded / pow;
  const rounded = m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10;
  return rounded * pow;
}

export function buildYTicks(yMax: number): number[] {
  const nice = niceMax(yMax);
  return [nice, nice * 0.75, nice * 0.5, nice * 0.25, 0];
}
