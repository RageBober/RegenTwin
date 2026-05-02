/**
 * Shared Plotly layout defaults matching the RegenTwin design system.
 * Use: Plot({ ...data }, { ...getPlotlyLayout(isDark), title: 'My Plot' })
 */

const LIGHT = {
  paper_bgcolor: 'rgba(255,255,255,0)',
  plot_bgcolor: 'rgba(248,250,251,0.8)',
  font: { color: '#1a2e2b', family: '"DM Sans", Inter, system-ui, sans-serif', size: 11 },
  gridcolor: 'rgba(15, 118, 110, 0.06)',
  zerolinecolor: 'rgba(15, 118, 110, 0.12)',
};

const DARK = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(35,46,45,0.8)',
  font: { color: '#c8d8d5', family: '"DM Sans", Inter, system-ui, sans-serif', size: 11 },
  gridcolor: 'rgba(125, 190, 180, 0.07)',
  zerolinecolor: 'rgba(125, 190, 180, 0.13)',
};

export const PLOTLY_COLORS = [
  '#0d9488', // primary-500
  '#f59e0b', // accent-500
  '#10b981', // emerald-500
  '#6366f1', // indigo-500
  '#ec4899', // pink-500
  '#8b5cf6', // violet-500
  '#14b8a6', // teal-500
  '#f97316', // orange-500
  '#06b6d4', // cyan-500
  '#84cc16', // lime-500
];

export function getPlotlyLayout(isDark: boolean): Record<string, unknown> {
  const theme = isDark ? DARK : LIGHT;
  return {
    paper_bgcolor: theme.paper_bgcolor,
    plot_bgcolor: theme.plot_bgcolor,
    font: theme.font,
    margin: { l: 50, r: 20, t: 30, b: 40 },
    xaxis: {
      gridcolor: theme.gridcolor,
      zerolinecolor: theme.zerolinecolor,
      tickfont: { size: 10 },
    },
    yaxis: {
      gridcolor: theme.gridcolor,
      zerolinecolor: theme.zerolinecolor,
      tickfont: { size: 10 },
    },
    colorway: PLOTLY_COLORS,
    legend: {
      font: { size: 10 },
      bgcolor: 'rgba(0,0,0,0)',
    },
    hoverlabel: {
      bgcolor: isDark ? '#2a3736' : '#ffffff',
      bordercolor: isDark ? 'rgba(125,190,180,0.18)' : 'rgba(15,118,110,0.15)',
      font: { color: isDark ? '#c8d8d5' : '#1a2e2b', size: 11 },
    },
  };
}
