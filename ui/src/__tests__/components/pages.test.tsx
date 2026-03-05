import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClientProvider, QueryClient } from '@tanstack/react-query';

// Mock i18next
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) => key,
    i18n: { language: 'en', changeLanguage: vi.fn() },
  }),
}));

// Mock axios
vi.mock('../../lib/api', () => ({
  apiClient: { get: vi.fn(), post: vi.fn() },
  API_V1: '/api/v1',
  updateApiBaseUrl: vi.fn(),
}));

// Mock plotly
vi.mock('react-plotly.js', () => ({
  default: () => <div data-testid="plotly-chart">Plotly</div>,
}));

// Helpers
function withProviders(ui: React.ReactElement, { route = '/' } = {}) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
}

describe('Page smoke tests', () => {
  it('Home renders title key', async () => {
    const Home = (await import('../../routes/Home')).default;
    withProviders(<Home />);
    expect(screen.getByText('home.title')).toBeInTheDocument();
  });

  it('Settings renders title key', async () => {
    const Settings = (await import('../../routes/Settings')).default;
    withProviders(<Settings />);
    expect(screen.getByText('settings.title')).toBeInTheDocument();
  });

  it('Dashboard renders upload and model sections', async () => {
    const Dashboard = (await import('../../routes/Dashboard')).default;
    withProviders(<Dashboard />);
    expect(screen.getByText('dashboard.upload.title')).toBeInTheDocument();
    expect(screen.getByText('dashboard.model.title')).toBeInTheDocument();
  });

  it('History renders title key', async () => {
    const History = (await import('../../routes/History')).default;
    withProviders(<History />);
    expect(screen.getByText('history.title')).toBeInTheDocument();
  });
});
