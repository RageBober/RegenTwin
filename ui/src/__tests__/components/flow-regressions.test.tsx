import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import UploadFCS from '../../components/Upload/UploadFCS';
import Results from '../../routes/Results';
import { useSimulationStore } from '../../stores/simulationStore';
import { DEFAULT_SIMULATION_PARAMS } from '../../types/api';

const { postMock } = vi.hoisted(() => ({
  postMock: vi.fn(),
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) => key,
    i18n: { language: 'en', changeLanguage: vi.fn() },
  }),
}));

vi.mock('../../lib/api', () => ({
  apiClient: { get: vi.fn(), post: postMock },
  API_V1: '/api/v1',
  API_VIZ: '/api/viz',
  updateApiBaseUrl: vi.fn(),
}));

vi.mock('../../hooks/useResults', () => ({
  useResults: vi.fn(),
  useSimulationMeta: vi.fn(),
}));

vi.mock('../../components/Visualization/PopulationCharts', () => ({ default: () => <div>PopulationCharts</div> }));
vi.mock('../../components/Visualization/CytokineCharts', () => ({ default: () => <div>CytokineCharts</div> }));
vi.mock('../../components/Visualization/ECMCharts', () => ({ default: () => <div>ECMCharts</div> }));
vi.mock('../../components/Visualization/PhaseTimeline', () => ({ default: () => <div>PhaseTimeline</div> }));
vi.mock('../../components/Visualization/TherapyComparison', () => ({ default: () => <div>TherapyComparison</div> }));
vi.mock('../../components/Visualization/CellHeatmap', () => ({ default: () => <div>CellHeatmap</div> }));
vi.mock('../../components/Visualization/InflammationMap', () => ({ default: () => <div>InflammationMap</div> }));
vi.mock('../../components/Visualization/AnimationPlayer', () => ({ default: () => <div>AnimationPlayer</div> }));
vi.mock('../../components/Visualization/SpatialView3D', () => ({ default: () => <div>SpatialView3D</div> }));
vi.mock('../../components/Results/ExportPanel', () => ({ default: () => <div>ExportPanel</div> }));

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
}

describe('Flow regressions', () => {
  beforeEach(() => {
    postMock.mockReset();
    useSimulationStore.setState({
      params: { ...DEFAULT_SIMULATION_PARAMS },
      uploadId: null,
      activeSimulationId: null,
    });
  });

  it('applies upload-derived initial conditions to the store', async () => {
    postMock.mockResolvedValue({
      data: {
        upload_id: 'upload-1',
        filename: 'sample.fcs',
        status: 'ready',
        created_at: '2026-03-18T00:00:00Z',
        metadata: {
          parameter_source: 'metadata_heuristic',
          initial_conditions: {
            F0: 111,
            S0: 22,
          },
        },
      },
    });

    const { container } = renderWithProviders(<UploadFCS />);
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(['fake'], 'sample.fcs', { type: 'application/octet-stream' });

    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => {
      const state = useSimulationStore.getState();
      expect(state.uploadId).toBe('upload-1');
      expect(state.params.upload_id).toBe('upload-1');
      expect(state.params.F0).toBe(111);
      expect(state.params.S0).toBe(22);
    });
  });

  it('shows only ABM-relevant result tabs for ABM runs', async () => {
    const { useResults, useSimulationMeta } = await import('../../hooks/useResults');
    vi.mocked(useResults).mockReturnValue({
      data: {
        simulation_id: 'sim-1',
        mode: 'abm',
        times: [0, 24, 48],
        variables: { stem: [1, 2, 3] },
        metadata: {},
      },
      isLoading: false,
    } as ReturnType<typeof useResults>);
    vi.mocked(useSimulationMeta).mockReturnValue({
      data: {
        simulation_id: 'sim-1',
        status: 'completed',
        progress: 100,
        message: 'done',
        created_at: '2026-03-18T00:00:00Z',
        completed_at: '2026-03-18T01:00:00Z',
        params_json: { ...DEFAULT_SIMULATION_PARAMS, mode: 'abm' },
      },
      isLoading: false,
    } as ReturnType<typeof useSimulationMeta>);

    renderWithProviders(
      <MemoryRouter initialEntries={['/results/sim-1']}>
        <Routes>
          <Route path="/results/:id" element={<Results />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.getByText('results.tabs.heatmap')).toBeInTheDocument();
    expect(screen.getByText('results.tabs.animation')).toBeInTheDocument();
    expect(screen.queryByText('results.tabs.cytokines')).not.toBeInTheDocument();
    expect(screen.queryByText('results.tabs.comparison')).not.toBeInTheDocument();
  });
});
