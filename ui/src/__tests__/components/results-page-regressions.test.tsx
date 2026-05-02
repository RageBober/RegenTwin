import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import Results from '../../routes/Results';

const { refetchResults, refetchMeta } = vi.hoisted(() => ({
  refetchResults: vi.fn(),
  refetchMeta: vi.fn(),
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) => key,
    i18n: { language: 'en', changeLanguage: vi.fn() },
  }),
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

function renderResultsRoute() {
  return render(
    <MemoryRouter initialEntries={['/results/sim-1']}>
      <Routes>
        <Route path="/results/:id" element={<Results />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('Results page regressions', () => {
  beforeEach(() => {
    refetchResults.mockReset();
    refetchMeta.mockReset();
  });

  it('renders an error state instead of an infinite spinner when results loading fails', async () => {
    const { useResults, useSimulationMeta } = await import('../../hooks/useResults');
    vi.mocked(useResults).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: { response: { data: { detail: 'Result files not found' } } },
      refetch: refetchResults,
    } as ReturnType<typeof useResults>);
    vi.mocked(useSimulationMeta).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: false,
      error: null,
      refetch: refetchMeta,
    } as ReturnType<typeof useSimulationMeta>);

    renderResultsRoute();

    expect(screen.getByText('results.states.loadFailedTitle')).toBeInTheDocument();
    expect(screen.getByText('Result files not found')).toBeInTheDocument();

    fireEvent.click(screen.getByText('common.retry'));
    expect(refetchResults).toHaveBeenCalledTimes(1);
    expect(refetchMeta).toHaveBeenCalledTimes(1);
  });

  it('renders an unavailable state when metadata cannot produce visualization params', async () => {
    const { useResults, useSimulationMeta } = await import('../../hooks/useResults');
    vi.mocked(useResults).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: false,
      error: null,
      refetch: refetchResults,
    } as ReturnType<typeof useResults>);
    vi.mocked(useSimulationMeta).mockReturnValue({
      data: {
        simulation_id: 'sim-1',
        status: 'completed',
        progress: 100,
        message: 'done',
        created_at: '2026-03-18T00:00:00Z',
        completed_at: '2026-03-18T01:00:00Z',
        params_json: undefined,
      },
      isLoading: false,
      isError: false,
      error: null,
      refetch: refetchMeta,
    } as ReturnType<typeof useSimulationMeta>);

    renderResultsRoute();

    expect(screen.getByText('results.states.unavailableTitle')).toBeInTheDocument();
    expect(screen.getByText('results.states.unavailableDescription')).toBeInTheDocument();
  });
});
