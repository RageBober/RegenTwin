import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ModelSelector from '../../components/Parameters/ModelSelector';
import { useSimulationStore } from '../../stores/simulationStore';
import { DEFAULT_SIMULATION_PARAMS } from '../../types/api';

// Mock i18next
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const map: Record<string, string> = {
        'dashboard.model.title': 'Model',
        'dashboard.model.mvp': 'MVP',
        'dashboard.model.extended': 'Extended',
        'dashboard.model.abm': 'ABM',
        'dashboard.model.integrated': 'Integrated',
      };
      return map[key] || key;
    },
  }),
}));

describe('ModelSelector', () => {
  beforeEach(() => {
    useSimulationStore.setState({ params: { ...DEFAULT_SIMULATION_PARAMS } });
  });

  it('renders all 4 mode options', () => {
    render(<ModelSelector />);
    expect(screen.getByText('MVP')).toBeInTheDocument();
    expect(screen.getByText('Extended')).toBeInTheDocument();
    expect(screen.getByText('ABM')).toBeInTheDocument();
    expect(screen.getByText('Integrated')).toBeInTheDocument();
  });

  it('selects the current mode from store', () => {
    render(<ModelSelector />);
    const extendedRadio = screen.getByDisplayValue('extended') as HTMLInputElement;
    expect(extendedRadio.checked).toBe(true);
  });

  it('updates store when mode changes', () => {
    render(<ModelSelector />);
    fireEvent.click(screen.getByDisplayValue('abm'));
    expect(useSimulationStore.getState().params.mode).toBe('abm');
  });
});
