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
        'dashboard.model.mvpDesc': 'Fast model',
        'dashboard.model.extended': 'Extended',
        'dashboard.model.extendedDesc': 'Full model',
        'dashboard.model.abm': 'ABM',
        'dashboard.model.abmDesc': 'Agent model',
        'dashboard.model.integrated': 'Integrated',
      };
      return map[key] || key;
    },
  }),
}));

// Mock framer-motion to avoid animation issues in tests
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: React.PropsWithChildren<Record<string, unknown>>) => {
      const { layoutId: _l, initial: _i, animate: _a, exit: _e, transition: _t, ...rest } = props;
      return <div {...rest}>{children}</div>;
    },
  },
  AnimatePresence: ({ children }: React.PropsWithChildren) => children,
}));

describe('ModelSelector', () => {
  beforeEach(() => {
    useSimulationStore.setState({ params: { ...DEFAULT_SIMULATION_PARAMS } });
  });

  it('renders only the currently supported mode options', () => {
    render(<ModelSelector />);
    expect(screen.getByText('MVP')).toBeInTheDocument();
    expect(screen.getByText('Extended')).toBeInTheDocument();
    expect(screen.getByText('ABM')).toBeInTheDocument();
    expect(screen.queryByText('Integrated')).not.toBeInTheDocument();
  });

  it('selects the current mode from store', () => {
    render(<ModelSelector />);
    // Default mode is 'extended' — the Extended button should exist
    expect(screen.getByText('Extended')).toBeInTheDocument();
    expect(useSimulationStore.getState().params.mode).toBe('extended');
  });

  it('updates store when mode changes', () => {
    render(<ModelSelector />);
    fireEvent.click(screen.getByText('ABM'));
    expect(useSimulationStore.getState().params.mode).toBe('abm');
  });
});
