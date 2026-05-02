import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import ExportPanel from '../../components/Results/ExportPanel';
import { DEFAULT_SIMULATION_PARAMS, toSimulationParams } from '../../types/api';
import { normalizeTraceName } from '../../components/Visualization/SpatialView3D';

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
  apiClient: { post: postMock },
  API_V1: '/api/v1',
  API_VIZ: '/api/viz',
}));

vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: { children: ReactNode }) => <div>{children}</div>,
}));

vi.mock('@react-three/drei', () => ({
  OrbitControls: () => null,
}));

vi.mock('../../hooks/useSpatialData', () => ({
  useSpatialScatter: vi.fn(() => ({ data: null, isLoading: false })),
}));

describe('Export and spatial regressions', () => {
  beforeEach(() => {
    postMock.mockReset();
    Object.defineProperty(globalThis.URL, 'createObjectURL', {
      configurable: true,
      writable: true,
      value: vi.fn(() => 'blob:mock'),
    });
    Object.defineProperty(globalThis.URL, 'revokeObjectURL', {
      configurable: true,
      writable: true,
      value: vi.fn(),
    });
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
  });

  it('shows inline export errors and clears them after a successful retry', async () => {
    postMock
      .mockRejectedValueOnce({ response: { data: { detail: 'Export backend failed' } } })
      .mockResolvedValueOnce({
        data: new Uint8Array([1, 2, 3]),
        headers: { 'content-disposition': 'attachment; filename="results.export.csv"' },
      });

    render(
      <ExportPanel
        simulationId="sim-1"
        params={toSimulationParams(DEFAULT_SIMULATION_PARAMS)}
        mode="extended"
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'results.export.csv' }));

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('Export backend failed');
    });

    fireEvent.click(screen.getByRole('button', { name: 'results.export.csv' }));

    await waitFor(() => {
      expect(postMock).toHaveBeenCalledTimes(2);
    });
    await waitFor(() => {
      expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    });
    expect(globalThis.URL.createObjectURL).toHaveBeenCalled();
  });

  it('normalizes localized and raw trace labels to backend agent types', () => {
    expect(normalizeTraceName('Макрофаги')).toBe('macro');
    expect(normalizeTraceName('Fibroblasts')).toBe('fibro');
    expect(normalizeTraceName('stem')).toBe('stem');
    expect(normalizeTraceName('Endothelial cells')).toBe('endothelial');
  });
});
