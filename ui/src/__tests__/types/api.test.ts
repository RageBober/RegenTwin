import { describe, it, expect } from 'vitest';
import { DEFAULT_SIMULATION_PARAMS, toSimulationParams } from '../../types/api';
import type { SimulationRequest } from '../../types/api';

describe('api types', () => {
  describe('DEFAULT_SIMULATION_PARAMS', () => {
    it('has all required fields', () => {
      const keys: (keyof SimulationRequest)[] = [
        'mode', 'P0', 'Ne0', 'M1_0', 'M2_0', 'F0', 'Mf0', 'E0', 'S0',
        'C_TNF0', 'C_IL10_0', 'D0', 'O2_0', 't_max_hours', 'dt',
        'prp_enabled', 'pemf_enabled', 'prp_intensity', 'pemf_frequency',
        'pemf_intensity', 'random_seed', 'n_trajectories', 'upload_id',
      ];
      for (const key of keys) {
        expect(DEFAULT_SIMULATION_PARAMS).toHaveProperty(key);
      }
    });

    it('therapy defaults are disabled', () => {
      expect(DEFAULT_SIMULATION_PARAMS.prp_enabled).toBe(false);
      expect(DEFAULT_SIMULATION_PARAMS.pemf_enabled).toBe(false);
    });
  });

  describe('toSimulationParams', () => {
    it('extracts SimulationParams from SimulationRequest', () => {
      const result = toSimulationParams(DEFAULT_SIMULATION_PARAMS);
      expect(result.P0).toBe(500);
      expect(result.t_max_hours).toBe(720);
      expect(result).not.toHaveProperty('mode');
      expect(result).not.toHaveProperty('n_trajectories');
      expect(result).not.toHaveProperty('upload_id');
    });

    it('preserves all numerical fields', () => {
      const custom: SimulationRequest = {
        ...DEFAULT_SIMULATION_PARAMS,
        P0: 1000,
        prp_enabled: true,
        prp_intensity: 2.5,
      };
      const result = toSimulationParams(custom);
      expect(result.P0).toBe(1000);
      expect(result.prp_enabled).toBe(true);
      expect(result.prp_intensity).toBe(2.5);
    });
  });
});
