import { describe, it, expect, beforeEach } from 'vitest';
import { useSimulationStore } from '../../stores/simulationStore';
import { DEFAULT_SIMULATION_PARAMS } from '../../types/api';

describe('simulationStore', () => {
  beforeEach(() => {
    useSimulationStore.setState({
      params: { ...DEFAULT_SIMULATION_PARAMS },
      uploadId: null,
      activeSimulationId: null,
    });
  });

  it('has correct default params matching schemas.py', () => {
    const { params } = useSimulationStore.getState();
    expect(params.P0).toBe(500);
    expect(params.Ne0).toBe(200);
    expect(params.M1_0).toBe(100);
    expect(params.M2_0).toBe(10);
    expect(params.F0).toBe(50);
    expect(params.Mf0).toBe(0);
    expect(params.E0).toBe(20);
    expect(params.S0).toBe(40);
    expect(params.C_TNF0).toBe(10);
    expect(params.C_IL10_0).toBe(0.5);
    expect(params.D0).toBe(5);
    expect(params.O2_0).toBe(80);
    expect(params.t_max_hours).toBe(720);
    expect(params.dt).toBe(0.1);
    expect(params.mode).toBe('extended');
  });

  it('setParam updates a single parameter', () => {
    useSimulationStore.getState().setParam('P0', 1000);
    expect(useSimulationStore.getState().params.P0).toBe(1000);
    // Other params unchanged
    expect(useSimulationStore.getState().params.Ne0).toBe(200);
  });

  it('setParam updates mode', () => {
    useSimulationStore.getState().setParam('mode', 'abm');
    expect(useSimulationStore.getState().params.mode).toBe('abm');
  });

  it('setParam updates therapy fields', () => {
    useSimulationStore.getState().setParam('prp_enabled', true);
    expect(useSimulationStore.getState().params.prp_enabled).toBe(true);

    useSimulationStore.getState().setParam('prp_intensity', 0.8);
    expect(useSimulationStore.getState().params.prp_intensity).toBe(0.8);
  });

  it('resetParams restores defaults', () => {
    useSimulationStore.getState().setParam('P0', 9999);
    useSimulationStore.getState().setParam('mode', 'integrated');
    useSimulationStore.getState().resetParams();

    const { params } = useSimulationStore.getState();
    expect(params.P0).toBe(500);
    expect(params.mode).toBe('extended');
  });

  it('setUploadId stores upload ID', () => {
    useSimulationStore.getState().setUploadId('upload-123');
    expect(useSimulationStore.getState().uploadId).toBe('upload-123');
  });

  it('setUploadId can clear upload ID', () => {
    useSimulationStore.getState().setUploadId('upload-123');
    useSimulationStore.getState().setUploadId(null);
    expect(useSimulationStore.getState().uploadId).toBeNull();
  });

  it('setActiveSimulationId stores simulation ID', () => {
    useSimulationStore.getState().setActiveSimulationId('sim-456');
    expect(useSimulationStore.getState().activeSimulationId).toBe('sim-456');
  });
});
