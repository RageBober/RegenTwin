import { create } from 'zustand';
import type { SimulationRequest } from '../types/api';
import { DEFAULT_SIMULATION_PARAMS } from '../types/api';

interface SimulationStore {
  params: SimulationRequest;
  setParam: <K extends keyof SimulationRequest>(key: K, value: SimulationRequest[K]) => void;
  setParams: (patch: Partial<SimulationRequest>) => void;
  resetParams: () => void;
  uploadId: string | null;
  setUploadId: (id: string | null) => void;
  applyUpload: (id: string, patch?: Partial<SimulationRequest> | null) => void;
  activeSimulationId: string | null;
  setActiveSimulationId: (id: string | null) => void;
}

export const useSimulationStore = create<SimulationStore>((set) => ({
  params: { ...DEFAULT_SIMULATION_PARAMS },

  setParam: (key, value) =>
    set((state) => ({
      params: { ...state.params, [key]: value },
    })),

  setParams: (patch) =>
    set((state) => ({
      params: { ...state.params, ...patch },
    })),

  resetParams: () =>
    set({ params: { ...DEFAULT_SIMULATION_PARAMS }, uploadId: null }),

  uploadId: null,
  setUploadId: (id) =>
    set((state) => ({
      uploadId: id,
      params: { ...state.params, upload_id: id },
    })),
  applyUpload: (id, patch) =>
    set((state) => ({
      uploadId: id,
      params: {
        ...state.params,
        ...(patch ?? {}),
        upload_id: id,
      },
    })),

  activeSimulationId: null,
  setActiveSimulationId: (id) => set({ activeSimulationId: id }),
}));
