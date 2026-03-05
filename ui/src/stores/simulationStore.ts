import { create } from 'zustand';
import type { SimulationRequest } from '../types/api';
import { DEFAULT_SIMULATION_PARAMS } from '../types/api';

interface SimulationStore {
  params: SimulationRequest;
  setParam: <K extends keyof SimulationRequest>(key: K, value: SimulationRequest[K]) => void;
  resetParams: () => void;
  uploadId: string | null;
  setUploadId: (id: string | null) => void;
  activeSimulationId: string | null;
  setActiveSimulationId: (id: string | null) => void;
}

export const useSimulationStore = create<SimulationStore>((set) => ({
  params: { ...DEFAULT_SIMULATION_PARAMS },

  setParam: (key, value) =>
    set((state) => ({
      params: { ...state.params, [key]: value },
    })),

  resetParams: () =>
    set({ params: { ...DEFAULT_SIMULATION_PARAMS } }),

  uploadId: null,
  setUploadId: (id) => set({ uploadId: id }),

  activeSimulationId: null,
  setActiveSimulationId: (id) => set({ activeSimulationId: id }),
}));
