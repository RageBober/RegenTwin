import { create } from 'zustand';

interface UIStore {
  sidebarCollapsed: boolean;
  theme: 'light' | 'dark';
  toggleSidebar: () => void;
  setTheme: (t: 'light' | 'dark') => void;
}

const savedTheme = (localStorage.getItem('regentwin-theme') as 'light' | 'dark') || 'light';

export const useUIStore = create<UIStore>((set) => ({
  sidebarCollapsed: false,
  theme: savedTheme,

  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),

  setTheme: (t) => {
    localStorage.setItem('regentwin-theme', t);
    if (t === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    set({ theme: t });
  },
}));
