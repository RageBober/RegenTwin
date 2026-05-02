import { useEffect, useMemo, useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import type { ComponentType, ReactNode, SVGProps } from 'react';
import { HomeIcon, ClockIcon, BeakerIcon, InformationCircleIcon, Cog6ToothIcon } from '@heroicons/react/24/outline';
import { apiClient, API_V1 } from '../lib/api';
import { useUIStore } from '../stores/uiStore';
import { useSimulationsList } from '../hooks/useSimulation';
import TopBar from './common/TopBar';

const MicroscopeIcon = (props: SVGProps<SVGSVGElement>) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} {...props}>
    <path d="M7 21h10M12 21v-4M9 3h6M12 3v4M9 7l-2 6h10l-2-6" strokeLinecap="round" strokeLinejoin="round" />
    <circle cx="12" cy="16" r="1" />
  </svg>
);

const ThemeIcon = (props: SVGProps<SVGSVGElement>) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} {...props}>
    <path
      d="M21.75 15.5A9.75 9.75 0 0 1 18 16.25 9.75 9.75 0 0 1 8.25 6.5c0-1.33.27-2.6.75-3.75A9.75 9.75 0 0 0 3 12.5c0 5.38 4.37 9.75 9.75 9.75 4.34 0 8-2.83 9.27-6.75Z"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const BrandMark = () => (
  <div
    className="h-8 w-8 rounded-lg flex items-center justify-center"
    style={{ background: 'var(--accent-soft)', border: '1px solid var(--accent-border)' }}
  >
    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="var(--accent)" strokeWidth="1.8">
      <path
        d="M12 3c-1.5 3-4 4.5-4 7s2.5 4 4 7c1.5-3 4-4.5 4-7s-2.5-4-4-7Z"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="12" cy="10" r="1.5" fill="var(--accent)" />
    </svg>
  </div>
);

type NavItem = {
  path: string;
  icon: ComponentType<SVGProps<SVGSVGElement>>;
  labelKey: string;
  trailing?: ReactNode;
};

const sections: readonly { label?: string; items: readonly NavItem[] }[] = [
  {
    items: [
      { path: '/', icon: HomeIcon, labelKey: 'nav.home' },
      { path: '/dashboard', icon: MicroscopeIcon, labelKey: 'nav.dashboard' },
    ],
  },
  {
    label: 'Работа',
    items: [
      { path: '/history', icon: ClockIcon, labelKey: 'nav.history' },
      { path: '/analysis', icon: BeakerIcon, labelKey: 'nav.analysis' },
    ],
  },
  {
    items: [
      { path: '/about', icon: InformationCircleIcon, labelKey: 'nav.about' },
      { path: '/settings', icon: Cog6ToothIcon, labelKey: 'nav.settings' },
    ],
  },
] as const;

export default function Layout() {
  const { t } = useTranslation();
  const theme = useUIStore((state) => state.theme);
  const setTheme = useUIStore((state) => state.setTheme);
  const [backendOnline, setBackendOnline] = useState(false);
  const { data: simulations } = useSimulationsList();

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await apiClient.get(`${API_V1}/health`);
        setBackendOnline(true);
      } catch {
        setBackendOnline(false);
      }
    };

    void checkHealth();
    const interval = window.setInterval(checkHealth, backendOnline ? 30000 : 15000);
    return () => window.clearInterval(interval);
  }, [backendOnline]);

  const navSections = useMemo(() => {
    const historyCount = simulations?.length ?? 0;
    return sections.map((section) => ({
      ...section,
      items: section.items.map((item) =>
        item.path === '/history'
          ? {
              ...item,
              trailing: (
                <span className="ml-auto text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
                  {historyCount}
                </span>
              ),
            }
          : item,
      ),
    }));
  }, [simulations]);

  return (
    <div className="flex h-screen w-full">
      <aside
        className="flex flex-col w-[220px] shrink-0 border-r"
        style={{ background: 'var(--surface-1)', borderColor: 'var(--border-default)' }}
      >
        <div className="flex items-center gap-2.5 px-4 py-4 border-b" style={{ borderColor: 'var(--border-default)' }}>
          <BrandMark />
          <span className="font-display text-base font-semibold tracking-tight" style={{ color: 'var(--text-primary)' }}>
            RegenTwin
          </span>
        </div>

        <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-5">
          {navSections.map((section, index) => (
            <div
              key={index}
              className={index > 0 ? 'pt-4 border-t space-y-0.5' : 'space-y-0.5'}
              style={index > 0 ? { borderColor: 'var(--border-default)' } : undefined}
            >
              {section.label ? <div className="section-label px-2 mb-2">{section.label}</div> : null}
              {section.items.map(({ path, icon: Icon, labelKey, trailing }) => (
                <NavLink key={path} to={path} end={path === '/'} className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
                  <Icon className="h-[17px] w-[17px]" />
                  <span>{t(labelKey)}</span>
                  {trailing}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>

        <div className="p-3 border-t" style={{ borderColor: 'var(--border-default)' }}>
          <button
            className="btn-outline w-full justify-center"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            type="button"
          >
            <ThemeIcon className="h-3.5 w-3.5" />
            <span>Сменить тему</span>
          </button>
        </div>
      </aside>

      <div className="flex-1 flex flex-col min-w-0">
        <TopBar backendOnline={backendOnline} />

        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
