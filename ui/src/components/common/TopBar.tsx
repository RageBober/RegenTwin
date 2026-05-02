import { useTranslation } from 'react-i18next';
import { useLocation } from 'react-router-dom';

const ROUTE_KEYS: Record<string, string> = {
  '/': 'nav.home',
  '/dashboard': 'nav.dashboard',
  '/analysis': 'nav.analysis',
  '/history': 'nav.history',
  '/settings': 'nav.settings',
  '/about': 'nav.about',
};

export default function TopBar({ backendOnline }: { backendOnline: boolean }) {
  const { t, i18n } = useTranslation();
  const location = useLocation();

  const toggleLanguage = () => {
    const nextLanguage = i18n.language === 'ru' ? 'en' : 'ru';
    void i18n.changeLanguage(nextLanguage);
    localStorage.setItem('regentwin-language', nextLanguage);
  };

  const pathSegments = location.pathname.split('/').filter(Boolean);
  const currentRouteKey = ROUTE_KEYS[location.pathname] || ROUTE_KEYS[`/${pathSegments[0]}`];

  return (
    <header
      className="h-12 flex items-center px-5 border-b"
      style={{ background: 'var(--surface-1)', borderColor: 'var(--border-default)' }}
    >
      <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-muted)' }}>
        <span>{t('nav.home')}</span>
        {currentRouteKey && currentRouteKey !== 'nav.home' ? (
          <>
            <span style={{ opacity: 0.4 }}>/</span>
            <span style={{ color: 'var(--text-primary)' }}>{t(currentRouteKey)}</span>
          </>
        ) : null}
      </div>

      <div className="flex-1" />

      <div className="flex items-center gap-3 text-xs">
        <div className="flex items-center gap-1.5">
          <div
            className="w-1.5 h-1.5 rounded-full"
            style={{
              background: backendOnline ? 'var(--success)' : 'var(--danger)',
              boxShadow: backendOnline ? '0 0 6px var(--success)' : '0 0 6px var(--danger)',
            }}
          />
          <span
            className="font-mono"
            style={{ color: backendOnline ? '#56d364' : '#ff7b72' }}
          >
            {backendOnline ? t('common.online') : t('common.offline')}
          </span>
        </div>

        <button className="btn-ghost font-mono text-xs" style={{ padding: '4px 8px' }} onClick={toggleLanguage} type="button">
          {i18n.language.toUpperCase()}
        </button>
      </div>
    </header>
  );
}
