import { test, expect } from '@playwright/test';
import { openHome } from './helpers/simulation';

const ROUTES: Array<{ path: string; match: RegExp }> = [
  { path: '/', match: /RegenTwin|Главная|Home/i },
  { path: '/dashboard', match: /Панель управления|Dashboard/i },
  { path: '/history', match: /История|History/i },
  { path: '/analysis', match: /Анализ|Analysis/i },
  { path: '/about', match: /О проекте|About/i },
  { path: '/settings', match: /Настройки|Settings/i },
];

test.describe('Навигация по основным роутам', () => {
  test('Каждый роут загружается без консольных ошибок', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(`pageerror: ${err.message}`));
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        const text = msg.text();
        if (!/ResizeObserver|Plotly|Failed to load resource|favicon/i.test(text)) {
          errors.push(`console.error: ${text}`);
        }
      }
    });

    for (const { path, match } of ROUTES) {
      await page.goto(path, { waitUntil: 'domcontentloaded' });
      await expect(page.getByRole('heading', { level: 1 }).first()).toBeVisible();
      await expect(page.getByRole('heading', { level: 1 }).first()).toContainText(match);
    }

    expect(errors, `Unexpected console errors: ${errors.join('\n')}`).toEqual([]);
  });

  test('Sidebar показывает все пункты меню на Home', async ({ page }) => {
    await openHome(page);
    const sidebar = page.locator('aside');
    await expect(sidebar).toBeVisible();
    for (const label of ['Главная', 'Панель управления', 'История', 'Анализ', 'Настройки', 'О проекте']) {
      await expect(sidebar.getByText(label, { exact: true }).first()).toBeVisible();
    }
  });

  test('Клик по sidebar-пункту меняет URL', async ({ page }) => {
    await openHome(page);
    const sidebar = page.locator('aside');
    await sidebar.getByText('История', { exact: true }).first().click();
    await expect(page).toHaveURL(/\/history/);
    await sidebar.getByText('Анализ', { exact: true }).first().click();
    await expect(page).toHaveURL(/\/analysis/);
    await sidebar.getByText('Главная', { exact: true }).first().click();
    await expect(page).toHaveURL(/\/$|\/#?$/);
  });
});
