import { test, expect } from '@playwright/test';
import { openDashboard } from './helpers/simulation';

test.describe('UI пути ошибок', () => {
  test('Upload не-FCS файла игнорируется (нативная фильтрация)', async ({ page }) => {
    await openDashboard(page);
    const fileInput = page.locator('input[type="file"][accept=".fcs"]');
    await fileInput.setInputFiles({
      name: 'not-an-fcs.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('not a real fcs file'),
    });
    await expect(page.locator('text=sample.fcs')).toHaveCount(0);
    const errorBadge = page.locator('text=/ошибка|error/i');
    const uploadedLabel = page.locator('text=not-an-fcs.txt');
    await expect(uploadedLabel).toHaveCount(0);
    await expect(errorBadge).toHaveCount(0);
  });

  test('Невалидный simulation_id в URL Results показывает ошибку/unavailable', async ({ page }) => {
    await page.goto('/results/00000000-0000-0000-0000-000000000000', { waitUntil: 'domcontentloaded' });
    const errorCard = page.locator(
      'text=/недоступн|unavailable|Не удалось загрузить|failed to load|Results unavailable/i',
    );
    await expect(errorCard.first()).toBeVisible({ timeout: 30_000 });
  });

  test('ErrorBoundary / fallback при отказе /api/v1/health', async ({ page }) => {
    await page.route('**/api/v1/health', (route) => route.abort());
    await page.goto('/', { waitUntil: 'domcontentloaded' });
    await expect(page.getByRole('heading', { level: 1 }).first()).toBeVisible();
  });
});
