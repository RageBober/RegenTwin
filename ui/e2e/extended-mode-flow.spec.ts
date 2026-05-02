import { test, expect } from '@playwright/test';
import { runSimulationViaUI } from './helpers/simulation';

test.describe('Extended SDE flow', () => {
  test('Полный цикл: upload → Extended → run → results с plotly charts', async ({ page }) => {
    test.setTimeout(240_000);
    await runSimulationViaUI(page, 'extended', { tMaxHours: 24 });

    await expect(page.getByText(/Результаты симуляции|Results/i).first()).toBeVisible();
    // Populations tab — первая по умолчанию
    await expect(page.locator('.js-plotly-plot').first()).toBeVisible({ timeout: 30_000 });

    // Переключиться на cytokines
    const cytoTab = page.getByRole('button', { name: /Цитокин|Cytokine/i });
    if (await cytoTab.count()) {
      await cytoTab.first().click();
      await expect(page.locator('.js-plotly-plot').first()).toBeVisible({ timeout: 20_000 });
    }

    // Переключиться на ECM
    const ecmTab = page.getByRole('button', { name: /ECM|Матрикс/i });
    if (await ecmTab.count()) {
      await ecmTab.first().click();
      await expect(page.locator('.js-plotly-plot').first()).toBeVisible({ timeout: 20_000 });
    }
  });
});
