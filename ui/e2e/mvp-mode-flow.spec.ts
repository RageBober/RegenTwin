import { test, expect } from '@playwright/test';
import { runSimulationViaUI } from './helpers/simulation';

test.describe('MVP flow', () => {
  test('Полный цикл: upload → MVP → run → results', async ({ page }) => {
    test.setTimeout(240_000);

    page.on('response', async (response) => {
      if (response.url().includes('/api/v1/simulate') && response.request().method() === 'POST') {
        const status = response.status();
        if (status !== 200) {
          const req = response.request();
          const body = req.postData();
          let respText = '';
          try { respText = await response.text(); } catch { /* empty */ }
          console.log(`[DEBUG] POST /simulate -> ${status}`);
          console.log(`[DEBUG] request body: ${body}`);
          console.log(`[DEBUG] response body: ${respText}`);
        }
      }
    });

    await runSimulationViaUI(page, 'mvp', { tMaxHours: 24 });

    // Results page: ищем populations-график
    await expect(page.getByText(/Результаты симуляции|Results/i).first()).toBeVisible();
    await expect(page.locator('.js-plotly-plot').first()).toBeVisible({ timeout: 30_000 });
  });
});
