import { test, expect } from '@playwright/test';
import {
  clickRun,
  fillPopulations,
  fillTMax,
  goToStep,
  openDashboard,
  openResultsFromRunner,
  selectMode,
  uploadSampleFcs,
  waitForSimulationComplete,
} from './helpers/simulation';

test.describe('ABM flow (key scenario)', () => {
  test('Upload → ABM → run → spatial visualizations', async ({ page }) => {
    test.setTimeout(600_000);

    await openDashboard(page);
    await uploadSampleFcs(page);

    await goToStep(page, 1);
    await selectMode(page, 'abm');

    // Короткий t_max и минимальные популяции — иначе ABM тянет тысячи шагов.
    await goToStep(page, 2);
    await fillTMax(page, 24);
    await fillPopulations(page, {
      P0: 10, Ne0: 5, M10: 2, M20: 1, F0: 5, Mf0: 0, E0: 2, S0: 2,
    });

    await goToStep(page, 3);
    await clickRun(page);

    await waitForSimulationComplete(page, 420_000);
    await openResultsFromRunner(page);

    // По умолчанию активен populations tab
    await expect(page.locator('.js-plotly-plot').first()).toBeVisible({ timeout: 30_000 });

    // Heatmap tab
    const heatmapTab = page.getByRole('button', { name: /Heatmap|Карта клеток|Тепловая/i });
    await heatmapTab.first().click();
    await expect(page.getByTestId('cell-heatmap')).toBeVisible({ timeout: 20_000 });

    // Inflammation tab
    const inflTab = page.getByRole('button', { name: /Воспаление|Inflammation/i });
    await inflTab.first().click();
    await expect(page.getByTestId('inflammation-map')).toBeVisible({ timeout: 20_000 });

    // Animation tab
    const animTab = page.getByRole('button', { name: /Анимация|Animation/i });
    await animTab.first().click();
    await expect(page.getByTestId('animation-player')).toBeVisible({ timeout: 20_000 });

    // Click play button and verify it toggles to pause
    const playButton = page.getByTestId('animation-play');
    await expect(playButton).toBeVisible();
    await playButton.click();
    await expect(page.getByTestId('animation-pause')).toBeVisible({ timeout: 5_000 });
    // Pause back
    await page.getByTestId('animation-pause').click();
    await expect(page.getByTestId('animation-play')).toBeVisible({ timeout: 5_000 });

    // 3D tab — проверяем наличие canvas и data-testid root
    const threeDTab = page.getByRole('button', { name: /3D|Простран/i });
    await threeDTab.first().click();
    await expect(page.getByTestId('spatial-view-3d')).toBeVisible({ timeout: 20_000 });
    await expect(page.getByTestId('spatial-view-3d').locator('canvas')).toBeVisible({ timeout: 20_000 });
  });
});
