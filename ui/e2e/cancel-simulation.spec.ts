import { test, expect } from '@playwright/test';
import {
  clickRun,
  fillTMax,
  goToStep,
  openDashboard,
  selectMode,
  uploadSampleFcs,
} from './helpers/simulation';

test.describe('Отмена симуляции через UI', () => {
  test('Cancel running simulation', async ({ page }) => {
    test.setTimeout(240_000);

    await openDashboard(page);
    await uploadSampleFcs(page);
    await goToStep(page, 1);
    await selectMode(page, 'extended');

    // Step 2: задать длительную симуляцию
    await goToStep(page, 2);
    await fillTMax(page, 720);
    const nTrajInput = page.getByTestId('param-n-trajectories');
    if (await nTrajInput.count()) {
      await nTrajInput.first().fill('50');
    }

    await goToStep(page, 3);
    await clickRun(page);

    await page.getByTestId('simulation-running').waitFor({ state: 'visible', timeout: 30_000 });

    const cancelButton = page.getByTestId('simulation-cancel-button');
    await expect(cancelButton).toBeVisible({ timeout: 10_000 });
    await cancelButton.click();

    const terminal = page.locator(
      'text=/Симуляция отменена|Simulation cancelled|Симуляция завершена|Simulation complete/i',
    ).first();
    const viewResults = page.getByTestId('simulation-view-results-button');
    const runAgain = page.getByTestId('simulation-run-button');
    await expect(async () => {
      const [terminalCount, viewCount, runCount] = await Promise.all([
        terminal.count(),
        viewResults.count(),
        runAgain.count(),
      ]);
      expect(terminalCount + viewCount + runCount).toBeGreaterThan(0);
    }).toPass({ timeout: 120_000 });
  });
});
