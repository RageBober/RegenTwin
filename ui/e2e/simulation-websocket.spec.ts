import { test, expect } from '@playwright/test';
import {
  clickRun,
  fillTMax,
  goToStep,
  openDashboard,
  selectMode,
  uploadSampleFcs,
  waitForSimulationComplete,
} from './helpers/simulation';

test.describe('WebSocket прогресс', () => {
  test('WS отдаёт progress и complete события', async ({ page }) => {
    test.setTimeout(240_000);

    await openDashboard(page);
    await uploadSampleFcs(page);
    await goToStep(page, 1);
    await selectMode(page, 'extended');
    await goToStep(page, 2);
    await fillTMax(page, 720);
    await page.getByRole('button', { name: /Монте-Карло|Monte Carlo/i }).first().click();
    const nTrajInput = page.getByTestId('param-n-trajectories');
    await nTrajInput.waitFor({ state: 'attached', timeout: 10_000 });
    await nTrajInput.first().fill('10');
    await goToStep(page, 3);

    const simulateResponsePromise = page.waitForResponse(
      (r) => r.request().method() === 'POST' && r.url().endsWith('/api/v1/simulate'),
      { timeout: 10_000 },
    );
    await clickRun(page);
    const simulateResponse = await simulateResponsePromise;
    const simulateBody = await simulateResponse.json();
    const simulationId = simulateBody.simulation_id as string;
    expect(simulationId).toMatch(/^[0-9a-f-]{36}$/);

    // Открываем WebSocket напрямую из контекста страницы, чтобы не зависеть
    // от UI-хука, который подключается только после status poll со status=running.
    const wsResult = await page.evaluate(async (simId: string) => {
      return await new Promise<{ frames: string[]; closeCode: number | null; error: string | null }>(
        (resolve) => {
          const frames: string[] = [];
          let closeCode: number | null = null;
          let settled = false;
          const url = `ws://${window.location.host}/api/v1/simulate/${simId}/ws`;
          const ws = new WebSocket(url);
          const finish = (error: string | null) => {
            if (settled) return;
            settled = true;
            try {
              ws.close();
            } catch {
              /* noop */
            }
            resolve({ frames, closeCode, error });
          };
          ws.onmessage = (ev) => {
            if (typeof ev.data === 'string') {
              frames.push(ev.data);
              if (/"event"\s*:\s*"(complete|failed|cancelled|not_found)"/.test(ev.data)) {
                finish(null);
              }
            }
          };
          ws.onclose = (ev) => {
            closeCode = ev.code;
            finish(null);
          };
          ws.onerror = () => {
            finish('ws error');
          };
          // Safety timeout: даём бэкенду до 180s на прогон симуляции.
          setTimeout(() => finish('timeout'), 180_000);
        },
      );
    }, simulationId);

    console.log(`[E2E] WS frames: ${wsResult.frames.length}, close=${wsResult.closeCode}, err=${wsResult.error}`);
    if (wsResult.frames.length > 0) {
      console.log(`[E2E] First WS frame: ${wsResult.frames[0].slice(0, 200)}`);
      console.log(`[E2E] Last WS frame: ${wsResult.frames[wsResult.frames.length - 1].slice(0, 200)}`);
    }

    const progressEvents = wsResult.frames.filter((f) => /"event"\s*:\s*"progress"/.test(f));
    const completeEvents = wsResult.frames.filter((f) => /"event"\s*:\s*"complete"/.test(f));
    expect(progressEvents.length, 'expected at least one WS progress frame').toBeGreaterThan(0);
    expect(completeEvents.length, 'expected at least one WS complete frame').toBeGreaterThan(0);

    // Дополнительно ждём, чтобы UI корректно закрыл симуляцию (view-results button).
    await waitForSimulationComplete(page, 30_000).catch(() => {});
  });
});
