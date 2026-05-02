import { expect, Page } from '@playwright/test';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export const FIXTURE_FCS = join(__dirname, '..', 'fixtures', 'sample.fcs');

export type UIMode = 'mvp' | 'extended' | 'abm';

const MODE_LABEL: Record<UIMode, RegExp> = {
  mvp: /MVP/i,
  extended: /SDE/i,
  abm: /Агент|Agent/i,
};

/** Ждём, пока фронтенд смонтируется. */
export async function openHome(page: Page): Promise<void> {
  await page.goto('/', { waitUntil: 'domcontentloaded' });
  await expect(page.getByRole('heading', { level: 1 }).first()).toBeVisible();
}

export async function openDashboard(page: Page): Promise<void> {
  await page.goto('/dashboard', { waitUntil: 'domcontentloaded' });
  await expect(page.getByRole('heading', { level: 1 }).first()).toBeVisible();
}

/** Загружает FCS через скрытый `<input type="file">`. */
export async function uploadSampleFcs(page: Page): Promise<void> {
  const fileInput = page.locator('input[type="file"][accept=".fcs"]');
  await fileInput.setInputFiles(FIXTURE_FCS);
  await expect(page.locator('text=sample.fcs').first()).toBeVisible({ timeout: 20_000 });
}

/** Переход между шагами степпера по data-testid. */
export async function goToStep(page: Page, step: 0 | 1 | 2 | 3): Promise<void> {
  await page.getByTestId(`dashboard-step-${step}`).click();
}

/** Выбирает режим симуляции в ModelSelector. */
export async function selectMode(page: Page, mode: UIMode): Promise<void> {
  const button = page.getByRole('button', { name: MODE_LABEL[mode] });
  await button.first().click();
}

/** Клик по кнопке Run на шаге 3. */
export async function clickRun(page: Page): Promise<void> {
  const run = page.getByTestId('simulation-run-button');
  await expect(run).toBeVisible();
  await run.click();
}

/** Ждёт состояние running (симуляция стартовала). */
export async function waitForSimulationRunning(page: Page, timeout = 30_000): Promise<void> {
  await page.getByTestId('simulation-running').waitFor({ state: 'visible', timeout });
}

/** Ждёт terminal-состояние симуляции (view-results button появился). */
export async function waitForSimulationComplete(page: Page, timeout = 120_000): Promise<void> {
  await page.getByTestId('simulation-view-results-button').waitFor({ state: 'visible', timeout });
}

/** Переходит к экрану Results после завершения симуляции. */
export async function openResultsFromRunner(page: Page): Promise<string> {
  await page.getByTestId('simulation-view-results-button').click();
  await page.waitForURL(/\/results\/[0-9a-f-]{36}/);
  const match = page.url().match(/\/results\/([0-9a-f-]{36})/);
  return match ? match[1] : '';
}

/** Надёжно заполняет t_max через testid (ждёт монтирования DOM). */
export async function fillTMax(page: Page, tMaxHours: number): Promise<void> {
  const input = page.getByTestId('param-t-max');
  await input.waitFor({ state: 'attached', timeout: 10_000 });
  await input.first().fill(String(tMaxHours));
}

export interface PopulationOverrides {
  P0?: number;
  Ne0?: number;
  M10?: number;
  M20?: number;
  F0?: number;
  Mf0?: number;
  E0?: number;
  S0?: number;
}

/** Заполняет начальные популяции через testid (ждёт монтирования DOM). */
export async function fillPopulations(page: Page, values: PopulationOverrides): Promise<void> {
  const keys: Array<keyof PopulationOverrides> = [
    'P0', 'Ne0', 'M10', 'M20', 'F0', 'Mf0', 'E0', 'S0',
  ];
  // Дождаться появления первого input в секции Populations.
  await page.getByTestId('param-P0').waitFor({ state: 'attached', timeout: 10_000 });
  for (const key of keys) {
    const value = values[key];
    if (value === undefined) continue;
    await page.getByTestId(`param-${key}`).first().fill(String(value));
  }
}

/** Полный сценарий: upload → model → run → wait complete. Возвращает simulation_id. */
export async function runSimulationViaUI(
  page: Page,
  mode: UIMode,
  opts: { tMaxHours?: number; timeout?: number; populations?: PopulationOverrides } = {},
): Promise<string> {
  const { tMaxHours = 24, timeout = 180_000, populations } = opts;

  await openDashboard(page);

  // Step 0: upload
  await uploadSampleFcs(page);

  // Step 1: model
  await goToStep(page, 1);
  await selectMode(page, mode);

  // Step 2: params
  await goToStep(page, 2);
  await fillTMax(page, tMaxHours);
  if (populations) {
    await fillPopulations(page, populations);
  }

  // Step 3: run
  await goToStep(page, 3);
  await clickRun(page);

  await waitForSimulationComplete(page, timeout);
  return openResultsFromRunner(page);
}
