import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function ensureFixture(): Promise<void> {
  const repoRoot = join(__dirname, '..', '..');
  const fixtureDir = join(__dirname, 'fixtures');
  const fixturePath = join(fixtureDir, 'sample.fcs');
  const scriptPath = join(repoRoot, 'scripts', 'generate_e2e_fixtures.py');

  mkdirSync(fixtureDir, { recursive: true });

  if (!existsSync(scriptPath)) {
    throw new Error(`generate_e2e_fixtures.py not found at ${scriptPath}`);
  }

  const result = spawnSync('python', [scriptPath], { cwd: repoRoot, stdio: 'inherit' });
  if (result.status !== 0) {
    throw new Error(`Failed to generate FCS fixture (exit ${result.status})`);
  }
  if (!existsSync(fixturePath)) {
    throw new Error(`FCS fixture not generated at ${fixturePath}`);
  }
}

export default async function globalSetup(): Promise<void> {
  await ensureFixture();
}
