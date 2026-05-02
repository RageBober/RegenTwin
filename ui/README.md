# RegenTwin UI

Frontend часть RegenTwin построена на `React 19 + TypeScript + Vite + Tauri`.

## Что делает UI

- Загружает `.fcs` и применяет upload-derived initial conditions в store.
- Запускает `mvp`, `extended` и `abm` симуляции.
- Слушает backend progress через polling и WebSocket terminal events.
- Показывает mode-aware results pages.
- Дает live preview/export и cached export completed runs.
- Показывает только рабочую аналитическую поверхность: Sobol sensitivity.

## Что намеренно скрыто

- `integrated` mode не предлагается в UI.
- `Morris` method не предлагается в sensitivity UI.
- parameter estimation вкладка убрана из обычного интерфейса.
- spatial tabs показываются только для `abm` results.

## Скрипты

```bash
npm run dev         # Vite dev server
npm run dev:full    # backend через uv + frontend
npm run build       # tsc + vite build
npm run lint        # ESLint
npm run test        # Vitest
npm run tauri:dev   # Tauri desktop dev
npm run tauri:build # Tauri desktop build
```

## Локальная разработка

```bash
cd ui
npm install
npm run dev
```

Если нужен backend рядом с frontend:

```bash
cd ui
npm run dev:full
```

`dev:full` использует:
- `uv run python scripts/kill_port.py 8000`
- `uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000`
- `vite`

## Desktop / Tauri

Конфигурация Tauri хранится в `ui/src-tauri`.

Важные детали текущего состояния:
- backend launcher сначала пробует `uv run uvicorn ...`
- затем ищет project-local `.venv`
- затем делает fallback на `python`
- cargo target-dir локальный: `ui/src-tauri/target-tauri`

Проверенный smoke-check:
- `cargo check` проходит

## Основные директории

```text
ui/
├── src/
│   ├── components/
│   ├── hooks/
│   ├── routes/
│   ├── stores/
│   ├── types/
│   └── __tests__/
├── src-tauri/
└── package.json
```

## Regression coverage

Добавлены целевые frontend-regression тесты для:
- upload-derived initial conditions
- mode-aware results tabs

Проверенные команды:
- `npm run lint`
- `tsc -b --pretty false`
- `npm run test -- src/__tests__/components/flow-regressions.test.tsx`

## Ограничения

- live preview export по-прежнему работает отдельно от cached export completed runs
- `integrated` surface отключен до появления реальной реализации в backend
- часть более глубоких UI E2E сценариев еще остается отдельным тестовым треком
