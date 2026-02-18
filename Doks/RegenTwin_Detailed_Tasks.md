# RegenTwin: Детализированная декомпозиция задач (Tauri + React)

---

## Методология: TDD (Test-Driven Development)

Для каждой задачи следуем циклу **Red → Green → Refactor**:

1. **RED** — Пишем тест, который падает (функционал ещё не реализован)
2. **GREEN** — Пишем минимальный код, чтобы тест прошёл
3. **REFACTOR** — Улучшаем код, сохраняя тесты зелёными

### Структура тестов

```
tests/
├── api/
│   ├── test_main.py
│   ├── test_upload.py
│   ├── test_simulate.py
│   └── test_results.py
├── integration/
│   └── test_e2e.py
└── conftest.py          # Fixtures
```

```
ui/src/
├── __tests__/           # React тесты
│   ├── components/
│   └── pages/
└── setupTests.ts
```

---

## Фаза 3: FastAPI Backend

**Цель:** Создать REST API для взаимодействия React-фронтенда с математическим ядром

---

### Задача 3.1: Базовый FastAPI сервер

**Зависимости:** Фаза 0 (инфраструктура)

#### TDD: Тесты (RED)

```python
# tests/api/test_main.py

def test_health_endpoint_returns_ok(client):
    """Health check должен возвращать status: ok"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_cors_headers_present(client):
    """CORS headers должны быть в ответе"""
    response = client.options("/api/v1/health")
    assert "access-control-allow-origin" in response.headers

def test_docs_available(client):
    """Swagger UI должен быть доступен"""
    response = client.get("/docs")
    assert response.status_code == 200
```

#### Подзадачи (GREEN)

1. **Создание главного файла приложения**
   - Файл: `src/api/main.py`
   - FastAPI app инициализация
   - CORS middleware (для React)
   - Подключение роутеров
   - Description: `Description/description_api_main.md`

2. **Конфигурация CORS**
   - Разрешённые origins: `http://localhost:5173`, `tauri://localhost`
   - Методы: GET, POST, OPTIONS
   - Headers: Content-Type, Authorization

3. **Health check endpoint**
   - `GET /api/v1/health` → `{"status": "ok"}`

#### Критерии готовности
- [ ] Все тесты проходят (pytest tests/api/test_main.py)
- [ ] `uvicorn src.api.main:app --reload` запускается
- [ ] http://localhost:8000/docs показывает Swagger UI

---

### Задача 3.2: Pydantic модели (schemas)

**Зависимости:** 3.1

#### TDD: Тесты (RED)

```python
# tests/api/test_schemas.py

def test_prp_config_defaults():
    """PRPConfig должен иметь значения по умолчанию"""
    config = PRPConfig()
    assert config.enabled == True
    assert config.initial_concentration == 1.0
    assert config.decay_rate == 0.1

def test_prp_config_validation():
    """PRPConfig должен валидировать диапазоны"""
    with pytest.raises(ValidationError):
        PRPConfig(initial_concentration=-1.0)  # Отрицательное значение

def test_simulation_request_requires_upload_id():
    """SimulationRequest требует upload_id"""
    with pytest.raises(ValidationError):
        SimulationRequest()  # Без upload_id

def test_dynamics_data_serialization():
    """DynamicsData должен корректно сериализоваться"""
    data = DynamicsData(
        times=[0.0, 1.0],
        N_mean=[100.0, 150.0],
        N_lower=[90.0, 140.0],
        N_upper=[110.0, 160.0],
        C_mean=[1.0, 0.9]
    )
    json_data = data.model_dump_json()
    assert "times" in json_data
```

#### Подзадачи (GREEN)

1. **Создание файла моделей**
   - Файл: `src/api/models/schemas.py`
   - Description: `Description/description_api_schemas.md`

2. **Модели с валидацией:**

```python
class PRPConfig(BaseModel):
    enabled: bool = True
    initial_concentration: float = Field(default=1.0, ge=0.1, le=10.0)
    decay_rate: float = Field(default=0.1, ge=0.01, le=1.0)
    injection_times: list[float] = [0.0]

class PEMFConfig(BaseModel):
    enabled: bool = True
    frequency: float = Field(default=15.0, ge=1.0, le=100.0)
    intensity: float = Field(default=1.0, ge=0.1, le=10.0)
    session_duration: float = Field(default=30.0, ge=5.0, le=120.0)
    sessions_per_day: int = Field(default=2, ge=1, le=6)
```

#### Критерии готовности
- [ ] Все тесты проходят (pytest tests/api/test_schemas.py)
- [ ] Валидация работает (Pydantic v2)

---

### Задача 3.3: Upload endpoint

**Зависимости:** 3.2

#### TDD: Тесты (RED)

```python
# tests/api/test_upload.py

def test_upload_fcs_file(client, sample_fcs_file):
    """Загрузка FCS файла должна вернуть upload_id"""
    response = client.post(
        "/api/v1/upload",
        files={"file": sample_fcs_file}
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["status"] == "completed"
    assert data["channels"] is not None

def test_upload_invalid_format(client, invalid_file):
    """Загрузка неверного формата должна вернуть ошибку"""
    response = client.post(
        "/api/v1/upload",
        files={"file": invalid_file}
    )
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

def test_upload_too_large(client, large_file):
    """Файл > 100MB должен быть отклонён"""
    response = client.post(
        "/api/v1/upload",
        files={"file": large_file}
    )
    assert response.status_code == 413

def test_get_upload_status(client, uploaded_file_id):
    """Получение статуса загрузки по ID"""
    response = client.get(f"/api/v1/upload/{uploaded_file_id}")
    assert response.status_code == 200
    assert response.json()["id"] == uploaded_file_id

def test_get_upload_not_found(client):
    """Несуществующий upload должен вернуть 404"""
    response = client.get("/api/v1/upload/nonexistent-id")
    assert response.status_code == 404
```

#### Fixtures (conftest.py)

```python
# tests/conftest.py

@pytest.fixture
def sample_fcs_file():
    """Мок FCS файл для тестов"""
    # Создаём минимальный валидный FCS
    return ("test.fcs", b"FCS3.0...", "application/octet-stream")

@pytest.fixture
def invalid_file():
    return ("test.txt", b"not a fcs file", "text/plain")
```

#### Подзадачи (GREEN)

1. **Создание роутера загрузки**
   - Файл: `src/api/routes/upload.py`
   - Description: `Description/description_routes_upload.md`

2. **Endpoints:**
   - `POST /api/v1/upload`
   - `GET /api/v1/upload/{upload_id}`

3. **Сервис обработки файлов**
   - Файл: `src/api/services/file_service.py`

#### Критерии готовности
- [ ] Все тесты проходят (pytest tests/api/test_upload.py)
- [ ] Файлы загружаются и валидируются

---

### Задача 3.4: Simulate endpoint

**Зависимости:** 3.3

#### TDD: Тесты (RED)

```python
# tests/api/test_simulate.py

def test_start_simulation(client, uploaded_file_id):
    """Запуск симуляции должен вернуть simulation_id"""
    response = client.post("/api/v1/simulate", json={
        "upload_id": uploaded_file_id,
        "prp_config": {"enabled": True},
        "pemf_config": {"enabled": True},
        "simulation_params": {"duration_days": 7, "n_trajectories": 10}
    })
    assert response.status_code == 200
    assert "simulation_id" in response.json()

def test_simulation_with_invalid_upload_id(client):
    """Симуляция с несуществующим upload_id должна вернуть 404"""
    response = client.post("/api/v1/simulate", json={
        "upload_id": "nonexistent",
        "prp_config": {},
        "pemf_config": {},
        "simulation_params": {}
    })
    assert response.status_code == 404

def test_get_simulation_status(client, running_simulation_id):
    """Получение статуса симуляции"""
    response = client.get(f"/api/v1/simulate/{running_simulation_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["pending", "running", "completed", "error"]
    assert 0.0 <= data["progress"] <= 1.0

@pytest.mark.asyncio
async def test_websocket_progress(client, running_simulation_id):
    """WebSocket должен отправлять прогресс"""
    async with client.websocket_connect(
        f"/api/v1/simulate/{running_simulation_id}/ws"
    ) as ws:
        data = await ws.receive_json()
        assert "progress" in data
        assert "stage" in data
```

#### Подзадачи (GREEN)

1. **Создание роутера симуляции**
   - Файл: `src/api/routes/simulate.py`

2. **Endpoints:**
   - `POST /api/v1/simulate`
   - `GET /api/v1/simulate/{simulation_id}`
   - `WS /api/v1/simulate/{simulation_id}/ws`

3. **Сервис симуляции**
   - Файл: `src/api/services/simulation_service.py`

#### Критерии готовности
- [ ] Все тесты проходят (pytest tests/api/test_simulate.py)
- [ ] WebSocket работает

---

### Задача 3.5: Results endpoint

**Зависимости:** 3.4

#### TDD: Тесты (RED)

```python
# tests/api/test_results.py

def test_get_results(client, completed_simulation_id):
    """Получение результатов завершённой симуляции"""
    response = client.get(f"/api/v1/results/{completed_simulation_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["dynamics"] is not None
    assert len(data["dynamics"]["times"]) > 0

def test_get_results_not_ready(client, running_simulation_id):
    """Результаты незавершённой симуляции"""
    response = client.get(f"/api/v1/results/{running_simulation_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["dynamics"] is None

def test_export_csv(client, completed_simulation_id):
    """Экспорт в CSV"""
    response = client.post(
        f"/api/v1/export/{completed_simulation_id}",
        json={"format": "csv"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv"

def test_export_pdf(client, completed_simulation_id):
    """Экспорт в PDF"""
    response = client.post(
        f"/api/v1/export/{completed_simulation_id}",
        json={"format": "pdf"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
```

#### Подзадачи (GREEN)

1. **Создание роутера результатов**
   - Файл: `src/api/routes/results.py`

2. **Endpoints:**
   - `GET /api/v1/results/{simulation_id}`
   - `POST /api/v1/export/{simulation_id}`

#### Критерии готовности
- [ ] Все тесты проходят (pytest tests/api/test_results.py)
- [ ] Экспорт работает

---

## Фаза 4: Tauri + React Frontend

**Цель:** Создать кросс-платформенное приложение (веб + десктоп)

### Тестирование React (Vitest + React Testing Library)

```bash
npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom
```

---

### Задача 4.1: Инициализация Tauri проекта

**Зависимости:** Node.js 18+, Rust

#### Подзадачи

1. **Создание проекта**
   ```bash
   cd c:\Users\Compic\Documents\RegenTwin
   npm create tauri-app@latest ui -- --template react-ts
   cd ui
   ```

2. **Установка зависимостей**
   ```bash
   npm install react-router-dom axios zustand @tanstack/react-query
   npm install plotly.js react-plotly.js d3 three @react-three/fiber
   npm install tailwindcss postcss autoprefixer @headlessui/react
   npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom
   npm install -D @types/d3 @types/three
   ```

3. **Настройка Vitest**
   ```typescript
   // vite.config.ts
   export default defineConfig({
     test: {
       globals: true,
       environment: 'jsdom',
       setupFiles: './src/setupTests.ts',
     },
   })
   ```

#### Критерии готовности
- [ ] `npm run tauri dev` открывает окно
- [ ] `npm run test` запускает тесты

---

### Задача 4.3: API клиент и State Management

**Зависимости:** 4.2, 3.1

#### TDD: Тесты (RED)

```typescript
// ui/src/__tests__/services/api.test.ts

import { api } from '../../services/api';
import { vi } from 'vitest';

describe('API client', () => {
  it('should upload file and return upload_id', async () => {
    const mockFile = new File([''], 'test.fcs');
    const response = await api.upload(mockFile);
    expect(response.data.id).toBeDefined();
  });

  it('should handle upload error', async () => {
    const invalidFile = new File([''], 'test.txt');
    await expect(api.upload(invalidFile)).rejects.toThrow();
  });
});
```

```typescript
// ui/src/__tests__/stores/simulationStore.test.ts

import { useSimulationStore } from '../../stores/simulationStore';

describe('Simulation Store', () => {
  it('should have initial state', () => {
    const state = useSimulationStore.getState();
    expect(state.uploadId).toBeNull();
    expect(state.simulationStatus).toBe('idle');
  });

  it('should update upload state', () => {
    const { setUploadId } = useSimulationStore.getState();
    setUploadId('test-id');
    expect(useSimulationStore.getState().uploadId).toBe('test-id');
  });

  it('should reset state', () => {
    const { setUploadId, reset } = useSimulationStore.getState();
    setUploadId('test-id');
    reset();
    expect(useSimulationStore.getState().uploadId).toBeNull();
  });
});
```

#### Подзадачи (GREEN)

1. **API клиент:** `ui/src/services/api.ts`
2. **Zustand store:** `ui/src/stores/simulationStore.ts`

#### Критерии готовности
- [ ] Тесты проходят (npm run test)

---

### Задача 4.4: Компонент Upload

**Зависимости:** 4.3

#### TDD: Тесты (RED)

```typescript
// ui/src/__tests__/components/Upload/UploadFCS.test.tsx

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { UploadFCS } from '../../../components/Upload/UploadFCS';

describe('UploadFCS', () => {
  it('should render drop zone', () => {
    render(<UploadFCS />);
    expect(screen.getByText(/drag.*drop/i)).toBeInTheDocument();
  });

  it('should accept .fcs files', async () => {
    render(<UploadFCS />);
    const file = new File([''], 'test.fcs', { type: 'application/octet-stream' });
    const input = screen.getByTestId('file-input');

    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => {
      expect(screen.getByText('test.fcs')).toBeInTheDocument();
    });
  });

  it('should reject non-.fcs files', async () => {
    render(<UploadFCS />);
    const file = new File([''], 'test.txt', { type: 'text/plain' });
    const input = screen.getByTestId('file-input');

    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => {
      expect(screen.getByText(/invalid.*format/i)).toBeInTheDocument();
    });
  });

  it('should show upload progress', async () => {
    render(<UploadFCS />);
    const file = new File([''], 'test.fcs');
    const input = screen.getByTestId('file-input');

    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => {
      expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });
  });
});
```

#### Подзадачи (GREEN)

1. **UploadFCS компонент:** `ui/src/components/Upload/UploadFCS.tsx`
2. **ScatterPreview компонент:** `ui/src/components/Upload/ScatterPreview.tsx`

#### Критерии готовности
- [ ] Тесты проходят
- [ ] Drag & drop работает

---

### Задача 4.5: Компонент Parameters

**Зависимости:** 4.4

#### TDD: Тесты (RED)

```typescript
// ui/src/__tests__/components/Parameters/TherapyConfig.test.tsx

import { render, screen, fireEvent } from '@testing-library/react';
import { TherapyConfig } from '../../../components/Parameters/TherapyConfig';

describe('TherapyConfig', () => {
  it('should render PRP and PEMF sections', () => {
    render(<TherapyConfig />);
    expect(screen.getByText('PRP Therapy')).toBeInTheDocument();
    expect(screen.getByText('PEMF Therapy')).toBeInTheDocument();
  });

  it('should toggle PRP enabled', () => {
    render(<TherapyConfig />);
    const toggle = screen.getByTestId('prp-toggle');

    fireEvent.click(toggle);

    expect(toggle).not.toBeChecked();
  });

  it('should update concentration slider', () => {
    const onChange = vi.fn();
    render(<TherapyConfig onChange={onChange} />);

    const slider = screen.getByTestId('prp-concentration-slider');
    fireEvent.change(slider, { target: { value: 2.5 } });

    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({
        prp: expect.objectContaining({ initial_concentration: 2.5 })
      })
    );
  });
});
```

#### Подзадачи (GREEN)

1. **TherapyConfig:** `ui/src/components/Parameters/TherapyConfig.tsx`
2. **PRPSettings:** `ui/src/components/Parameters/PRPSettings.tsx`
3. **PEMFSettings:** `ui/src/components/Parameters/PEMFSettings.tsx`

#### Критерии готовности
- [ ] Тесты проходят
- [ ] Все параметры настраиваются

---

### Задача 4.6: Компонент Simulation

**Зависимости:** 4.5, 3.4

#### TDD: Тесты (RED)

```typescript
// ui/src/__tests__/components/Simulation/SimulationRunner.test.tsx

describe('SimulationRunner', () => {
  it('should show start button when idle', () => {
    render(<SimulationRunner />);
    expect(screen.getByText('Start Simulation')).toBeInTheDocument();
  });

  it('should show progress when running', () => {
    // Mock store with running status
    useSimulationStore.setState({ simulationStatus: 'running', progress: 0.5 });

    render(<SimulationRunner />);

    expect(screen.getByRole('progressbar')).toHaveAttribute('value', '50');
    expect(screen.getByText(/running/i)).toBeInTheDocument();
  });

  it('should redirect to results when completed', async () => {
    const navigate = vi.fn();
    vi.mock('react-router-dom', () => ({ useNavigate: () => navigate }));

    useSimulationStore.setState({ simulationStatus: 'completed', simulationId: 'test-id' });

    render(<SimulationRunner />);

    await waitFor(() => {
      expect(navigate).toHaveBeenCalledWith('/results/test-id');
    });
  });
});
```

#### Подзадачи (GREEN)

1. **SimulationRunner:** `ui/src/components/Simulation/SimulationRunner.tsx`
2. **useSimulationProgress hook:** `ui/src/hooks/useSimulationProgress.ts`

#### Критерии готовности
- [ ] Тесты проходят
- [ ] WebSocket прогресс работает

---

### Задача 4.7: Компоненты Visualization

**Зависимости:** 4.6, 3.5

#### TDD: Тесты (RED)

```typescript
// ui/src/__tests__/components/Visualization/GrowthChart.test.tsx

describe('GrowthChart', () => {
  const mockData = {
    times: [0, 1, 2, 3],
    N_mean: [100, 120, 150, 180],
    N_lower: [90, 110, 140, 170],
    N_upper: [110, 130, 160, 190]
  };

  it('should render plotly chart', () => {
    render(<GrowthChart data={mockData} />);
    expect(screen.getByTestId('plotly-chart')).toBeInTheDocument();
  });

  it('should show confidence interval', () => {
    render(<GrowthChart data={mockData} showConfidence={true} />);
    // Проверяем что есть fill between
    const chart = screen.getByTestId('plotly-chart');
    expect(chart).toHaveAttribute('data-traces', '3'); // mean + upper + lower
  });
});
```

#### Подзадачи (GREEN)

1. **GrowthChart:** `ui/src/components/Visualization/GrowthChart.tsx`
2. **CellHeatmap:** `ui/src/components/Visualization/CellHeatmap.tsx`
3. **SpatialView3D:** `ui/src/components/Visualization/SpatialView3D.tsx`

#### Критерии готовности
- [ ] Тесты проходят
- [ ] Графики интерактивны

---

### Задача 4.8: Компонент Results и Export

**Зависимости:** 4.7

#### TDD: Тесты (RED)

```typescript
// ui/src/__tests__/components/Results/ExportPanel.test.tsx

describe('ExportPanel', () => {
  it('should render export buttons', () => {
    render(<ExportPanel simulationId="test-id" />);
    expect(screen.getByText('Export PDF')).toBeInTheDocument();
    expect(screen.getByText('Export CSV')).toBeInTheDocument();
  });

  it('should call export API on click', async () => {
    const exportMock = vi.spyOn(api, 'exportResults');
    render(<ExportPanel simulationId="test-id" />);

    fireEvent.click(screen.getByText('Export PDF'));

    await waitFor(() => {
      expect(exportMock).toHaveBeenCalledWith('test-id', 'pdf');
    });
  });

  it('should show loading state during export', async () => {
    render(<ExportPanel simulationId="test-id" />);

    fireEvent.click(screen.getByText('Export PDF'));

    expect(screen.getByText(/exporting/i)).toBeInTheDocument();
  });
});
```

#### Подзадачи (GREEN)

1. **ResultsSummary:** `ui/src/components/Results/ResultsSummary.tsx`
2. **ExportPanel:** `ui/src/components/Results/ExportPanel.tsx`
3. **HistoryList:** `ui/src/components/Results/HistoryList.tsx`

#### Критерии готовности
- [ ] Тесты проходят
- [ ] Экспорт работает

---

## Фаза 5: Интеграция и сборка

### Задача 5.3: E2E тестирование

#### TDD: Integration Tests

```python
# tests/integration/test_e2e.py

@pytest.mark.integration
class TestFullWorkflow:
    """E2E тесты полного workflow"""

    def test_upload_simulate_export(self, client, sample_fcs_file):
        """Полный цикл: загрузка → симуляция → экспорт"""
        # 1. Upload
        upload_response = client.post("/api/v1/upload", files={"file": sample_fcs_file})
        assert upload_response.status_code == 200
        upload_id = upload_response.json()["id"]

        # 2. Start simulation
        sim_response = client.post("/api/v1/simulate", json={
            "upload_id": upload_id,
            "prp_config": {"enabled": True},
            "pemf_config": {"enabled": True},
            "simulation_params": {"duration_days": 7, "n_trajectories": 10}
        })
        assert sim_response.status_code == 200
        sim_id = sim_response.json()["simulation_id"]

        # 3. Wait for completion
        for _ in range(30):  # Max 30 секунд
            status = client.get(f"/api/v1/simulate/{sim_id}").json()
            if status["status"] == "completed":
                break
            time.sleep(1)

        assert status["status"] == "completed"

        # 4. Get results
        results = client.get(f"/api/v1/results/{sim_id}").json()
        assert results["dynamics"] is not None
        assert len(results["dynamics"]["times"]) > 0

        # 5. Export
        export = client.post(f"/api/v1/export/{sim_id}", json={"format": "csv"})
        assert export.status_code == 200
```

---

## Итоговая сводка

| Фаза | Задачи | Тестов | Файлов |
|------|--------|--------|--------|
| 3. FastAPI Backend | 5 задач | ~20 тестов | ~10 файлов |
| 4. Tauri + React | 9 задач | ~30 тестов | ~25 файлов |
| 5. Интеграция | 3 задачи | ~5 тестов | ~5 файлов |
| **Итого** | **17 задач** | **~55 тестов** | **~40 файлов** |

---

## Команды для запуска тестов

```bash
# Backend тесты
pytest tests/api/ -v                    # Unit тесты API
pytest tests/integration/ -v -m integration  # E2E тесты
pytest --cov=src --cov-report=html      # С покрытием

# Frontend тесты
cd ui
npm run test                            # Все тесты
npm run test -- --watch                 # Watch mode
npm run test -- --coverage              # С покрытием
```

---

*Документ создан: Февраль 2026*
*Версия: 2.1 (TDD)*
