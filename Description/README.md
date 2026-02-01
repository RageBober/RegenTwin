# Папка Description

Здесь хранятся детальные описания функционала для каждого модуля проекта.

## Методология

Проект следует методологии **Stub-First**:

### 1. Stub (Заглушка)

Создаём скелет файла с:
- Сигнатурами функций и классов
- `raise NotImplementedError("Stub: требуется реализация")`
- Типизацией параметров и возвращаемых значений

```python
def process_data(input_file: str) -> pd.DataFrame:
    """
    Обработка входных данных.

    Подробное описание: Description/description_module.md#process_data
    """
    raise NotImplementedError("Stub: требуется реализация")
```

### 2. Description (Описание)

Создаём файл `description_<module_name>.md` с:
- Обзором модуля
- Описанием каждого класса и метода
- Параметрами и возвращаемыми значениями
- Алгоритмами и бизнес-логикой
- Примерами использования

### 3. Docstrings (Документация в коде)

Добавляем краткие docstrings со ссылками:

```python
"""
Краткое описание функции.

Подробное описание: Description/description_module.md#function_name
"""
```

---

## Структура файлов описания

```
Description/
├── README.md                          # Этот файл
├── description_fcs_parser.md          # src/data/fcs_parser.py
├── description_gating.md              # src/data/gating.py
├── description_parameter_extraction.md # src/data/parameter_extraction.py
├── description_sde_model.md           # src/core/sde_model.py
├── description_abm_model.md           # src/core/abm_model.py
├── description_integration.md         # src/core/integration.py
├── description_monte_carlo.md         # src/core/monte_carlo.py
├── description_plots.md               # src/visualization/plots.py
├── description_spatial.md             # src/visualization/spatial.py
├── description_export.md              # src/visualization/export.py
├── description_api_main.md            # src/api/main.py
├── description_routes_upload.md       # src/api/routes/upload.py
├── description_routes_simulate.md     # src/api/routes/simulate.py
├── description_api_models.md          # src/api/models/
└── description_generate_mock_data.md  # data/mock/generate_mock_data.py
```

---

## Шаблон файла описания

```markdown
# Описание: <filename>.py

## Обзор

Краткое описание назначения модуля.

---

## Классы

### ClassName

**Назначение:** Что делает класс.

**Атрибуты:**

| Атрибут | Тип | Описание |
|---------|-----|----------|
| attr1 | str | Описание |

---

## Методы

### method_name

**Назначение:** Что делает метод.

**Сигнатура:**

\`\`\`python
def method_name(param1: type1, param2: type2) -> return_type
\`\`\`

**Параметры:**

| Параметр | Тип | Описание | По умолчанию |
|----------|-----|----------|--------------|
| param1 | type1 | Описание | - |

**Возвращает:** Описание возвращаемого значения.

**Алгоритм:**

1. Шаг 1
2. Шаг 2

---

## Примеры использования

\`\`\`python
# Пример кода
\`\`\`
```

---

## Преимущества методологии

1. **Планирование перед кодированием** — продумываем архитектуру заранее
2. **Документация как часть процесса** — не откладываем на потом
3. **Чёткие контракты** — интерфейсы определены до реализации
4. **Параллельная разработка** — разные разработчики могут работать над разными частями
5. **Простое ревью** — описания помогают понять намерения автора
