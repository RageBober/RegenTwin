# Шаблон разработки Stubs + TDD Descriptions для RegenTwin

## Контекст проекта

### Структура
```
src/
  data/           # Phase 1: Data Pipeline (6 модулей)
  core/           # Phase 2: Mathematical Core (4 модуля)
  api/            # Phase 3: API (будущее)
Description/
  Phase1/         # Описания для data pipeline
  Phase2/         # Описания для math core
tests/
  unit/
    data/         # Тесты Phase 1
    core/         # Тесты Phase 2
Doks/             # Документация (Mathematical Framework)
developers_plans/ # Планы разработки
```

### 3-этапный workflow (из instructions.md)
1. **Этап 1** — Stubs: скелеты классов и методов с `raise NotImplementedError`
2. **Этап 2** — TDD Tests: тесты на основе Description файлов
3. **Этап 3** — Implementation: реализация функционала

### Математическая модель (20 переменных)
- **8 клеточных популяций:** P (тромбоциты), Ne (нейтрофилы), M1/M2 (макрофаги), F (фибробласты), Mf (миофибробласты), E (эндотелиальные), S (стволовые)
- **7 цитокинов:** TNF-α, IL-10, PDGF, VEGF, TGF-β, MCP-1, IL-8
- **3 ECM:** rho_collagen, C_MMP, rho_fibrin
- **2 вспомогательные:** D (damage signal), O2 (кислород)

### Конвенции кода
- **Docstrings:** на русском языке
- **Имена переменных:** на английском (snake_case)
- **Line length:** 100 символов
- **Tools:** ruff + black + mypy (Python 3.11+)
- **Stubs:** `raise NotImplementedError("Stub: требуется реализация в Этап 3")`

---

## Шаблон: Stub-метод

```python
def method_name(
    self,
    arg1: type1,
    arg2: type2 = default_value,
) -> ReturnType:
    """Краткое описание что делает метод (1-3 предложения).

    Детали поведения: алгоритм, логика, что возвращает при корректных данных.
    Контекст: зачем этот метод нужен в модели регенерации.

    Args:
        arg1: Описание аргумента
        arg2: Описание аргумента (по умолчанию default_value)

    Returns:
        Описание возвращаемого значения

    Raises:
        ValueError: Когда аргумент некорректен
        KeyError: Когда ключ не найден

    Подробное описание: Description/PhaseN/description_module.md#method_name
    """
    raise NotImplementedError("Stub: требуется реализация в Этап 3")
```

---

## Шаблон: Stub-класс

```python
class ClassName(BaseClass):
    """Краткое назначение класса (1-2 предложения).

    Свойства:
    - Свойство 1 (что делает)
    - Свойство 2 (что делает)
    - Биологический контекст (если применимо)

    Подробное описание: Description/PhaseN/description_module.md#ClassName
    """

    # Константы класса
    CONSTANT_1: type = value  # Комментарий
    CONSTANT_2: type = value  # Комментарий

    def __init__(self, param1: type1, param2: type2) -> None:
        """Инициализация.

        Args:
            param1: Описание
            param2: Описание

        Подробное описание: Description/PhaseN/description_module.md#ClassName.__init__
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")
```

---

## Шаблон: Dataclass

```python
@dataclass
class DataClassName:
    """Описание структуры данных (1-2 предложения).

    Подробное описание: Description/PhaseN/description_module.md#DataClassName
    """

    field1: type  # Описание
    field2: type = default  # Описание
    field3: list[type] = field(default_factory=list)  # Описание

    @property
    def computed_property(self) -> type:
        """Вычисляемое свойство."""
        raise NotImplementedError("Stub: требуется реализация в Этап 3")
```

---

## Шаблон: TDD-секция в Description файле

```markdown
### MethodName

**Назначение:** Что делает метод и зачем.

**Сигнатура:**

\```python
def method_name(self, arg1: type1, arg2: type2 = default) -> ReturnType
\```

**Поведение:**
1. Шаг 1 алгоритма
2. Шаг 2 алгоритма
3. Возвращает результат

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Нормальный ввод (описание) | Конкретный результат |
| Граничный случай 1 | Конкретный результат |
| Граничный случай 2 | Конкретный результат |
| Ошибочный ввод | ValueError / KeyError |

**Edge cases:**
- Пустой ввод → описание поведения
- Нулевые значения → описание поведения
- Максимальные значения → описание поведения
- NaN/Inf → описание поведения

**Ошибки:**
- `ValueError`: когда condition
- `KeyError`: когда condition

**Инварианты:**
- Условие, которое ВСЕГДА выполняется (напр. `0 <= result <= 1`)
- Тип возвращаемого значения всегда один и тот же
- Связь между входом и выходом (напр. `len(output) == len(input)`)
```

---

## Шаблон: Тестовая таблица для класса

```markdown
### ClassName

**Константы класса:**

| Константа | Значение | Описание |
|-----------|----------|----------|
| CONST_1 | value | Описание |
| CONST_2 | value | Описание |

**Тестовые сценарии:**

| Сценарий | Ожидание |
|----------|----------|
| Создание с defaults | field1 == default1, field2 == default2 |
| Создание с кастомными значениями | Корректный объект |
| validate() при корректных данных | True |
| validate() при некорректных данных | ValueError |
| to_dict() | dict с >= N ключами |

**Инварианты:**
- CLASS_CONSTANT == expected_value
- Все числовые поля >= 0
```

---

## Чеклист для каждого нового элемента

- [ ] Stub в .py файле с информативным docstring
- [ ] Ссылка на Description файл в docstring
- [ ] TDD-секция в Description файле (сигнатура, сценарии, edge cases, инварианты)
- [ ] Экспорт в `__init__.py`
- [ ] Импорт проверен: `python -c "from src.module import ClassName"`
- [ ] Существующие тесты проходят: `pytest tests/unit/ -v`
- [ ] Линтер чист: `ruff check src/`
