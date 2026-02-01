# RegenTwin

Программный инструмент для симуляции регенерации тканей с использованием данных flow cytometry и моделирования терапий PRP/PEMF.

## Обзор

RegenTwin — это веб-приложение для:
- Загрузки и анализа данных flow cytometry (.fcs файлы)
- Моделирования регенерации тканей с использованием SDE + ABM
- Симуляции терапий PRP (Platelet-Rich Plasma) и PEMF (Pulsed Electromagnetic Field)
- Визуализации результатов и генерации отчётов

## Технологический стек

| Компонент | Технология |
|-----------|------------|
| Язык | Python 3.11+ |
| Пакетный менеджер | UV |
| Flow cytometry | FlowKit |
| Численные методы | NumPy, SciPy |
| Визуализация | Matplotlib, Plotly |
| Backend | FastAPI |
| Frontend | Streamlit |
| База данных | SQLite (MVP) → PostgreSQL |

## Установка

### Требования

- Python 3.11+
- UV (пакетный менеджер)

### Шаги установки

```bash
# Клонирование репозитория
git clone https://github.com/your-org/regentwin.git
cd regentwin

# Установка зависимостей через UV
uv sync

# Установка dev-зависимостей
uv sync --extra dev
```

## Запуск

### Backend API

```bash
uvicorn src.api.main:app --reload
```

API будет доступен по адресу: http://localhost:8000

### Frontend

```bash
streamlit run frontend/app.py
```

Интерфейс будет доступен по адресу: http://localhost:8501

## Структура проекта

```
regentwin/
├── src/
│   ├── core/           # Математическое ядро (SDE, ABM)
│   ├── data/           # Парсинг и обработка данных
│   ├── api/            # FastAPI backend
│   ├── visualization/  # Графики и визуализация
│   └── utils/          # Вспомогательные функции
├── frontend/           # Streamlit интерфейс
├── Description/        # Файлы описания функционала
├── notebooks/          # Jupyter для экспериментов
├── data/               # Датасеты
├── docs/               # Документация
└── docker/             # Docker конфигурации
```

## Методология разработки

Проект следует методологии Stub-First:

1. **Stub** — создаём скелет-заглушку с сигнатурами
2. **Description** — документируем функционал в `Description/`
3. **Docstrings** — добавляем краткие описания со ссылками

Подробнее: [Description/README.md](Description/README.md)

## Математическая модель

### SDE (Stochastic Differential Equations)

Уравнение Ланжевена для динамики клеток:

```
dNₜ = [rNₜ(1 - Nₜ/K) + αf(PRPₜ) + βg(PEMFₜ)]dt + σNₜdWₜ
```

### ABM (Agent-Based Model)

Агенты:
- Стволовые клетки (CD34+)
- Макрофаги (CD14+/CD68+)
- Фибробласты

## Документация

- [План разработки](Doks/RegenTwin_Development_Plan.md)
- [Детализация задач](Doks/RegenTwin_Detailed_Tasks.md)
- [Описания функционала](Description/)

## Лицензия

MIT License

## Авторы

RegenTwin Team
