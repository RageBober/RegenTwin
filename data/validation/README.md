# Validation Data

Директория для валидационных данных моделей RegenTwin.

## Структура

```
data/validation/
    README.md           # Этот файл
    fcs/               # Flow cytometry файлы (.fcs)
    time_series/       # Временные ряды (.csv, .json)
```

## Формат FCS данных

- Стандарт FCS 2.0/3.0/3.1
- Обязательные каналы: FSC-A, FSC-H, SSC-A, CD34, Annexin-V
- Опциональные каналы: CD14, CD68, CD66b, CD31
- Минимум 100 событий

## Формат временных рядов

CSV с колонками:
- `time` (часы) — обязательная, монотонно возрастающая
- `cell_count` — общее количество клеток
- `wound_area` — нормализованная площадь раны (0=зажила, 1=начальная)
- Цитокины: TNF_alpha, IL_10, PDGF, VEGF, TGF_beta, MCP_1, IL_8

## Источники данных

| Источник | ID | Тип данных |
|----------|-----|-----------|
| FlowRepository | FR-FCM-* | Flow cytometry (рана) |
| GEO (NCBI) | GSE* | Транскриптомика (временной ряд) |
| Wound Healing Society | — | Клинические данные заживления |
| Human Protein Atlas | — | Экспрессия белков в коже |
