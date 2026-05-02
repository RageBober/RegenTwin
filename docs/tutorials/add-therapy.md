# Туториал: добавление новой терапии

Текущие поддерживаемые терапии: PRP и PEMF (см. `src/core/therapy_models.py`).

## Шаги для добавления, например, **Negative Pressure Wound Therapy (NPWT)**:

1. Расширить `TherapyProtocol`:
   ```python
   @dataclass
   class TherapyProtocol:
       prp_enabled: bool = False
       prp_intensity: float = 0.0
       pemf_enabled: bool = False
       pemf_intensity: float = 0.0
       # NEW:
       npwt_enabled: bool = False
       npwt_pressure: float = -125.0  # mmHg
   ```

2. Добавить modifier-функцию (по образцу PRP):
   ```python
   def npwt_modifier(state, dt, pressure):
       # Например, NPWT уменьшает edema → ускоряет VEGF-индуцированный ангиогенез
       return {"E_boost": 1.2, "C_VEGF_boost": 1.1}
   ```

3. Подключить в `ExtendedSDEModel.simulate(...)` — добавить вызов modifier
   в цикл интегрирования.

4. Расширить Pydantic-схему API (`src/api/models/schemas.py`) — добавить
   `npwt_*` поля в request schema.

5. Добавить тесты в `tests/unit/core/test_therapy_models.py`.

6. Документировать в [docs/architecture/sde-model.md](../architecture/sde-model.md)
   — секция «Терапии».

!!! note "Биологическая обоснованность"
    Перед мерджем — обновить `Doks/RegenTwin_Mathematical_Framework.md` с
    источниками литературы.
