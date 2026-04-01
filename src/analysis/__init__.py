"""Модуль анализа и валидации моделей RegenTwin.

Предоставляет метрики качества предсказаний для 20-переменной SDE-модели:
- DTW + CRPS: временные ряды с фазовыми сдвигами
- ArviZ PPC: байесовская проверка предсказаний
- Changepoint detection (ruptures): автоматическое обнаружение фаз
- Kendall's τ: ранговая корреляция рейтингов чувствительности

Подробное описание: Description/Phase3/description_validation.md
"""

from __future__ import annotations

from src.analysis.validation import (
    DTWCRPSResult,
    PhaseBreakpoint,
    PhaseTimingResult,
    PPCResult,
    RankingComparison,
    SensitivityRankingResult,
    ValidationConfig,
    ValidationResult,
    ValidationRunner,
    validate_model,
)

__all__ = [
    "DTWCRPSResult",
    "PhaseBreakpoint",
    "PhaseTimingResult",
    "PPCResult",
    "RankingComparison",
    "SensitivityRankingResult",
    "ValidationConfig",
    "ValidationResult",
    "ValidationRunner",
    "validate_model",
]
