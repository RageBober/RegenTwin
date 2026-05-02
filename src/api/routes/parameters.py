"""API endpoint для получения границ параметров модели."""

from __future__ import annotations

from fastapi import APIRouter, Query

from src.api.models.schemas import ParameterBoundItem, ParameterBoundsResponse
from src.core.bounds import PARAM_GROUPS
from src.core.parameters import ParameterSet

router = APIRouter(prefix="/api/v1", tags=["parameters"])


@router.get("/parameters/bounds", response_model=ParameterBoundsResponse)
def get_parameter_bounds(
    names: list[str] | None = Query(default=None),
) -> ParameterBoundsResponse:
    """Получить границы параметров для анализа чувствительности.

    Без параметра `names` возвращает все ~100 параметров (кроме численных).
    С `?names=r_F&names=K_F` — только запрошенные.
    """
    bounds = ParameterSet.get_bounds(names)
    items = [
        ParameterBoundItem(
            name=b.name,
            lower=b.lower,
            upper=b.upper,
            nominal=b.nominal if b.nominal is not None else 0.0,
            group=PARAM_GROUPS.get(b.name, "other"),
        )
        for b in bounds
    ]
    return ParameterBoundsResponse(bounds=items, total=len(items))
