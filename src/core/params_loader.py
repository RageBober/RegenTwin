"""YAML-загрузчик параметров RegenTwin v2.0.

Поднимает `params.yaml` (источник правды для всех численных параметров
математического фреймворка v2.0) в Python-структуру и помогает построить
плоский словарь, совместимый с `ParameterSet.from_dict`.

Формат значений в `params.yaml`:
    s_TNF_M1: {value: 5.0e-6, units: "ng/(cells·h)", source: "Beutler_1985"}

Загрузчик распознаёт такие словари и заменяет их на значение `value`.
Скаляры и булевы значения (например `noise.sigma_P: 0.1`, `numerics.multirate_subcycling: true`)
передаются как есть.

См. описание Phase 0 в `C:\\Users\\dzume\\.claude\\plans\\keen-cuddling-tome.md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS: tuple[str, ...] = (
    "cells",
    "cytokines",
    "ecm",
    "prp",
    "pemf",
    "damage",
    "oxygen",
    "noise",
    "initial_conditions",
    "numerics",
)


def _is_value_node(node: Any) -> bool:
    return isinstance(node, dict) and "value" in node and "units" in node


def _extract_value(node: Any) -> Any:
    """Достать `value` из словаря-метаданных, либо вернуть узел как есть."""
    if _is_value_node(node):
        return node["value"]
    return node


def _walk_extract(node: Any) -> Any:
    """Рекурсивно прогуляться по структуре и заменить value-узлы их значениями."""
    if _is_value_node(node):
        return node["value"]
    if isinstance(node, dict):
        return {k: _walk_extract(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_walk_extract(x) for x in node]
    return node


def load_params_yaml(path: Path | str = "params.yaml") -> dict[str, Any]:
    """Прочитать params.yaml и вернуть вложенную структуру с распакованными значениями.

    Args:
        path: путь к YAML (по умолчанию `params.yaml` в текущей директории).

    Returns:
        Словарь верхнего уровня, в котором каждое значение либо вложенный dict
        (для секций cells/cytokines/...), либо итоговое число/bool/строка.

    Raises:
        FileNotFoundError: если файла нет.
        ValueError: если корень не словарь или отсутствует обязательная секция.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"params.yaml не найден по пути {p.resolve()}")

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"params.yaml должен быть YAML-словарём на верхнем уровне, получено {type(raw).__name__}"
        )

    missing = [s for s in REQUIRED_SECTIONS if s not in raw]
    if missing:
        raise ValueError(f"params.yaml: отсутствуют обязательные секции: {missing}")

    return _walk_extract(raw)


# Карта переименований YAML-имя → ParameterSet-имя.
# Используется для совместимости с историческими именами полей. Каждый ряд
# имеет комментарий с тегом фикса, который окончательно унифицирует именование.
_FIELD_RENAMES: dict[str, str] = {
    # damage
    "D_0": "D0",
    # oxygen — L_0 в YAML, L_diffusion в коде; FIX-11 заменит на L(E)
    "L_0": "L_diffusion",
    # ecm — k_MMP в YAML, k_MMP_deg в коде
    "k_MMP": "k_MMP_deg",
    # cells — k_deg_P в YAML, k_deg в коде (FIX-02)
    "k_deg_P": "k_deg",
    # cells — n_IL8 в YAML, n_hill в коде (FIX-25 поменяет значение)
    "n_IL8": "n_hill",
    # cells — K_PDGF_prolif в YAML, K_PDGF в коде
    "K_PDGF_prolif": "K_PDGF",
    # cells — K_switch/K_reverse в YAML vs K_switch_half/K_reverse_half в коде
    "K_switch": "K_switch_half",
    "K_reverse": "K_reverse_half",
    # pemf — B_0 в YAML; в коде пока нет, оставим имя как есть
    "B_0": "B_0",
}


def flatten_for_parameter_set(data: dict[str, Any]) -> dict[str, Any]:
    """Расплющить вложенную структуру YAML в плоский dict для `ParameterSet.from_dict`.

    Применяет переименования из `_FIELD_RENAMES` для исторической совместимости.
    Сложные узлы (например `prp.factors`, `oxygen.weights`, `numerics.benchmark_GBM`)
    разворачиваются: вложенные `weights.w_Ne` → ключ `w_Ne`. Сложные структуры
    типа `prp.factors` (карта фактор→{phi, tau_burst, tau_sustained}) НЕ включаются —
    их потребляет PRPModel напрямую, не через ParameterSet.

    Args:
        data: результат `load_params_yaml`.

    Returns:
        Плоский словарь {имя_поля: значение}. Любые поля, не существующие в
        `ParameterSet`, будут автоматически проигнорированы `from_dict`.
    """
    flat: dict[str, Any] = {}

    def _section(name: str) -> dict[str, Any]:
        sec = data.get(name, {})
        if not isinstance(sec, dict):
            return {}
        return sec

    def _put(key: Any, value: Any) -> None:
        key_str = str(key)
        flat[_FIELD_RENAMES.get(key_str, key_str)] = value

    for k, v in _section("cells").items():
        _put(k, v)

    for k, v in _section("cytokines").items():
        _put(k, v)

    for k, v in _section("ecm").items():
        _put(k, v)

    for k, v in _section("prp").items():
        if str(k) == "factors":
            continue
        _put(k, v)

    for k, v in _section("pemf").items():
        _put(k, v)

    for k, v in _section("damage").items():
        _put(k, v)

    for k, v in _section("oxygen").items():
        if str(k) == "weights" and isinstance(v, dict):
            for wk, wv in v.items():
                flat[str(wk)] = wv
            continue
        _put(k, v)

    for k, v in _section("noise").items():
        flat[str(k)] = v

    # numerics — scheme/interpretation/dt_fast/dt_slow/multirate_subcycling/X_min
    # (benchmark_GBM — вложенный словарь, нужен только в convergence-тестах,
    # хранить как есть; ParameterSet проигнорирует — не страшно)
    for k, v in _section("numerics").items():
        flat[str(k)] = v

    return flat


def get_initial_conditions(data: dict[str, Any]) -> dict[str, float]:
    """Вернуть IC для state vector как плоский словарь, готовый к скармливанию `ExtendedSDEState`.

    Это отдельная функция, потому что IC не часть `ParameterSet`, а часть состояния модели.
    """
    ic = data.get("initial_conditions", {})
    if not isinstance(ic, dict):
        raise ValueError("initial_conditions должна быть словарём")
    return {k: float(v) for k, v in ic.items()}


__all__ = [
    "REQUIRED_SECTIONS",
    "_extract_value",
    "load_params_yaml",
    "flatten_for_parameter_set",
    "get_initial_conditions",
]
