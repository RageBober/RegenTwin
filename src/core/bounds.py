"""Границы параметров и метаданные для анализа чувствительности.

Единый источник истины для bounds всех параметров ParameterSet.
Используется как core-модулями (SensitivityAnalyzer), так и API-слоем.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParameterBounds:
    """Границы одного параметра для анализа чувствительности.

    Задаёт диапазон варьирования параметра при сэмплировании.
    Номинальное значение используется для локальной чувствительности.
    """

    name: str
    lower: float
    upper: float
    nominal: float | None = None


# Параметры, исключённые из анализа чувствительности (численные/служебные/целочисленные)
NUMERICAL_PARAMS: frozenset[str] = frozenset({"dt", "t_max", "epsilon", "n_hill"})

# Куратированные границы для параметров, где ±50% от номинала
# даёт физически бессмысленные значения
CURATED_BOUNDS: dict[str, tuple[float, float]] = {
    # Carrying capacity — логарифмическая шкала
    "K_F": (1e4, 2e6),
    "K_E": (5e3, 5e5),
    "K_S": (5e2, 5e4),
    # Инициальная концентрация тромбоцитов
    "P_max": (1e3, 1e5),
    # O2 — физиологический диапазон (mmHg)
    "O2_blood": (60.0, 150.0),
    # Дистанция диффузии (мкм)
    "L_diffusion": (50.0, 300.0),
    # sigma — cap at 0.5 (>100% волатильность нефизична)
    "sigma_P": (0.005, 0.5),
    "sigma_Ne": (0.005, 0.5),
    "sigma_M": (0.005, 0.5),
    "sigma_M1": (0.005, 0.5),
    "sigma_M2": (0.005, 0.5),
    "sigma_F": (0.005, 0.5),
    "sigma_Mf": (0.005, 0.5),
    "sigma_E": (0.005, 0.5),
    "sigma_S": (0.005, 0.5),
    "sigma_TNF": (0.005, 0.5),
    "sigma_IL10": (0.005, 0.5),
    "sigma_PDGF": (0.005, 0.5),
    "sigma_VEGF": (0.005, 0.5),
    "sigma_TGF": (0.005, 0.5),
    "sigma_MCP1": (0.005, 0.5),
    "sigma_IL8": (0.005, 0.5),
}

# Группировка параметров для UI
PARAM_GROUPS: dict[str, str] = {
    # Пролиферация
    "r_F": "proliferation",
    "r_E": "proliferation",
    "r_S": "proliferation",
    # Смерть
    "delta_P": "death",
    "delta_Ne": "death",
    "delta_M": "death",
    "delta_F": "death",
    "delta_Mf": "death",
    "delta_E": "death",
    "delta_S": "death",
    # Переключение и активация
    "k_switch": "switching",
    "k_reverse": "switching",
    "k_act": "switching",
    # Carrying capacity
    "K_F": "capacity",
    "K_E": "capacity",
    "K_S": "capacity",
    # Тромбоциты
    "P_max": "platelets",
    "tau_P": "platelets",
    "k_deg": "platelets",
    # Рекрутирование
    "R_Ne_max": "recruitment",
    "K_IL8": "recruitment",
    "n_hill": "recruitment",
    "R_M_max": "recruitment",
    "K_MCP1": "recruitment",
    # Фагоцитоз
    "k_phag": "phagocytosis",
    "K_phag": "phagocytosis",
    # Стволовые клетки
    "k_diff_S": "stem_cells",
    "K_diff": "stem_cells",
    # Фибробласты
    "K_PDGF": "fibroblasts",
    "K_TGFb_prolif": "fibroblasts",
    "alpha_TGF": "fibroblasts",
    # Миофибробласты
    "K_activ": "myofibroblasts",
    "K_survival": "myofibroblasts",
    # Ангиогенез
    "K_VEGF": "angiogenesis",
    "K_O2": "angiogenesis",
    # PRP
    "alpha_PRP_S": "prp",
    # Переключение макрофагов (Hill)
    "K_switch_half": "macrophage_switching",
    "K_reverse_half": "macrophage_switching",
    # Деградация цитокинов
    "gamma_TNF": "cytokine_degradation",
    "gamma_IL10": "cytokine_degradation",
    "gamma_PDGF": "cytokine_degradation",
    "gamma_VEGF": "cytokine_degradation",
    "gamma_TGF": "cytokine_degradation",
    "gamma_MCP1": "cytokine_degradation",
    "gamma_IL8": "cytokine_degradation",
    "gamma_MMP": "cytokine_degradation",
    # Секреция цитокинов
    "s_TNF_M1": "cytokine_secretion",
    "s_TNF_Ne": "cytokine_secretion",
    "s_IL10_M2": "cytokine_secretion",
    "s_IL10_efferocytosis": "cytokine_secretion",
    "s_PDGF_P": "cytokine_secretion",
    "s_PDGF_M": "cytokine_secretion",
    "s_VEGF_M2": "cytokine_secretion",
    "s_VEGF_F": "cytokine_secretion",
    "alpha_hypoxia": "cytokine_secretion",
    "s_TGF_P": "cytokine_secretion",
    "s_TGF_M2": "cytokine_secretion",
    "s_TGF_Mf": "cytokine_secretion",
    "s_MCP1_damage": "cytokine_secretion",
    "s_MCP1_M1": "cytokine_secretion",
    "s_IL8_damage": "cytokine_secretion",
    "s_IL8_M1": "cytokine_secretion",
    "s_IL8_Ne": "cytokine_secretion",
    # Ингибирование и связывание
    "k_inhib_IL10": "inhibition",
    "K_inhib": "inhibition",
    "k_bind_F": "inhibition",
    "k_bind_E": "inhibition",
    # ECM
    "q_F": "ecm",
    "q_Mf": "ecm",
    "rho_c_max": "ecm",
    "k_MMP_deg": "ecm",
    "K_MMP_sub": "ecm",
    "s_MMP_M1": "ecm",
    "s_MMP_M2": "ecm",
    "alpha_MMP_M2": "ecm",
    "s_MMP_F": "ecm",
    "k_TIMP": "ecm",
    "C_TIMP": "ecm",
    "k_fibrinolysis": "ecm",
    "k_remodel": "ecm",
    # Шум
    "sigma_P": "noise",
    "sigma_Ne": "noise",
    "sigma_M": "noise",
    "sigma_M1": "noise",
    "sigma_M2": "noise",
    "sigma_F": "noise",
    "sigma_Mf": "noise",
    "sigma_E": "noise",
    "sigma_S": "noise",
    "sigma_TNF": "noise",
    "sigma_IL10": "noise",
    "sigma_PDGF": "noise",
    "sigma_VEGF": "noise",
    "sigma_TGF": "noise",
    "sigma_MCP1": "noise",
    "sigma_IL8": "noise",
    # Damage
    "D0": "damage",
    "tau_damage": "damage",
    # Кислород
    "D_O2": "oxygen",
    "O2_blood": "oxygen",
    "L_diffusion": "oxygen",
    "k_consumption": "oxygen",
    "K_O2_consume": "oxygen",
    "k_angio": "oxygen",
}
