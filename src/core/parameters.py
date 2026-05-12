"""Полный набор параметров модели регенерации тканей.

Содержит все 80+ параметров из математического фреймворка §8:
- Клеточные параметры (пролиферация, смерть, переключение)
- Цитокиновые параметры (секреция, деградация)
- ECM параметры (продукция коллагена, MMP-деградация)
- Параметры шума (sigma для всех переменных)
- Вспомогательные (damage, oxygen)
- Численные параметры (dt, t_max, epsilon)
- v2.0 (FIX-01..FIX-25): добавлены carrying-capacity floors, sprouting/homing,
  PEMF LF/RF, TIMP/SDF-1, метаболические веса, multirate subcycling.

Подробное описание: Description/Phase2/description_parameters.md
Phase 0 plan: C:\\Users\\dzume\\.claude\\plans\\keen-cuddling-tome.md
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.bounds import CURATED_BOUNDS, NUMERICAL_PARAMS, ParameterBounds


def _coerce_field_value(raw: Any, field_type: Any) -> Any:
    """Привести значение из YAML к типу dataclass-поля.

    Поддерживаемые типы полей: `float`, `int`, `bool`, `str`. Для строкового
    представления типа (когда `from __future__ import annotations` превращает
    аннотацию в строку) сравнение делается по имени.

    Если приведение не удалось — возвращаем raw без модификации; ошибку поймает
    конструктор dataclass и упадёт с понятной TypeError.
    """
    type_name = (
        field_type
        if isinstance(field_type, str)
        else getattr(field_type, "__name__", str(field_type))
    )

    # bool строго отличаем от int — оба isinstance(True, int) == True
    if type_name == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() in {"true", "1", "yes", "on"}
        return bool(raw)
    if type_name == "str":
        return str(raw)
    if type_name == "int":
        return int(raw)
    if type_name == "float":
        return float(raw)
    # неизвестный тип — оставляем как есть
    return raw


@dataclass
class ParameterSet:
    """Полный набор параметров 20-переменной SDE системы.

    Все значения по умолчанию из §8 RegenTwin Mathematical Framework.
    Группы: клеточные, цитокиновые, ECM, шум, вспомогательные, численные.

    Подробное описание: Description/Phase2/description_parameters.md#ParameterSet
    """

    # ===== Клеточные параметры: пролиферация (ч⁻¹) =====
    r_F: float = 0.03  # Пролиферация фибробластов (Xue 2009, Flegg 2010)
    r_E: float = 0.02  # Пролиферация эндотелия (Anderson 1998)
    r_S: float = 0.01  # Самообновление стволовых (Badiavas 2003)

    # ===== Клеточные параметры: смерть (ч⁻¹) =====
    delta_P: float = 0.1  # Клиренс тромбоцитов (Nurden 2008)
    delta_Ne: float = 0.05  # Апоптоз нейтрофилов (Kolaczkowska 2013)
    delta_M: float = 0.01  # Апоптоз макрофагов (Murray 2017)
    delta_F: float = 0.003  # Апоптоз фибробластов (Hinz 2007)
    delta_Mf: float = 0.01  # Апоптоз миофибробластов (Desmouliere 2005)
    delta_E: float = 0.005  # Апоптоз эндотелия
    delta_S: float = 0.005  # Апоптоз стволовых

    # ===== Клеточные: переключение и активация (ч⁻¹) =====
    k_switch: float = 0.02  # M1→M2 переключение (Mantovani 2004)
    k_reverse: float = 0.005  # M2→M1 обратное переключение
    k_act: float = 0.01  # F→Mf активация (Hinz 2007)

    # ===== Carrying capacity =====
    # Текущие значения откалиброваны для cells/мкл (legacy v1.0).
    # v2.0 целевые (params.yaml): K_F=5e8, K_E=5e7, K_S=1e6 cells/ml (FIX-01).
    # Переход выполняется по фазам синхронно с рекалибровкой s_X / k_bind_* (FIX-19).
    K_F: float = 5e5  # Carrying capacity F+Mf (Flegg 2015)
    K_E: float = 1e5  # Carrying capacity E
    K_S: float = 1e4  # Carrying capacity S

    # ===== Тромбоциты =====
    P_max: float = 1e4  # Макс. концентрация при активации
    tau_P: float = 2.0  # Временная константа активации (ч)
    k_deg: float = 0.05  # Скорость дегрануляции (ч⁻¹)

    # ===== Нейтрофилы: рекрутирование (Hill-функция) =====
    R_Ne_max: float = 100.0  # Макс. скорость рекрутирования
    K_IL8: float = 1.0  # Константа полунасыщения IL-8
    n_hill: int = 2  # Коэффициент Хилла

    # ===== Моноциты: рекрутирование =====
    R_M_max: float = 50.0  # Макс. скорость рекрутирования
    K_MCP1: float = 1.0  # Константа полунасыщения MCP-1

    # ===== Фагоцитоз (Michaelis-Menten) =====
    k_phag: float = 0.01  # Скорость фагоцитоза
    K_phag: float = 100.0  # Константа полунасыщения

    # ===== Стволовые клетки: дифференциация =====
    k_diff_S: float = 0.005  # Скорость дифференциации S→F
    K_diff: float = 1.0  # Полунасыщение дифференциации

    # ===== Фибробласты: митогенная стимуляция =====
    K_PDGF: float = 1.0  # Полунасыщение PDGF
    K_TGFb_prolif: float = 2.0  # Полунасыщение TGF-β (пролиф.)
    alpha_TGF: float = 0.5  # Усиление TGF-β пролиферации

    # ===== Миофибробласты =====
    K_activ: float = 2.0  # Полунасыщение активации F→Mf (Hill)
    K_survival: float = 1.0  # Полунасыщение TGF-β выживания Mf

    # ===== Эндотелий: ангиогенез =====
    K_VEGF: float = 1.0  # Полунасыщение VEGF (Hill n=2)
    K_O2: float = 5.0  # Порог кислорода для гипоксии

    # ===== PRP: эффект на стволовые =====
    alpha_PRP_S: float = 0.5  # Коэффициент PRP стимуляции

    # ===== Макрофаги: переключение (Hill-функция) =====
    K_switch_half: float = 1.0  # Полунасыщение M1→M2
    K_reverse_half: float = 1.0  # Полунасыщение M2→M1

    # ===== Цитокины: деградация (ч⁻¹) =====
    gamma_TNF: float = 0.5  # TNF-α (Bradley 2008)
    gamma_IL10: float = 0.3  # IL-10 (Mosser 2008)
    gamma_PDGF: float = 0.2  # PDGF (Heldin 1999)
    gamma_VEGF: float = 0.3  # VEGF (Ferrara 2004)
    gamma_TGF: float = 0.15  # TGF-β (Leask 2004)
    gamma_MCP1: float = 0.4  # MCP-1
    gamma_IL8: float = 0.5  # IL-8
    gamma_MMP: float = 0.1  # MMP

    # ===== Цитокины: секреция (нг/(мл·кл·ч)) =====
    s_TNF_M1: float = 0.01  # TNF от M1 (Bradley 2008)
    s_TNF_Ne: float = 0.005  # TNF от нейтрофилов
    s_IL10_M2: float = 0.008  # IL-10 от M2 (Mosser 2008)
    s_IL10_efferocytosis: float = 0.005  # IL-10 при эффероцитозе
    s_PDGF_P: float = 0.02  # PDGF от тромбоцитов
    s_PDGF_M: float = 0.005  # PDGF от макрофагов
    s_VEGF_M2: float = 0.01  # VEGF от M2
    s_VEGF_F: float = 0.003  # VEGF от фибробластов
    alpha_hypoxia: float = 2.0  # Усиление VEGF гипоксией
    s_TGF_P: float = 0.015  # TGF-β от тромбоцитов
    s_TGF_M2: float = 0.008  # TGF-β от M2
    s_TGF_Mf: float = 0.01  # TGF-β от миофибробластов
    s_MCP1_damage: float = 0.1  # MCP-1 от DAMPs
    s_MCP1_M1: float = 0.01  # MCP-1 от M1
    s_IL8_damage: float = 0.1  # IL-8 от DAMPs
    s_IL8_M1: float = 0.01  # IL-8 от M1
    s_IL8_Ne: float = 0.005  # IL-8 от нейтрофилов (аутокрин)

    # ===== Цитокины: ингибирование и связывание =====
    k_inhib_IL10: float = 0.01  # Ингибирование TNF через IL-10
    K_inhib: float = 1.0  # Полунасыщение ингибирования
    k_bind_F: float = 0.001  # Связывание PDGF фибробластами
    k_bind_E: float = 0.001  # Связывание VEGF эндотелием

    # ===== ECM параметры =====
    q_F: float = 0.005  # Продукция коллагена F (Xue 2009)
    q_Mf: float = 0.015  # Продукция коллагена Mf (Desmouliere 2005)
    rho_c_max: float = 1.0  # Макс. плотность коллагена (норм.)
    k_MMP_deg: float = 0.02  # Деградация коллагена MMP (Gill 2008)
    K_MMP_sub: float = 0.5  # Константа Михаэлиса MMP-субстрат
    s_MMP_M1: float = 0.01  # Секреция MMP M1
    s_MMP_M2: float = 0.003  # Секреция MMP M2
    alpha_MMP_M2: float = 0.3  # Коэффициент MMP M2 (< 1)
    s_MMP_F: float = 0.005  # Секреция MMP фибробластами
    k_TIMP: float = 0.01  # Ингибирование TIMP
    C_TIMP: float = 0.5  # Концентрация TIMP (константа)
    k_fibrinolysis: float = 0.01  # Фибринолиз MMP
    k_remodel: float = 0.005  # Ремоделирование фибрина

    # ===== Параметры шума (sigma) =====
    sigma_P: float = 0.05  # Тромбоциты
    sigma_Ne: float = 0.05  # Нейтрофилы
    sigma_M: float = 0.03  # Макрофаги (legacy, не используется)
    sigma_M1: float = 0.03  # M1 макрофаги (провоспалительные)
    sigma_M2: float = 0.03  # M2 макрофаги (противовоспалительные)
    sigma_F: float = 0.02  # Фибробласты
    sigma_Mf: float = 0.02  # Миофибробласты
    sigma_E: float = 0.02  # Эндотелий
    sigma_S: float = 0.02  # Стволовые
    sigma_TNF: float = 0.05  # TNF-α
    sigma_IL10: float = 0.03  # IL-10
    sigma_PDGF: float = 0.03  # PDGF
    sigma_VEGF: float = 0.03  # VEGF
    sigma_TGF: float = 0.02  # TGF-β
    sigma_MCP1: float = 0.05  # MCP-1
    sigma_IL8: float = 0.05  # IL-8

    # ===== Вспомогательные: damage signal =====
    D0: float = 1.0  # Начальная интенсивность повреждения
    tau_damage: float = 36.0  # Затухание DAMPs (ч)

    # ===== Вспомогательные: кислород =====
    D_O2: float = 2.0e-5  # Диффузия O₂ (см²/с)
    O2_blood: float = 100.0  # Давление O₂ крови (mmHg)
    L_diffusion: float = 100.0  # Дистанция диффузии (мкм)
    k_consumption: float = 0.01  # Потребление O₂ клетками
    K_O2_consume: float = 5.0  # Полунасыщение потребления O₂
    k_angio: float = 0.001  # Перфузия от ангиогенеза

    # ===== Численные параметры =====
    dt: float = 0.01  # Шаг времени (ч)
    t_max: float = 720.0  # Макс. время (ч = 30 дней)
    epsilon: float = 1e-10  # Защита от деления на 0

    # =================================================================
    # v2.0 (FIX-01..FIX-25) — НОВЫЕ ПОЛЯ
    # Значения по умолчанию совпадают с params.yaml. Используются
    # уравнениями только после применения соответствующих фиксов в
    # Phase 3..8. До тех пор живут как inert defaults — не влияют на
    # текущую динамику.
    # =================================================================

    # ----- FIX-03: миофибробласты, floor смертности -----
    delta_floor: float = 0.1  # Минимальная относительная смертность Mf при насыщающем TGF-β

    # ----- FIX-21: дефолт M1 при базальных цитокинах -----
    phi_baseline: float = 0.1  # ng/ml — смещение φ₁ в сторону M1 при C_TNF≈C_IL10≈0

    # ----- FIX-22: миграция фибробластов -----
    J_F_migration: float = 50.0  # cells/(ml·h)
    K_chi: float = 1.0  # ng/ml — полунасыщение PDGF-хемотаксиса фибробластов

    # ----- FIX-05: sprouting эндотелия -----
    J_sprouting: float = 1.0  # cells/(ml·h)
    K_xi: float = 0.5  # ng/ml — полунасыщение sprouting по VEGF

    # ----- FIX-06: хоминг стволовых клеток -----
    J_homing: float = 5.0  # cells/(ml·h) — поток CD34+ из костного мозга
    K_SDF1: float = 1.0  # ng/ml — полунасыщение SDF-1/CXCR4
    alpha_PRP_homing: float = 2.0  # коэффициент PRP-усиления хоминга

    # ----- FIX-07: множитель ингибиции на источнике (n=1.5) -----
    n_inhib: float = 1.5  # коэффициент Хилла для IL-10-ингибиции TNF/IL-8

    # ----- FIX-08: рецепторное потребление TGF-β -----
    k_bind_TGF: float = 0.001  # 1/h
    K_TGF_bind: float = 0.5  # ng/ml

    # ----- Дополнительные Kd рецепторного связывания -----
    K_PDGF_bind: float = 0.1  # ng/ml (отдельно от K_PDGF_prolif)
    K_VEGF_bind: float = 0.5  # ng/ml

    # ----- SDF-1 (новое уравнение, FIX-06) -----
    s_SDF1_F: float = 5.0e-7  # ng/(cells·h)
    s_SDF1_E: float = 2.0e-6  # ng/(cells·h)
    alpha_hypoxia_SDF: float = 5.0  # усиление SDF-1 гипоксией
    gamma_SDF1: float = 0.2  # 1/h

    # ----- TIMP (новое уравнение, FIX-09) -----
    s_TIMP_F: float = 2.0e-9  # unit/(cells·h)
    s_TIMP_M2: float = 1.0e-9  # unit/(cells·h)
    alpha_TGF_TIMP: float = 3.0  # усиление TIMP TGF-β
    K_TIMP: float = 1.0  # ng/ml — полунасыщение TGF-β для TIMP
    gamma_TIMP: float = 0.05  # 1/h

    # ----- PRP объём и нормализация Бэйтмена (FIX-12) -----
    V_wound: float = 5.0  # ml — объём раны
    D_PRP: float = 1.0  # ml — введённая доза PRP

    # ----- PEMF, два частотных окна (FIX-13) -----
    f_LF: float = 75.0  # Hz — центральная частота LF
    sigma_LF: float = 30.0  # Hz — ширина LF-окна
    f_RF: float = 2.712e7  # Hz — 27.12 MHz, центр RF
    sigma_RF_log: float = 0.3  # log10(Hz) — ширина RF-окна
    epsilon_LF_max: float = 0.5  # макс прирост пролиферации LF-PEMF
    epsilon_RF_max: float = 0.4  # макс снижение TNF RF-PEMF
    B_half: float = 0.5  # mT — полунасыщение по амплитуде B
    B_0: float = 0.1  # mT — пороговая амплитуда

    # ----- PRP+PEMF синергия (FIX-14) -----
    beta_synergy: float = 1.5  # коэффициент синергии
    Theta_PRP_ref: float = 1.0  # ng/(ml·h) — нормировка PRP

    # ----- FIX-11: оксигенация, диффузия как функция E -----
    alpha_E: float = 1.0  # уменьшение L диффузии с ростом E
    # Метаболические веса (взвешенная сумма потребления O₂)
    w_Ne: float = 100.0
    w_M: float = 10.0
    w_F: float = 1.0
    w_E: float = 5.0
    w_S: float = 0.5

    # ----- FIX-20: шум SDF-1 (остальные σ уже есть) -----
    sigma_SDF1: float = 0.2

    # ----- FIX-16: multirate subcycling -----
    dt_fast: float = 0.02  # ч — шаг цитокинов
    dt_slow: float = 1.0  # ч — шаг клеток/ECM
    multirate_subcycling: bool = True
    X_min: float = 1.0e-10  # пол положительности

    # ----- FIX-24: интерпретация SDE -----
    interpretation: str = "Ito"  # "Ito" | "Stratonovich"

    # ----- FIX-20: ECM детерминистично (без шума) -----
    ecm_deterministic: bool = True

    def validate(self) -> bool:
        """Валидация физической осмысленности всех параметров.

        Проверяет:
        - Все скорости (r_*, delta_*, gamma_*) > 0
        - Все carrying capacity (K_*) > 0
        - Все sigma >= 0
        - Все константы полунасыщения > 0
        - dt > 0, t_max > 0, epsilon > 0

        Returns:
            True если все параметры валидны

        Raises:
            ValueError: Если параметр физически бессмысленный

        Подробное описание:
            Description/Phase2/description_parameters.md#ParameterSet.validate
        """
        # Параметры, которые должны быть строго > 0
        strictly_positive = {
            # Пролиферация
            "r_F": self.r_F,
            "r_E": self.r_E,
            "r_S": self.r_S,
            # Смерть
            "delta_P": self.delta_P,
            "delta_Ne": self.delta_Ne,
            "delta_M": self.delta_M,
            "delta_F": self.delta_F,
            "delta_Mf": self.delta_Mf,
            "delta_E": self.delta_E,
            "delta_S": self.delta_S,
            # Переключение и активация
            "k_switch": self.k_switch,
            "k_reverse": self.k_reverse,
            "k_act": self.k_act,
            # Carrying capacity
            "K_F": self.K_F,
            "K_E": self.K_E,
            "K_S": self.K_S,
            # Тромбоциты
            "P_max": self.P_max,
            "tau_P": self.tau_P,
            "k_deg": self.k_deg,
            # Рекрутирование
            "R_Ne_max": self.R_Ne_max,
            "K_IL8": self.K_IL8,
            "n_hill": self.n_hill,
            "R_M_max": self.R_M_max,
            "K_MCP1": self.K_MCP1,
            # Фагоцитоз
            "k_phag": self.k_phag,
            "K_phag": self.K_phag,
            # Дифференциация
            "k_diff_S": self.k_diff_S,
            "K_diff": self.K_diff,
            # Полунасыщение
            "K_PDGF": self.K_PDGF,
            "K_TGFb_prolif": self.K_TGFb_prolif,
            "K_activ": self.K_activ,
            "K_survival": self.K_survival,
            "K_VEGF": self.K_VEGF,
            "K_O2": self.K_O2,
            "K_switch_half": self.K_switch_half,
            "K_reverse_half": self.K_reverse_half,
            "K_inhib": self.K_inhib,
            "K_MMP_sub": self.K_MMP_sub,
            "K_O2_consume": self.K_O2_consume,
            # Деградация цитокинов
            "gamma_TNF": self.gamma_TNF,
            "gamma_IL10": self.gamma_IL10,
            "gamma_PDGF": self.gamma_PDGF,
            "gamma_VEGF": self.gamma_VEGF,
            "gamma_TGF": self.gamma_TGF,
            "gamma_MCP1": self.gamma_MCP1,
            "gamma_IL8": self.gamma_IL8,
            "gamma_MMP": self.gamma_MMP,
            # Ингибирование и связывание
            "k_inhib_IL10": self.k_inhib_IL10,
            "k_bind_F": self.k_bind_F,
            "k_bind_E": self.k_bind_E,
            # ECM
            "k_MMP_deg": self.k_MMP_deg,
            "k_TIMP": self.k_TIMP,
            "k_fibrinolysis": self.k_fibrinolysis,
            "k_remodel": self.k_remodel,
            "rho_c_max": self.rho_c_max,
            # Вспомогательные
            "D0": self.D0,
            "tau_damage": self.tau_damage,
            "D_O2": self.D_O2,
            "O2_blood": self.O2_blood,
            "L_diffusion": self.L_diffusion,
            "k_consumption": self.k_consumption,
            "k_angio": self.k_angio,
            # Численные
            "dt": self.dt,
            "t_max": self.t_max,
            "epsilon": self.epsilon,
        }
        for name, value in strictly_positive.items():
            if value <= 0:
                raise ValueError(f"Параметр {name} должен быть > 0, получено {value}")

        # Параметры, которые должны быть >= 0
        non_negative = {
            "sigma_P": self.sigma_P,
            "sigma_Ne": self.sigma_Ne,
            "sigma_M1": self.sigma_M1,
            "sigma_M2": self.sigma_M2,
            "sigma_F": self.sigma_F,
            "sigma_Mf": self.sigma_Mf,
            "sigma_E": self.sigma_E,
            "sigma_S": self.sigma_S,
            "sigma_TNF": self.sigma_TNF,
            "sigma_IL10": self.sigma_IL10,
            "sigma_PDGF": self.sigma_PDGF,
            "sigma_VEGF": self.sigma_VEGF,
            "sigma_TGF": self.sigma_TGF,
            "sigma_MCP1": self.sigma_MCP1,
            "sigma_IL8": self.sigma_IL8,
        }
        for name, value in non_negative.items():
            if value < 0:
                raise ValueError(f"Параметр {name} должен быть >= 0, получено {value}")

        # Верхняя граница для sigma (относительная волатильность > 100% нефизична)
        import warnings

        for name, value in non_negative.items():
            if name.startswith("sigma_") and value > 1.0:
                warnings.warn(
                    f"Параметр {name} = {value} > 1.0: "
                    f"относительная волатильность > 100% может привести "
                    f"к нестабильной симуляции",
                    stacklevel=2,
                )

        return True

    def to_dict(self) -> dict[str, float | int | bool | str]:
        """Сериализация всех параметров в словарь.

        v2.0: возвращает смешанные типы — float/int для численных параметров,
        bool для флагов (multirate_subcycling, ecm_deterministic), str для
        категориальных (interpretation). Для каллеров, требующих только
        числовые значения (Bayesian inference, sensitivity bounds, прайор-построение),
        используйте `to_numeric_dict()`.

        Returns:
            Словарь {имя_параметра: значение} для всех полей

        Подробное описание:
            Description/Phase2/description_parameters.md#ParameterSet.to_dict
        """
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    def to_numeric_dict(self) -> dict[str, float | int]:
        """Сериализация только численных параметров (float/int).

        v2.0: специализированная версия `to_dict`, отфильтровывающая bool- и
        str-поля (multirate_subcycling, ecm_deterministic, interpretation).
        Удобно для Bayesian priors, sensitivity analysis, parameter estimation —
        там, где обработка нечисловых значений недопустима.

        Returns:
            Словарь {имя_параметра: значение} только для float/int полей.
            bool явно исключён, хотя `isinstance(True, int)` истинно.
        """
        result: dict[str, float | int] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            # bool — подкласс int в Python; исключаем явно.
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                result[f.name] = value
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, float | int | bool | str]) -> ParameterSet:
        """Создание ParameterSet из словаря.

        Игнорирует неизвестные ключи, использует defaults для отсутствующих.
        Поля типа `float`/`int` принимают только числовые значения; для полей
        `bool` принимаются bool/числовое 0|1; для полей `str` принимается str.

        Args:
            data: Словарь с параметрами

        Returns:
            Новый ParameterSet

        Raises:
            TypeError: Если значение несовместимо с типом поля (например, строка
                     передана в `float`-поле).

        Подробное описание:
            Description/Phase2/description_parameters.md#ParameterSet.from_dict
        """
        fields_by_name = {f.name: f for f in dataclasses.fields(cls)}
        filtered: dict[str, Any] = {}
        for k, v in data.items():
            if k not in fields_by_name:
                continue
            field_type = fields_by_name[k].type
            type_name = (
                field_type
                if isinstance(field_type, str)
                else getattr(field_type, "__name__", str(field_type))
            )
            if type_name in ("float", "int"):
                # bool isinstance int — отвергаем; str — отвергаем
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise TypeError(
                        f"Значение для {k} должно быть числом, получено {type(v).__name__}"
                    )
                filtered[k] = v
            elif type_name == "bool":
                if not isinstance(v, (bool, int, float)):
                    raise TypeError(
                        f"Значение для {k} должно быть bool, получено {type(v).__name__}"
                    )
                filtered[k] = bool(v)
            elif type_name == "str":
                if not isinstance(v, str):
                    raise TypeError(
                        f"Значение для {k} должно быть str, получено {type(v).__name__}"
                    )
                filtered[k] = v
            else:
                # неизвестный тип — пропускаем без приведения
                filtered[k] = v
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: Path | str = "params.yaml") -> ParameterSet:
        """Создать `ParameterSet`, загрузив значения из params.yaml (v2.0 источник правды).

        Подгружает `params.yaml` через `params_loader.load_params_yaml`, расплющивает
        вложенные секции, и для каждого поля dataclass'а подставляет соответствующее
        значение (с приведением к типу: float/bool/str). Поля, отсутствующие в YAML,
        остаются с захардкоженным defaults — это гарантирует обратную совместимость
        в Phase 0 (до применения FIX'ов на этих параметрах).

        Если файл не найден — возвращается экземпляр со всеми текущими defaults
        и пишется warning (поведение fallback).

        Args:
            path: путь к params.yaml. По умолчанию — `./params.yaml`.

        Returns:
            Новый ParameterSet с YAML-значениями там, где YAML их предоставляет.

        Note:
            В Phase 0 этот метод существует, но `ParameterSet()` по-прежнему
            возвращает старые literature-defaults. Активный переход на YAML
            будет в Phase 1 (FIX-01 units).
        """
        # Локальный import, чтобы избежать циклов и сделать pyyaml soft-dep.
        from src.core.params_loader import flatten_for_parameter_set, load_params_yaml

        try:
            raw = load_params_yaml(path)
        except FileNotFoundError:
            import warnings

            warnings.warn(
                f"params.yaml не найден по пути {path!r}; возвращаются hardcoded defaults",
                stacklevel=2,
            )
            return cls()

        flat = flatten_for_parameter_set(raw)
        valid_fields: dict[str, Any] = {f.name: f.type for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {}
        for name, raw_value in flat.items():
            if name not in valid_fields:
                continue
            kwargs[name] = _coerce_field_value(raw_value, valid_fields[name])
        return cls(**kwargs)

    @classmethod
    def get_literature_defaults(cls) -> ParameterSet:
        """Создание ParameterSet с литературными значениями по умолчанию.

        Возвращает экземпляр с параметрами из §8 Mathematical Framework.
        Идентичен ParameterSet() (все defaults уже литературные).

        Returns:
            ParameterSet с базовыми значениями

        Подробное описание:
            Description/Phase2/description_parameters.md#get_literature_defaults
        """
        return cls()

    @classmethod
    def get_bounds(cls, names: list[str] | None = None) -> list[ParameterBounds]:
        """Получить границы параметров для анализа чувствительности.

        Использует куратированные bounds для ключевых параметров,
        ±50% от номинала для остальных. Исключает численные (dt, t_max, epsilon).

        Args:
            names: Список имён параметров (None → все кроме численных)

        Returns:
            Список ParameterBounds для запрошенных параметров
        """
        defaults = cls()
        param_dict = defaults.to_dict()
        if names is not None:
            target_names = [n for n in names if n not in NUMERICAL_PARAMS]
        else:
            target_names = [
                f.name for f in dataclasses.fields(cls) if f.name not in NUMERICAL_PARAMS
            ]
        result: list[ParameterBounds] = []
        for name in target_names:
            nominal = param_dict.get(name)
            if nominal is None or not isinstance(nominal, (int, float)):
                continue
            if name in CURATED_BOUNDS:
                lo, hi = CURATED_BOUNDS[name]
            else:
                lo, hi = nominal * 0.5, nominal * 2.0
            # Защита от вырожденных bounds (nominal == 0)
            if lo >= hi:
                continue
            result.append(
                ParameterBounds(
                    name=name,
                    lower=lo,
                    upper=hi,
                    nominal=float(nominal),
                )
            )
        return result
