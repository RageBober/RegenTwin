"""Полный набор параметров модели регенерации тканей.

Содержит все 80+ параметров из математического фреймворка §8:
- Клеточные параметры (пролиферация, смерть, переключение)
- Цитокиновые параметры (секреция, деградация)
- ECM параметры (продукция коллагена, MMP-деградация)
- Параметры шума (sigma для всех переменных)
- Вспомогательные (damage, oxygen)
- Численные параметры (dt, t_max, epsilon)

Подробное описание: Description/Phase2/description_parameters.md
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass
class ParameterSet:
    """Полный набор параметров 20-переменной SDE системы.

    Все значения по умолчанию из §8 RegenTwin Mathematical Framework.
    Группы: клеточные, цитокиновые, ECM, шум, вспомогательные, численные.

    Подробное описание: Description/Phase2/description_parameters.md#ParameterSet
    """

    # ===== Клеточные параметры: пролиферация (ч⁻¹) =====
    r_F: float = 0.03   # Пролиферация фибробластов (Vodovotz 2006)
    r_E: float = 0.02   # Пролиферация эндотелия (Anderson 1998)
    r_S: float = 0.01   # Самообновление стволовых (Badiavas 2003)

    # ===== Клеточные параметры: смерть (ч⁻¹) =====
    delta_P: float = 0.1     # Клиренс тромбоцитов (Nurden 2008)
    delta_Ne: float = 0.05   # Апоптоз нейтрофилов (Kolaczkowska 2013)
    delta_M: float = 0.01    # Апоптоз макрофагов (Murray 2017)
    delta_F: float = 0.003   # Апоптоз фибробластов (Hinz 2007)
    delta_Mf: float = 0.01   # Апоптоз миофибробластов (Desmouliere 2005)
    delta_E: float = 0.005   # Апоптоз эндотелия
    delta_S: float = 0.005   # Апоптоз стволовых

    # ===== Клеточные: переключение и активация (ч⁻¹) =====
    k_switch: float = 0.02    # M1→M2 переключение (Mantovani 2004)
    k_reverse: float = 0.005  # M2→M1 обратное переключение
    k_act: float = 0.01       # F→Mf активация (Hinz 2007)

    # ===== Carrying capacity (клеток/мкл) =====
    K_F: float = 5e5    # Carrying capacity F+Mf (Flegg 2015)
    K_E: float = 1e5    # Carrying capacity E
    K_S: float = 1e4    # Carrying capacity S

    # ===== Тромбоциты =====
    P_max: float = 1e4   # Макс. концентрация при активации
    tau_P: float = 2.0   # Временная константа активации (ч)
    k_deg: float = 0.05  # Скорость дегрануляции (ч⁻¹)

    # ===== Нейтрофилы: рекрутирование (Hill-функция) =====
    R_Ne_max: float = 100.0  # Макс. скорость рекрутирования
    K_IL8: float = 1.0       # Константа полунасыщения IL-8
    n_hill: int = 2           # Коэффициент Хилла

    # ===== Моноциты: рекрутирование =====
    R_M_max: float = 50.0  # Макс. скорость рекрутирования
    K_MCP1: float = 1.0    # Константа полунасыщения MCP-1

    # ===== Фагоцитоз (Michaelis-Menten) =====
    k_phag: float = 0.01    # Скорость фагоцитоза
    K_phag: float = 100.0   # Константа полунасыщения

    # ===== Стволовые клетки: дифференциация =====
    k_diff_S: float = 0.005  # Скорость дифференциации S→F
    K_diff: float = 1.0      # Полунасыщение дифференциации

    # ===== Фибробласты: митогенная стимуляция =====
    K_PDGF: float = 1.0         # Полунасыщение PDGF
    K_TGFb_prolif: float = 2.0  # Полунасыщение TGF-β (пролиф.)
    alpha_TGF: float = 0.5      # Усиление TGF-β пролиферации

    # ===== Миофибробласты =====
    K_activ: float = 2.0     # Полунасыщение активации F→Mf (Hill)
    K_survival: float = 1.0  # Полунасыщение TGF-β выживания Mf

    # ===== Эндотелий: ангиогенез =====
    K_VEGF: float = 1.0  # Полунасыщение VEGF (Hill n=2)
    K_O2: float = 5.0    # Порог кислорода для гипоксии

    # ===== PRP: эффект на стволовые =====
    alpha_PRP_S: float = 0.5  # Коэффициент PRP стимуляции

    # ===== Макрофаги: переключение (Hill-функция) =====
    K_switch_half: float = 1.0    # Полунасыщение M1→M2
    K_reverse_half: float = 1.0   # Полунасыщение M2→M1

    # ===== Цитокины: деградация (ч⁻¹) =====
    gamma_TNF: float = 0.5    # TNF-α (Bradley 2008)
    gamma_IL10: float = 0.3   # IL-10 (Mosser 2008)
    gamma_PDGF: float = 0.2   # PDGF (Heldin 1999)
    gamma_VEGF: float = 0.3   # VEGF (Ferrara 2004)
    gamma_TGF: float = 0.15   # TGF-β (Leask 2004)
    gamma_MCP1: float = 0.4   # MCP-1
    gamma_IL8: float = 0.5    # IL-8
    gamma_MMP: float = 0.1    # MMP

    # ===== Цитокины: секреция (нг/(мл·кл·ч)) =====
    s_TNF_M1: float = 0.01           # TNF от M1 (Bradley 2008)
    s_TNF_Ne: float = 0.005          # TNF от нейтрофилов
    s_IL10_M2: float = 0.008         # IL-10 от M2 (Mosser 2008)
    s_IL10_efferocytosis: float = 0.005  # IL-10 при эффероцитозе
    s_PDGF_P: float = 0.02           # PDGF от тромбоцитов
    s_PDGF_M: float = 0.005          # PDGF от макрофагов
    s_VEGF_M2: float = 0.01          # VEGF от M2
    s_VEGF_F: float = 0.003          # VEGF от фибробластов
    alpha_hypoxia: float = 2.0       # Усиление VEGF гипоксией
    s_TGF_P: float = 0.015           # TGF-β от тромбоцитов
    s_TGF_M2: float = 0.008          # TGF-β от M2
    s_TGF_Mf: float = 0.01           # TGF-β от миофибробластов
    s_MCP1_damage: float = 0.1       # MCP-1 от DAMPs
    s_MCP1_M1: float = 0.01          # MCP-1 от M1
    s_IL8_damage: float = 0.1        # IL-8 от DAMPs
    s_IL8_M1: float = 0.01           # IL-8 от M1
    s_IL8_Ne: float = 0.005          # IL-8 от нейтрофилов (аутокрин)

    # ===== Цитокины: ингибирование и связывание =====
    k_inhib_IL10: float = 0.01  # Ингибирование TNF через IL-10
    K_inhib: float = 1.0        # Полунасыщение ингибирования
    k_bind_F: float = 0.001     # Связывание PDGF фибробластами
    k_bind_E: float = 0.001     # Связывание VEGF эндотелием

    # ===== ECM параметры =====
    q_F: float = 0.005         # Продукция коллагена F (Xue 2009)
    q_Mf: float = 0.015        # Продукция коллагена Mf (Desmouliere 2005)
    rho_c_max: float = 1.0     # Макс. плотность коллагена (норм.)
    k_MMP_deg: float = 0.02    # Деградация коллагена MMP (Gill 2008)
    K_MMP_sub: float = 0.5     # Константа Михаэлиса MMP-субстрат
    s_MMP_M1: float = 0.01     # Секреция MMP M1
    s_MMP_M2: float = 0.003    # Секреция MMP M2
    alpha_MMP_M2: float = 0.3  # Коэффициент MMP M2 (< 1)
    s_MMP_F: float = 0.005     # Секреция MMP фибробластами
    k_TIMP: float = 0.01       # Ингибирование TIMP
    C_TIMP: float = 0.5        # Концентрация TIMP (константа)
    k_fibrinolysis: float = 0.01   # Фибринолиз MMP
    k_remodel: float = 0.005       # Ремоделирование фибрина

    # ===== Параметры шума (sigma) =====
    sigma_P: float = 0.05    # Тромбоциты
    sigma_Ne: float = 0.05   # Нейтрофилы
    sigma_M: float = 0.03    # Макрофаги (M1 и M2)
    sigma_F: float = 0.02    # Фибробласты
    sigma_Mf: float = 0.02   # Миофибробласты
    sigma_E: float = 0.02    # Эндотелий
    sigma_S: float = 0.02    # Стволовые
    sigma_TNF: float = 0.05  # TNF-α
    sigma_IL10: float = 0.03  # IL-10
    sigma_PDGF: float = 0.03  # PDGF
    sigma_VEGF: float = 0.03  # VEGF
    sigma_TGF: float = 0.02  # TGF-β
    sigma_MCP1: float = 0.05  # MCP-1
    sigma_IL8: float = 0.05  # IL-8

    # ===== Вспомогательные: damage signal =====
    D0: float = 1.0           # Начальная интенсивность повреждения
    tau_damage: float = 36.0  # Затухание DAMPs (ч)

    # ===== Вспомогательные: кислород =====
    D_O2: float = 2.0e-5          # Диффузия O₂ (см²/с)
    O2_blood: float = 100.0       # Давление O₂ крови (mmHg)
    L_diffusion: float = 100.0    # Дистанция диффузии (мкм)
    k_consumption: float = 0.01   # Потребление O₂ клетками
    K_O2_consume: float = 5.0     # Полунасыщение потребления O₂
    k_angio: float = 0.001        # Перфузия от ангиогенеза

    # ===== Численные параметры =====
    dt: float = 0.01         # Шаг времени (ч)
    t_max: float = 720.0     # Макс. время (ч = 30 дней)
    epsilon: float = 1e-10   # Защита от деления на 0

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
            "r_F": self.r_F, "r_E": self.r_E, "r_S": self.r_S,
            # Смерть
            "delta_P": self.delta_P, "delta_Ne": self.delta_Ne,
            "delta_M": self.delta_M, "delta_F": self.delta_F,
            "delta_Mf": self.delta_Mf, "delta_E": self.delta_E,
            "delta_S": self.delta_S,
            # Переключение и активация
            "k_switch": self.k_switch, "k_reverse": self.k_reverse,
            "k_act": self.k_act,
            # Carrying capacity
            "K_F": self.K_F, "K_E": self.K_E, "K_S": self.K_S,
            # Тромбоциты
            "P_max": self.P_max, "tau_P": self.tau_P, "k_deg": self.k_deg,
            # Рекрутирование
            "R_Ne_max": self.R_Ne_max, "K_IL8": self.K_IL8,
            "n_hill": self.n_hill,
            "R_M_max": self.R_M_max, "K_MCP1": self.K_MCP1,
            # Фагоцитоз
            "k_phag": self.k_phag, "K_phag": self.K_phag,
            # Дифференциация
            "k_diff_S": self.k_diff_S, "K_diff": self.K_diff,
            # Полунасыщение
            "K_PDGF": self.K_PDGF, "K_TGFb_prolif": self.K_TGFb_prolif,
            "K_activ": self.K_activ, "K_survival": self.K_survival,
            "K_VEGF": self.K_VEGF, "K_O2": self.K_O2,
            "K_switch_half": self.K_switch_half,
            "K_reverse_half": self.K_reverse_half,
            "K_inhib": self.K_inhib, "K_MMP_sub": self.K_MMP_sub,
            "K_O2_consume": self.K_O2_consume,
            # Деградация цитокинов
            "gamma_TNF": self.gamma_TNF, "gamma_IL10": self.gamma_IL10,
            "gamma_PDGF": self.gamma_PDGF, "gamma_VEGF": self.gamma_VEGF,
            "gamma_TGF": self.gamma_TGF, "gamma_MCP1": self.gamma_MCP1,
            "gamma_IL8": self.gamma_IL8, "gamma_MMP": self.gamma_MMP,
            # Ингибирование и связывание
            "k_inhib_IL10": self.k_inhib_IL10,
            "k_bind_F": self.k_bind_F, "k_bind_E": self.k_bind_E,
            # ECM
            "k_MMP_deg": self.k_MMP_deg, "k_TIMP": self.k_TIMP,
            "k_fibrinolysis": self.k_fibrinolysis,
            "k_remodel": self.k_remodel,
            "rho_c_max": self.rho_c_max,
            # Вспомогательные
            "D0": self.D0, "tau_damage": self.tau_damage,
            "D_O2": self.D_O2, "O2_blood": self.O2_blood,
            "L_diffusion": self.L_diffusion,
            "k_consumption": self.k_consumption, "k_angio": self.k_angio,
            # Численные
            "dt": self.dt, "t_max": self.t_max, "epsilon": self.epsilon,
        }
        for name, value in strictly_positive.items():
            if value <= 0:
                raise ValueError(
                    f"Параметр {name} должен быть > 0, получено {value}"
                )

        # Параметры, которые должны быть >= 0
        non_negative = {
            "sigma_P": self.sigma_P, "sigma_Ne": self.sigma_Ne,
            "sigma_M": self.sigma_M, "sigma_F": self.sigma_F,
            "sigma_Mf": self.sigma_Mf, "sigma_E": self.sigma_E,
            "sigma_S": self.sigma_S,
            "sigma_TNF": self.sigma_TNF, "sigma_IL10": self.sigma_IL10,
            "sigma_PDGF": self.sigma_PDGF, "sigma_VEGF": self.sigma_VEGF,
            "sigma_TGF": self.sigma_TGF, "sigma_MCP1": self.sigma_MCP1,
            "sigma_IL8": self.sigma_IL8,
        }
        for name, value in non_negative.items():
            if value < 0:
                raise ValueError(
                    f"Параметр {name} должен быть >= 0, получено {value}"
                )

        return True

    def to_dict(self) -> dict[str, float | int]:
        """Сериализация всех параметров в словарь.

        Returns:
            Словарь {имя_параметра: значение} для всех полей

        Подробное описание:
            Description/Phase2/description_parameters.md#ParameterSet.to_dict
        """
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
        }

    @classmethod
    def from_dict(cls, data: dict[str, float | int]) -> ParameterSet:
        """Создание ParameterSet из словаря.

        Игнорирует неизвестные ключи, использует defaults для отсутствующих.

        Args:
            data: Словарь с параметрами

        Returns:
            Новый ParameterSet

        Raises:
            TypeError: Если значения имеют некорректный тип

        Подробное описание:
            Description/Phase2/description_parameters.md#ParameterSet.from_dict
        """
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {}
        for k, v in data.items():
            if k not in valid_fields:
                continue
            if not isinstance(v, (int, float)):
                raise TypeError(
                    f"Значение для {k} должно быть числом, "
                    f"получено {type(v).__name__}"
                )
            filtered[k] = v
        return cls(**filtered)

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
