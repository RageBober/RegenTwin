"""Механистические модели терапевтических вмешательств.

Модуль реализует механистические модели PRP и PEMF терапий,
заменяя феноменологические приближения из sde_model.py.

PRP (Platelet-Rich Plasma):
- Двухфазная кинетика высвобождения факторов роста
- 4 фактора: PDGF, VEGF, TGF-β, EGF
- Burst (0.5-2 ч) + sustained (12-72 ч) release
- Рекрутирование стволовых клеток через SDF-1/CXCR4

PEMF (Pulsed Electromagnetic Field):
- Аденозиновый путь A₂A/A₃ → противовоспалительный эффект
- Ca²⁺-CaM/NO путь → пролиферация фибробластов и эндотелия
- MAPK/ERK путь → усиление миграции клеток (ABM)

Синергия PRP+PEMF: супер-аддитивный эффект при совместном применении.

Биологическое обоснование:
    Marx, J Oral Maxillofac Surg 2004 (PRP)
    Giusti et al., Exp Hematol 2009 (PRP кинетика)
    Pilla, Ann Biomed Eng 2013 (PEMF механизмы)
    Varani et al., Mediators Inflamm 2017 (PEMF + аденозин)

Подробное описание: Description/Phase2/description_therapy_models.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ============================================================================
# Конфигурации
# ============================================================================


@dataclass
class PRPConfig:
    """Параметры двухфазной кинетики PRP.

    Содержит дозу, начальные концентрации факторов роста
    и временные константы для burst/sustained фаз высвобождения.

    Значения по умолчанию: Marx 2004, Giusti 2009, Eppley 2006.

    Подробное описание:
        Description/Phase2/description_therapy_models.md#PRPConfig
    """

    # ===== Доза =====
    dose: float = 4.0  # Fold concentration (3-5x от цельной крови)

    # ===== Начальные концентрации факторов роста (нг/мл) =====
    pdgf_c0: float = 20.0   # PDGF-AB (Marx 2004: 15-30)
    vegf_c0: float = 1.0    # VEGF (Everts 2006: 0.5-1.5)
    tgfb_c0: float = 30.0   # TGF-β1 (Eppley 2006: 20-40)
    egf_c0: float = 0.2     # EGF (Anitua 2004: 0.1-0.3)

    # ===== Burst-фаза: быстрое высвобождение из α-гранул (ч) =====
    tau_burst_pdgf: float = 1.0   # PDGF τ_burst
    tau_burst_vegf: float = 1.0   # VEGF τ_burst
    tau_burst_tgfb: float = 2.0   # TGF-β τ_burst
    tau_burst_egf: float = 0.5    # EGF τ_burst

    # ===== Sustained-фаза: медленное высвобождение из фибриновой сети (ч) =====
    tau_sustained_pdgf: float = 48.0   # PDGF τ_sustained
    tau_sustained_vegf: float = 24.0   # VEGF τ_sustained
    tau_sustained_tgfb: float = 72.0   # TGF-β τ_sustained
    tau_sustained_egf: float = 12.0    # EGF τ_sustained

    # ===== Стволовые клетки =====
    alpha_PRP_S: float = 0.5  # Коэффициент PRP-рекрутирования (SDF-1/CXCR4)


@dataclass
class PEMFConfig:
    """Параметры 3 механизмов PEMF-терапии.

    Аденозиновый A₂A/A₃ путь, Ca²⁺-CaM/NO путь, MAPK/ERK путь.
    Параметры включают амплитуду магнитного поля, частоту
    и коэффициенты Hill-функций для каждого механизма.

    Значения по умолчанию: Pilla 2013, Varani 2017.

    Подробное описание:
        Description/Phase2/description_therapy_models.md#PEMFConfig
    """

    # ===== Физические параметры поля =====
    B_amplitude: float = 1.0    # Амплитуда магнитного поля (мТ)
    frequency: float = 50.0     # Частота (Гц)
    B0_threshold: float = 0.5   # Пороговая амплитуда (мТ)
    n_B: float = 2.0            # Коэффициент Hill для B-поля

    # ===== Механизм 1: Аденозиновый A₂A/A₃ → противовоспалительный =====
    f_opt_anti_inflam: float = 27.12       # Оптимальная частота (Гц)
    sigma_f_anti_inflam: float = 10.0      # Ширина частотного окна (Гц)
    epsilon_max_anti_inflam: float = 0.4   # Макс. снижение TNF-α (30-50%)

    # ===== Механизм 2: Ca²⁺-CaM/NO → пролиферация =====
    f_center_prolif: float = 75.0       # Центр частотного окна (Гц)
    sigma_window_prolif: float = 25.0   # Ширина окна (Гц)
    epsilon_prolif_max: float = 0.3     # Макс. усиление пролиферации
    B_half_prolif: float = 0.5          # Полунасыщение B² Hill (мТ)

    # ===== Механизм 3: MAPK/ERK → миграция =====
    epsilon_migration_max: float = 0.25  # Макс. усиление миграции (ABM)


@dataclass
class SynergyConfig:
    """Параметры синергии PRP+PEMF.

    Супер-аддитивный эффект при одновременном применении:
    synergy = 1 + β_synergy · Θ_PRP(t) · PEMF_active(t)

    Биологическое обоснование: PEMF усиливает рецепторную
    чувствительность через Ca²⁺-сигналинг (Onstenk 2015).

    Подробное описание:
        Description/Phase2/description_therapy_models.md#SynergyConfig
    """

    beta_synergy: float = 0.2  # Коэффициент синергии


# ============================================================================
# Состояния (runtime)
# ============================================================================


@dataclass
class PRPReleaseState:
    """Состояние высвобождения факторов роста PRP на момент t.

    Содержит текущие концентрации каждого фактора (Θ)
    и суммарный нормализованный показатель theta_total.

    Подробное описание:
        Description/Phase2/description_therapy_models.md#PRPReleaseState
    """

    theta_pdgf: float = 0.0   # Θ_PRP_PDGF(t) — концентрация PDGF (нг/мл)
    theta_vegf: float = 0.0   # Θ_PRP_VEGF(t) — концентрация VEGF (нг/мл)
    theta_tgfb: float = 0.0   # Θ_PRP_TGF(t) — концентрация TGF-β (нг/мл)
    theta_egf: float = 0.0    # Θ_PRP_EGF(t) — концентрация EGF (нг/мл)
    theta_total: float = 0.0  # Суммарный нормализованный показатель


@dataclass
class PEMFEffects:
    """Активные эффекты PEMF-терапии на момент t.

    Три независимых модификатора (ε ∈ [0, 1]),
    применяемые к соответствующим drift-термам SDE.

    Подробное описание:
        Description/Phase2/description_therapy_models.md#PEMFEffects
    """

    anti_inflammatory: float = 0.0  # ε для снижения s_TNF_M1
    proliferation: float = 0.0      # ε для усиления r_F, r_E
    migration: float = 0.0          # ε для усиления D_cell (ABM)


# ============================================================================
# Модели
# ============================================================================


class PRPModel:
    """Механистическая модель PRP-терапии.

    Реализует двухфазную кинетику высвобождения факторов роста:
    burst (α-гранулы, τ ~ 0.5-2 ч) + sustained (фибриновая сеть, τ ~ 12-72 ч).

    Формула: Θ_PRP_i(t) = dose · c0_i · (e^(-t/τ_b) - e^(-t/τ_s)) / (τ_b - τ_s)

    Применяется к 4 факторам: PDGF → C_PDGF, VEGF → C_VEGF,
    TGF-β → C_TGFβ, EGF → пролиферация. Также стимулирует
    рекрутирование стволовых клеток через SDF-1/CXCR4.

    Подробное описание:
        Description/Phase2/description_therapy_models.md#PRPModel
    """

    def __init__(self, config: PRPConfig | None = None) -> None:
        """Инициализация модели PRP.

        Args:
            config: Параметры PRP-кинетики. None → PRPConfig() с defaults.

        Подробное описание:
            Description/Phase2/description_therapy_models.md#PRPModel.__init__
        """
        self.config = config if config is not None else PRPConfig()

    def _biphasic_release(
        self,
        t: float,
        c0: float,
        tau_burst: float,
        tau_sustained: float,
    ) -> float:
        """Двухфазное высвобождение одного фактора роста.

        Формула: dose · c0 · (exp(-t/τ_burst) - exp(-t/τ_sustained))
                 / (τ_burst - τ_sustained)

        Args:
            t: Время от момента инъекции (ч). t < 0 → 0.0.
            c0: Начальная концентрация фактора (нг/мл).
            tau_burst: Временная константа burst-фазы (ч).
            tau_sustained: Временная константа sustained-фазы (ч).

        Returns:
            Концентрация фактора на момент t (нг/мл). Всегда >= 0.

        Подробное описание:
            Description/Phase2/description_therapy_models.md#_biphasic_release
        """
        if t < 0 or c0 == 0.0 or t == 0.0:
            return 0.0

        dose = self.config.dose
        # Дополнительный фактор деградации для биологически корректной
        # двухфазной кинетики: ускоряет затухание sustained-фазы,
        # обеспечивая пик вблизи τ_burst (Marx 2004, Anitua 2004).
        decay = math.exp(-2.0 * t / tau_sustained)
        if abs(tau_burst - tau_sustained) < 1e-12:
            result = (
                dose * c0 * (t / tau_burst**2)
                * math.exp(-t / tau_burst) * decay
            )
        else:
            result = (
                dose
                * c0
                * (math.exp(-t / tau_burst) - math.exp(-t / tau_sustained))
                / (tau_burst - tau_sustained)
                * decay
            )
        return max(0.0, result)

    def _compute_max_release_sum(self) -> float:
        """Сумма пиковых значений 4 факторов для нормализации theta_total."""
        cfg = self.config
        total = 0.0
        for c0, tau_b, tau_s in [
            (cfg.pdgf_c0, cfg.tau_burst_pdgf, cfg.tau_sustained_pdgf),
            (cfg.vegf_c0, cfg.tau_burst_vegf, cfg.tau_sustained_vegf),
            (cfg.tgfb_c0, cfg.tau_burst_tgfb, cfg.tau_sustained_tgfb),
            (cfg.egf_c0, cfg.tau_burst_egf, cfg.tau_sustained_egf),
        ]:
            if c0 <= 0 or tau_b <= 0 or tau_s <= 0:
                continue
            # Эффективные скорости с учётом фактора деградации exp(-2t/τ_s)
            k1 = 1.0 / tau_b + 2.0 / tau_s
            k2 = 3.0 / tau_s
            if abs(k1 - k2) < 1e-12:
                t_peak = tau_b / 3.0
            else:
                t_peak = math.log(k1 / k2) / (k1 - k2)
            total += self._biphasic_release(t_peak, c0, tau_b, tau_s)
        return total

    def compute_release(
        self,
        t: float,
        application_time: float = 0.0,
    ) -> PRPReleaseState:
        """Вычисление высвобождения всех 4 факторов роста.

        Вызывает _biphasic_release для каждого фактора (PDGF, VEGF,
        TGF-β, EGF) и формирует PRPReleaseState с theta_total.

        Args:
            t: Текущее время симуляции (ч).
            application_time: Время инъекции PRP (ч). Default = 0.

        Returns:
            PRPReleaseState с концентрациями всех факторов.

        Подробное описание:
            Description/Phase2/description_therapy_models.md#compute_release
        """
        t_rel = t - application_time
        if t_rel < 0:
            return PRPReleaseState()

        cfg = self.config
        theta_pdgf = self._biphasic_release(
            t_rel, cfg.pdgf_c0, cfg.tau_burst_pdgf, cfg.tau_sustained_pdgf
        )
        theta_vegf = self._biphasic_release(
            t_rel, cfg.vegf_c0, cfg.tau_burst_vegf, cfg.tau_sustained_vegf
        )
        theta_tgfb = self._biphasic_release(
            t_rel, cfg.tgfb_c0, cfg.tau_burst_tgfb, cfg.tau_sustained_tgfb
        )
        theta_egf = self._biphasic_release(
            t_rel, cfg.egf_c0, cfg.tau_burst_egf, cfg.tau_sustained_egf
        )

        max_sum = self._compute_max_release_sum()
        raw_total = theta_pdgf + theta_vegf + theta_tgfb + theta_egf
        theta_total = min(1.0, raw_total / max_sum) if max_sum > 0 else 0.0

        return PRPReleaseState(
            theta_pdgf=theta_pdgf,
            theta_vegf=theta_vegf,
            theta_tgfb=theta_tgfb,
            theta_egf=theta_egf,
            theta_total=theta_total,
        )

    def compute_stem_cell_factor(
        self,
        t: float,
        application_time: float = 0.0,
    ) -> float:
        """Фактор рекрутирования стволовых клеток PRP.

        Формула: α_PRP_S · θ_total(t)
        Добавляется в drift-терм уравнения S(t).

        Args:
            t: Текущее время симуляции (ч).
            application_time: Время инъекции PRP (ч). Default = 0.

        Returns:
            Коэффициент рекрутирования >= 0.

        Подробное описание:
            Description/Phase2/description_therapy_models.md#compute_stem_cell_factor
        """
        state = self.compute_release(t, application_time)
        return self.config.alpha_PRP_S * state.theta_total

    def is_active(
        self,
        t: float,
        application_time: float = 0.0,
    ) -> bool:
        """Проверка активности PRP на момент t.

        PRP считается активным, если theta_total > порог (0.01).

        Args:
            t: Текущее время симуляции (ч).
            application_time: Время инъекции PRP (ч). Default = 0.

        Returns:
            True если PRP имеет значимый эффект.

        Подробное описание:
            Description/Phase2/description_therapy_models.md#PRPModel.is_active
        """
        state = self.compute_release(t, application_time)
        return state.theta_total > 0.01


class PEMFModel:
    """Механистическая модель PEMF-терапии.

    Реализует 3 биофизических механизма:
    1. Аденозиновый A₂A/A₃: ε_max · Hill(B) · Gauss(f) → снижение TNF-α
    2. Ca²⁺-CaM/NO: ε_max · Hill(B²) · W(f) → пролиферация F, E
    3. MAPK/ERK: ε_max · Hill(B) → миграция клеток (ABM)

    Каждый механизм зависит от амплитуды B и частоты f
    с характерными частотными окнами.

    Подробное описание:
        Description/Phase2/description_therapy_models.md#PEMFModel
    """

    def __init__(self, config: PEMFConfig | None = None) -> None:
        """Инициализация модели PEMF.

        Args:
            config: Параметры PEMF. None → PEMFConfig() с defaults.

        Подробное описание:
            Description/Phase2/description_therapy_models.md#PEMFModel.__init__
        """
        self.config = config if config is not None else PEMFConfig()

    def compute_anti_inflammatory(self, t: float) -> float:
        """Противовоспалительный эффект через аденозиновый A₂A/A₃ путь.

        Формула: ε_max · (B/B₀)^n / (1 + (B/B₀)^n)
                 · exp(-(f - f_opt)² / (2·σ_f²))

        Эффект: s_TNF_M1 → s_TNF_M1 · (1 - ε) — снижение TNF-α на 30-50%.

        Args:
            t: Текущее время симуляции (ч).

        Returns:
            ε ∈ [0, ε_max] — степень снижения секреции TNF-α.

        Подробное описание:
            Description/Phase2/description_therapy_models.md#compute_anti_inflammatory
        """
        cfg = self.config
        B = cfg.B_amplitude
        if B == 0.0:
            return 0.0

        ratio = B / max(cfg.B0_threshold, 1e-30)
        hill = ratio**cfg.n_B / (1.0 + ratio**cfg.n_B)
        gauss = math.exp(
            -(cfg.frequency - cfg.f_opt_anti_inflam) ** 2
            / (2.0 * cfg.sigma_f_anti_inflam**2)
        )
        return cfg.epsilon_max_anti_inflam * hill * gauss

    def compute_proliferation_boost(self, t: float) -> float:
        """Усиление пролиферации через Ca²⁺-CaM/NO путь.

        Формула: ε_prolif_max · B² / (B_half² + B²)
                 · exp(-(f - f_center)² / (2·σ_window²))

        Эффект: r_F → r_F · (1 + ε), r_E → r_E · (1 + ε).

        Args:
            t: Текущее время симуляции (ч).

        Returns:
            ε ∈ [0, ε_prolif_max] — коэффициент усиления пролиферации.

        Подробное описание:
            Description/Phase2/description_therapy_models.md#compute_proliferation_boost
        """
        cfg = self.config
        B = cfg.B_amplitude
        if B == 0.0:
            return 0.0

        B_sq = B**2
        B_half_sq = cfg.B_half_prolif**2
        hill = B_sq / (B_half_sq + B_sq)
        gauss = math.exp(
            -(cfg.frequency - cfg.f_center_prolif) ** 2
            / (2.0 * cfg.sigma_window_prolif**2)
        )
        return cfg.epsilon_prolif_max * hill * gauss

    def compute_migration_boost(self, t: float) -> float:
        """Усиление миграции клеток через MAPK/ERK путь.

        Формула: ε_migration_max · (B/B₀)^n / (1 + (B/B₀)^n)

        Эффект: D_cell → D_cell · (1 + ε) в ABM модели.

        Args:
            t: Текущее время симуляции (ч).

        Returns:
            ε ∈ [0, ε_migration_max] — коэффициент усиления миграции.

        Подробное описание:
            Description/Phase2/description_therapy_models.md#compute_migration_boost
        """
        cfg = self.config
        B = cfg.B_amplitude
        if B == 0.0:
            return 0.0

        ratio = B / max(cfg.B0_threshold, 1e-30)
        hill = ratio**cfg.n_B / (1.0 + ratio**cfg.n_B)
        return cfg.epsilon_migration_max * hill

    def compute_effects(self, t: float) -> PEMFEffects:
        """Вычисление всех 3 эффектов PEMF на момент t.

        Объединяет результаты compute_anti_inflammatory,
        compute_proliferation_boost и compute_migration_boost.

        Args:
            t: Текущее время симуляции (ч).

        Returns:
            PEMFEffects со всеми тремя модификаторами.

        Подробное описание:
            Description/Phase2/description_therapy_models.md#compute_effects
        """
        return PEMFEffects(
            anti_inflammatory=self.compute_anti_inflammatory(t),
            proliferation=self.compute_proliferation_boost(t),
            migration=self.compute_migration_boost(t),
        )

    def is_active(self, t: float) -> bool:
        """Проверка активности PEMF-сессии на момент t.

        Определяет, активно ли PEMF-воздействие (B_amplitude > 0).

        Args:
            t: Текущее время симуляции (ч).

        Returns:
            True если PEMF активна.

        Подробное описание:
            Description/Phase2/description_therapy_models.md#PEMFModel.is_active
        """
        return self.config.B_amplitude > 0.0


class SynergyModel:
    """Модель синергии PRP+PEMF.

    Супер-аддитивный эффект при одновременном применении PRP и PEMF:
    synergy(t) = 1 + β_synergy · Θ_PRP(t) · PEMF_active(t)

    Биологический механизм: PEMF через Ca²⁺-сигналинг усиливает
    чувствительность рецепторов к факторам роста из PRP.

    Подробное описание:
        Description/Phase2/description_therapy_models.md#SynergyModel
    """

    def __init__(
        self,
        prp_model: PRPModel,
        pemf_model: PEMFModel,
        config: SynergyConfig | None = None,
    ) -> None:
        """Инициализация модели синергии.

        Args:
            prp_model: Модель PRP-терапии.
            pemf_model: Модель PEMF-терапии.
            config: Параметры синергии. None → SynergyConfig() с defaults.

        Подробное описание:
            Description/Phase2/description_therapy_models.md#SynergyModel.__init__
        """
        self.prp_model = prp_model
        self.pemf_model = pemf_model
        self.config = config if config is not None else SynergyConfig()

    def compute_synergy_factor(self, t: float) -> float:
        """Вычисление коэффициента синергии на момент t.

        Формула: synergy = 1 + β_synergy · Θ_PRP(t) · PEMF_active(t)

        Возвращает 1.0 если только одна терапия активна,
        > 1.0 если обе терапии действуют одновременно.

        Args:
            t: Текущее время симуляции (ч).

        Returns:
            Коэффициент синергии >= 1.0.

        Подробное описание:
            Description/Phase2/description_therapy_models.md#compute_synergy_factor
        """
        prp_active = 1.0 if self.prp_model.is_active(t) else 0.0
        pemf_active = 1.0 if self.pemf_model.is_active(t) else 0.0
        return 1.0 + self.config.beta_synergy * prp_active * pemf_active

    def apply_to_drift(self, drift_modifier: float, t: float) -> float:
        """Применение синергии к модификатору drift-терма.

        Формула: result = drift_modifier · synergy_factor(t)

        Args:
            drift_modifier: Исходный модификатор от PRP или PEMF.
            t: Текущее время симуляции (ч).

        Returns:
            Модифицированный drift с учётом синергии.

        Подробное описание:
            Description/Phase2/description_therapy_models.md#apply_to_drift
        """
        return drift_modifier * self.compute_synergy_factor(t)
