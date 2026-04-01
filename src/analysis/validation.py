"""Метрики валидации 20-переменной SDE модели регенерации тканей.

Четыре группы метрик за единым интерфейсом ValidationRunner:
- DTW + CRPS: временные ряды с биологическими фазовыми сдвигами
- ArviZ PPC: байесовская проверка предсказаний (LOO, HDI, envelope)
- Changepoint detection (ruptures.Pelt): автоматическое обнаружение фаз заживления
- Kendall's τ (scipy.stats): ранговая корреляция рейтингов чувствительности

Заменяет «Temporal R²» из раздела 6.2 Mathematical Framework.

Подробное описание: Description/Phase3/description_validation.md
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from loguru import logger
except ImportError:
    import logging as _logging

    logger = _logging.getLogger(__name__)  # type: ignore[assignment]


# =====================================================================
# Dataclasses — результаты отдельных метрик
# =====================================================================


@dataclass
class DTWCRPSResult:
    """Результат DTW + CRPS валидации по временным рядам.

    DTW измеряет расстояние между формами временных рядов, нечувствительное
    к фазовым сдвигам. CRPS оценивает качество вероятностных предсказаний.

    Подробное описание: Description/Phase3/description_validation.md#DTWCRPSResult
    """

    variable_names: list[str]
    dtw_distances: dict[str, float]  # {var: dtw_distance}
    crps_scores: dict[str, float]  # {var: mean_crps}
    mean_dtw: float
    mean_crps: float
    n_observations: int
    elapsed_seconds: float


@dataclass
class PhaseBreakpoint:
    """Точка смены фазы заживления (changepoint).

    Подробное описание: Description/Phase3/description_validation.md#PhaseBreakpoint
    """

    time_hours: float  # Момент смены фазы (ч)
    phase_before: str  # Фаза до смены
    phase_after: str  # Фаза после смены
    confidence: float  # Уверенность [0..1]


@dataclass
class PhaseTimingResult:
    """Результат changepoint detection для определения фаз заживления.

    Подробное описание: Description/Phase3/description_validation.md#PhaseTimingResult
    """

    detected_breakpoints: list[PhaseBreakpoint]
    expected_breakpoints: list[PhaseBreakpoint] | None  # из данных (если заданы)
    timing_mae_hours: float | None  # MAE по времени (если expected задан)
    n_phases_detected: int
    algorithm: str  # "Pelt+BIC"
    elapsed_seconds: float


@dataclass
class RankingComparison:
    """Сравнение ранга одного параметра по Sobol и Morris.

    Подробное описание: Description/Phase3/description_validation.md#RankingComparison
    """

    parameter_name: str
    rank_sobol: int  # Ранг по ST (1 = наиболее влиятельный)
    rank_morris: int  # Ранг по μ* (1 = наиболее влиятельный)


@dataclass
class SensitivityRankingResult:
    """Результат сравнения ранжирования Sobol vs Morris через Kendall's τ.

    Подробное описание: Description/Phase3/description_validation.md#SensitivityRankingResult
    """

    kendall_tau: float  # [-1, 1]
    p_value: float  # [0, 1]
    ranking_comparisons: list[RankingComparison]
    n_parameters: int  # Число общих параметров
    elapsed_seconds: float


@dataclass
class PPCResult:
    """Результат Posterior Predictive Check.

    Два пути: ArviZ (только PyMC) и MC envelope fallback (любые MC данные).

    Подробное описание: Description/Phase3/description_validation.md#PPCResult
    """

    loo_elpd: float | None  # ArviZ LOO ELPD (только ArviZ path)
    loo_se: float | None  # Стандартная ошибка LOO
    coverage_95: dict[str, float]  # {var: доля наблюдений в 95% HDI}
    mean_coverage: float  # Среднее по всем переменным
    backend: str  # "arviz" или "mc_envelope"
    elapsed_seconds: float


@dataclass
class ValidationResult:
    """Агрегированный результат всех метрик валидации.

    Подробное описание: Description/Phase3/description_validation.md#ValidationResult
    """

    dtw_crps: DTWCRPSResult | None
    ppc: PPCResult | None
    phase_timing: PhaseTimingResult | None
    sensitivity_ranking: SensitivityRankingResult | None
    overall_score: float  # Взвешенное среднее [0..1]
    elapsed_seconds: float

    def get_summary(self) -> dict[str, Any]:
        """Краткое резюме всех метрик в виде плоского словаря.

        Returns:
            dict с ключами: overall_score, dtw_crps, ppc, phase_timing,
            sensitivity_ranking и их основными значениями.

        Подробное описание:
            Description/Phase3/description_validation.md#ValidationResult.get_summary
        """
        summary: dict[str, Any] = {
            "overall_score": self.overall_score,
            "elapsed_seconds": self.elapsed_seconds,
        }

        if self.dtw_crps is not None:
            summary["dtw_crps"] = {
                "mean_dtw": self.dtw_crps.mean_dtw,
                "mean_crps": self.dtw_crps.mean_crps,
                "n_observations": self.dtw_crps.n_observations,
            }
        else:
            summary["dtw_crps"] = None

        if self.ppc is not None:
            summary["ppc"] = {
                "mean_coverage": self.ppc.mean_coverage,
                "backend": self.ppc.backend,
                "loo_elpd": self.ppc.loo_elpd,
            }
        else:
            summary["ppc"] = None

        if self.phase_timing is not None:
            summary["phase_timing"] = {
                "n_phases_detected": self.phase_timing.n_phases_detected,
                "timing_mae_hours": self.phase_timing.timing_mae_hours,
                "algorithm": self.phase_timing.algorithm,
            }
        else:
            summary["phase_timing"] = None

        if self.sensitivity_ranking is not None:
            summary["sensitivity_ranking"] = {
                "kendall_tau": self.sensitivity_ranking.kendall_tau,
                "p_value": self.sensitivity_ranking.p_value,
                "n_parameters": self.sensitivity_ranking.n_parameters,
            }
        else:
            summary["sensitivity_ranking"] = None

        return summary


# =====================================================================
# ValidationConfig
# =====================================================================


@dataclass
class ValidationConfig:
    """Единая конфигурация для всех метрик валидации.

    Подробное описание: Description/Phase3/description_validation.md#ValidationConfig
    """

    # Флаги включения метрик
    run_dtw_crps: bool = True
    run_ppc: bool = True
    run_phase_timing: bool = True
    run_sensitivity_ranking: bool = True

    # DTW + CRPS
    dtw_variables: list[str] = field(default_factory=lambda: ["F", "M1", "M2", "Ne"])
    crps_n_samples: int = 1000  # (зарезервировано для ансамблевого CRPS)

    # PPC
    ppc_variables: list[str] = field(default_factory=lambda: ["F", "M1", "M2"])
    hdi_prob: float = 0.95

    # Changepoint
    ruptures_model: str = "rbf"  # модель сигнала для ruptures.Pelt
    ruptures_penalty_scale: float = 1.0  # множитель для BIC-штрафа

    # Веса для overall_score (должны суммироваться в 1.0)
    weight_dtw: float = 0.3
    weight_ppc: float = 0.3
    weight_timing: float = 0.2
    weight_ranking: float = 0.2

    rng_seed: int | None = 42

    def validate(self) -> None:
        """Проверяет, что сумма весов равна 1.0.

        Raises:
            ValueError: Если веса не суммируются в 1.0.

        Подробное описание:
            Description/Phase3/description_validation.md#ValidationConfig.validate
        """
        total = self.weight_dtw + self.weight_ppc + self.weight_timing + self.weight_ranking
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Сумма весов должна быть 1.0, получено {total:.10f}. "
                f"(weight_dtw={self.weight_dtw}, weight_ppc={self.weight_ppc}, "
                f"weight_timing={self.weight_timing}, weight_ranking={self.weight_ranking})"
            )


# =====================================================================
# Вспомогательные функции
# =====================================================================


def _dtw_distance(predicted: np.ndarray, observed: np.ndarray) -> float:
    """DTW-расстояние между двумя временными рядами.

    Lazy import dtaidistance. При отсутствии библиотеки бросает ImportError.
    """
    try:
        from dtaidistance import dtw as _dtw_lib
    except ImportError as e:
        raise ImportError("Установите dtaidistance: pip install dtaidistance>=2.3.0") from e

    pred_f = predicted.astype(np.float64)
    obs_f = observed.astype(np.float64)
    return float(_dtw_lib.distance(pred_f, obs_f))


def _crps_gaussian(observed: np.ndarray, mu: np.ndarray) -> float:
    """CRPS для гауссовского ансамбля (σ = 10% среднего).

    Lazy import properscoring.
    """
    try:
        from properscoring import crps_gaussian as _crps_gauss
    except ImportError as e:
        raise ImportError("Установите properscoring: pip install properscoring>=0.1") from e

    sigma = 0.1 * float(np.abs(mu).mean()) + 1e-6
    sigma_arr = np.full_like(mu, sigma, dtype=np.float64)
    values = _crps_gauss(observed.astype(np.float64), mu.astype(np.float64), sigma_arr)
    return float(np.mean(values))


def _bic_penalty(signal: np.ndarray, scale: float = 1.0) -> float:
    """BIC-вдохновлённый штраф для ruptures.Pelt.

    pen = n_features * log(n_samples) * scale
    """
    n_samples = signal.shape[0]
    n_features = signal.shape[1] if signal.ndim > 1 else 1
    return float(n_features * np.log(max(n_samples, 2)) * scale)


def _compute_coverage(
    observed: np.ndarray,
    obs_times: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    pred_times: np.ndarray,
) -> float:
    """Доля наблюдений, попадающих в интервал [lower, upper].

    lower/upper интерполируются на obs_times из pred_times.
    """
    lo_interp = np.interp(obs_times, pred_times, lower)
    hi_interp = np.interp(obs_times, pred_times, upper)
    inside = np.sum((observed >= lo_interp) & (observed <= hi_interp))
    return float(inside) / max(len(observed), 1)


# =====================================================================
# ValidationRunner
# =====================================================================


class ValidationRunner:
    """Запускает метрики валидации для SDE-модели регенерации.

    Паттерн: config → runner.run_*() → @dataclass result.

    Подробное описание: Description/Phase3/description_validation.md#ValidationRunner
    """

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self._config = config or ValidationConfig()

    @property
    def config(self) -> ValidationConfig:
        return self._config

    # ------------------------------------------------------------------
    # DTW + CRPS
    # ------------------------------------------------------------------

    def run_dtw_crps(
        self,
        trajectory: Any,  # ExtendedSDETrajectory (ленивый импорт)
        observed: Any,  # TimeSeriesData
    ) -> DTWCRPSResult:
        """DTW-расстояние и CRPS для каждой переменной из config.dtw_variables.

        Пропускает переменные, отсутствующие в observed.values.

        Args:
            trajectory: ExtendedSDETrajectory с методом get_variable(name)
            observed: TimeSeriesData с time_points и values

        Returns:
            DTWCRPSResult

        Подробное описание:
            Description/Phase3/description_validation.md#ValidationRunner.run_dtw_crps
        """
        t0 = time.perf_counter()

        dtw_distances: dict[str, float] = {}
        crps_scores: dict[str, float] = {}
        n_obs = int(len(observed.time_points))

        for var in self._config.dtw_variables:
            # Пропускаем переменные, которых нет в наблюдениях
            if var not in observed.values:
                continue

            obs_arr = observed.values[var].astype(np.float64)
            obs_times = observed.time_points

            # Извлечь и интерполировать траекторию
            pred_raw = trajectory.get_variable(var)
            pred_interp = np.interp(obs_times, trajectory.times, pred_raw).astype(np.float64)

            dtw_distances[var] = _dtw_distance(pred_interp, obs_arr)
            crps_scores[var] = _crps_gaussian(obs_arr, pred_interp)

        var_names = list(dtw_distances.keys())
        mean_dtw = float(np.mean(list(dtw_distances.values()))) if dtw_distances else 0.0
        mean_crps = float(np.mean(list(crps_scores.values()))) if crps_scores else 0.0

        return DTWCRPSResult(
            variable_names=var_names,
            dtw_distances=dtw_distances,
            crps_scores=crps_scores,
            mean_dtw=mean_dtw,
            mean_crps=mean_crps,
            n_observations=n_obs,
            elapsed_seconds=time.perf_counter() - t0,
        )

    # ------------------------------------------------------------------
    # PPC
    # ------------------------------------------------------------------

    def run_ppc(
        self,
        mc_results: Any,  # MonteCarloResults
        observed: Any,  # TimeSeriesData
        estimation_result: Any | None = None,  # EstimationResult
    ) -> PPCResult:
        """Posterior Predictive Check.

        ArviZ path: estimation_result.inference_data is not None
                    и содержит posterior_predictive.
        MC envelope fallback: использует variable_quantiles из MonteCarloResults.

        Args:
            mc_results: MonteCarloResults с variable_quantiles
            observed: TimeSeriesData с наблюдениями
            estimation_result: EstimationResult (опционально, для ArviZ path)

        Returns:
            PPCResult

        Подробное описание:
            Description/Phase3/description_validation.md#ValidationRunner.run_ppc
        """
        t0 = time.perf_counter()

        # Попытаться ArviZ path
        idata = estimation_result.inference_data if estimation_result is not None else None
        use_arviz = idata is not None and hasattr(idata, "posterior_predictive")

        if use_arviz:
            return self._run_ppc_arviz(idata, observed, t0)
        return self._run_ppc_mc_envelope(mc_results, observed, t0)

    def _run_ppc_arviz(
        self,
        idata: Any,
        observed: Any,
        t0: float,
    ) -> PPCResult:
        """ArviZ path: LOO + HDI coverage."""
        try:
            import arviz as az
        except ImportError as e:
            raise ImportError("pip install arviz>=0.17.0") from e

        loo_result = az.loo(idata, pointwise=False)
        loo_elpd = float(loo_result.elpd_loo)
        loo_se = float(loo_result.se)

        coverage_95: dict[str, float] = {}
        for var in self._config.ppc_variables:
            if var not in observed.values:
                continue
            try:
                pp = idata.posterior_predictive[var].values.reshape(-1, -1)
                hdi_bounds = az.hdi(pp, hdi_prob=self._config.hdi_prob)
                obs_arr = observed.values[var]
                lo = hdi_bounds[..., 0]
                hi = hdi_bounds[..., 1]
                coverage_95[var] = _compute_coverage(
                    obs_arr, observed.time_points, lo, hi, observed.time_points
                )
            except Exception:
                pass

        mean_cov = float(np.mean(list(coverage_95.values()))) if coverage_95 else 0.0
        return PPCResult(
            loo_elpd=loo_elpd,
            loo_se=loo_se,
            coverage_95=coverage_95,
            mean_coverage=mean_cov,
            backend="arviz",
            elapsed_seconds=time.perf_counter() - t0,
        )

    def _run_ppc_mc_envelope(
        self,
        mc_results: Any,
        observed: Any,
        t0: float,
    ) -> PPCResult:
        """MC envelope fallback: coverage через quantile_quantiles."""
        coverage_95: dict[str, float] = {}
        pred_times = mc_results.times

        alpha = 1.0 - self._config.hdi_prob
        q_lo = alpha / 2.0
        q_hi = 1.0 - alpha / 2.0

        for var in self._config.ppc_variables:
            if var not in observed.values:
                continue

            obs_arr = observed.values[var]
            obs_times = observed.time_points

            vq = mc_results.variable_quantiles.get(var)
            if vq:
                # Ищем ближайшие доступные квантили
                available = sorted(vq.keys())
                lo_key = min(available, key=lambda q: abs(q - q_lo))
                hi_key = min(available, key=lambda q: abs(q - q_hi))
                lo_arr = vq[lo_key]
                hi_arr = vq[hi_key]
            else:
                # Gaussian approximation через mean/std
                mean_var = mc_results.variable_means.get(var, mc_results.mean_N)
                std_var = mc_results.variable_stds.get(var, mc_results.std_N)
                z = float(np.abs(np.quantile(np.array([0.0, 1.0]), q_hi) - 0.5) * 2)
                # z ≈ 1.96 для 0.95
                z = 1.96
                lo_arr = mean_var - z * std_var
                hi_arr = mean_var + z * std_var

            coverage_95[var] = _compute_coverage(obs_arr, obs_times, lo_arr, hi_arr, pred_times)

        mean_cov = float(np.mean(list(coverage_95.values()))) if coverage_95 else 0.0
        return PPCResult(
            loo_elpd=None,
            loo_se=None,
            coverage_95=coverage_95,
            mean_coverage=mean_cov,
            backend="mc_envelope",
            elapsed_seconds=time.perf_counter() - t0,
        )

    # ------------------------------------------------------------------
    # Phase timing (ruptures changepoint detection)
    # ------------------------------------------------------------------

    def run_phase_timing(
        self,
        trajectory: Any,  # ExtendedSDETrajectory
        observed_breakpoints: list[PhaseBreakpoint] | None = None,
    ) -> PhaseTimingResult:
        """Обнаружение фаз заживления через ruptures.Pelt.

        Стек переменных: Ne, M1, M2, F. Каждая точка разрыва конвертируется
        из индекса в часы через trajectory.times.

        Args:
            trajectory: ExtendedSDETrajectory
            observed_breakpoints: опциональные эталонные точки разрыва
                                  для вычисления timing_mae_hours

        Returns:
            PhaseTimingResult

        Подробное описание:
            Description/Phase3/description_validation.md#ValidationRunner.run_phase_timing
        """
        t0 = time.perf_counter()

        try:
            import ruptures as rpt
        except ImportError as e:
            raise ImportError("pip install ruptures>=1.1.0") from e

        # Собрать многомерный сигнал: Ne, M1, M2, F
        signal_vars = [v for v in ["Ne", "M1", "M2", "F"] if len(trajectory.get_variable(v)) > 0]
        cols = [trajectory.get_variable(v).astype(np.float64) for v in signal_vars]

        if not cols or len(cols[0]) < 4:
            return PhaseTimingResult(
                detected_breakpoints=[],
                expected_breakpoints=observed_breakpoints,
                timing_mae_hours=None,
                n_phases_detected=0,
                algorithm="Pelt+BIC",
                elapsed_seconds=time.perf_counter() - t0,
            )

        signal = np.column_stack(cols)

        # Нормализация: предотвращает доминирование одной переменной
        std = signal.std(axis=0)
        std[std == 0] = 1.0
        signal = (signal - signal.mean(axis=0)) / std

        pen = _bic_penalty(signal, scale=self._config.ruptures_penalty_scale)

        try:
            algo = rpt.Pelt(model=self._config.ruptures_model).fit(signal)
            bkpt_indices = algo.predict(pen=pen)
        except Exception:
            bkpt_indices = []

        # Последний элемент ruptures всегда = n_samples (не точка разрыва)
        bkpt_indices = [i for i in bkpt_indices if i < len(trajectory.times)]

        phase_labels = ["hemostasis", "inflammation", "proliferation", "remodeling"]
        detected: list[PhaseBreakpoint] = []
        for idx, bkpt_idx in enumerate(bkpt_indices):
            t_hours = float(trajectory.times[min(bkpt_idx, len(trajectory.times) - 1)])
            before = phase_labels[min(idx, len(phase_labels) - 1)]
            after = phase_labels[min(idx + 1, len(phase_labels) - 1)]
            detected.append(
                PhaseBreakpoint(
                    time_hours=t_hours,
                    phase_before=before,
                    phase_after=after,
                    confidence=0.7,  # ruptures не даёт вероятности → фиксированное значение
                )
            )

        # Вычислить timing MAE если заданы expected_breakpoints
        timing_mae: float | None = None
        if observed_breakpoints and detected:
            errors = []
            for exp_bp in observed_breakpoints:
                if detected:
                    nearest = min(detected, key=lambda d: abs(d.time_hours - exp_bp.time_hours))
                    errors.append(abs(nearest.time_hours - exp_bp.time_hours))
            timing_mae = float(np.mean(errors)) if errors else None

        return PhaseTimingResult(
            detected_breakpoints=detected,
            expected_breakpoints=observed_breakpoints,
            timing_mae_hours=timing_mae,
            n_phases_detected=len(detected),
            algorithm="Pelt+BIC",
            elapsed_seconds=time.perf_counter() - t0,
        )

    # ------------------------------------------------------------------
    # Sensitivity ranking (Kendall's τ)
    # ------------------------------------------------------------------

    def run_sensitivity_ranking(
        self,
        sobol: Any,  # SobolResult
        morris: Any,  # MorrisResult
    ) -> SensitivityRankingResult:
        """Kendall's τ между рейтингами Sobol (ST) и Morris (μ*).

        Сравниваются только общие параметры (пересечение parameter_names).

        Args:
            sobol: SobolResult с полями parameter_names, ST
            morris: MorrisResult с полями parameter_names, mu_star

        Returns:
            SensitivityRankingResult

        Подробное описание:
            Description/Phase3/description_validation.md#ValidationRunner.run_sensitivity_ranking
        """
        from scipy.stats import kendalltau  # scipy уже в зависимостях

        t0 = time.perf_counter()

        # Найти пересечение
        sobol_names = sobol.parameter_names
        morris_names = morris.parameter_names
        common = [n for n in sobol_names if n in morris_names]

        if len(common) < 2:
            # Kendall's τ недоопределён при < 2 точках
            comparisons = [RankingComparison(n, rank_sobol=1, rank_morris=1) for n in common]
            return SensitivityRankingResult(
                kendall_tau=float("nan"),
                p_value=1.0,
                ranking_comparisons=comparisons,
                n_parameters=len(common),
                elapsed_seconds=time.perf_counter() - t0,
            )

        # Ранги Sobol: индексы в порядке убывания ST (ранг 1 = наибольший ST)
        sobol_idx = {n: i for i, n in enumerate(sobol_names)}
        sobol_st = np.array([sobol.ST[sobol_idx[n]] for n in common])
        sobol_rank_order = np.argsort(-sobol_st)  # убывание
        sobol_ranks = np.empty(len(common), dtype=int)
        sobol_ranks[sobol_rank_order] = np.arange(1, len(common) + 1)

        # Ранги Morris: индексы в порядке убывания mu_star (ранг 1 = наибольший)
        morris_idx = {n: i for i, n in enumerate(morris_names)}
        morris_mu = np.array([morris.mu_star[morris_idx[n]] for n in common])
        morris_rank_order = np.argsort(-morris_mu)  # убывание
        morris_ranks = np.empty(len(common), dtype=int)
        morris_ranks[morris_rank_order] = np.arange(1, len(common) + 1)

        kt_result = kendalltau(sobol_ranks, morris_ranks)
        tau = float(kt_result[0])
        p_val = float(kt_result[1])

        comparisons = [
            RankingComparison(
                parameter_name=n,
                rank_sobol=int(sobol_ranks[i]),
                rank_morris=int(morris_ranks[i]),
            )
            for i, n in enumerate(common)
        ]

        return SensitivityRankingResult(
            kendall_tau=tau,
            p_value=p_val,
            ranking_comparisons=comparisons,
            n_parameters=len(common),
            elapsed_seconds=time.perf_counter() - t0,
        )

    # ------------------------------------------------------------------
    # run_all
    # ------------------------------------------------------------------

    def run_all(
        self,
        trajectory: Any | None = None,
        mc_results: Any | None = None,
        observed: Any | None = None,
        estimation_result: Any | None = None,
        sobol: Any | None = None,
        morris: Any | None = None,
        observed_breakpoints: list[PhaseBreakpoint] | None = None,
    ) -> ValidationResult:
        """Запустить все включённые метрики и агрегировать результат.

        Метрика пропускается, если:
        - она отключена в config (run_* = False)
        - не переданы необходимые данные

        overall_score вычисляется как взвешенное среднее по доступным метрикам
        (веса ренормализуются, если часть метрик недоступна).

        Args:
            trajectory: ExtendedSDETrajectory (для DTW/CRPS и phase_timing)
            mc_results: MonteCarloResults (для PPC MC fallback)
            observed: TimeSeriesData (для DTW/CRPS и PPC)
            estimation_result: EstimationResult (для ArviZ PPC path)
            sobol: SobolResult (для ranking)
            morris: MorrisResult (для ranking)
            observed_breakpoints: эталонные точки разрыва (для phase_timing MAE)

        Returns:
            ValidationResult

        Подробное описание:
            Description/Phase3/description_validation.md#ValidationRunner.run_all
        """
        t0 = time.perf_counter()
        cfg = self._config

        dtw_crps: DTWCRPSResult | None = None
        ppc: PPCResult | None = None
        phase_timing: PhaseTimingResult | None = None
        sensitivity_ranking: SensitivityRankingResult | None = None

        if cfg.run_dtw_crps and trajectory is not None and observed is not None:
            try:
                dtw_crps = self.run_dtw_crps(trajectory, observed)
            except Exception as exc:
                logger.warning(f"run_dtw_crps failed: {exc}")

        if cfg.run_ppc and mc_results is not None and observed is not None:
            try:
                ppc = self.run_ppc(mc_results, observed, estimation_result)
            except Exception as exc:
                logger.warning(f"run_ppc failed: {exc}")

        if cfg.run_phase_timing and trajectory is not None:
            try:
                phase_timing = self.run_phase_timing(trajectory, observed_breakpoints)
            except Exception as exc:
                logger.warning(f"run_phase_timing failed: {exc}")

        if cfg.run_sensitivity_ranking and sobol is not None and morris is not None:
            try:
                sensitivity_ranking = self.run_sensitivity_ranking(sobol, morris)
            except Exception as exc:
                logger.warning(f"run_sensitivity_ranking failed: {exc}")

        overall_score = _compute_overall_score(
            cfg, dtw_crps, ppc, phase_timing, sensitivity_ranking
        )

        return ValidationResult(
            dtw_crps=dtw_crps,
            ppc=ppc,
            phase_timing=phase_timing,
            sensitivity_ranking=sensitivity_ranking,
            overall_score=overall_score,
            elapsed_seconds=time.perf_counter() - t0,
        )


# =====================================================================
# Вычисление overall_score
# =====================================================================


def _compute_overall_score(
    cfg: ValidationConfig,
    dtw_crps: DTWCRPSResult | None,
    ppc: PPCResult | None,
    phase_timing: PhaseTimingResult | None,
    sensitivity_ranking: SensitivityRankingResult | None,
) -> float:
    """Взвешенное среднее нормализованных метрик.

    Каждая метрика конвертируется в score ∈ [0, 1]:
    - DTW/CRPS: exp(-mean_dtw / scale) (ближе к 0 → лучше → score ближе к 1)
    - PPC: mean_coverage ∈ [0, 1]
    - Phase timing: exp(-timing_mae / scale) (нет MAE → 0.5)
    - Sensitivity ranking: (kendall_tau + 1) / 2 ∈ [0, 1]

    Веса недоступных метрик ренормализуются.
    """
    scores: dict[str, float] = {}
    weights: dict[str, float] = {}

    # DTW/CRPS → нормализованный score через экспоненту
    if dtw_crps is not None:
        dtw_scale = max(dtw_crps.mean_dtw, 1.0)
        dtw_score = float(np.exp(-1.0))  # нормируем на "1 scale unit" → 0.37
        # Лучший score когда DTW мало
        dtw_score = float(np.exp(-dtw_crps.mean_dtw / dtw_scale))
        scores["dtw"] = max(0.0, min(1.0, dtw_score))
        weights["dtw"] = cfg.weight_dtw

    # PPC → mean_coverage напрямую
    if ppc is not None:
        scores["ppc"] = max(0.0, min(1.0, ppc.mean_coverage))
        weights["ppc"] = cfg.weight_ppc

    # Phase timing → экспоненциальный штраф за ошибку (или 0.5 без MAE)
    if phase_timing is not None:
        if phase_timing.timing_mae_hours is not None:
            scale_h = 24.0  # один день
            scores["timing"] = float(np.exp(-phase_timing.timing_mae_hours / scale_h))
        else:
            scores["timing"] = 0.5  # нет эталона → нейтральный score
        scores["timing"] = max(0.0, min(1.0, scores["timing"]))
        weights["timing"] = cfg.weight_timing

    # Sensitivity ranking → τ ∈ [-1,1] → (τ+1)/2 ∈ [0,1]
    if sensitivity_ranking is not None:
        tau = sensitivity_ranking.kendall_tau
        if not np.isnan(tau):
            scores["ranking"] = (tau + 1.0) / 2.0
        else:
            scores["ranking"] = 0.5
        scores["ranking"] = max(0.0, min(1.0, scores["ranking"]))
        weights["ranking"] = cfg.weight_ranking

    if not scores:
        return 0.0

    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0

    overall = sum(scores[k] * weights[k] for k in scores) / total_weight
    return max(0.0, min(1.0, overall))


# =====================================================================
# Convenience function
# =====================================================================


def validate_model(
    trajectory: Any,  # ExtendedSDETrajectory
    observed: Any,  # TimeSeriesData
    *,
    mc_results: Any | None = None,  # MonteCarloResults
    estimation_result: Any | None = None,  # EstimationResult
    sobol: Any | None = None,  # SobolResult
    morris: Any | None = None,  # MorrisResult
    config: ValidationConfig | None = None,
) -> ValidationResult:
    """Валидация SDE-модели по всем включённым метрикам.

    Удобная обёртка над ValidationRunner.run_all().

    Args:
        trajectory: ExtendedSDETrajectory (обязательно)
        observed: TimeSeriesData с наблюдениями (обязательно)
        mc_results: MonteCarloResults для PPC (опционально)
        estimation_result: EstimationResult для ArviZ PPC (опционально)
        sobol: SobolResult для sensitivity ranking (опционально)
        morris: MorrisResult для sensitivity ranking (опционально)
        config: ValidationConfig (по умолчанию создаётся с параметрами по умолчанию)

    Returns:
        ValidationResult с overall_score и компонентами

    Подробное описание:
        Description/Phase3/description_validation.md#validate_model
    """
    runner = ValidationRunner(config)
    return runner.run_all(
        trajectory=trajectory,
        mc_results=mc_results,
        observed=observed,
        estimation_result=estimation_result,
        sobol=sobol,
        morris=morris,
    )
