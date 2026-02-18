# RegenTwin: Полная Математическая Модель Регенерации Тканей

## Мультимасштабный фреймворк: система связанных SDE + Agent-Based модель

---

## 0. Позиционирование текущей реализации как MVP

Текущая модель RegenTwin использует 2-компонентную SDE:

```
dN = [rN(1 - N/K) + αf(PRP) + βg(PEMF) - δN]dt + σ_n·N·dW₁
dC = [ηN - γC + S_PRP(t)]dt + σ_c·C·dW₂
```

Где N — единая "плотность клеток", C — единая "концентрация цитокинов".

**Почему это валидный MVP:**
- Логистический рост — стандартная отправная точка для моделей популяционной динамики [Verhulst 1838, Murray 2002 "Mathematical Biology"]
- Euler-Maruyama — корректный численный метод для SDE [Kloeden & Platen 1992]
- ABM с 3 типами агентов покрывает ключевые клеточные типы
- Operator splitting для SDE↔ABM — устоявшийся метод мультимасштабной интеграции [Strang 1968]

**Что упрощено и почему это проблема для публикации:**

| Упрощение | Реальная биология | Влияние на предсказания |
|-----------|-------------------|------------------------|
| Одна переменная N(t) | 6-10 различных клеточных популяций с разной кинетикой | Невозможно предсказать фазовые переходы заживления |
| Один "цитокин" C(t) | >20 сигнальных молекул, часто с противоположными эффектами | Нельзя отделить воспаление от пролиферации |
| Нет ECM динамики | Коллаген, фибрин, MMPs — ключевые для исхода заживления | Нет предсказания качества ткани (рубец vs регенерация) |
| PRP = экспоненциальное затухание | PRP содержит >300 белков с разной кинетикой | Грубое приближение терапевтического эффекта |
| PEMF = сигмоида от частоты | Механотрансдукция через Ca²⁺, аденозин, MAPK/ERK | Нет механистического обоснования |
| Нет фаз заживления | 4 перекрывающиеся фазы с разной динамикой | Модель не различает стадии |

Ниже — полная модель, устраняющая эти упрощения.

---

## 1. Биологическая основа: фазы заживления раны

Регенерация ткани проходит через 4 перекрывающиеся фазы [Gurtner et al., Nature 2008; Eming et al., Science Translational Medicine 2014]:

**Фаза I — Гемостаз (0–6 часов):**
Тромбоциты агрегируются, формируя фибриновый сгусток. Дегрануляция тромбоцитов высвобождает PDGF, TGF-β, VEGF.

**Фаза II — Воспаление (6 ч – 4-6 дней):**
Нейтрофилы мигрируют первыми (хемотаксис по градиенту IL-8). Моноциты дифференцируются в макрофаги. Макрофаги поляризуются M1 (провоспалительный) → M2 (противовоспалительный) переключение.

**Фаза III — Пролиферация (4-21 день):**
Фибробласты мигрируют и пролиферируют (стимуляция PDGF, TGF-β). Ангиогенез (VEGF-зависимый). Формирование грануляционной ткани. Стволовые клетки дифференцируются.

**Фаза IV — Ремоделирование (21 день – 1 год):**
Коллаген III → коллаген I замещение. MMP/TIMP баланс. Апоптоз миофибробластов. Созревание рубца.

---

## 2. Полная система уравнений

### 2.1. Клеточные популяции

Обозначения:
- **P(t)** — тромбоциты (platelets), клеток/мкл
- **Nₑ(t)** — нейтрофилы, клеток/мкл  
- **M₁(t)** — M1 макрофаги (провоспалительные), клеток/мкл
- **M₂(t)** — M2 макрофаги (репаративные), клеток/мкл
- **F(t)** — фибробласты, клеток/мкл
- **Mf(t)** — миофибробласты, клеток/мкл
- **E(t)** — эндотелиальные клетки, клеток/мкл
- **S(t)** — стволовые/прогениторные клетки (CD34+), клеток/мкл

---

#### 2.1.1. Тромбоциты P(t)

```
dP = [S_P(t) - δ_P · P - k_deg · P]dt + σ_P · P · dW_P
```

Где:
- **S_P(t)** — источник тромбоцитов: S_P(t) = P_max · exp(-t/τ_P) — быстрая активация в первые часы после повреждения
- **δ_P** — скорость естественного клиренса ≈ 0.1 ч⁻¹ (время жизни тромбоцита в ране ~8-10 ч)
- **k_deg** — скорость дегрануляции (высвобождение PDGF, TGF-β, VEGF)
- **P_max** — максимальная концентрация при активации
- **τ_P** — временная константа активации ≈ 1-2 ч

**Обоснование:** Тромбоциты — первый ответ на повреждение. Их кинетика хорошо описывается экспоненциальным затуханием после начального пика [Nurden et al., Blood Reviews 2008]. Дегрануляция α-гранул высвобождает >300 биоактивных молекул [Blair & Flaumenhaft, Blood Reviews 2009].

---

#### 2.1.2. Нейтрофилы Nₑ(t)

```
dNₑ = [R_Ne(C_IL8) - δ_Ne · Nₑ - k_phag · M_total · Nₑ/(Nₑ + K_phag)]dt + σ_Ne · Nₑ · dW_Ne
```

Где:
- **R_Ne(C_IL8)** — рекрутирование: R_Ne = R_Ne_max · C_IL8ⁿ / (K_IL8ⁿ + C_IL8ⁿ) — Hill-функция хемотаксиса
- **n** — коэффициент Хилла ≈ 2 (кооперативность рецепторного связывания)
- **δ_Ne** — скорость апоптоза нейтрофилов ≈ 0.05 ч⁻¹ (t₁/₂ ≈ 12-14 ч в ткани)
- **k_phag · M_total · Nₑ/(Nₑ + K_phag)** — фагоцитоз апоптотических нейтрофилов макрофагами (кинетика Михаэлиса-Ментен)
- **M_total = M₁ + M₂** — общее количество макрофагов

**Обоснование:** Нейтрофилы рекрутируются по градиенту IL-8/CXCL8 в первые 24-48 ч [Kolaczkowska & Kubes, Nature Reviews Immunology 2013]. Фагоцитоз апоптотических нейтрофилов (эффероцитоз) — критический триггер для M1→M2 переключения макрофагов [Serhan & Savill, Nature Immunology 2005]. Кинетика Михаэлиса-Ментен стандартна для рецептор-опосредованных процессов.

---

#### 2.1.3. Макрофаги: M1 (провоспалительные)

```
dM₁ = [R_M(C_MCP1) · φ₁(C_TNF, C_IL10) - k_switch · ψ(C_IL10, C_TGFβ) · M₁ + k_reverse · ζ(C_TNF) · M₂ - δ_M · M₁]dt + σ_M · M₁ · dW_M1
```

Где:
- **R_M(C_MCP1)** — рекрутирование моноцитов: R_M = R_M_max · C_MCP1 / (K_MCP1 + C_MCP1)
- **φ₁(C_TNF, C_IL10)** — доля поляризации в M1: φ₁ = C_TNF / (C_TNF + C_IL10 + ε)
  - ε — малая константа для предотвращения деления на 0
- **k_switch** — скорость M1→M2 переключения ≈ 0.02 ч⁻¹
- **ψ(C_IL10, C_TGFβ)** — функция переключения: ψ = (C_IL10 + C_TGFβ)ⁿ / (K_switchⁿ + (C_IL10 + C_TGFβ)ⁿ)
- **k_reverse** — скорость обратного M2→M1 ≈ 0.005 ч⁻¹ (медленнее прямого)
- **ζ(C_TNF)** — обратное переключение при высоком TNF: ζ = C_TNF² / (K_reverse² + C_TNF²)
- **δ_M** — скорость апоптоза ≈ 0.01 ч⁻¹ (t₁/₂ ≈ 3-5 дней)

**Обоснование:** M1/M2 — это спектр поляризации, но бинарная модель с переключением хорошо описывает макроскопическую динамику [Murray, Annual Review Physiology 2017]. MCP-1/CCL2 — основной хемоаттрактант для моноцитов [Deshmane et al., Journal of Interferon & Cytokine Research 2009]. TNF-α поддерживает M1 состояние, IL-10 и TGF-β управляют M2 переключением [Mantovani et al., Trends in Immunology 2004].

---

#### 2.1.4. Макрофаги: M2 (репаративные)

```
dM₂ = [R_M(C_MCP1) · φ₂(C_TNF, C_IL10) + k_switch · ψ(C_IL10, C_TGFβ) · M₁ - k_reverse · ζ(C_TNF) · M₂ - δ_M · M₂]dt + σ_M · M₂ · dW_M2
```

Где:
- **φ₂ = 1 - φ₁** — доля поляризации в M2 (баланс с M1)
- Остальные параметры идентичны M1 (симметрия переключения)

**Обоснование:** M2 макрофаги продуцируют IL-10, TGF-β, VEGF — факторы, необходимые для перехода от воспаления к пролиферации. Нарушение M1→M2 переключения → хроническая рана [Sindrilaru et al., Journal of Clinical Investigation 2011].

---

#### 2.1.5. Фибробласты F(t)

```
dF = [r_F · F · (1 - (F + Mf)/K_F) · H(C_PDGF, C_TGFβ) + k_diff_S · S · g_diff(C_TGFβ) - k_act · F · A(C_TGFβ) - δ_F · F]dt + σ_F · F · dW_F
```

Где:
- **r_F** — максимальная скорость пролиферации ≈ 0.03 ч⁻¹ (время удвоения ~24 ч)
- **(1 - (F + Mf)/K_F)** — контактное ингибирование (общая carrying capacity для F + Mf)
- **H(C_PDGF, C_TGFβ)** — митогенная стимуляция:
  ```
  H = (C_PDGF / (K_PDGF + C_PDGF)) · (1 + α_TGF · C_TGFβ / (K_TGFβ_prolif + C_TGFβ))
  ```
  PDGF — основной митоген для фибробластов, TGF-β усиливает (но не инициирует) пролиферацию
- **k_diff_S · S · g_diff** — приток из дифференциации стволовых клеток
- **k_act · F · A(C_TGFβ)** — потеря за счёт активации в миофибробласты
  ```
  A(C_TGFβ) = C_TGFβ² / (K_activ² + C_TGFβ²)
  ```
  (Hill n=2: кооперативность TGF-β рецепторов Smad2/3)
- **δ_F** — скорость апоптоза ≈ 0.003 ч⁻¹ (t₁/₂ ~ 10 дней)

**Обоснование:** PDGF — первичный митоген для фибробластов [Heldin & Westermark, Physiological Reviews 1999]. TGF-β в высоких концентрациях активирует Smad2/3 путь, вызывая α-SMA экспрессию и трансформацию в миофибробласт [Hinz et al., American Journal of Pathology 2007]. Контактное ингибирование описывается общей carrying capacity [Vodovotz et al., PLoS Computational Biology 2006].

---

#### 2.1.6. Миофибробласты Mf(t)

```
dMf = [k_act · F · A(C_TGFβ) - δ_Mf · Mf · (1 - C_TGFβ/(K_survival + C_TGFβ))]dt + σ_Mf · Mf · dW_Mf
```

Где:
- **k_act · F · A(C_TGFβ)** — приток из активации фибробластов (тот же терм что и потеря в ур. F)
- **δ_Mf · (1 - C_TGFβ/(K_survival + C_TGFβ))** — апоптоз, подавляемый TGF-β:
  - При высоком TGF-β → апоптоз замедляется (TGF-β — сигнал выживания для миофибробластов)
  - При снижении TGF-β → массовый апоптоз (критически для разрешения фиброза)

**Обоснование:** Миофибробласты — ключевые эффекторы ремоделирования и потенциальные драйверы фиброза. Их апоптоз при снижении механического стресса и TGF-β — основной механизм разрешения фиброза [Desmouliere et al., International Journal of Biochemistry & Cell Biology 2005]. Нарушение этого процесса → патологический рубец/келоид.

---

#### 2.1.7. Эндотелиальные клетки E(t) — Ангиогенез

```
dE = [r_E · E · (1 - E/K_E) · V(C_VEGF) · (1 - θ_hypoxia(O₂)) - δ_E · E]dt + σ_E · E · dW_E
```

Где:
- **r_E** — скорость пролиферации ≈ 0.02 ч⁻¹
- **V(C_VEGF)** — VEGF-зависимая активация:
  ```
  V = C_VEGF² / (K_VEGF² + C_VEGF²)
  ```
  (Hill n=2: VEGFR2 димеризация)
- **θ_hypoxia(O₂)** — гипоксический стимул: θ = O₂/(K_O2 + O₂)
  - При низком O₂ → θ мало → (1-θ) велико → стимуляция ангиогенеза
- **K_E** — carrying capacity (определяется плотностью капилляров)

**Обоснование:** Ангиогенез критически зависит от VEGF (через VEGFR2/KDR) и гипоксии (через HIF-1α) [Ferrara et al., Nature Medicine 2003]. Модель Anderson & Chaplain [Bulletin of Mathematical Biology 1998] — классическая референция для математического описания. Hill n=2 обоснован димеризацией VEGFR2.

---

#### 2.1.8. Стволовые / прогениторные клетки S(t)

```
dS = [r_S · S · (1 - S/K_S) · (1 + α_PRP_S · Θ_PRP(t)) - k_diff_S · S · g_diff(C_TGFβ) - δ_S · S]dt + σ_S · S · dW_S
```

Где:
- **r_S** — скорость самообновления ≈ 0.01 ч⁻¹ (медленная пролиферация)
- **Θ_PRP(t)** — эффект PRP на мобилизацию стволовых клеток (см. раздел 4)
- **g_diff(C_TGFβ)** — вероятность дифференциации:
  ```
  g_diff = C_TGFβ / (K_diff + C_TGFβ)
  ```
- **δ_S** — скорость апоптоза/истощения ≈ 0.005 ч⁻¹

**Обоснование:** CD34+ клетки в контексте регенерации кожи обладают ограниченной пролиферацией и дифференцируются преимущественно в фибробласты и эндотелиальные клетки [Badiavas & Falanga, Archives of Dermatology 2003]. PRP стимулирует мобилизацию через SDF-1/CXCR4 ось [Ranzato et al., Journal of Cellular and Molecular Medicine 2009].

---

### 2.2. Сигнальные молекулы (цитокины и факторы роста)

Обозначения: все концентрации в нг/мл. Каждая молекула: продукция клетками — деградация — потребление рецепторами.

---

#### 2.2.1. TNF-α (провоспалительный цитокин)

```
dC_TNF = [s_TNF_M1 · M₁ + s_TNF_Ne · Nₑ - γ_TNF · C_TNF - k_inhib_IL10 · C_IL10 · C_TNF/(K_inhib + C_TNF)]dt + σ_TNF · C_TNF · dW_TNF
```

Где:
- **s_TNF_M1** — секреция M1 макрофагами ≈ 0.01 нг/(мл·клетка·ч)
- **s_TNF_Ne** — секреция нейтрофилами ≈ 0.005
- **γ_TNF** — деградация ≈ 0.5 ч⁻¹ (t₁/₂ ≈ 20 мин в плазме, ~1-2 ч в ткани)
- **k_inhib_IL10 · C_IL10** — ингибирование продукции TNF-α через IL-10 (отрицательная обратная связь)

**Обоснование:** TNF-α — центральный медиатор воспалительной фазы, продуцируется M1 макрофагами через NF-κB путь [Bradley, Journal of Pathology 2008]. IL-10 ингибирует NF-κB транскрипцию, снижая продукцию TNF-α [Mosser & Zhang, Immunological Reviews 2008].

---

#### 2.2.2. IL-10 (противовоспалительный цитокин)

```
dC_IL10 = [s_IL10_M2 · M₂ + s_IL10_efferocytosis · k_phag · M_total · Nₑ/(Nₑ + K_phag) - γ_IL10 · C_IL10]dt + σ_IL10 · C_IL10 · dW_IL10
```

Где:
- **s_IL10_M2** — секреция M2 макрофагами ≈ 0.008
- **s_IL10_efferocytosis** — дополнительная продукция IL-10 при эффероцитозе нейтрофилов
- **γ_IL10** — деградация ≈ 0.3 ч⁻¹

**Обоснование:** IL-10 — ключевой противовоспалительный цитокин. Эффероцитоз (фагоцитоз апоптотических нейтрофилов) — сильнейший стимул для продукции IL-10 и M2 поляризации [Fadok et al., Journal of Clinical Investigation 1998]. Это критическая обратная связь: нейтрофилы → апоптоз → эффероцитоз → IL-10 → M2 → разрешение воспаления.

---

#### 2.2.3. PDGF (фактор роста тромбоцитарного происхождения)

```
dC_PDGF = [s_PDGF_P · k_deg · P + s_PDGF_M · (M₁ + M₂) + Θ_PRP_PDGF(t) - γ_PDGF · C_PDGF - k_bind_F · F · C_PDGF/(K_PDGF + C_PDGF)]dt + σ_PDGF · C_PDGF · dW_PDGF
```

Где:
- **s_PDGF_P · k_deg · P** — высвобождение при дегрануляции тромбоцитов
- **s_PDGF_M** — секреция макрофагами
- **Θ_PRP_PDGF(t)** — дополнительный PDGF из PRP-инъекции (см. раздел 4)
- **k_bind_F · F · C_PDGF/(K_PDGF + C_PDGF)** — потребление фибробластами (рецепторное связывание)
- **γ_PDGF** — деградация ≈ 0.2 ч⁻¹ (t₁/₂ ~ 2-4 ч)

**Обоснование:** PDGF — основной хемоаттрактант и митоген для фибробластов. Высвобождается из α-гранул тромбоцитов [Heldin & Westermark, Physiological Reviews 1999]. PRP содержит 3-5x концентрацию PDGF по сравнению с цельной кровью [Marx, Journal of Oral & Maxillofacial Surgery 2004].

---

#### 2.2.4. VEGF (фактор роста эндотелия сосудов)

```
dC_VEGF = [s_VEGF_M2 · M₂ · (1 + α_hypoxia · (1 - θ_hypoxia)) + s_VEGF_F · F + Θ_PRP_VEGF(t) - γ_VEGF · C_VEGF - k_bind_E · E · C_VEGF/(K_VEGF + C_VEGF)]dt + σ_VEGF · C_VEGF · dW_VEGF
```

Где:
- **s_VEGF_M2 · (1 + α_hypoxia · (1 - θ_hypoxia))** — секреция M2, усиленная гипоксией (через HIF-1α)
- **s_VEGF_F** — секреция фибробластами (основной дополнительный источник)
- **k_bind_E** — потребление эндотелиальными клетками (VEGFR2-связывание)
- **γ_VEGF** — деградация ≈ 0.3 ч⁻¹

**Обоснование:** VEGF-A индуцируется через HIF-1α при гипоксии — классический механизм ангиогенеза [Ferrara, Endocrine Reviews 2004]. M2 макрофаги — основной клеточный источник VEGF в ране [Mantovani et al., Trends in Immunology 2004].

---

#### 2.2.5. TGF-β (трансформирующий фактор роста бета)

```
dC_TGFβ = [s_TGF_P · k_deg · P + s_TGF_M2 · M₂ + s_TGF_Mf · Mf + Θ_PRP_TGF(t) - γ_TGF · C_TGFβ]dt + σ_TGF · C_TGFβ · dW_TGF
```

Где:
- **s_TGF_P** — из тромбоцитов (латентная форма → активация MMP-2/9)
- **s_TGF_M2** — из M2 макрофагов
- **s_TGF_Mf** — из миофибробластов (положительная обратная связь: TGF-β → Mf → больше TGF-β)
- **γ_TGF** — деградация ≈ 0.15 ч⁻¹ (t₁/₂ ~ 3-4 ч)

**Обоснование:** TGF-β — плейотропный цитокин: противовоспалительный + профибротический. Положительная обратная связь миофибробласт↔TGF-β — ключевой механизм фиброза [Leask & Abraham, FASEB Journal 2004]. Это бистабильный переключатель: при нарушении баланса → патологический рубец.

⚠️ **Бистабильность TGF-β системы** — критически важная особенность. Положительная обратная связь Mf → TGF-β → Mf создаёт два устойчивых состояния: (1) разрешение фиброза (нормальное заживление) и (2) хронический фиброз. Эта бистабильность отсутствует в простой 2-переменной модели и является одним из главных аргументов за расширение системы.

---

#### 2.2.6. MCP-1/CCL2 (хемоаттрактант для моноцитов)

```
dC_MCP1 = [s_MCP1_damage · D(t) + s_MCP1_M1 · M₁ - γ_MCP1 · C_MCP1]dt + σ_MCP1 · C_MCP1 · dW_MCP1
```

Где:
- **D(t)** — сигнал повреждения (damage-associated molecular patterns, DAMPs): D(t) = D₀ · exp(-t/τ_damage)
- **s_MCP1_M1** — секреция M1 макрофагами (положительная обратная связь рекрутирования)
- **γ_MCP1** — деградация ≈ 0.4 ч⁻¹

**Обоснование:** MCP-1/CCL2 — основной хемоаттрактант для моноцитов. Его продукция управляется NF-κB путём и DAMPs от повреждённых клеток [Deshmane et al., Journal of Interferon & Cytokine Research 2009].

---

#### 2.2.7. IL-8/CXCL8 (хемоаттрактант для нейтрофилов)

```
dC_IL8 = [s_IL8_damage · D(t) + s_IL8_M1 · M₁ + s_IL8_Ne · Nₑ - γ_IL8 · C_IL8]dt + σ_IL8 · C_IL8 · dW_IL8
```

Где:
- **s_IL8_damage · D(t)** — продукция повреждёнными клетками
- **s_IL8_M1** — продукция M1 макрофагами
- **s_IL8_Ne** — аутокринная продукция нейтрофилами (положительная обратная связь)
- **γ_IL8** — деградация ≈ 0.5 ч⁻¹

**Обоснование:** IL-8 — ключевой рекрутёр нейтрофилов через CXCR1/CXCR2 [Kolaczkowska & Kubes, Nature Reviews Immunology 2013]. Аутокринная петля усиления — важная особенность ранней воспалительной фазы.

---

### 2.3. Внеклеточный матрикс (ECM)

#### 2.3.1. Коллаген ρ_c(t)

```
dρ_c = [q_F · F · (1 - ρ_c/ρ_c_max) + q_Mf · Mf · (1 - ρ_c/ρ_c_max) - k_MMP · C_MMP · ρ_c/(K_MMP_sub + ρ_c)]dt
```

Где:
- **q_F** — скорость продукции фибробластами (преимущественно коллаген III)
- **q_Mf** — скорость продукции миофибробластами (2-3x выше, преимущественно коллаген I)
- **(1 - ρ_c/ρ_c_max)** — насыщение (механическая обратная связь)
- **k_MMP · C_MMP · ρ_c/(K_MMP_sub + ρ_c)** — деградация матриксными металлопротеиназами (кинетика Михаэлиса-Ментен)

**Обоснование:** Коллаген — основной структурный белок ECM. Баланс продукции (фибробласты/миофибробласты) и деградации (MMPs) определяет исход заживления [Xue et al., PLoS Computational Biology 2009]. Модель Michaelis-Menten для MMP-субстратной кинетики — стандарт [Vempati et al., PLoS Computational Biology 2014].

---

#### 2.3.2. Матриксные металлопротеиназы C_MMP(t)

```
dC_MMP = [s_MMP_M1 · M₁ + s_MMP_M2 · M₂ · α_MMP_M2 + s_MMP_F · F - k_TIMP · C_TIMP · C_MMP - γ_MMP · C_MMP]dt
```

Где:
- **s_MMP_M1** — секреция M1 (MMP-1, MMP-9 — деградация фибрина и коллагена)
- **s_MMP_M2 · α_MMP_M2** — секреция M2 (другой спектр MMPs, обычно α_MMP_M2 < 1)
- **s_MMP_F** — секреция фибробластами (MMP-1, MMP-2)
- **k_TIMP · C_TIMP · C_MMP** — ингибирование TIMPs (tissue inhibitors of metalloproteinases)
- **γ_MMP** — деградация ≈ 0.1 ч⁻¹

**Обоснование:** MMP/TIMP баланс — ключевой регулятор ремоделирования ECM [Gill & Parks, International Journal of Biochemistry & Cell Biology 2008]. Нарушение баланса → избыточное рубцевание (↑TIMPs) или хроническая рана (↑MMPs).

---

#### 2.3.3. Фибрин ρ_f(t) — временная матрица

```
dρ_f = [-k_fibrinolysis · C_MMP · ρ_f - k_remodel · F · ρ_f]dt
```

Где:
- **k_fibrinolysis** — скорость MMP-опосредованного фибринолиза
- **k_remodel** — замещение фибробластами (фибрин → коллаген)

**Обоснование:** Фибриновый сгусток — временный каркас, постепенно замещаемый коллагеном. Скорость замещения определяет механические свойства ткани [Clark, Annals of the New York Academy of Sciences 2001].

---

### 2.4. Вспомогательные переменные

#### 2.4.1. Сигнал повреждения D(t)

```
D(t) = D₀ · exp(-t/τ_damage)
```

- **D₀** — начальная интенсивность повреждения (зависит от размера/глубины раны)
- **τ_damage** — временная константа затухания ≈ 24-48 ч

DAMPs (damage-associated molecular patterns) включают ATP, HMGB1, фрагменты гиалуроновой кислоты [Bianchi, Journal of Leukocyte Biology 2007].

#### 2.4.2. Кислород O₂(t) (упрощённая модель)

```
dO₂/dt = D_O2 · (O₂_blood - O₂) / L² - k_consumption · (all_cells) · O₂/(K_O2_consume + O₂) + k_angio · E
```

- **D_O2** — диффузия кислорода
- **O₂_blood** — давление O₂ в крови (нормоксия)
- **L** — характерная дистанция диффузии от ближайшего капилляра
- **k_angio · E** — улучшение перфузии от ангиогенеза

---

## 3. Механизмы терапевтических вмешательств

### 3.1. PRP (Platelet-Rich Plasma) — механистическая модель

**Текущая реализация:** f(PRP) = C₀·exp(-λt) — одна экспонента.

**Полная модель:**

PRP — это концентрированная суспензия тромбоцитов (3-5x выше нормы). При активации высвобождаются факторы роста с **разной кинетикой**:

```
Θ_PRP_i(t) = Σⱼ [PRP_dose · φᵢⱼ · (exp(-t/τ_burst_j) - exp(-t/τ_sustained_j)) / (τ_burst_j - τ_sustained_j)]
```

Где для каждого фактора роста i (PDGF, VEGF, TGF-β, EGF):
- **PRP_dose** — доза PRP (кратность концентрации × объём)
- **φᵢⱼ** — содержание i-го фактора в j-м пуле высвобождения
- **τ_burst** — быстрое высвобождение из α-гранул ≈ 0.5-2 ч
- **τ_sustained** — медленное высвобождение из фибриновой сети ≈ 24-72 ч

| Фактор | Концентрация в PRP (нг/мл) | τ_burst (ч) | τ_sustained (ч) | Источник |
|--------|---------------------------|-------------|-----------------|----------|
| PDGF-AB | 15-30 | 1 | 48 | Marx 2004 |
| TGF-β1 | 20-40 | 2 | 72 | Eppley et al. 2006 |
| VEGF | 0.5-1.5 | 1 | 24 | Everts et al. 2006 |
| EGF | 0.1-0.3 | 0.5 | 12 | Anitua et al. 2004 |

**Обоснование:** Двухфазная кинетика PRP подтверждена in vitro: быстрый выброс при активации (минуты-часы) + замедленное высвобождение из фибриновой матрицы (дни) [Giusti et al., Experimental Hematology 2009]. Простая экспонента не описывает этот двухфазный характер.

Дополнительные эффекты PRP:
- **Рекрутирование стволовых клеток** через SDF-1/CXCR4: α_PRP_S · Θ_PRP(t) в уравнении S(t)
- **Антибактериальный эффект** (не моделируем для стерильных условий)

---

### 3.2. PEMF (Pulsed Electromagnetic Field) — механистическая модель

**Текущая реализация:** g(PEMF) = sigmoid(frequency) — чисто феноменологическая.

**Полная модель:**

PEMF воздействует через несколько механизмов [Pilla, Annals of Biomedical Engineering 2013]:

**Механизм 1: Аденозиновый путь A₂A/A₃**
PEMF усиливает связывание аденозина с A₂A рецепторами → противовоспалительный эффект:

```
ε_PEMF_anti_inflam(f, B) = ε_max · (B/B₀)^n_B / (1 + (B/B₀)^n_B) · exp(-(f - f_opt)²/(2·σ_f²))
```

Где:
- **B** — амплитуда магнитного поля (мТл)
- **B₀** — пороговая амплитуда ≈ 0.1-1 мТл
- **f** — частота (Гц)
- **f_opt** — оптимальная частота ≈ 27.12 МГц (для A₂A) или 50-75 Гц (для низкочастотного PEMF)
- **σ_f** — ширина частотного окна

Эффект: **снижение продукции TNF-α** на 30-50% [Varani et al., Mediators of Inflammation 2017]:
```
s_TNF_M1 → s_TNF_M1 · (1 - ε_PEMF_anti_inflam) когда PEMF активен
```

**Механизм 2: Ca²⁺-CaM путь**
PEMF усиливает внутриклеточный кальций → CaM-зависимая NO синтаза → NO → стимуляция пролиферации:

```
ε_PEMF_prolif(f, B) = ε_prolif_max · B²/(B_half² + B²) · W(f)
```

Где W(f) — оконная функция (пропускает частоты 50-100 Гц):
```
W(f) = exp(-(f - f_center)²/(2·σ_window²))
```

Эффект: **усиление пролиферации фибробластов и эндотелиальных клеток** [Pilla 2013]:
```
r_F → r_F · (1 + ε_PEMF_prolif) когда PEMF активен
r_E → r_E · (1 + ε_PEMF_prolif) когда PEMF активен
```

**Механизм 3: MAPK/ERK сигнализация**
PEMF активирует ERK1/2 → усиление миграции клеток. Моделируется как увеличение коэффициента диффузии в ABM:
```
D_cell → D_cell · (1 + ε_PEMF_migration) когда PEMF активен
```

**Синергия PRP + PEMF:**
```
synergy(t) = 1 + β_synergy · Θ_PRP(t) · PEMF_active(t)
```

Обоснование: комбинация PRP + PEMF показывает супер-аддитивный эффект in vivo [Onstenk et al., Journal of Orthopaedic Research 2015], вероятно через усиление рецепторной чувствительности PEMF-индуцированным Ca²⁺.

---

## 4. Мультимасштабная интеграция SDE ↔ ABM

### 4.1. Текущий подход (валидный для MVP)

Текущая реализация использует operator splitting:
1. SDE шаг (макроуровень, глобальные N, C)
2. ABM шаг (микроуровень, индивидуальные агенты)
3. Синхронизация: N_corrected = N_sde + α·(N_abm - N_sde)

### 4.2. Расширенный подход: Equation-Free Framework

Для полной модели рекомендуется Equation-Free Framework [Kevrekidis et al., Communications in Mathematical Sciences 2003]:

```
Макро-SDE: dX = F_macro(X)dt + G(X)dW    (X = вектор всех популяций и цитокинов)
                     ↕ lifting/restricting
Микро-ABM: агенты с правилами поведения
```

**Lifting** (макро→микро): распределение агентов по пространству так, чтобы их агрегированные свойства соответствовали макро-переменным.

**Restricting** (микро→макро): X_macro = Σ(agent_states) / volume — среднее по агентам.

### 4.3. Сопряжение шкал

| Уровень | Масштаб | Модель | Переменные |
|---------|---------|--------|-----------|
| Макро (ткань) | мм–см, часы–дни | SDE система (§2) | P, Ne, M1, M2, F, Mf, E, S, цитокины, ECM |
| Мезо (клеточная популяция) | 10-100 мкм, минуты–часы | ABM + реакция-диффузия | Позиции агентов, локальные поля цитокинов |
| Микро (внутриклеточный) | нм–мкм, секунды–минуты | ODE сигнальных путей | NF-κB, Smad2/3, ERK1/2 (опционально) |

Для MVP достаточно макро+мезо. Микро-уровень добавляется для предсказания эффектов конкретных лекарств.

---

## 5. Численные методы

### 5.1. Для SDE системы

**Текущий метод:** Euler-Maruyama (EM) — порядок сходимости 0.5 (сильная) / 1.0 (слабая).

**Рекомендуемое улучшение:** Milstein scheme — порядок 1.0 (сильная):

```
Xₙ₊₁ = Xₙ + μ(Xₙ)·Δt + σ(Xₙ)·ΔWₙ + ½·σ(Xₙ)·σ'(Xₙ)·(ΔWₙ² - Δt)
```

Для мультимерной системы — Milstein с перекрёстными термами или Stochastic Runge-Kutta SRI2W1 [Rößler, SIAM Journal on Numerical Analysis 2010].

**Адаптивный шаг:** Для стиффных систем (fast TNF dynamics + slow collagen remodeling) — implicit-explicit (IMEX) splitting:
- Explicit: медленные переменные (ECM, клетки)
- Implicit: быстрые переменные (цитокины с γ >> 1)

### 5.2. Для ABM

Текущая реализация корректна. Рекомендуемые улучшения:
- **KD-Tree** вместо SpatialHash для поиска соседей (лучше при неравномерном распределении)
- **Subcycling:** цитокиновое поле обновляется с большим dt чем агенты

### 5.3. Верификация

- **Method of Manufactured Solutions** — подставить известное решение, проверить порядок сходимости
- **Сравнение SDE-only vs ABM-only** — при большом числе агентов ABM должна сходиться к SDE (закон больших чисел)
- **Conservation checks** — общее число клеток ≈ интеграл потоков (баланс рождения/смерти)

---

## 6. Стратегия валидации на данных

### 6.1. Публичные датасеты для валидации

| Источник | Данные | Применение |
|----------|--------|-----------|
| FlowRepository (FR-FCM-) | Flow cytometry раны | Начальные условия: CD34+%, CD14+%, апоптоз |
| GEO (NCBI) | Транскриптомы ран по времени | Валидация динамики цитокинов |
| Wound Healing Society datasets | Клинические данные заживления | Валидация скорости закрытия |
| Human Protein Atlas | Экспрессия в коже | Базовые уровни белков |

### 6.2. Метрики валидации

1. **Temporal R²** — корреляция предсказанных и наблюдаемых траекторий клеточных популяций
2. **Phase timing** — правильность предсказания длительности фаз (воспаление → пролиферация → ремоделирование)
3. **Sensitivity analysis** — Sobol indices для ключевых параметров
4. **Uncertainty quantification** — Monte Carlo envelopes покрывают наблюдения

---

## 7. Связь с текущим кодом: план миграции

### Фаза 1: Расширение SDE (2-4 недели)

```python
# Текущее: 2 переменных
class SDEState:
    N: float   # единая плотность
    C: float   # единый цитокин

# Целевое: 17+ переменных
class ExtendedSDEState:
    # Клеточные популяции
    P: float      # тромбоциты
    Ne: float     # нейтрофилы
    M1: float     # M1 макрофаги
    M2: float     # M2 макрофаги
    F: float      # фибробласты
    Mf: float     # миофибробласты
    E: float      # эндотелиальные
    S: float      # стволовые
    
    # Цитокины
    C_TNF: float
    C_IL10: float
    C_PDGF: float
    C_VEGF: float
    C_TGFb: float
    C_MCP1: float
    C_IL8: float
    
    # ECM
    rho_collagen: float
    C_MMP: float
    rho_fibrin: float
    
    # Вспомогательные
    D: float      # сигнал повреждения
    O2: float     # кислород
```

### Фаза 2: Рефакторинг ABM (2-3 недели)
- Добавить типы агентов: Neutrophil, EndothelialCell
- Внутриклеточные переменные: polarization_state как continuous (0=M1, 1=M2)
- Механотрансдукция в ABM: миофибробластная активация зависит от механического стресса

### Фаза 3: Интеграция + Monte Carlo (1-2 недели)
- Обновить IntegrationConfig для 17-мерного вектора
- Parameter estimation: Bayesian inference (PyMC / emcee)

### Фаза 4: Валидация (4-8 недель)
- Подобрать параметры на публичных данных
- Sensitivity analysis (SALib)
- Написать препринт для bioRxiv

---

## 8. Таблица параметров (базовые значения)

### 8.1. Клеточные параметры

| Параметр | Значение | Единицы | Описание | Источник |
|----------|----------|---------|----------|----------|
| r_F | 0.03 | ч⁻¹ | Пролиферация фибробластов | Vodovotz 2006 |
| r_E | 0.02 | ч⁻¹ | Пролиферация эндотелия | Anderson 1998 |
| r_S | 0.01 | ч⁻¹ | Самообновление стволовых | Badiavas 2003 |
| δ_P | 0.1 | ч⁻¹ | Клиренс тромбоцитов | Nurden 2008 |
| δ_Ne | 0.05 | ч⁻¹ | Апоптоз нейтрофилов | Kolaczkowska 2013 |
| δ_M | 0.01 | ч⁻¹ | Апоптоз макрофагов | Murray 2017 |
| δ_F | 0.003 | ч⁻¹ | Апоптоз фибробластов | Hinz 2007 |
| k_switch | 0.02 | ч⁻¹ | M1→M2 переключение | Mantovani 2004 |
| k_act | 0.01 | ч⁻¹ | F→Mf активация | Hinz 2007 |
| K_F | 5×10⁵ | клеток/мкл | Carrying capacity F+Mf | Flegg 2015 |

### 8.2. Параметры цитокинов

| Параметр | Значение | Единицы | Описание | Источник |
|----------|----------|---------|----------|----------|
| γ_TNF | 0.5 | ч⁻¹ | Деградация TNF-α | Bradley 2008 |
| γ_IL10 | 0.3 | ч⁻¹ | Деградация IL-10 | Mosser 2008 |
| γ_PDGF | 0.2 | ч⁻¹ | Деградация PDGF | Heldin 1999 |
| γ_VEGF | 0.3 | ч⁻¹ | Деградация VEGF | Ferrara 2004 |
| γ_TGF | 0.15 | ч⁻¹ | Деградация TGF-β | Leask 2004 |
| s_TNF_M1 | 0.01 | нг/(мл·кл·ч) | Секреция TNF M1 | Bradley 2008 |
| s_IL10_M2 | 0.008 | нг/(мл·кл·ч) | Секреция IL-10 M2 | Mosser 2008 |

### 8.3. Параметры ECM

| Параметр | Значение | Единицы | Описание | Источник |
|----------|----------|---------|----------|----------|
| q_F | 0.005 | ед/ч | Продукция коллагена F | Xue 2009 |
| q_Mf | 0.015 | ед/ч | Продукция коллагена Mf | Desmouliere 2005 |
| k_MMP | 0.02 | ч⁻¹ | Деградация коллагена MMP | Gill 2008 |

### 8.4. Параметры PRP

| Параметр | Значение | Единицы | Описание | Источник |
|----------|----------|---------|----------|----------|
| PRP_dose | 3-5x | кратность | Концентрация тромбоцитов | Marx 2004 |
| τ_burst | 1-2 | ч | Быстрое высвобождение | Giusti 2009 |
| τ_sustained | 24-72 | ч | Замедленное высвобождение | Giusti 2009 |

---

## 9. Список литературы

1. Anderson, A.R.A. & Chaplain, M.A.J. (1998). Continuous and discrete mathematical models of tumor-induced angiogenesis. *Bulletin of Mathematical Biology*, 60(5), 857-899.
2. Anitua, E. et al. (2004). Autologous platelets as a source of proteins for healing and tissue regeneration. *Thrombosis and Haemostasis*, 91(1), 4-15.
3. Badiavas, E.V. & Falanga, V. (2003). Treatment of chronic wounds with bone marrow-derived cells. *Archives of Dermatology*, 139(4), 510-516.
4. Bianchi, M.E. (2007). DAMPs, PAMPs and alarmins. *Journal of Leukocyte Biology*, 81(1), 1-5.
5. Blair, P. & Flaumenhaft, R. (2009). Platelet α-granules: basic biology and clinical correlates. *Blood Reviews*, 23(4), 177-189.
6. Bradley, J.R. (2008). TNF-mediated inflammatory disease. *Journal of Pathology*, 214(2), 149-160.
7. Clark, R.A.F. (2001). Fibrin and wound healing. *Annals of the New York Academy of Sciences*, 936, 355-367.
8. Desmoulière, A. et al. (2005). Apoptosis mediates the decrease in cellularity during the transition between granulation tissue and scar. *American Journal of Pathology*, 146(1), 56-66.
9. Deshmane, S.L. et al. (2009). Monocyte chemoattractant protein-1 (MCP-1): an overview. *Journal of Interferon & Cytokine Research*, 29(6), 313-326.
10. Eming, S.A. et al. (2014). Wound repair and regeneration: mechanisms, signaling, and translation. *Science Translational Medicine*, 6(265), 265sr6.
11. Eppley, B.L. et al. (2006). Platelet-rich plasma: a review of biology and applications. *Plastic and Reconstructive Surgery*, 118(6), 147e-159e.
12. Everts, P.A.M. et al. (2006). Platelet-rich plasma and platelet gel: a review. *Journal of Extra-Corporeal Technology*, 38(2), 174-187.
13. Fadok, V.A. et al. (1998). Macrophages that have ingested apoptotic cells in vitro inhibit proinflammatory cytokine production. *Journal of Clinical Investigation*, 101(4), 890-898.
14. Ferrara, N. et al. (2003). The biology of VEGF and its receptors. *Nature Medicine*, 9(6), 669-676.
15. Ferrara, N. (2004). Vascular endothelial growth factor: basic science and clinical progress. *Endocrine Reviews*, 25(4), 581-611.
16. Flegg, J.A. et al. (2015). Mathematical model of hyperbaric oxygen therapy applied to chronic diabetic wounds. *Bulletin of Mathematical Biology*, 77(10), 1901-1928.
17. Gill, S.E. & Parks, W.C. (2008). Metalloproteinases and their inhibitors: regulators of wound healing. *International Journal of Biochemistry & Cell Biology*, 40(6-7), 1334-1347.
18. Giusti, I. et al. (2009). Identification of an optimal concentration of platelet gel for promoting angiogenesis. *Experimental Hematology*, 37(4), 423-435.
19. Gurtner, G.C. et al. (2008). Wound repair and regeneration. *Nature*, 453(7193), 314-321.
20. Heldin, C.H. & Westermark, B. (1999). Mechanism of action and in vivo role of platelet-derived growth factor. *Physiological Reviews*, 79(4), 1283-1316.
21. Hinz, B. et al. (2007). The myofibroblast: one function, multiple origins. *American Journal of Pathology*, 170(6), 1807-1816.
22. Kevrekidis, I.G. et al. (2003). Equation-free, coarse-grained multiscale computation. *Communications in Mathematical Sciences*, 1(4), 715-762.
23. Kloeden, P.E. & Platen, E. (1992). *Numerical Solution of Stochastic Differential Equations*. Springer.
24. Kolaczkowska, E. & Kubes, P. (2013). Neutrophil recruitment and function in health and inflammation. *Nature Reviews Immunology*, 13(3), 159-175.
25. Leask, A. & Abraham, D.J. (2004). TGF-β signaling and the fibrotic response. *FASEB Journal*, 18(7), 816-827.
26. Mantovani, A. et al. (2004). The chemokine system in diverse forms of macrophage activation and polarization. *Trends in Immunology*, 25(12), 677-686.
27. Marx, R.E. (2004). Platelet-rich plasma: evidence to support its use. *Journal of Oral & Maxillofacial Surgery*, 62(4), 489-496.
28. Mosser, D.M. & Zhang, X. (2008). Interleukin-10: new perspectives on an old cytokine. *Immunological Reviews*, 226, 205-218.
29. Murray, J.D. (2002). *Mathematical Biology I: An Introduction*. 3rd ed. Springer.
30. Murray, P.J. (2017). Macrophage polarization. *Annual Review of Physiology*, 79, 541-566.
31. Nurden, A.T. et al. (2008). Platelets in wound healing and regenerative medicine. *Blood Reviews*, 22(6), 299-333.
32. Pilla, A.A. (2013). Nonthermal electromagnetic fields: from first messenger to therapeutic applications. *Electromagnetic Biology and Medicine*, 32(2), 123-136.
33. Ranzato, E. et al. (2009). Platelet lysate promotes in vitro wound scratch closure of human dermal fibroblasts. *Journal of Cellular and Molecular Medicine*, 13(8b), 2448-2454.
34. Rößler, A. (2010). Runge–Kutta Methods for the Strong Approximation of Solutions of Stochastic Differential Equations. *SIAM Journal on Numerical Analysis*, 48(3), 922-952.
35. Serhan, C.N. & Savill, J. (2005). Resolution of inflammation: the beginning programs the end. *Nature Immunology*, 6(12), 1191-1197.
36. Sindrilaru, A. et al. (2011). An unrestrained proinflammatory M1 macrophage population induced by iron impairs wound healing in humans and mice. *Journal of Clinical Investigation*, 121(3), 985-997.
37. Strang, G. (1968). On the construction and comparison of difference schemes. *SIAM Journal on Numerical Analysis*, 5(3), 506-517.
38. Varani, K. et al. (2017). Pulsed electromagnetic field stimulation in osteogenesis and chondrogenesis. *Mediators of Inflammation*, 2017, 8045926.
39. Vempati, P. et al. (2014). Quantifying the proteolytic release of extracellular matrix-sequestered VEGF. *PLoS Computational Biology*, 10(1), e1003426.
40. Vodovotz, Y. et al. (2006). Mathematical models of the acute inflammatory response. *Current Opinion in Critical Care*, 12(4), 325-332.
41. Xue, C. et al. (2009). A mathematical model of ischemic cutaneous wounds. *PLoS Computational Biology*, 5(4), e1000348.

---

*Документ подготовлен для проекта RegenTwin. Каждое уравнение обосновано ссылкой на рецензированную публикацию. Документ может служить основой для раздела "Methods" препринта на bioRxiv.*