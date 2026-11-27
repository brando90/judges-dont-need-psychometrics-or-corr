import Mathlib

/-!
# Claims audit of the real-data plan (Lean 4 + Mathlib), 11-26-2026

Checks the mathematical claims of the "5-dataset real judging suite" plan that Brando sent to the
Grok bot and the Cursor agent.  Zero `sorry`; the axioms of every theorem are printed at the end of the
file (`propext`, `Classical.choice`, `Quot.sound` only).

* Section A: the per-sample test risk `R̂ = (1/m) ∑_c (1/k_c) ∑_j L(q̂(c), y_{c,j})` is unbiased for
  every loss `L`, every `k_c ≥ 1` (so also `k_c = 1`), with different laws per prompt; plus the case
  where it is NOT unbiased (the number of ratings depends on the ratings, as in HelpSteer2).
* Section B: anti-properness of correlation PROVED from a finite joint law.  Both correlations are
  computed from raw joint moments; nothing about `Var(μ)/Var(Y)` is assumed.
* Section C: the ordinal `{0,…,4}` example (Brier 4×, RPS 2×, log-loss `ln 2` versus `+∞`).
* Section D: `√RPS` must be taken after averaging (Jensen).
* Section E: the binary example (Pearson tie 0.6532, Brier 0.1433 versus 0.4119, ratio 2.87).

Not repeated here, already compiled elsewhere in this repository: Brier and RPS strict propriety
(`extended_checks_11-26-2026/lean/ProperScoring`), the standardized L2 identity, the MSE
decomposition and affine invariance of Pearson (`claims_audit_11-26-2026/AuditClaims.lean`), the exact
empirical-risk rate `(1 - ‖p‖²)/N` and the plug-in bias `(1/k)(1 - ‖p‖²)` (experiment 36
`SanmiPitch.lean`, over `Rat`, theorems `empirical_l2_rate` and `plugin_DB_bias`).
-/

open Finset BigOperators

namespace PlanAudit

/-! ## Section A. The per-sample test risk is unbiased for any loss, any `k_c ≥ 1`, even `k_c = 1`.

`expV n p f` is the expectation of `f s` when the `n` ratings `s 0, …, s (n-1)` are independent and
rating `j` has law `p j` on `Fin K`.  (A different law per rating lets different items have
different `p*(· | c)`.)  Independence is used only to build this model: the unbiasedness theorem
below needs only linearity of expectation and the correct marginal of each rating. -/

noncomputable def expV {K : ℕ} : (n : ℕ) → (Fin n → Fin K → ℝ) → ((Fin n → Fin K) → ℝ) → ℝ
  | 0, _, f => f Fin.elim0
  | n + 1, p, f => ∑ a : Fin K, p 0 a * expV n (fun i => p i.succ) (fun t => f (Fin.cons a t))

theorem expV_add {K : ℕ} : ∀ (n : ℕ) (p : Fin n → Fin K → ℝ) (f g : (Fin n → Fin K) → ℝ),
    expV n p (fun s => f s + g s) = expV n p f + expV n p g
  | 0, p, f, g => by simp [expV]
  | n + 1, p, f, g => by
      simp only [expV]
      rw [← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl (fun a _ => ?_)
      rw [expV_add n (fun i => p i.succ) (fun t => f (Fin.cons a t)) (fun t => g (Fin.cons a t))]
      ring

theorem expV_smul {K : ℕ} : ∀ (n : ℕ) (p : Fin n → Fin K → ℝ) (c : ℝ) (f : (Fin n → Fin K) → ℝ),
    expV n p (fun s => c * f s) = c * expV n p f
  | 0, p, c, f => by simp [expV]
  | n + 1, p, c, f => by
      simp only [expV]
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl (fun a _ => ?_)
      rw [expV_smul n (fun i => p i.succ) c (fun t => f (Fin.cons a t))]
      ring

theorem expV_const {K : ℕ} : ∀ (n : ℕ) (p : Fin n → Fin K → ℝ), (∀ i, ∑ a, p i a = 1) →
    ∀ c : ℝ, expV n p (fun _ => c) = c
  | 0, p, _, c => by simp [expV]
  | n + 1, p, hp, c => by
      simp only [expV]
      rw [show (∑ a : Fin K, p 0 a * expV n (fun i => p i.succ) (fun _ => c))
            = ∑ a : Fin K, p 0 a * c from
          Finset.sum_congr rfl (fun a _ => by
            rw [expV_const n (fun i => p i.succ) (fun i => hp i.succ) c])]
      rw [← Finset.sum_mul, hp 0, one_mul]

/-- Each rating has its own law as its marginal. -/
theorem expV_coord {K : ℕ} : ∀ (n : ℕ) (p : Fin n → Fin K → ℝ), (∀ i, ∑ a, p i a = 1) →
    ∀ (j : Fin n) (g : Fin K → ℝ), expV n p (fun s => g (s j)) = ∑ a, p j a * g a
  | 0, p, _, j, g => j.elim0
  | n + 1, p, hp, j, g => by
      refine Fin.cases ?_ ?_ j
      · simp only [expV, Fin.cons_zero]
        refine Finset.sum_congr rfl (fun a _ => ?_)
        rw [expV_const n (fun i => p i.succ) (fun i => hp i.succ) (g a)]
      · intro j'
        simp only [expV, Fin.cons_succ]
        rw [show (∑ a : Fin K, p 0 a * expV n (fun i => p i.succ) (fun t => g (t j')))
              = ∑ a : Fin K, p 0 a * (∑ b, p j'.succ b * g b) from
            Finset.sum_congr rfl (fun a _ => by
              rw [expV_coord n (fun i => p i.succ) (fun i => hp i.succ) j' g])]
        rw [← Finset.sum_mul, hp 0, one_mul]

/-- `expV` as a linear functional. -/
noncomputable def expLin {K : ℕ} (n : ℕ) (p : Fin n → Fin K → ℝ) :
    ((Fin n → Fin K) → ℝ) →ₗ[ℝ] ℝ where
  toFun := expV n p
  map_add' f g := expV_add n p f g
  map_smul' c f := by
    change expV n p (fun s => c * f s) = c * expV n p f
    exact expV_smul n p c f

/-- **Unbiasedness, general form.**  Any linear expectation with the right marginals gives
`E[∑ w_j ℓ_j(s_j)] = ∑ w_j ∑_a p_j(a) ℓ_j(a)`.  No independence between ratings is needed. -/
theorem unbiased_weighted_sum {n K : ℕ} (E : ((Fin n → Fin K) → ℝ) →ₗ[ℝ] ℝ)
    (p : Fin n → Fin K → ℝ)
    (hmarg : ∀ (j : Fin n) (g : Fin K → ℝ), E (fun s => g (s j)) = ∑ a, p j a * g a)
    (w : Fin n → ℝ) (ℓ : Fin n → Fin K → ℝ) :
    E (fun s => ∑ j, w j * ℓ j (s j)) = ∑ j, w j * ∑ a, p j a * ℓ j a := by
  have h1 : (fun s : Fin n → Fin K => ∑ j, w j * ℓ j (s j))
      = ∑ j, w j • (fun s : Fin n → Fin K => ℓ j (s j)) := by
    funext s
    simp [Finset.sum_apply]
  rw [h1, map_sum]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  rw [map_smul, hmarg j (ℓ j)]
  simp

/-! ### The plan's estimator, with items and unequal `k_c`. -/

/-- Number of ratings of item `i`. -/
def kCount {n m : ℕ} (item : Fin n → Fin m) (i : Fin m) : ℕ :=
  (Finset.univ.filter (fun j => item j = i)).card

/-- `R̂ = (1/m) ∑_c (1/k_c) ∑_{j of c} L(q̂(c), y_{c,j})`.  `L i a` is the loss of item `i`'s
forecast when the realized rating is `a`; rating `j` belongs to item `item j`. -/
noncomputable def planEstimator {n m K : ℕ} (item : Fin n → Fin m)
    (L : Fin m → Fin K → ℝ) (s : Fin n → Fin K) : ℝ :=
  (1 / (m : ℝ)) * ∑ i : Fin m, (1 / (kCount item i : ℝ)) *
    ∑ j ∈ Finset.univ.filter (fun j => item j = i), L i (s j)

/-- **The plan's claim.**  If every item has at least one rating and every rating of item `i` has
marginal law `P i`, then the estimator's expectation is the average over items of the true expected
loss `∑_a P i a * L i a`.  This holds for each `k_c ≥ 1`, in particular `k_c = 1`. -/
theorem plan_estimator_unbiased {n m K : ℕ}
    (E : ((Fin n → Fin K) → ℝ) →ₗ[ℝ] ℝ) (P : Fin m → Fin K → ℝ) (item : Fin n → Fin m)
    (hmarg : ∀ (j : Fin n) (g : Fin K → ℝ), E (fun s => g (s j)) = ∑ a, P (item j) a * g a)
    (hk : ∀ i, 1 ≤ kCount item i) (L : Fin m → Fin K → ℝ) :
    E (planEstimator item L) = (1 / (m : ℝ)) * ∑ i : Fin m, ∑ a, P i a * L i a := by
  have hF : ∀ i : Fin m,
      E (fun s => ∑ j ∈ Finset.univ.filter (fun j => item j = i), L i (s j))
        = (kCount item i : ℝ) * ∑ a, P i a * L i a := by
    intro i
    have h1 : (fun s : Fin n → Fin K => ∑ j ∈ Finset.univ.filter (fun j => item j = i), L i (s j))
        = ∑ j ∈ Finset.univ.filter (fun j => item j = i),
            (fun s : Fin n → Fin K => L i (s j)) := by
      funext s; simp [Finset.sum_apply]
    rw [h1, map_sum]
    rw [Finset.sum_congr rfl (fun j hj => by
        have hji : item j = i := (Finset.mem_filter.mp hj).2
        rw [hmarg j (L i), hji])]
    rw [Finset.sum_const, nsmul_eq_mul]
    rfl
  have h2 : planEstimator item L
      = (1 / (m : ℝ)) • ∑ i : Fin m, (1 / (kCount item i : ℝ)) •
          (fun s : Fin n → Fin K => ∑ j ∈ Finset.univ.filter (fun j => item j = i), L i (s j)) := by
    funext s
    simp [planEstimator, Finset.sum_apply]
  rw [h2, map_smul, map_sum]
  simp only [map_smul, hF, smul_eq_mul]
  congr 1
  refine Finset.sum_congr rfl (fun i _ => ?_)
  have hk' : (kCount item i : ℝ) ≠ 0 :=
    (Nat.cast_pos.mpr (lt_of_lt_of_le one_pos (hk i))).ne'
  field_simp

/-- The independent-ratings model satisfies the marginal hypothesis, so the theorem applies to it. -/
theorem plan_estimator_unbiased_product {n m K : ℕ} (P : Fin m → Fin K → ℝ)
    (hP : ∀ i, ∑ a, P i a = 1) (item : Fin n → Fin m)
    (hk : ∀ i, 1 ≤ kCount item i) (L : Fin m → Fin K → ℝ) :
    expV n (fun j => P (item j)) (planEstimator item L)
      = (1 / (m : ℝ)) * ∑ i : Fin m, ∑ a, P i a * L i a :=
  plan_estimator_unbiased (expLin n (fun j => P (item j))) P item
    (fun j g => expV_coord n (fun j => P (item j)) (fun j => hP (item j)) j g) hk L

/-! ### The hypotheses can hold: two prompts, one with `k = 1` rating and one with `k = 3`. -/

/-- Prompt 0: `P₀ = (1/4, 3/4)`, forecast `(1/2, 1/2)`, Brier loss `(1/2, 1/2)`.
Prompt 1: `P₁ = (1/2, 1/2)`, forecast `(1/5, 4/5)`, Brier loss `(32/25, 2/25)`. -/
theorem instance_unbiased :
    expV 4 (fun j => (![![(1:ℝ)/4, 3/4], ![1/2, 1/2]] : Fin 2 → Fin 2 → ℝ) (![0, 1, 1, 1] j))
      (planEstimator (![0, 1, 1, 1] : Fin 4 → Fin 2)
        (![![(1:ℝ)/2, 1/2], ![32/25, 2/25]] : Fin 2 → Fin 2 → ℝ))
      = 59 / 100 := by
  have hP : ∀ i, ∑ a, (![![(1:ℝ)/4, 3/4], ![1/2, 1/2]] : Fin 2 → Fin 2 → ℝ) i a = 1 := by
    intro i; fin_cases i <;> simp [Fin.sum_univ_two] <;> norm_num
  have hk : ∀ i, 1 ≤ kCount (![0, 1, 1, 1] : Fin 4 → Fin 2) i := by
    intro i; fin_cases i <;> decide
  rw [plan_estimator_unbiased_product _ hP _ hk]
  simp [Fin.sum_univ_two]
  norm_num

/-! ### What breaks it: a number of ratings that depends on the ratings (adaptive stopping). -/

/-- Rating law of a Bernoulli(`θ`) rating. -/
def pr (θ : ℝ) : Bool → ℝ
  | true => θ
  | false => 1 - θ

def val : Bool → ℝ
  | true => 1
  | false => 0

/-- Loss `ℓ(y) = y` (the Brier loss of the forecast `0`), so the true risk is `θ`.
Draw two ratings; if they agree use their mean; if they disagree draw two more and use the mean of
all four.  This is how HelpSteer2 collects (extra annotators only when the first ones disagree). -/
noncomputable def adaptiveExp (θ : ℝ) : ℝ :=
  ∑ y1 : Bool, ∑ y2 : Bool, pr θ y1 * pr θ y2 *
    (if y1 = y2 then (val y1 + val y2) / 2
     else ∑ y3 : Bool, ∑ y4 : Bool, pr θ y3 * pr θ y4 * ((val y1 + val y2 + val y3 + val y4) / 4))

theorem adaptive_stopping_biased : adaptiveExp (3 / 10) = 129 / 500 ∧ adaptiveExp (3 / 10) ≠ 3 / 10 := by
  have h : adaptiveExp (3 / 10) = 129 / 500 := by
    simp [adaptiveExp, pr, val]
    norm_num
  exact ⟨h, by rw [h]; norm_num⟩


/-! ## Section B. Anti-properness of correlation, proved from a finite joint law.

Nothing is assumed about `Corr(X_{p*}, Y) = Var(μ_Y(C)) / Var(Y)`: both correlations below are
computed from the raw joint expectations of the finite model, then compared. -/

section AntiProper

variable {C Y : Type*} [Fintype C] [Fintype Y]

/-- Prompts `c` with weights `π`, conditional rating laws `p c`, rating values `v`. -/
structure Model (C Y : Type*) [Fintype C] [Fintype Y] where
  π : C → ℝ
  p : C → Y → ℝ
  v : Y → ℝ
  π_nonneg : ∀ c, 0 ≤ π c
  p_nonneg : ∀ c y, 0 ≤ p c y
  π_sum : ∑ c, π c = 1
  p_sum : ∀ c, ∑ y, p c y = 1

namespace Model
variable (M : Model C Y)

/-- Conditional mean `μ_Y(c)`. -/
noncomputable def mu (c : C) : ℝ := ∑ y, M.p c y * M.v y
/-- Conditional second moment. -/
noncomputable def s2 (c : C) : ℝ := ∑ y, M.p c y * M.v y ^ 2
noncomputable def EY : ℝ := ∑ c, M.π c * M.mu c
noncomputable def EY2 : ℝ := ∑ c, M.π c * M.s2 c
noncomputable def varY : ℝ := M.EY2 - M.EY ^ 2
/-- `Var(μ_Y(C))`. -/
noncomputable def varMu : ℝ := ∑ c, M.π c * M.mu c ^ 2 - M.EY ^ 2
/-- `E[Var(Y | C)]`. -/
noncomputable def eVar : ℝ := ∑ c, M.π c * (M.s2 c - M.mu c ^ 2)

/-- Law of total variance. -/
theorem total_variance : M.varY = M.eVar + M.varMu := by
  unfold varY varMu eVar EY2
  simp only [mul_sub, Finset.sum_sub_distrib]
  ring

/-- Expectation when the judge draws `x ~ p c` independently of the human draw `y ~ p c`. -/
noncomputable def Ej (f : C → Y → Y → ℝ) : ℝ :=
  ∑ c, M.π c * ∑ x, M.p c x * ∑ y, M.p c y * f c x y

/-- Expectation when the judge's output is a function of `c` alone. -/
noncomputable def Ed (f : C → Y → ℝ) : ℝ := ∑ c, M.π c * ∑ y, M.p c y * f c y

lemma sum_p_mul_const (c : C) (t : ℝ) : ∑ y, M.p c y * t = t := by
  rw [← Finset.sum_mul, M.p_sum c, one_mul]

lemma Ej_x : M.Ej (fun _ x _ => M.v x) = M.EY := by
  unfold Ej EY mu
  refine Finset.sum_congr rfl (fun c _ => ?_)
  congr 1
  refine Finset.sum_congr rfl (fun x _ => ?_)
  rw [M.sum_p_mul_const c (M.v x)]

lemma Ej_y : M.Ej (fun _ _ y => M.v y) = M.EY := by
  unfold Ej EY mu
  refine Finset.sum_congr rfl (fun c _ => ?_)
  congr 1
  rw [← Finset.sum_mul, M.p_sum c, one_mul]

lemma Ej_xx : M.Ej (fun _ x _ => M.v x ^ 2) = M.EY2 := by
  unfold Ej EY2 s2
  refine Finset.sum_congr rfl (fun c _ => ?_)
  congr 1
  refine Finset.sum_congr rfl (fun x _ => ?_)
  rw [M.sum_p_mul_const c (M.v x ^ 2)]

lemma Ej_yy : M.Ej (fun _ _ y => M.v y ^ 2) = M.EY2 := by
  unfold Ej EY2 s2
  refine Finset.sum_congr rfl (fun c _ => ?_)
  congr 1
  rw [← Finset.sum_mul, M.p_sum c, one_mul]

lemma Ej_xy : M.Ej (fun _ x y => M.v x * M.v y) = ∑ c, M.π c * M.mu c ^ 2 := by
  unfold Ej
  refine Finset.sum_congr rfl (fun c _ => ?_)
  congr 1
  have h1 : ∀ x, ∑ y, M.p c y * (M.v x * M.v y) = M.v x * M.mu c := by
    intro x
    unfold mu
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl (fun y _ => by ring)
  simp_rw [h1]
  have h2 : ∑ x, M.p c x * (M.v x * M.mu c) = (∑ x, M.p c x * M.v x) * M.mu c := by
    rw [Finset.sum_mul]
    exact Finset.sum_congr rfl (fun x _ => by ring)
  rw [h2]
  unfold mu
  ring

lemma Ed_pred (f : C → ℝ) : M.Ed (fun c _ => f c) = ∑ c, M.π c * f c := by
  unfold Ed
  refine Finset.sum_congr rfl (fun c _ => ?_)
  rw [M.sum_p_mul_const c]

lemma Ed_y : M.Ed (fun _ y => M.v y) = M.EY := rfl

lemma Ed_yy : M.Ed (fun _ y => M.v y ^ 2) = M.EY2 := rfl

lemma Ed_gy (a b : ℝ) :
    M.Ed (fun c y => (a * M.mu c + b) * M.v y) = a * ∑ c, M.π c * M.mu c ^ 2 + b * M.EY := by
  unfold Ed
  have h1 : ∀ c, ∑ y, M.p c y * ((a * M.mu c + b) * M.v y) = (a * M.mu c + b) * M.mu c := by
    intro c
    have h : ∀ y, M.p c y * ((a * M.mu c + b) * M.v y)
        = (a * M.mu c + b) * (M.p c y * M.v y) := fun y => by ring
    simp_rw [h]
    rw [← Finset.mul_sum]
    rfl
  simp_rw [h1]
  unfold EY
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl (fun c _ => by ring)

lemma Ed_g (a b : ℝ) : M.Ed (fun c _ => a * M.mu c + b) = a * M.EY + b := by
  rw [Ed_pred]
  unfold EY
  have h1 : ∀ c, M.π c * (a * M.mu c + b) = a * (M.π c * M.mu c) + b * M.π c := fun c => by ring
  simp_rw [h1]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, M.π_sum]
  ring

lemma Ed_gg (a b : ℝ) : M.Ed (fun c _ => (a * M.mu c + b) ^ 2)
    = a ^ 2 * ∑ c, M.π c * M.mu c ^ 2 + 2 * a * b * M.EY + b ^ 2 := by
  rw [Ed_pred]
  have h1 : ∀ c, M.π c * (a * M.mu c + b) ^ 2
      = a ^ 2 * (M.π c * M.mu c ^ 2) + 2 * a * b * (M.π c * M.mu c) + b ^ 2 * M.π c :=
    fun c => by ring
  simp_rw [h1]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
    ← Finset.mul_sum, M.π_sum]
  unfold EY
  ring

/-- Correlation as `cov / (σ_X σ_Y)`. -/
noncomputable def corrOf (cov vx vy : ℝ) : ℝ := cov / (Real.sqrt vx * Real.sqrt vy)

/-- `Corr(X_{p*}, Y)` for the judge that samples `X ~ p*(· | c)` independently of `Y`,
computed from the raw joint moments `E[XY] - E[X]E[Y]`, `E[X²] - E[X]²`, `E[Y²] - E[Y]²`. -/
noncomputable def corrTrue : ℝ :=
  corrOf (M.Ej (fun _ x y => M.v x * M.v y) - M.Ej (fun _ x _ => M.v x) * M.Ej (fun _ _ y => M.v y))
    (M.Ej (fun _ x _ => M.v x ^ 2) - M.Ej (fun _ x _ => M.v x) ^ 2)
    (M.Ej (fun _ _ y => M.v y ^ 2) - M.Ej (fun _ _ y => M.v y) ^ 2)

/-- `Corr(X_δ, Y)` for the collapsed judge `X_δ(c) = a μ_Y(c) + b`. -/
noncomputable def corrCollapsed (a b : ℝ) : ℝ :=
  corrOf
    (M.Ed (fun c y => (a * M.mu c + b) * M.v y)
      - M.Ed (fun c _ => a * M.mu c + b) * M.Ed (fun _ y => M.v y))
    (M.Ed (fun c _ => (a * M.mu c + b) ^ 2) - M.Ed (fun c _ => a * M.mu c + b) ^ 2)
    (M.Ed (fun _ y => M.v y ^ 2) - M.Ed (fun _ y => M.v y) ^ 2)

theorem corrTrue_eq (hY : 0 < M.varY) : M.corrTrue = M.varMu / M.varY := by
  unfold corrTrue corrOf
  rw [Ej_xy, Ej_x, Ej_y, Ej_xx, Ej_yy]
  have hc : ∑ c, M.π c * M.mu c ^ 2 - M.EY * M.EY = M.varMu := by unfold varMu; ring
  have hv : M.EY2 - M.EY ^ 2 = M.varY := rfl
  rw [hc, hv, Real.mul_self_sqrt hY.le]

theorem corrCollapsed_eq (a b : ℝ) (ha : 0 < a) (hvM : 0 < M.varMu) (hY : 0 < M.varY) :
    M.corrCollapsed a b = Real.sqrt M.varMu / Real.sqrt M.varY := by
  unfold corrCollapsed corrOf
  rw [Ed_gy, Ed_g, Ed_gg, Ed_y, Ed_yy]
  have hc : a * ∑ c, M.π c * M.mu c ^ 2 + b * M.EY - (a * M.EY + b) * M.EY = a * M.varMu := by
    unfold varMu; ring
  have hx : a ^ 2 * ∑ c, M.π c * M.mu c ^ 2 + 2 * a * b * M.EY + b ^ 2 - (a * M.EY + b) ^ 2
      = a ^ 2 * M.varMu := by
    unfold varMu; ring
  have hv : M.EY2 - M.EY ^ 2 = M.varY := rfl
  rw [hc, hx, hv, Real.sqrt_mul (sq_nonneg a), Real.sqrt_sq ha.le]
  have h1 : 0 < Real.sqrt M.varMu := Real.sqrt_pos.mpr hvM
  have h2 : 0 < Real.sqrt M.varY := Real.sqrt_pos.mpr hY
  have h3 : M.varMu = Real.sqrt M.varMu * Real.sqrt M.varMu := (Real.mul_self_sqrt hvM.le).symm
  field_simp
  nth_rewrite 1 [h3]
  ring

/-- **Anti-properness of correlation.**  If `Var(μ_Y(C)) > 0` and `E[Var(Y|C)] > 0`, then
`Corr(X_{p*}, Y) = Var(μ_Y(C)) / Var(Y) ∈ (0, 1)`, and the collapsed judge `X_δ = a μ_Y(C) + b`
(`a > 0`) has `Corr(X_δ, Y)² = Corr(X_{p*}, Y)`, hence a strictly larger correlation. -/
theorem anti_proper (a b : ℝ) (ha : 0 < a) (hvM : 0 < M.varMu) (hE : 0 < M.eVar) :
    M.corrTrue = M.varMu / M.varY ∧ 0 < M.corrTrue ∧ M.corrTrue < 1 ∧
    M.corrCollapsed a b ^ 2 = M.corrTrue ∧ M.corrTrue < M.corrCollapsed a b := by
  have hY : 0 < M.varY := by rw [total_variance]; linarith
  have hT := M.corrTrue_eq hY
  have hD := M.corrCollapsed_eq a b ha hvM hY
  have hpos : 0 < M.varMu / M.varY := div_pos hvM hY
  have hlt : M.varMu / M.varY < 1 := by
    rw [div_lt_one hY, total_variance]; linarith
  have hsq : (Real.sqrt M.varMu / Real.sqrt M.varY) ^ 2 = M.varMu / M.varY := by
    rw [div_pow, Real.sq_sqrt hvM.le, Real.sq_sqrt hY.le]
  have hDpos : 0 < Real.sqrt M.varMu / Real.sqrt M.varY :=
    div_pos (Real.sqrt_pos.mpr hvM) (Real.sqrt_pos.mpr hY)
  refine ⟨hT, by rw [hT]; exact hpos, by rw [hT]; exact hlt, ?_, ?_⟩
  · rw [hD, hT]; exact hsq
  · rw [hD, hT]
    set d := Real.sqrt M.varMu / Real.sqrt M.varY with hd
    have hd1 : d < 1 := by nlinarith
    rw [← hsq]
    nlinarith

end Model

/-! ### The hypotheses can hold.  Prompt A: ratings 0 or 2 (probability 1/2 each).
Prompt B: ratings 2 or 4.  The two prompts are equally likely. -/

noncomputable def exampleModel : Model (Fin 2) (Fin 5) where
  π := fun _ => 1 / 2
  p := ![![1 / 2, 0, 1 / 2, 0, 0], ![0, 0, 1 / 2, 0, 1 / 2]]
  v := ![0, 1, 2, 3, 4]
  π_nonneg := by intro c; norm_num
  p_nonneg := by
    intro c y
    fin_cases c <;> fin_cases y <;> norm_num
  π_sum := by norm_num [Fin.sum_univ_two]
  p_sum := by
    intro c
    fin_cases c <;> norm_num [Fin.sum_univ_five]

theorem example_moments :
    exampleModel.varMu = 1 ∧ exampleModel.eVar = 1 ∧ exampleModel.varY = 2 := by
  have hπ : ∀ c, exampleModel.π c = 1 / 2 := fun _ => rfl
  have hmu0 : exampleModel.mu 0 = 1 := by
    norm_num [Model.mu, exampleModel, Fin.sum_univ_five]
  have hmu1 : exampleModel.mu 1 = 3 := by
    norm_num [Model.mu, exampleModel, Fin.sum_univ_five]
  have hs0 : exampleModel.s2 0 = 2 := by
    norm_num [Model.s2, exampleModel, Fin.sum_univ_five]
  have hs1 : exampleModel.s2 1 = 10 := by
    norm_num [Model.s2, exampleModel, Fin.sum_univ_five]
  have hEY : exampleModel.EY = 2 := by
    norm_num [Model.EY, Fin.sum_univ_two, hπ, hmu0, hmu1]
  have hEY2 : exampleModel.EY2 = 6 := by
    norm_num [Model.EY2, Fin.sum_univ_two, hπ, hs0, hs1]
  refine ⟨?_, ?_, ?_⟩
  · norm_num [Model.varMu, Fin.sum_univ_two, hπ, hmu0, hmu1, hEY]
  · norm_num [Model.eVar, Fin.sum_univ_two, hπ, hs0, hs1, hmu0, hmu1]
  · norm_num [Model.varY, hEY, hEY2]

/-- For the example: `Corr(X_{p*}, Y) = 1/2` and the collapsed judge has `Corr² = 1/2`, so
`Corr(X_δ, Y) = √(1/2) ≈ 0.7071 > 0.5`. -/
theorem example_anti_proper :
    exampleModel.corrTrue = 1 / 2 ∧ exampleModel.corrCollapsed 1 0 ^ 2 = 1 / 2 ∧
    (1 / 2 : ℝ) < exampleModel.corrCollapsed 1 0 := by
  obtain ⟨h1, h2, h3⟩ := example_moments
  have h := Model.anti_proper exampleModel 1 0 one_pos (by rw [h1]; norm_num) (by rw [h2]; norm_num)
  obtain ⟨hT, _, _, hsq, hlt⟩ := h
  have hT' : exampleModel.corrTrue = 1 / 2 := by rw [hT, h1, h3]
  rw [hT'] at hlt
  exact ⟨hT', by rw [hsq, hT'], hlt⟩

end AntiProper

/-! ## Section C. The ordinal `{0,…,4}` example: Brier 4×, RPS 2×, cross-entropy `ln 2` versus `+∞`.

Same law as `exampleModel`.  The true forecast is `q = p*`; the collapsed forecast is the point mass
at the conditional mean (`1` for prompt A, `3` for prompt B). -/

section Scores

/-- Brier loss `‖q − e_y‖²`. -/
def brierLoss (q : Fin 5 → ℝ) (y : Fin 5) : ℝ := ∑ k, (q k - if k = y then 1 else 0) ^ 2

/-- `F_q(t) = ∑_{k ≤ t} q k`. -/
def cdf (q : Fin 5 → ℝ) (t : Fin 5) : ℝ := ∑ k, if k ≤ t then q k else 0

/-- Ranked probability score `‖F_q − F_{e_y}‖²` (the term `t = 4` is `(1 − 1)² = 0`). -/
def rpsLoss (q : Fin 5 → ℝ) (y : Fin 5) : ℝ := ∑ t, (cdf q t - if y ≤ t then 1 else 0) ^ 2

/-- Expected loss of the forecasts `q c` under the example's rating law. -/
noncomputable def exampleRisk (loss : (Fin 5 → ℝ) → Fin 5 → ℝ) (q : Fin 2 → Fin 5 → ℝ) : ℝ :=
  ∑ c, exampleModel.π c * ∑ y, exampleModel.p c y * loss (q c) y

noncomputable def qTrue : Fin 2 → Fin 5 → ℝ := exampleModel.p

def qCollapsed : Fin 2 → Fin 5 → ℝ := ![![0, 1, 0, 0, 0], ![0, 0, 0, 1, 0]]

theorem example_scores :
    exampleRisk brierLoss qTrue = 1 / 2 ∧ exampleRisk brierLoss qCollapsed = 2 ∧
    exampleRisk rpsLoss qTrue = 1 / 2 ∧ exampleRisk rpsLoss qCollapsed = 1 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
  · norm_num [exampleRisk, brierLoss, rpsLoss, cdf, qTrue, qCollapsed, exampleModel,
      Fin.sum_univ_two, Fin.sum_univ_five]

/-- The collapsed forecast gives probability `0` to every rating that can occur, so its
log-loss `−log 0` is `+∞` on every draw; the true forecast gives probability `1/2`. -/
theorem example_ce_support :
    (∀ c y, exampleModel.p c y ≠ 0 → qCollapsed c y = 0) ∧
    (∀ c y, exampleModel.p c y ≠ 0 → qTrue c y = 1 / 2) := by
  constructor <;>
  · intro c y
    fin_cases c <;> fin_cases y <;> norm_num [qCollapsed, qTrue, exampleModel]

/-- Expected log-loss of the true forecast is `ln 2 ≈ 0.6931`. -/
theorem example_ce_true :
    (∑ c, exampleModel.π c * ∑ y, exampleModel.p c y * (-Real.log (qTrue c y))) = Real.log 2 := by
  have h : Real.log (1 / 2) = -Real.log 2 := by rw [one_div, Real.log_inv]
  norm_num [qTrue, exampleModel, Fin.sum_univ_two, Fin.sum_univ_five, h]
  ring

theorem example_ln2 : (0.6931 : ℝ) < Real.log 2 ∧ Real.log 2 < 0.6932 := by
  constructor
  · have := Real.log_two_gt_d9; linarith
  · have := Real.log_two_lt_d9; linarith

end Scores

/-! ## Section D. Smaller claims. -/

section Small

/-- Taking `√` per sample and then averaging under-states `√` of the average risk:
`√RPS` must be applied to the *average* RPS, not averaged per rating. -/
theorem sqrt_mean_lt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a ≠ b) :
    (Real.sqrt a + Real.sqrt b) / 2 < Real.sqrt ((a + b) / 2) := by
  have h1 : 0 ≤ (Real.sqrt a + Real.sqrt b) / 2 := by positivity
  rw [Real.lt_sqrt h1]
  have hsa := Real.sq_sqrt ha
  have hsb := Real.sq_sqrt hb
  have hne : Real.sqrt a ≠ Real.sqrt b := fun h => hab (by rw [← hsa, ← hsb, h])
  nlinarith [sq_pos_of_ne_zero (sub_ne_zero.mpr hne)]

theorem sqrt_mean_example : (Real.sqrt 0 + Real.sqrt 1) / 2 < Real.sqrt ((0 + 1) / 2) :=
  sqrt_mean_lt 0 1 le_rfl zero_le_one (by norm_num)

end Small

/-! ## Section E. The binary example `p* ∈ {0.1, 0.5, 0.9}`, `q₂ = 0.01 p* + 0.90`. -/

section Binary

noncomputable def binModel : Model (Fin 3) (Fin 2) where
  π := fun _ => 1 / 3
  p := ![![9 / 10, 1 / 10], ![1 / 2, 1 / 2], ![1 / 10, 9 / 10]]
  v := ![0, 1]
  π_nonneg := by intro c; norm_num
  p_nonneg := by
    intro c y
    fin_cases c <;> fin_cases y <;> norm_num
  π_sum := by norm_num [Fin.sum_univ_three]
  p_sum := by
    intro c
    fin_cases c <;> norm_num [Fin.sum_univ_two]

theorem binary_moments : binModel.varMu = 8 / 75 ∧ binModel.varY = 1 / 4 ∧ binModel.eVar = 43 / 300 := by
  have hπ : ∀ c, binModel.π c = 1 / 3 := fun _ => rfl
  have hmu : ∀ c, binModel.mu c = ![1 / 10, 1 / 2, 9 / 10] c := by
    intro c; fin_cases c <;> norm_num [Model.mu, binModel, Fin.sum_univ_two]
  have hs : ∀ c, binModel.s2 c = ![1 / 10, 1 / 2, 9 / 10] c := by
    intro c; fin_cases c <;> norm_num [Model.s2, binModel, Fin.sum_univ_two]
  have hEY : binModel.EY = 1 / 2 := by
    norm_num [Model.EY, Fin.sum_univ_three, hπ, hmu]
  have hEY2 : binModel.EY2 = 1 / 2 := by
    norm_num [Model.EY2, Fin.sum_univ_three, hπ, hs]
  refine ⟨?_, ?_, ?_⟩
  · norm_num [Model.varMu, Fin.sum_univ_three, hπ, hmu, hEY]
  · norm_num [Model.varY, hEY, hEY2]
  · norm_num [Model.eVar, Fin.sum_univ_three, hπ, hs, hmu]

/-- Judge 1 reports `q₁ = p*`; judge 2 reports `q₂ = 0.01 p* + 0.90`.  Both are affine in `μ_Y(c) = p*`,
so they have the same Pearson correlation with `Y`, namely `√(Var(p*)/Var(Y)) = √(32/75) ≈ 0.6532`. -/
theorem binary_pearson_tie :
    binModel.corrCollapsed 1 0 = binModel.corrCollapsed (1 / 100) (9 / 10) ∧
    (0.6531 : ℝ) < binModel.corrCollapsed 1 0 ∧ binModel.corrCollapsed 1 0 < 0.6532 := by
  obtain ⟨h1, h2, _⟩ := binary_moments
  have hv : 0 < binModel.varMu := by rw [h1]; norm_num
  have hY : 0 < binModel.varY := by rw [h2]; norm_num
  have e1 := binModel.corrCollapsed_eq 1 0 one_pos hv hY
  have e2 := binModel.corrCollapsed_eq (1 / 100) (9 / 10) (by norm_num) hv hY
  have hq : binModel.corrCollapsed 1 0 = Real.sqrt (32 / 75) := by
    rw [e1, ← Real.sqrt_div hv.le, h1, h2]; norm_num
  refine ⟨by rw [e1, e2], ?_, ?_⟩
  · rw [hq]
    exact Real.lt_sqrt (by norm_num) |>.mpr (by norm_num)
  · rw [hq]
    exact (Real.sqrt_lt' (by norm_num)).mpr (by norm_num)

/-- Binary Brier risk `E[(q(c) − Y)²]` of a forecast `q c` of `P(Y = 1 | c)`. -/
noncomputable def binBrier (q : Fin 3 → ℝ) : ℝ :=
  ∑ c, binModel.π c * ∑ y, binModel.p c y * (q c - binModel.v y) ^ 2

theorem binary_brier :
    binBrier ![1 / 10, 1 / 2, 9 / 10] = 43 / 300 ∧
    binBrier (fun c => (1 / 100) * (![1 / 10, 1 / 2, 9 / 10] : Fin 3 → ℝ) c + 9 / 10)
      = 1235707 / 3000000 ∧
    (2.87 : ℝ) < (1235707 / 3000000) / (43 / 300) ∧ (1235707 / 3000000 : ℝ) / (43 / 300) < 2.88 := by
  refine ⟨?_, ?_, by norm_num, by norm_num⟩ <;>
  · norm_num [binBrier, binModel, Fin.sum_univ_three, Fin.sum_univ_two]

end Binary

end PlanAudit

#print axioms PlanAudit.expV_coord
#print axioms PlanAudit.unbiased_weighted_sum
#print axioms PlanAudit.plan_estimator_unbiased
#print axioms PlanAudit.plan_estimator_unbiased_product
#print axioms PlanAudit.instance_unbiased
#print axioms PlanAudit.adaptive_stopping_biased
#print axioms PlanAudit.Model.total_variance
#print axioms PlanAudit.Model.corrTrue_eq
#print axioms PlanAudit.Model.corrCollapsed_eq
#print axioms PlanAudit.Model.anti_proper
#print axioms PlanAudit.example_moments
#print axioms PlanAudit.example_anti_proper
#print axioms PlanAudit.example_scores
#print axioms PlanAudit.example_ce_support
#print axioms PlanAudit.example_ce_true
#print axioms PlanAudit.example_ln2
#print axioms PlanAudit.sqrt_mean_lt
#print axioms PlanAudit.sqrt_mean_example
#print axioms PlanAudit.binary_moments
#print axioms PlanAudit.binary_pearson_tie
#print axioms PlanAudit.binary_brier
