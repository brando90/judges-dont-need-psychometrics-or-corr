import Mathlib

/-!
# Experiment 40: output-space boundary, elicitation, moment-blindness, and the reliability ceiling

What each statement covers (corrected 11-27-2026 after an independent review; the earlier header said "every theorem is general", which was false):
* general in the number of classes `K` or items `n`: `mean_not_identifying` (every K ≥ 3), `sampled_accuracy_le_max`,
  `sampled_accuracy_eq_max_iff`, `truthful_sampled_accuracy_optimal_iff` (any K; the bound `m` is any upper bound of `p`),
  `total_variance`, `icc_eq_rho`, `cov_sq_le`, `covPooled_eq_covF`, `ceiling_sq`, `ceiling_rho`, `ceiling_pooled_rho`, `ceiling_attained`.
* about exactly two classes by construction: `binary_mean_eq`, `binary_mean_identifies`, `brier_binary_local`, `decision_accuracy_proper`.
* INSTANCES (one concrete example each, stated as an existence claim, not a general theorem): `accuracy_prefers_overconfidence` (K = 2),
  `decision_accuracy_not_strict` (K = 2), `brier_not_local_three` (K = 3 only), `same_two_moments_different_law` (K = 5).
No `sorry`, no custom axiom. The compiler output and `#print axioms` are saved next to this file.

Notation. A rating law on K ordered classes with values 0,…,K−1 is a `p : Fin K → ℝ` with `p i ≥ 0` and `∑ p i = 1`.
The mean of `p` is `∑ i, (i : ℝ) * p i`.

Part A  (K = 2 versus K ≥ 3).   The mean identifies the law exactly when K = 2.
Part B  (what accuracy elicits). Accuracy rewards the mode and nothing else; sampled accuracy punishes honest forecasts.
Part C  (Brier is local for K = 2; an instance at K = 3 where it is not).
Part D  (moment-blindness).     An instance: two laws with the same mean AND the same variance that differ (squared distance 7/40).
Part E  (reliability).          Law of total variance on a finite population; the one-way intraclass correlation equals ρ.
Part F  (the ceiling).          On a finite population of items with rating laws, the squared correlation of any item-level score
                                with a single pooled rating is at most ρ (`ceiling_pooled_rho`), and the item mean attains ρ
                                (`ceiling_attained_pooled`). The tower step Cov(f(c), Y) = Cov(f(c), μ(c)) is the theorem `covPooled_eq_covF`.
-/

namespace E40

open Finset

/-! ## Part A: for K = 2 the mean identifies the law; for every K ≥ 3 it does not -/

/-- On two classes `{0,1}` the mean of a law is the probability of class 1. -/
theorem binary_mean_eq (p : Fin 2 → ℝ) : ∑ i : Fin 2, ((i : ℕ) : ℝ) * p i = p 1 := by
  simp [Fin.sum_univ_two]

/-- **K = 2: the mean identifies the law.** Two laws on `{0,1}` with the same mean are equal. -/
theorem binary_mean_identifies (p q : Fin 2 → ℝ) (hp : p 0 + p 1 = 1) (hq : q 0 + q 1 = 1)
    (h : ∑ i : Fin 2, ((i : ℕ) : ℝ) * p i = ∑ i : Fin 2, ((i : ℕ) : ℝ) * q i) : p = q := by
  rw [binary_mean_eq, binary_mean_eq] at h
  funext i
  fin_cases i
  · show p 0 = q 0
    linarith
  · exact h

/-- **K ≥ 3 (every K): the mean does not identify the law.** There are two different laws on `{0,…,K-1}`
with the same mean: a point mass at 1, and one half at 0 plus one half at 2. -/
theorem mean_not_identifying (K : ℕ) (hK : 3 ≤ K) :
    ∃ p q : Fin K → ℝ,
      (∀ i, 0 ≤ p i) ∧ (∀ i, 0 ≤ q i) ∧ ∑ i, p i = 1 ∧ ∑ i, q i = 1 ∧
      ∑ i : Fin K, ((i : ℕ) : ℝ) * p i = ∑ i : Fin K, ((i : ℕ) : ℝ) * q i ∧ p ≠ q := by
  obtain ⟨a0, ha0⟩ : ∃ a : Fin K, (a : ℕ) = 0 := ⟨⟨0, by omega⟩, rfl⟩
  obtain ⟨a1, ha1⟩ : ∃ a : Fin K, (a : ℕ) = 1 := ⟨⟨1, by omega⟩, rfl⟩
  obtain ⟨a2, ha2⟩ : ∃ a : Fin K, (a : ℕ) = 2 := ⟨⟨2, by omega⟩, rfl⟩
  have h12 : a1 ≠ a2 := by
    intro h; have := congrArg Fin.val h; omega
  have h10 : a1 ≠ a0 := by
    intro h; have := congrArg Fin.val h; omega
  refine ⟨fun i => if i = a1 then 1 else 0,
          fun i => (if i = a0 then (1 / 2 : ℝ) else 0) + (if i = a2 then (1 / 2 : ℝ) else 0), ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro i
    show (0 : ℝ) ≤ if i = a1 then 1 else 0
    split_ifs <;> norm_num
  · intro i
    have h1 : (0 : ℝ) ≤ if i = a0 then (1 / 2 : ℝ) else 0 := by split_ifs <;> norm_num
    have h2 : (0 : ℝ) ≤ if i = a2 then (1 / 2 : ℝ) else 0 := by split_ifs <;> norm_num
    exact add_nonneg h1 h2
  · simp
  · simp [Finset.sum_add_distrib]
    norm_num
  · simp [mul_add, mul_ite, Finset.sum_add_distrib, ha0, ha1, ha2]
  · intro h
    have := congrFun h a1
    simp [h10, h12] at this

/-! ## Part B: accuracy elicits only the mode -/

/-- Expected *sampled* accuracy of a forecast `q` when the truth is `p`: the forecaster draws a label from `q`. -/
def sampledAccuracy {K : ℕ} (p q : Fin K → ℝ) : ℝ := ∑ i, q i * p i

/-- No forecast can beat the largest class probability `m`. -/
theorem sampled_accuracy_le_max {K : ℕ} (p q : Fin K → ℝ) (hq0 : ∀ i, 0 ≤ q i) (hq1 : ∑ i, q i = 1)
    (m : ℝ) (hm : ∀ i, p i ≤ m) : sampledAccuracy p q ≤ m := by
  unfold sampledAccuracy
  calc ∑ i, q i * p i ≤ ∑ i, q i * m :=
        Finset.sum_le_sum (fun i _ => mul_le_mul_of_nonneg_left (hm i) (hq0 i))
    _ = m := by rw [← Finset.sum_mul, hq1, one_mul]

/-- For any upper bound `m` of `p` (in particular `m = max p`), the bound is attained exactly by forecasts that put all their
mass on classes of probability exactly `m`. (If `m` is a strict upper bound, both sides are false.) -/
theorem sampled_accuracy_eq_max_iff {K : ℕ} (p q : Fin K → ℝ) (hq0 : ∀ i, 0 ≤ q i) (hq1 : ∑ i, q i = 1)
    (m : ℝ) (hm : ∀ i, p i ≤ m) :
    sampledAccuracy p q = m ↔ ∀ i, q i ≠ 0 → p i = m := by
  unfold sampledAccuracy
  constructor
  · intro h i hi
    have hzero : ∑ j, q j * (m - p j) = 0 := by
      have : ∑ j, q j * (m - p j) = m - ∑ j, q j * p j := by
        have h1 : ∑ j, q j * (m - p j) = ∑ j, (q j * m - q j * p j) := by
          apply Finset.sum_congr rfl; intro j _; ring
        rw [h1, Finset.sum_sub_distrib, ← Finset.sum_mul, hq1, one_mul]
      rw [this, h]; ring
    have hnn : ∀ j ∈ (Finset.univ : Finset (Fin K)), 0 ≤ q j * (m - p j) :=
      fun j _ => mul_nonneg (hq0 j) (sub_nonneg.mpr (hm j))
    have := (Finset.sum_eq_zero_iff_of_nonneg hnn).mp hzero i (Finset.mem_univ i)
    rcases mul_eq_zero.mp this with h0 | h0
    · exact absurd h0 hi
    · linarith
  · intro h
    have : ∀ i ∈ (Finset.univ : Finset (Fin K)), q i * p i = q i * m := by
      intro i _
      by_cases hi : q i = 0
      · simp [hi]
      · rw [h i hi]
    rw [Finset.sum_congr rfl this, ← Finset.sum_mul, hq1, one_mul]

/-- **Truthful reporting reaches the bound `m` exactly when every class of positive probability has probability `m`.** For
`m = max p` this says: truthful reporting maximizes sampled accuracy only for laws that are uniform on their support. -/
theorem truthful_sampled_accuracy_optimal_iff {K : ℕ} (p : Fin K → ℝ) (hp0 : ∀ i, 0 ≤ p i) (hp1 : ∑ i, p i = 1)
    (m : ℝ) (hm : ∀ i, p i ≤ m) :
    sampledAccuracy p p = m ↔ ∀ i, p i ≠ 0 → p i = m :=
  sampled_accuracy_eq_max_iff p p hp0 hp1 m hm

/-- Instance: the honest forecast `(3/5, 2/5)` has sampled accuracy `13/25`, below the `3/5` of the overconfident
point mass on the mode. -/
theorem accuracy_prefers_overconfidence :
    sampledAccuracy (![3 / 5, 2 / 5] : Fin 2 → ℝ) (![3 / 5, 2 / 5] : Fin 2 → ℝ) = 13 / 25 ∧
    sampledAccuracy (![3 / 5, 2 / 5] : Fin 2 → ℝ) (![1, 0] : Fin 2 → ℝ) = 3 / 5 ∧
    (13 / 25 : ℝ) < 3 / 5 := by
  refine ⟨?_, ?_, by norm_num⟩
  · simp [sampledAccuracy, Fin.sum_univ_two]; norm_num
  · simp [sampledAccuracy, Fin.sum_univ_two]

/-- Decision accuracy (take the class with the larger forecast mass, ties to 0) on two classes. -/
noncomputable def decisionAccuracy (p q : Fin 2 → ℝ) : ℝ := p (if q 1 ≤ q 0 then 0 else 1)

/-- **Decision accuracy is proper (K = 2).** No forecast `q` earns more expected decision accuracy than the true law `p` itself. -/
theorem decision_accuracy_proper (p q : Fin 2 → ℝ) : decisionAccuracy p q ≤ decisionAccuracy p p := by
  unfold decisionAccuracy
  by_cases hq : q 1 ≤ q 0 <;> by_cases hp : p 1 ≤ p 0 <;> simp [hq, hp] <;> linarith

/-- **INSTANCE (K = 2): decision accuracy is not strictly proper.** The honest forecast `(3/5, 2/5)` and the different point-mass
forecast `(1, 0)` with the same argmax earn exactly the same expected decision accuracy. (Propriety itself is
`decision_accuracy_proper` above; this theorem shows one example where a different forecast ties.) -/
theorem decision_accuracy_not_strict :
    decisionAccuracy (![3 / 5, 2 / 5] : Fin 2 → ℝ) (![3 / 5, 2 / 5] : Fin 2 → ℝ)
      = decisionAccuracy (![3 / 5, 2 / 5] : Fin 2 → ℝ) (![1, 0] : Fin 2 → ℝ) ∧
    (![3 / 5, 2 / 5] : Fin 2 → ℝ) ≠ (![1, 0] : Fin 2 → ℝ) := by
  refine ⟨?_, ?_⟩
  · simp [decisionAccuracy]
    norm_num
  · intro h
    have := congrFun h 0
    norm_num at this

/-! ## Part C: the Brier score of the realized outcome depends only on the probability given to it when K = 2 -/

/-- Brier score of forecast `q` at outcome `y`: squared distance to the one-hot vector. -/
def brier {K : ℕ} (q : Fin K → ℝ) (y : Fin K) : ℝ := ∑ j, (q j - (if j = y then 1 else 0)) ^ 2

/-- **K = 2: Brier is local.** For forecasts on two classes, the score at outcome `y` is `2 (1 - q y)²`. -/
theorem brier_binary_local (q : Fin 2 → ℝ) (hq : q 0 + q 1 = 1) (y : Fin 2) :
    brier q y = 2 * (1 - q y) ^ 2 := by
  unfold brier
  fin_cases y
  · simp [Fin.sum_univ_two]
    have : q 1 = 1 - q 0 := by linarith
    rw [this]; ring
  · simp [Fin.sum_univ_two]
    have : q 0 = 1 - q 1 := by linarith
    rw [this]; ring

/-- **INSTANCE (K = 3 only): Brier is not local.** Two forecasts give class 0 the same probability `1/2` but receive different
Brier scores when class 0 occurs (`1/2` versus `3/8`). Padding both forecasts with zeros gives the same for every larger K;
that generalization is not proved here. -/
theorem brier_not_local_three :
    ∃ q q' : Fin 3 → ℝ, ∑ i, q i = 1 ∧ ∑ i, q' i = 1 ∧ q 0 = q' 0 ∧ brier q 0 ≠ brier q' 0 := by
  refine ⟨![1 / 2, 1 / 2, 0], ![1 / 2, 1 / 4, 1 / 4], ?_, ?_, ?_, ?_⟩
  · simp [Fin.sum_univ_three]; norm_num
  · simp [Fin.sum_univ_three]; norm_num
  · simp
  · simp [brier, Fin.sum_univ_three]
    norm_num

/-! ## Part D: moment-blindness (what a variance-components study cannot see) -/

/-- **INSTANCE (K = 5): same two moments, different law.** Two different laws on `{0,1,2,3,4}` with the same mean (2) and the same
second moment (hence the same variance), whose squared distance (the Brier divergence) is `7/40`. A generalizability study sees
only item means and within-item variances, so it cannot tell these two laws apart. -/
theorem same_two_moments_different_law :
    ∃ p q : Fin 5 → ℝ,
      (∀ i, 0 ≤ p i) ∧ (∀ i, 0 ≤ q i) ∧ ∑ i, p i = 1 ∧ ∑ i, q i = 1 ∧
      ∑ i : Fin 5, ((i : ℕ) : ℝ) * p i = ∑ i : Fin 5, ((i : ℕ) : ℝ) * q i ∧
      ∑ i : Fin 5, ((i : ℕ) : ℝ) ^ 2 * p i = ∑ i : Fin 5, ((i : ℕ) : ℝ) ^ 2 * q i ∧
      p ≠ q ∧ ∑ i, (p i - q i) ^ 2 = 7 / 40 := by
  refine ⟨![1 / 4, 0, 1 / 2, 0, 1 / 4], ![1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5], ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro i; fin_cases i <;> simp
  · intro i; fin_cases i <;> simp
  · simp [Fin.sum_univ_five]; norm_num
  · simp [Fin.sum_univ_five]; norm_num
  · simp [Fin.sum_univ_five]; norm_num
  · simp [Fin.sum_univ_five]; norm_num
  · intro h
    have := congrFun h 0
    norm_num at this
  · simp [Fin.sum_univ_five]; norm_num

/-! ## Part E: law of total variance on a finite population, and ICC(1) = ρ -/

section Reliability

variable {n K : ℕ}

/-- Item mean: items `c : Fin n` carry rating laws `p c : Fin K → ℝ` on class values `v`. -/
def itemMean (v : Fin K → ℝ) (p : Fin n → Fin K → ℝ) (c : Fin n) : ℝ := ∑ y, v y * p c y

/-- Item second moment. -/
def itemSecond (v : Fin K → ℝ) (p : Fin n → Fin K → ℝ) (c : Fin n) : ℝ := ∑ y, v y ^ 2 * p c y

/-- Variance of the pooled rating `Y` when an item is drawn uniformly and then a rating from its law. -/
noncomputable def totalVar (v : Fin K → ℝ) (p : Fin n → Fin K → ℝ) : ℝ :=
  (∑ c, itemSecond v p c) / n - ((∑ c, itemMean v p c) / n) ^ 2

/-- Variance across items of the item mean (the between-item component σ_c²). -/
noncomputable def betweenVar (v : Fin K → ℝ) (p : Fin n → Fin K → ℝ) : ℝ :=
  (∑ c, itemMean v p c ^ 2) / n - ((∑ c, itemMean v p c) / n) ^ 2

/-- Average within-item variance (the within-item component σ_w²). -/
noncomputable def withinVar (v : Fin K → ℝ) (p : Fin n → Fin K → ℝ) : ℝ :=
  (∑ c, (itemSecond v p c - itemMean v p c ^ 2)) / n

/-- **Law of total variance** on a finite population: `Var(Y) = Var(μ(c)) + E[Var(Y | c)]`. -/
theorem total_variance (v : Fin K → ℝ) (p : Fin n → Fin K → ℝ) :
    totalVar v p = betweenVar v p + withinVar v p := by
  unfold totalVar betweenVar withinVar
  rw [Finset.sum_sub_distrib]
  ring

/-- **The one-way intraclass correlation is ρ.** With `σ_c² = Var(μ(c))`, `σ_w² = E Var(Y|c)` the one-way reliability of
a single rating, `σ_c² / (σ_c² + σ_w²)`, equals `Var(μ(c)) / Var(Y)`, the number the repository calls ρ. -/
theorem icc_eq_rho (v : Fin K → ℝ) (p : Fin n → Fin K → ℝ) :
    betweenVar v p / (betweenVar v p + withinVar v p) = betweenVar v p / totalVar v p := by
  rw [total_variance]

end Reliability

/-! ## Part F: the reliability ceiling (finite population, equal item weights) -/

section Ceiling

variable {n : ℕ}

/-- Population covariance of two item-level quantities under the uniform law on items. -/
noncomputable def covF (f g : Fin n → ℝ) : ℝ := (∑ c, f c * g c) / n - ((∑ c, f c) / n) * ((∑ c, g c) / n)

/-- Population variance of an item-level quantity. -/
noncomputable def varF (f : Fin n → ℝ) : ℝ := covF f f

/-- The covariance is the mean product of the centered quantities. -/
theorem covF_centered (hn : 0 < n) (a b : Fin n → ℝ) :
    covF a b = (∑ c, (a c - (∑ d, a d) / n) * (b c - (∑ d, b d) / n)) / n := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  unfold covF
  have hexp : ∀ c, (a c - (∑ d, a d) / n) * (b c - (∑ d, b d) / n)
      = a c * b c - ((∑ d, b d) / n) * a c - ((∑ d, a d) / n) * b c
        + ((∑ d, a d) / n) * ((∑ d, b d) / n) := by
    intro c; ring
  rw [Finset.sum_congr rfl (fun c _ => hexp c), Finset.sum_add_distrib, Finset.sum_sub_distrib,
    Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  field_simp
  ring

/-- **Cauchy–Schwarz for item-level quantities:** `Cov(f, g)² ≤ Var(f) Var(g)`. -/
theorem cov_sq_le (hn : 0 < n) (f g : Fin n → ℝ) : covF f g ^ 2 ≤ varF f * varF g := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hnpos : (0 : ℝ) < (n : ℝ) ^ 2 := by positivity
  unfold varF
  rw [covF_centered hn f g, covF_centered hn f f, covF_centered hn g g]
  have cs := Finset.sum_mul_sq_le_sq_mul_sq (Finset.univ : Finset (Fin n))
      (fun c => f c - (∑ d, f d) / n) (fun c => g c - (∑ d, g d) / n)
  have e1 : ∀ x : Fin n → ℝ, ∑ c, (x c - (∑ d, x d) / n) * (x c - (∑ d, x d) / n)
      = ∑ c, (x c - (∑ d, x d) / n) ^ 2 := by
    intro x; apply Finset.sum_congr rfl; intro c _; ring
  rw [e1 f, e1 g]
  set Sfg := ∑ c, (f c - (∑ d, f d) / n) * (g c - (∑ d, g d) / n) with hSfg
  set Sff := ∑ c, (f c - (∑ d, f d) / n) ^ 2 with hSff
  set Sgg := ∑ c, (g c - (∑ d, g d) / n) ^ 2 with hSgg
  calc (Sfg / n) ^ 2 = Sfg ^ 2 / (n : ℝ) ^ 2 := by ring
    _ ≤ (Sff * Sgg) / (n : ℝ) ^ 2 := by gcongr
    _ = Sff / n * (Sgg / n) := by ring

end Ceiling

section CeilingReliability

variable {n K : ℕ}

/-- The between-item variance is the variance of the item mean. -/
theorem betweenVar_eq_varF (v : Fin K → ℝ) (p : Fin n → Fin K → ℝ) :
    betweenVar v p = varF (itemMean v p) := by
  unfold betweenVar varF covF
  simp [sq]

/-- Covariance, over the pooled population "draw an item `c` uniformly, then a rating `y` with probability `p c y`", of an item-level
score `f(c)` with the rating value `v y`: `E[f(c) Y] − E[f(c)] E[Y]`, computed from the double sum (it does not use the item mean). -/
noncomputable def covPooled (v : Fin K → ℝ) (p : Fin n → Fin K → ℝ) (f : Fin n → ℝ) : ℝ :=
  (∑ c, ∑ y, p c y * (f c * v y)) / n - ((∑ c, ∑ y, p c y * f c) / n) * ((∑ c, ∑ y, p c y * v y) / n)

/-- **Tower step.** If every item's law sums to one, the pooled covariance of an item-level score with a single rating equals its
covariance with the item mean: `Cov(f(c), Y) = Cov(f(c), μ(c))`. -/
theorem covPooled_eq_covF (v : Fin K → ℝ) (p : Fin n → Fin K → ℝ) (f : Fin n → ℝ) (hp : ∀ c, ∑ y, p c y = 1) :
    covPooled v p f = covF f (itemMean v p) := by
  unfold covPooled covF itemMean
  have h1 : ∀ c, ∑ y, p c y * (f c * v y) = f c * ∑ y, v y * p c y := by
    intro c
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro y _
    ring
  have h2 : ∀ c, ∑ y, p c y * f c = f c := by
    intro c
    rw [← Finset.sum_mul, hp c, one_mul]
  have h3 : ∀ c, ∑ y, p c y * v y = ∑ y, v y * p c y := by
    intro c
    apply Finset.sum_congr rfl
    intro y _
    ring
  simp only [h1, h2, h3]

/-- **Cauchy–Schwarz bound on the covariance with the item mean.** For any item-level score `f`,
`Cov(f(c), μ(c))² ≤ Var(f) · Var(μ(c))`. By the tower step `covPooled_eq_covF`, `Cov(f(c), μ(c))` is the covariance of `f(c)` with a
single pooled rating `Y`; `ceiling_pooled_rho` below states the bound in that form. -/
theorem ceiling_sq (hn : 0 < n) (v : Fin K → ℝ) (p : Fin n → Fin K → ℝ) (f : Fin n → ℝ) :
    covF f (itemMean v p) ^ 2 ≤ varF f * betweenVar v p := by
  rw [betweenVar_eq_varF]
  exact cov_sq_le hn f (itemMean v p)

/-- Dividing by `Var(f) · Var(Y)`: `Cov(f(c), μ(c))² / (Var(f) Var(Y)) ≤ ρ = Var(μ(c)) / Var(Y)`. (Here the numerator is the covariance
with the item mean; `ceiling_pooled_rho` restates it as the squared correlation with a single pooled rating.) -/
theorem ceiling_rho (hn : 0 < n) (v : Fin K → ℝ) (p : Fin n → Fin K → ℝ) (f : Fin n → ℝ)
    (hf : 0 < varF f) (hT : 0 < totalVar v p) :
    covF f (itemMean v p) ^ 2 / (varF f * totalVar v p) ≤ betweenVar v p / totalVar v p := by
  have h := ceiling_sq hn v p f
  have hpos : 0 < varF f * totalVar v p := mul_pos hf hT
  calc covF f (itemMean v p) ^ 2 / (varF f * totalVar v p)
      ≤ (varF f * betweenVar v p) / (varF f * totalVar v p) := by gcongr
    _ = betweenVar v p / totalVar v p := by field_simp

/-- **Reliability ceiling, pooled form.** If every item's law sums to one, then for every item-level score `f` with positive variance
the squared correlation of `f(c)` with a single pooled rating `Y` is at most `ρ = Var(μ(c)) / Var(Y)`. -/
theorem ceiling_pooled_rho (hn : 0 < n) (v : Fin K → ℝ) (p : Fin n → Fin K → ℝ) (f : Fin n → ℝ) (hp : ∀ c, ∑ y, p c y = 1)
    (hf : 0 < varF f) (hT : 0 < totalVar v p) :
    covPooled v p f ^ 2 / (varF f * totalVar v p) ≤ betweenVar v p / totalVar v p := by
  rw [covPooled_eq_covF v p f hp]
  exact ceiling_rho hn v p f hf hT

/-- The Cauchy–Schwarz bound holds with equality for the deterministic item mean itself. -/
theorem ceiling_attained (v : Fin K → ℝ) (p : Fin n → Fin K → ℝ) :
    covF (itemMean v p) (itemMean v p) ^ 2 = varF (itemMean v p) * betweenVar v p := by
  rw [betweenVar_eq_varF]
  unfold varF
  ring

/-- The ceiling `ρ` is attained by the item mean as a judge: with `f = μ`, the squared correlation with a single pooled rating equals `ρ`. -/
theorem ceiling_attained_pooled (v : Fin K → ℝ) (p : Fin n → Fin K → ℝ) (hp : ∀ c, ∑ y, p c y = 1)
    (hf : 0 < varF (itemMean v p)) (hT : 0 < totalVar v p) :
    covPooled v p (itemMean v p) ^ 2 / (varF (itemMean v p) * totalVar v p) = betweenVar v p / totalVar v p := by
  rw [covPooled_eq_covF v p _ hp, betweenVar_eq_varF]
  have hf' : varF (itemMean v p) ≠ 0 := ne_of_gt hf
  have hT' : totalVar v p ≠ 0 := ne_of_gt hT
  have e : covF (itemMean v p) (itemMean v p) = varF (itemMean v p) := rfl
  rw [e]
  field_simp

end CeilingReliability

end E40

#print axioms E40.binary_mean_eq
#print axioms E40.binary_mean_identifies
#print axioms E40.mean_not_identifying
#print axioms E40.sampled_accuracy_le_max
#print axioms E40.sampled_accuracy_eq_max_iff
#print axioms E40.truthful_sampled_accuracy_optimal_iff
#print axioms E40.accuracy_prefers_overconfidence
#print axioms E40.decision_accuracy_not_strict
#print axioms E40.brier_binary_local
#print axioms E40.brier_not_local_three
#print axioms E40.same_two_moments_different_law
#print axioms E40.total_variance
#print axioms E40.icc_eq_rho
#print axioms E40.cov_sq_le
#print axioms E40.covF_centered
#print axioms E40.betweenVar_eq_varF
#print axioms E40.ceiling_sq
#print axioms E40.ceiling_rho
#print axioms E40.ceiling_attained
#print axioms E40.decision_accuracy_proper
#print axioms E40.covPooled_eq_covF
#print axioms E40.ceiling_pooled_rho
#print axioms E40.ceiling_attained_pooled
