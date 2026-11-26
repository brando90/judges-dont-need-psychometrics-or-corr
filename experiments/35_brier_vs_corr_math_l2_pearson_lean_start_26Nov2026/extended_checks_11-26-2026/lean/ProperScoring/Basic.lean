/-
Experiment 35: proper scoring, Brier, squared error (L2) and correlation.
Finite-sample / finite-support statements only. No `sorry`, no new axioms.
-/
import Mathlib

open Finset BigOperators

namespace ProperScoring

/-! ## P0-1. Brier score is strictly proper (K categories, any Fintype). -/

section Brier
variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Brier score of forecast `p` when the outcome is `y`: `∑ₖ (pₖ - 1[y = k])²`. -/
noncomputable def brier (p : ι → ℝ) (y : ι) : ℝ :=
  ∑ k, (p k - if k = y then 1 else 0) ^ 2

/-- Expected Brier score of forecast `p` when the outcome has law `q`. -/
noncomputable def expBrier (q p : ι → ℝ) : ℝ := ∑ y, q y * brier p y

lemma brier_expand (p : ι → ℝ) (y : ι) :
    brier p y = (∑ k, p k ^ 2) - 2 * p y + 1 := by
  unfold brier
  have h : ∀ k, (p k - if k = y then (1:ℝ) else 0) ^ 2
      = p k ^ 2 - 2 * (if k = y then p k else 0) + (if k = y then 1 else 0) := by
    intro k; split_ifs <;> ring
  simp only [h, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum,
    Finset.sum_ite_eq', Finset.mem_univ, ite_true]

/-- Brier decomposition: the excess expected score is the squared distance to the truth.
Only `∑ q = 1` is used (nonnegativity is not needed for the identity). -/
theorem brier_excess (q p : ι → ℝ) (hq : ∑ y, q y = 1) :
    expBrier q p - expBrier q q = ∑ k, (p k - q k) ^ 2 := by
  unfold expBrier
  simp only [brier_expand]
  have e1 : ∀ r : ι → ℝ, ∑ y, q y * ((∑ k, r k ^ 2) - 2 * r y + 1)
      = (∑ k, r k ^ 2) * (∑ y, q y) - 2 * ∑ y, q y * r y + ∑ y, q y := by
    intro r
    have hy : ∀ y, q y * ((∑ k, r k ^ 2) - 2 * r y + 1)
        = (∑ k, r k ^ 2) * q y - 2 * (q y * r y) + q y := by intro y; ring
    rw [Finset.sum_congr rfl (fun y _ => hy y)]
    simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
  rw [e1 p, e1 q, hq]
  have e2 : ∑ k, (p k - q k) ^ 2 = ∑ k, p k ^ 2 - 2 * ∑ k, q k * p k + ∑ k, q k ^ 2 := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl ?_
    intro k _; ring
  have e3 : ∑ y, q y * q y = ∑ k, q k ^ 2 := by
    refine Finset.sum_congr rfl ?_; intro k _; ring
  rw [e2, e3]; ring

/-- Strict propriety of the Brier score. -/
theorem brier_strictly_proper (q p : ι → ℝ) (hq : ∑ y, q y = 1) :
    expBrier q q ≤ expBrier q p ∧ (expBrier q p = expBrier q q ↔ p = q) := by
  have h := brier_excess q p hq
  have hnn : 0 ≤ ∑ k, (p k - q k) ^ 2 := Finset.sum_nonneg (fun k _ => sq_nonneg _)
  refine ⟨by linarith, ?_⟩
  constructor
  · intro heq
    have h0 : ∑ k, (p k - q k) ^ 2 = 0 := by linarith
    rw [Finset.sum_eq_zero_iff_of_nonneg (fun k _ => sq_nonneg _)] at h0
    funext k
    have := h0 k (Finset.mem_univ k)
    have : p k - q k = 0 := by simpa using this
    linarith
  · intro hpq; rw [hpq]

/-- L2 on the one-hot vector is the Brier score (definitional restatement). -/
theorem l2_onehot_eq_brier (p : ι → ℝ) (y : ι) :
    (∑ k, (p k - (Pi.single y (1:ℝ) : ι → ℝ) k) ^ 2) = brier p y := by
  unfold brier
  refine Finset.sum_congr rfl ?_
  intro k _
  by_cases h : k = y
  · subst h; simp
  · simp [h]

end Brier

/-! ## P0-2. Squared error: decomposition and unique minimizer at the mean.
Finite distribution with weights `w ≥ 0`, `∑ w = 1`; no Gaussian assumption. -/

section L2
variable {ι : Type*} [Fintype ι]

noncomputable def wmean (w v : ι → ℝ) : ℝ := ∑ i, w i * v i
noncomputable def risk (w v : ι → ℝ) (f : ℝ) : ℝ := ∑ i, w i * (f - v i) ^ 2

theorem l2_decomposition (w v : ι → ℝ) (hw : ∑ i, w i = 1) (f : ℝ) :
    risk w v f = (f - wmean w v) ^ 2 + risk w v (wmean w v) := by
  unfold risk
  set μ := wmean w v with hμ
  have h : ∀ i, w i * (f - v i) ^ 2
      = w i * (μ - v i) ^ 2 + ((f - μ) ^ 2 * w i + 2 * (f - μ) * (w i * μ - w i * v i)) := by
    intro i; ring
  simp only [h, Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_sub_distrib, ← Finset.sum_mul]
  have : ∑ i, w i * v i = μ := rfl
  rw [hw, this]; ring

theorem l2_unique_minimizer (w v : ι → ℝ) (hw : ∑ i, w i = 1) (f : ℝ) :
    risk w v (wmean w v) ≤ risk w v f ∧ (risk w v f = risk w v (wmean w v) ↔ f = wmean w v) := by
  rw [l2_decomposition w v hw f]
  refine ⟨by nlinarith [sq_nonneg (f - wmean w v)], ?_⟩
  constructor
  · intro h
    have : (f - wmean w v) ^ 2 = 0 := by linarith
    have := pow_eq_zero_iff (n := 2) (by norm_num) |>.mp this
    linarith
  · intro h; rw [h]; ring

end L2

/-! ## P0-3. MSE = bias² + scale² + 2 σx σy (1 - ρ); standardized form 2(1 - ρ). -/

section Corr
variable {ι : Type*} [Fintype ι]

noncomputable def cov (w x y : ι → ℝ) : ℝ := ∑ i, w i * (x i - wmean w x) * (y i - wmean w y)
noncomputable def var (w x : ι → ℝ) : ℝ := cov w x x
noncomputable def corr (w x y : ι → ℝ) : ℝ := cov w x y / Real.sqrt (var w x * var w y)

lemma centered_sum (w x : ι → ℝ) (hw : ∑ i, w i = 1) :
    ∑ i, w i * (x i - wmean w x) = 0 := by
  have : ∑ i, w i * (x i - wmean w x) = ∑ i, w i * x i - wmean w x * ∑ i, w i := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl ?_; intro i _; ring
  rw [this, hw]; unfold wmean; ring

/-- Covariance form (no division). -/
theorem mse_cov_form (w x y : ι → ℝ) (hw : ∑ i, w i = 1) :
    ∑ i, w i * (x i - y i) ^ 2
      = (wmean w x - wmean w y) ^ 2 + var w x + var w y - 2 * cov w x y := by
  unfold var cov
  set mx := wmean w x
  set my := wmean w y
  have h : ∀ i, w i * (x i - y i) ^ 2
      = w i * (x i - mx) * (x i - mx) + w i * (y i - my) * (y i - my)
        - 2 * (w i * (x i - mx) * (y i - my))
        + 2 * (mx - my) * (w i * (x i - mx)) - 2 * (mx - my) * (w i * (y i - my))
        + (mx - my) ^ 2 * w i := by
    intro i; ring
  simp only [h, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.sum_mul]
  rw [centered_sum w x hw, centered_sum w y hw, hw]; ring

lemma var_nonneg (w x : ι → ℝ) (hw0 : ∀ i, 0 ≤ w i) : 0 ≤ var w x := by
  unfold var cov
  refine Finset.sum_nonneg (fun i _ => ?_)
  have := hw0 i
  nlinarith [mul_self_nonneg (x i - wmean w x)]

/-- MSE = (μx - μy)² + (σx - σy)² + 2 σx σy (1 - ρ), for positive variances. -/
theorem mse_corr_form (w x y : ι → ℝ) (hw : ∑ i, w i = 1)
    (hx : 0 < var w x) (hy : 0 < var w y) :
    ∑ i, w i * (x i - y i) ^ 2
      = (wmean w x - wmean w y) ^ 2
        + (Real.sqrt (var w x) - Real.sqrt (var w y)) ^ 2
        + 2 * Real.sqrt (var w x) * Real.sqrt (var w y) * (1 - corr w x y) := by
  rw [mse_cov_form w x y hw]
  unfold corr
  have sx := Real.sq_sqrt hx.le
  have sy := Real.sq_sqrt hy.le
  have px : 0 < Real.sqrt (var w x) := Real.sqrt_pos.mpr hx
  have py : 0 < Real.sqrt (var w y) := Real.sqrt_pos.mpr hy
  rw [Real.sqrt_mul hx.le]
  field_simp
  nlinarith [sx, sy]

/-- Standardized corollary: mean squared difference of z-scores is 2(1 - ρ). -/
theorem standardized_mse (w x y : ι → ℝ) (hw : ∑ i, w i = 1)
    (hx : 0 < var w x) (hy : 0 < var w y) :
    ∑ i, w i * ((x i - wmean w x) / Real.sqrt (var w x)
                - (y i - wmean w y) / Real.sqrt (var w y)) ^ 2
      = 2 * (1 - corr w x y) := by
  have sx := Real.sq_sqrt hx.le
  have sy := Real.sq_sqrt hy.le
  have px : 0 < Real.sqrt (var w x) := Real.sqrt_pos.mpr hx
  have py : 0 < Real.sqrt (var w y) := Real.sqrt_pos.mpr hy
  set a := Real.sqrt (var w x)
  set b := Real.sqrt (var w y)
  have h : ∀ i, w i * ((x i - wmean w x) / a - (y i - wmean w y) / b) ^ 2
      = (w i * (x i - wmean w x) * (x i - wmean w x)) / a ^ 2
        + (w i * (y i - wmean w y) * (y i - wmean w y)) / b ^ 2
        - 2 * (w i * (x i - wmean w x) * (y i - wmean w y)) / (a * b) := by
    intro i; field_simp; ring
  simp only [h, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.sum_div, ← Finset.mul_sum]
  have hv : ∀ z, ∑ i, w i * (z i - wmean w z) * (z i - wmean w z) = var w z := fun z => rfl
  have hc : ∑ i, w i * (x i - wmean w x) * (y i - wmean w y) = cov w x y := rfl
  rw [hv x, hv y, hc, sx, sy]
  unfold corr
  rw [Real.sqrt_mul hx.le]
  field_simp
  ring

/-! ## P0-4. Affine invariance of correlation for a > 0. -/

lemma wmean_affine (w x : ι → ℝ) (hw : ∑ i, w i = 1) (a b : ℝ) :
    wmean w (fun i => a * x i + b) = a * wmean w x + b := by
  unfold wmean
  have : ∀ i, w i * (a * x i + b) = a * (w i * x i) + b * w i := by intro i; ring
  simp only [this, Finset.sum_add_distrib, ← Finset.mul_sum, hw]; ring

lemma cov_affine_left (w x y : ι → ℝ) (hw : ∑ i, w i = 1) (a b : ℝ) :
    cov w (fun i => a * x i + b) y = a * cov w x y := by
  unfold cov
  rw [wmean_affine w x hw, Finset.mul_sum]
  refine Finset.sum_congr rfl ?_; intro i _; ring

lemma var_affine (w x : ι → ℝ) (hw : ∑ i, w i = 1) (a b : ℝ) :
    var w (fun i => a * x i + b) = a ^ 2 * var w x := by
  unfold var cov
  rw [wmean_affine w x hw, Finset.mul_sum]
  refine Finset.sum_congr rfl ?_; intro i _; ring

theorem corr_affine_invariant (w x y : ι → ℝ) (hw : ∑ i, w i = 1) (a b : ℝ) (ha : 0 < a) :
    corr w (fun i => a * x i + b) y = corr w x y := by
  unfold corr
  rw [cov_affine_left w x y hw, var_affine w x hw]
  rw [show a ^ 2 * var w x * var w y = a ^ 2 * (var w x * var w y) by ring]
  rw [Real.sqrt_mul (sq_nonneg a), Real.sqrt_sq ha.le]
  rw [mul_div_mul_left _ _ ha.ne']

theorem corr_self (w x : ι → ℝ) (hx : 0 < var w x) : corr w x x = 1 := by
  unfold corr
  rw [Real.sqrt_mul_self hx.le]
  exact div_self hx.ne'

/-- Oracle affine recalibration: min over (a,b) of the weighted MSE of `y - (a x + b)` equals
`Var y (1 - ρ²)`, attained at `a = cov/var x`, `b = μy - a μx` (stated as a lower bound and
attainment; covariance form, so no square roots). -/
theorem oracle_affine_mse (w x y : ι → ℝ) (hw : ∑ i, w i = 1) (hx : 0 < var w x) (a b : ℝ) :
    var w y - cov w x y ^ 2 / var w x ≤ ∑ i, w i * (y i - (a * x i + b)) ^ 2 ∧
    ∑ i, w i * (y i - ((cov w x y / var w x) * x i
        + (wmean w y - (cov w x y / var w x) * wmean w x))) ^ 2
      = var w y - cov w x y ^ 2 / var w x := by
  have key : ∀ a b : ℝ, ∑ i, w i * (y i - (a * x i + b)) ^ 2
      = (wmean w y - (a * wmean w x + b)) ^ 2 + var w y + a ^ 2 * var w x - 2 * a * cov w x y := by
    intro a b
    have := mse_cov_form w y (fun i => a * x i + b) hw
    rw [this, wmean_affine w x hw, var_affine w x hw]
    have hc : cov w y (fun i => a * x i + b) = a * cov w x y := by
      unfold cov; rw [wmean_affine w x hw, Finset.mul_sum]
      refine Finset.sum_congr rfl ?_; intro i _; ring
    rw [hc]; ring
  constructor
  · rw [key a b]
    have hv := hx
    have : var w y - cov w x y ^ 2 / var w x
        ≤ var w y + a ^ 2 * var w x - 2 * a * cov w x y := by
      rw [sub_le_iff_le_add]
      have : 0 ≤ (a * var w x - cov w x y) ^ 2 / var w x := div_nonneg (sq_nonneg _) hv.le
      have e : (a * var w x - cov w x y) ^ 2 / var w x
          = a ^ 2 * var w x - 2 * a * cov w x y + cov w x y ^ 2 / var w x := by
        field_simp; ring
      linarith
    nlinarith [sq_nonneg (wmean w y - (a * wmean w x + b))]
  · rw [key]
    field_simp
    ring

end Corr

/-! ## P1. Threshold-Brier family: RPS and Brier are both instances. Strict propriety of RPS. -/

section Threshold
variable {α β : Type*} [DecidableEq α]

/-- Expected threshold score: outcomes in `S` with weights `p`, forecast probabilities `c j`
for the events `A j` (`j ∈ T`). -/
noncomputable def expThr (S : Finset α) (T : Finset β) (A : β → Finset α) (p : α → ℝ)
    (c : β → ℝ) : ℝ :=
  ∑ y ∈ S, p y * ∑ j ∈ T, (c j - if y ∈ A j then 1 else 0) ^ 2

theorem thr_excess (S : Finset α) (T : Finset β) (A : β → Finset α) (hA : ∀ j ∈ T, A j ⊆ S)
    (p : α → ℝ) (hp : ∑ y ∈ S, p y = 1) (c : β → ℝ) :
    expThr S T A p c - expThr S T A p (fun j => ∑ y ∈ A j, p y)
      = ∑ j ∈ T, (c j - ∑ y ∈ A j, p y) ^ 2 := by
  unfold expThr
  have per : ∀ (d : β → ℝ), ∑ y ∈ S, p y * ∑ j ∈ T, (d j - if y ∈ A j then 1 else 0) ^ 2
      = ∑ j ∈ T, (d j ^ 2 - 2 * d j * (∑ y ∈ A j, p y) + ∑ y ∈ A j, p y) := by
    intro d
    simp only [Finset.mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl ?_
    intro j hj
    have h1 : ∀ y, p y * (d j - if y ∈ A j then (1:ℝ) else 0) ^ 2
        = d j ^ 2 * p y - 2 * d j * (if y ∈ A j then p y else 0) + (if y ∈ A j then p y else 0) := by
      intro y; split_ifs <;> ring
    simp only [h1, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
    rw [Finset.sum_ite_mem, Finset.inter_eq_right.mpr (hA j hj), hp]
    ring
  rw [per c, per, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl ?_; intro j _; ring

/-- Ranked probability score (unnormalized, thresholds j = 0..K-1 with events {0..j}) over
ratings `0..K-1`; the last threshold contributes 0 for distributions. -/
noncomputable def expRPS (K : ℕ) (p q : ℕ → ℝ) : ℝ :=
  expThr (Finset.range K) (Finset.range K) (fun j => Finset.range (j + 1)) p
    (fun j => ∑ i ∈ Finset.range (j + 1), q i)

lemma cdf_eq_imp_eq (K : ℕ) (p q : ℕ → ℝ)
    (h : ∀ j < K, ∑ i ∈ Finset.range (j + 1), q i = ∑ i ∈ Finset.range (j + 1), p i) :
    ∀ k < K, q k = p k := by
  intro k hk
  cases k with
  | zero => simpa using h 0 hk
  | succ m =>
    have h1 := h (m + 1) hk
    have h0 := h m (by omega)
    rw [Finset.sum_range_succ q (m + 1), Finset.sum_range_succ p (m + 1)] at h1
    linarith

theorem rps_strictly_proper (K : ℕ) (p q : ℕ → ℝ) (hp : ∑ y ∈ Finset.range K, p y = 1) :
    expRPS K p p ≤ expRPS K p q ∧ (expRPS K p q = expRPS K p p → ∀ k < K, q k = p k) := by
  have hA : ∀ j ∈ Finset.range K, Finset.range (j + 1) ⊆ Finset.range K := by
    intro j hj; apply Finset.range_subset_range.mpr; simp at hj; omega
  have h := thr_excess (Finset.range K) (Finset.range K) (fun j => Finset.range (j + 1)) hA p hp
    (fun j => ∑ i ∈ Finset.range (j + 1), q i)
  unfold expRPS
  have hnn : 0 ≤ ∑ j ∈ Finset.range K,
      ((∑ i ∈ Finset.range (j + 1), q i) - ∑ y ∈ Finset.range (j + 1), p y) ^ 2 :=
    Finset.sum_nonneg (fun _ _ => sq_nonneg _)
  refine ⟨by linarith, ?_⟩
  intro heq
  have h0 : ∑ j ∈ Finset.range K,
      ((∑ i ∈ Finset.range (j + 1), q i) - ∑ y ∈ Finset.range (j + 1), p y) ^ 2 = 0 := by
    linarith
  rw [Finset.sum_eq_zero_iff_of_nonneg (fun _ _ => sq_nonneg _)] at h0
  apply cdf_eq_imp_eq
  intro j hj
  have := h0 j (Finset.mem_range.mpr hj)
  have : (∑ i ∈ Finset.range (j + 1), q i) - ∑ y ∈ Finset.range (j + 1), p y = 0 := by
    simpa using this
  linarith

end Threshold

end ProperScoring
