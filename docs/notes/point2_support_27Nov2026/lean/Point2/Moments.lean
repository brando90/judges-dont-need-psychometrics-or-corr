import Mathlib

/-!
# Point 2 note: moments of a finite law, the `r²` identity, affine invariance

A finite (or empirical) law is a weight vector `w` on a finite type `ι` (nonnegative, summing to 1).
Random variables are functions `ι → ℝ`. An empirical evaluation set of `n` items is the case
`ι = Fin n`, `w i = 1/n`.
-/

open Finset BigOperators

namespace Point2

structure FinLaw (ι : Type*) [Fintype ι] where
  w : ι → ℝ
  nonneg : ∀ i, 0 ≤ w i
  sum_one : ∑ i, w i = 1

namespace FinLaw
variable {ι : Type*} [Fintype ι] (P : FinLaw ι)

/-- Expectation. -/
def E (f : ι → ℝ) : ℝ := ∑ i, P.w i * f i
def cov (f g : ι → ℝ) : ℝ := P.E (fun i => f i * g i) - P.E f * P.E g
def var (f : ι → ℝ) : ℝ := P.cov f f
/-- Pearson correlation `Cov / (σ σ)`. -/
noncomputable def corr (f g : ι → ℝ) : ℝ := P.cov f g / (Real.sqrt (P.var f) * Real.sqrt (P.var g))

lemma E_add (f g : ι → ℝ) : P.E (fun i => f i + g i) = P.E f + P.E g := by
  simp [E, mul_add, sum_add_distrib]
lemma E_sub (f g : ι → ℝ) : P.E (fun i => f i - g i) = P.E f - P.E g := by
  simp [E, mul_sub, sum_sub_distrib]
lemma E_smul (c : ℝ) (f : ι → ℝ) : P.E (fun i => c * f i) = c * P.E f := by
  simp only [E, mul_sum]; exact sum_congr rfl (fun i _ => by ring)
lemma E_const (c : ℝ) : P.E (fun _ => c) = c := by
  simp only [E, ← sum_mul, P.sum_one, one_mul]
lemma E_nonneg (f : ι → ℝ) (h : ∀ i, 0 ≤ f i) : 0 ≤ P.E f :=
  sum_nonneg (fun i _ => mul_nonneg (P.nonneg i) (h i))

lemma var_eq_E_sq (f : ι → ℝ) : P.var f = P.E (fun i => (f i - P.E f) ^ 2) := by
  have : (fun i => (f i - P.E f) ^ 2) = fun i => (f i * f i + (-2 * P.E f) * f i) + P.E f ^ 2 := by
    funext i; ring
  rw [this, E_add, E_add, E_smul, E_const]; unfold var cov; ring

lemma var_nonneg (f : ι → ℝ) : 0 ≤ P.var f := by
  rw [var_eq_E_sq]; exact P.E_nonneg _ (fun i => sq_nonneg _)

/-- Expected squared error of the affine recalibration `a + b S`, expanded in moments. -/
lemma mse_affine (S Y : ι → ℝ) (a b : ℝ) :
    P.E (fun i => (Y i - a - b * S i) ^ 2)
      = P.var Y - 2 * b * P.cov S Y + b ^ 2 * P.var S + (P.E Y - b * P.E S - a) ^ 2 := by
  have : (fun i => (Y i - a - b * S i) ^ 2) = fun i =>
      ((Y i * Y i + (-2 * a) * Y i) + ((-2 * b) * (S i * Y i) + (b ^ 2) * (S i * S i)))
        + ((2 * a * b) * S i + a ^ 2) := by
    funext i; ring
  rw [this]
  simp only [E_add, E_smul, E_const]
  unfold var cov; ring

/-- **Prop. `prop:identity`, part 1.** For every `(a, b)`, the recalibrated squared error is at
least `Var(Y) - Cov(S,Y)²/Var(S)` (when `Var S > 0`). -/
theorem mse_affine_ge (S Y : ι → ℝ) (hS : 0 < P.var S) (a b : ℝ) :
    P.var Y - P.cov S Y ^ 2 / P.var S ≤ P.E (fun i => (Y i - a - b * S i) ^ 2) := by
  rw [P.mse_affine]
  have key : P.var Y - P.cov S Y ^ 2 / P.var S
      = P.var Y - 2 * b * P.cov S Y + b ^ 2 * P.var S - P.var S * (b - P.cov S Y / P.var S) ^ 2 := by
    field_simp; ring
  rw [key]
  nlinarith [mul_nonneg hS.le (sq_nonneg (b - P.cov S Y / P.var S)), sq_nonneg (P.E Y - b * P.E S - a)]

/-- The minimizer `b* = Cov/Var S`, `a* = E Y - b* E S` attains the bound. -/
theorem mse_affine_opt (S Y : ι → ℝ) (hS : 0 < P.var S) :
    P.E (fun i => (Y i - (P.E Y - P.cov S Y / P.var S * P.E S) - P.cov S Y / P.var S * S i) ^ 2)
      = P.var Y - P.cov S Y ^ 2 / P.var S := by
  rw [P.mse_affine]; field_simp; ring

lemma corr_sq (S Y : ι → ℝ) (hS : 0 < P.var S) (hY : 0 < P.var Y) :
    P.corr S Y ^ 2 = P.cov S Y ^ 2 / (P.var S * P.var Y) := by
  unfold corr
  rw [div_pow, mul_pow, Real.sq_sqrt hS.le, Real.sq_sqrt hY.le]

/-- **Prop. `prop:identity`** (`r² = 1 - min_{a,b} E(Y - a - bS)² / Var Y`), finite law,
`Var S > 0`, `Var Y > 0`: the set of attainable recalibrated squared errors has least element
`Var(Y)(1 - r²)`. -/
theorem r_sq_identity (S Y : ι → ℝ) (hS : 0 < P.var S) (hY : 0 < P.var Y) :
    IsLeast {m | ∃ a b : ℝ, m = P.E (fun i => (Y i - a - b * S i) ^ 2)}
      (P.var Y * (1 - P.corr S Y ^ 2)) := by
  have h : P.var Y * (1 - P.corr S Y ^ 2) = P.var Y - P.cov S Y ^ 2 / P.var S := by
    rw [P.corr_sq S Y hS hY]; field_simp
  rw [h]
  refine ⟨⟨_, _, (P.mse_affine_opt S Y hS).symm⟩, ?_⟩
  rintro m ⟨a, b, rfl⟩
  exact P.mse_affine_ge S Y hS a b

/-- Same statement in the note's form `r² = 1 - min / Var Y`. -/
theorem r_sq_eq_one_sub_min (S Y : ι → ℝ) (hS : 0 < P.var S) (hY : 0 < P.var Y) :
    P.corr S Y ^ 2 = 1 - (P.var Y - P.cov S Y ^ 2 / P.var S) / P.var Y := by
  rw [P.corr_sq S Y hS hY]; field_simp; ring

/-! ### Prop. `prop:affine`: affine invariance with sign, any real `a ≠ 0`. -/

lemma cov_affine (S Y : ι → ℝ) (a b : ℝ) : P.cov (fun i => a * S i + b) Y = a * P.cov S Y := by
  unfold cov
  have h1 : (fun i => (a * S i + b) * Y i) = fun i => a * (S i * Y i) + b * Y i := by funext i; ring
  rw [h1, E_add, E_smul, E_smul, E_add, E_smul, E_const]; ring

lemma var_affine (S : ι → ℝ) (a b : ℝ) : P.var (fun i => a * S i + b) = a ^ 2 * P.var S := by
  unfold var cov
  have h1 : (fun i => (a * S i + b) * (a * S i + b)) =
      fun i => (a ^ 2 * (S i * S i) + (2 * a * b) * S i) + b ^ 2 := by funext i; ring
  rw [h1, E_add, E_add, E_smul, E_smul, E_const, E_add, E_smul, E_const]; ring

theorem pearson_affine (S Y : ι → ℝ) (a b : ℝ) (ha : a ≠ 0) :
    P.corr (fun i => a * S i + b) Y = (a / |a|) * P.corr S Y := by
  unfold corr
  rw [P.cov_affine, P.var_affine, Real.sqrt_mul (sq_nonneg a), Real.sqrt_sq_eq_abs]
  have : |a| ≠ 0 := abs_ne_zero.mpr ha
  by_cases h0 : Real.sqrt (P.var S) * Real.sqrt (P.var Y) = 0
  · have : |a| * Real.sqrt (P.var S) * Real.sqrt (P.var Y) = 0 := by rw [mul_assoc, h0, mul_zero]
    rw [this, h0]; simp
  · field_simp

theorem pearson_affine_pos (S Y : ι → ℝ) (a b : ℝ) (ha : 0 < a) :
    P.corr (fun i => a * S i + b) Y = P.corr S Y := by
  rw [P.pearson_affine S Y a b ha.ne', abs_of_pos ha, div_self ha.ne', one_mul]

theorem pearson_affine_neg (S Y : ι → ℝ) (a b : ℝ) (ha : a < 0) :
    P.corr (fun i => a * S i + b) Y = - P.corr S Y := by
  rw [P.pearson_affine S Y a b ha.ne, abs_of_neg ha, div_neg, div_self ha.ne]; ring

end FinLaw
end Point2
