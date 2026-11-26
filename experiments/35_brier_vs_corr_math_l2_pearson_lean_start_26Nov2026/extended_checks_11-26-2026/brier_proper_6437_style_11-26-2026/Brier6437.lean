/-
Brier score in the style of the 6.437 properness argument.

Outcomes are `Fin K`; `p` is the true rating distribution of one item and `q` is a reported
distribution. Argument order follows the note BRIER_PROPER_6437_STYLE_11-26-2026.md:
`expBrier p q` is the expected Brier cost of reporting `q` when the outcome has law `p`.
(The extended-checks project `../lean/ProperScoring/Basic.lean` uses the opposite order,
`expBrier law forecast`, and states the same excess identity as `brier_excess`.)

(a) `expBrier_identity`  : expBrier p q = ∑ (q a - p a)^2 + (1 - ∑ p a ^ 2)
(b) `brier_proper`       : expBrier p p ≤ expBrier p q, with equality iff q = p
(c) `affine_brier_proper`: the same for A * Brier + B(x), A > 0
(d) `brier_two_local`, `brier_three_not_local`: local at L = 2, not local at L = 3
No `sorry`.
-/
import Mathlib

open Finset BigOperators

namespace Brier6437

variable {K : ℕ}

/-- Indicator `1{x = a}`. -/
noncomputable def ind (a x : Fin K) : ℝ := if a = x then 1 else 0

/-- Brier cost of report `q` when the outcome is `x`: `∑ₐ (q a - 1{x = a})²`. -/
noncomputable def brierCost (q : Fin K → ℝ) (x : Fin K) : ℝ :=
  ∑ a, (q a - ind a x) ^ 2

/-- Expected Brier cost of report `q` under the true law `p`. -/
noncomputable def expBrier (p q : Fin K → ℝ) : ℝ :=
  ∑ x, p x * brierCost q x

/-- Squaring an indicator leaves it unchanged, so the cost expands to `‖q‖² - 2 q(x) + 1`. -/
lemma brierCost_expand (q : Fin K → ℝ) (x : Fin K) :
    brierCost q x = (∑ a, q a ^ 2) - 2 * q x + 1 := by
  unfold brierCost ind
  have h : ∀ a, (q a - if a = x then (1:ℝ) else 0) ^ 2
      = q a ^ 2 - 2 * (if a = x then q a else 0) + (if a = x then 1 else 0) := by
    intro a; split_ifs <;> ring
  simp only [h, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum,
    Finset.sum_ite_eq', Finset.mem_univ, ite_true]

/-- (a) Expected Brier = squared Euclidean distance to `p` plus `1 - ‖p‖²`.
Only `∑ p = 1` is used. -/
theorem expBrier_identity (p q : Fin K → ℝ) (hp : ∑ x, p x = 1) :
    expBrier p q = ∑ a, (q a - p a) ^ 2 + (1 - ∑ a, p a ^ 2) := by
  unfold expBrier
  simp only [brierCost_expand]
  have hy : ∀ x, p x * ((∑ a, q a ^ 2) - 2 * q x + 1)
      = (∑ a, q a ^ 2) * p x - 2 * (p x * q x) + p x := by intro x; ring
  rw [Finset.sum_congr rfl (fun x _ => hy x)]
  simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, hp]
  have e : ∑ a, (q a - p a) ^ 2 = ∑ a, q a ^ 2 - 2 * ∑ a, p a * q a + ∑ a, p a ^ 2 := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl ?_
    intro a _; ring
  rw [e]; ring

/-- The excess over the truthful report is exactly the squared distance. -/
theorem expBrier_excess (p q : Fin K → ℝ) (hp : ∑ x, p x = 1) :
    expBrier p q - expBrier p p = ∑ a, (q a - p a) ^ 2 := by
  rw [expBrier_identity p q hp, expBrier_identity p p hp]
  simp

/-- (b) Strict propriety: `p` minimises expected Brier, uniquely. -/
theorem brier_proper (p q : Fin K → ℝ) (hp : ∑ x, p x = 1) :
    expBrier p p ≤ expBrier p q ∧ (expBrier p q = expBrier p p ↔ q = p) := by
  have h := expBrier_excess p q hp
  have hnn : 0 ≤ ∑ a, (q a - p a) ^ 2 := Finset.sum_nonneg (fun a _ => sq_nonneg _)
  refine ⟨by linarith, ?_⟩
  constructor
  · intro heq
    have h0 : ∑ a, (q a - p a) ^ 2 = 0 := by linarith
    rw [Finset.sum_eq_zero_iff_of_nonneg (fun a _ => sq_nonneg _)] at h0
    funext a
    have : q a - p a = 0 := by simpa using h0 a (Finset.mem_univ a)
    linarith
  · intro hqp; rw [hqp]

/-- The `A, B` family: `C(x, q) = A * Brier(x, q) + B(x)`. -/
noncomputable def expCostAB (A : ℝ) (B : Fin K → ℝ) (p q : Fin K → ℝ) : ℝ :=
  ∑ x, p x * (A * brierCost q x + B x)

theorem expCostAB_eq (A : ℝ) (B : Fin K → ℝ) (p q : Fin K → ℝ) :
    expCostAB A B p q = A * expBrier p q + ∑ x, p x * B x := by
  unfold expCostAB expBrier
  rw [Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl ?_
  intro x _; ring

/-- (c) For `A > 0` and any `B`, the excess is `A ‖q - p‖²` and `p` is the unique minimiser. -/
theorem affine_brier_proper (A : ℝ) (hA : 0 < A) (B : Fin K → ℝ) (p q : Fin K → ℝ)
    (hp : ∑ x, p x = 1) :
    expCostAB A B p q - expCostAB A B p p = A * ∑ a, (q a - p a) ^ 2 ∧
    expCostAB A B p p ≤ expCostAB A B p q ∧
    (expCostAB A B p q = expCostAB A B p p ↔ q = p) := by
  have hx := expBrier_excess p q hp
  have hb := brier_proper p q hp
  have e : expCostAB A B p q - expCostAB A B p p = A * ∑ a, (q a - p a) ^ 2 := by
    rw [expCostAB_eq, expCostAB_eq, ← hx]; ring
  refine ⟨e, ?_, ?_⟩
  · rw [expCostAB_eq, expCostAB_eq]
    have := mul_le_mul_of_nonneg_left hb.1 hA.le
    linarith
  · constructor
    · intro h
      have h1 : A * (expBrier p q - expBrier p p) = 0 := by
        rw [expCostAB_eq, expCostAB_eq] at h; linarith
      rcases mul_eq_zero.mp h1 with h2 | h2
      · exact absurd h2 hA.ne'
      · exact hb.2.mp (by linarith)
    · intro hqp; rw [hqp]

/-- (d, L = 2) On two outcomes the Brier cost is `2 (1 - q(x))²`: it is local. -/
theorem brier_two_eq (q : Fin 2 → ℝ) (hq : q 0 + q 1 = 1) (x : Fin 2) :
    brierCost q x = 2 * (1 - q x) ^ 2 := by
  unfold brierCost ind
  fin_cases x
  · simp [Fin.sum_univ_two]
    have : q 1 = 1 - q 0 := by linarith
    rw [this]; ring
  · simp [Fin.sum_univ_two]
    have : q 0 = 1 - q 1 := by linarith
    rw [this]; ring

theorem brier_two_local :
    ∃ φ : Fin 2 → ℝ → ℝ, ∀ q : Fin 2 → ℝ, q 0 + q 1 = 1 →
      ∀ x, brierCost q x = φ x (q x) :=
  ⟨fun _ ν => 2 * (1 - ν) ^ 2, fun q hq x => brier_two_eq q hq x⟩

/-- (d, L = 3) Exact counterexample: same outcome, same `q(x) = 1/2`, different cost. -/
noncomputable def qA : Fin 3 → ℝ := ![1/2, 1/2, 0]
noncomputable def qB : Fin 3 → ℝ := ![1/2, 1/4, 1/4]

theorem brier_qA : brierCost qA 0 = 1/2 := by
  simp [brierCost, ind, qA, Fin.sum_univ_three]; norm_num

theorem brier_qB : brierCost qB 0 = 3/8 := by
  simp [brierCost, ind, qB, Fin.sum_univ_three]; norm_num

theorem brier_three_not_local :
    ¬ ∃ φ : Fin 3 → ℝ → ℝ, ∀ q : Fin 3 → ℝ, (∑ a, q a = 1) →
      ∀ x, brierCost q x = φ x (q x) := by
  rintro ⟨φ, hφ⟩
  have hA : ∑ a, qA a = 1 := by simp [qA, Fin.sum_univ_three]; norm_num
  have hB : ∑ a, qB a = 1 := by simp [qB, Fin.sum_univ_three]; norm_num
  have h1 := hφ qA hA 0
  have h2 := hφ qB hB 0
  have hsame : qA 0 = qB 0 := by simp [qA, qB]
  rw [brier_qA] at h1
  rw [brier_qB, ← hsame, ← h1] at h2
  norm_num at h2

end Brier6437

#print axioms Brier6437.expBrier_identity
#print axioms Brier6437.brier_proper
#print axioms Brier6437.affine_brier_proper
#print axioms Brier6437.brier_two_local
#print axioms Brier6437.brier_three_not_local
