import Mathlib

/- Exact finite identities used in MATH_11-21-2026.md. -/
namespace PearsonLoss

theorem standardized_residual_loss {I : Type*} [Fintype I]
    (w z h : I → ℝ) (hz : ∑ i, w i * (z i)^2 = 1)
    (hh : ∑ i, w i * (h i)^2 = 1) :
    (∑ i, w i * (z i - h i)^2) / 2 = 1 - ∑ i, w i * z i * h i := by
  have expand : (∑ i, w i * (z i - h i)^2) =
      (∑ i, w i * (z i)^2) + (∑ i, w i * (h i)^2) - 2 * ∑ i, w i * z i * h i := by
    calc
      _ = ∑ i, (w i * (z i)^2 + w i * (h i)^2 - 2 * (w i * z i * h i)) := by
        apply Finset.sum_congr rfl
        intro i _
        ring
      _ = _ := by rw [Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum]
  rw [expand, hz, hh]
  ring

theorem unchanged_expected_rating {I : Type*} [Fintype I]
    (r p d : I → ℝ) (t : ℝ) (h : ∑ i, r i * d i = 0) :
    ∑ i, r i * (p i + t * d i) = ∑ i, r i * p i := by
  calc
    _ = ∑ i, (r i * p i + t * (r i * d i)) := by
      apply Finset.sum_congr rfl
      intro i _
      ring
    _ = _ := by rw [Finset.sum_add_distrib, ← Finset.mul_sum, h, mul_zero, add_zero]

end PearsonLoss
