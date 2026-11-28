/-
Experiment 53 (27 Nov 2026): the uncertainty term behind the advisor's note S7.

With one rating per item the mean per-item proper score is unbiased for the judge's expected score against p*.
The distance to p* (expected score minus the uncertainty term E_x[S(p*_x, p*_x)]) also needs the uncertainty term.
This file checks the finite facts used in the paper's sentence:
  1. two independent ratings of one item agree with probability sum_k p_k^2 (so the term is estimable from pairs);
  2. the normalized Brier uncertainty term is (1 - sum_k p_k^2)/2, a function of that agreement probability;
  3. for one cumulative threshold of RPS, the expected squared error of the truthful forecast P is P(1 - P);
  4. a witness that one rating per item cannot identify the term: two item populations with the same single-rating law
     (1/2, 1/2) but uncertainty terms 1/4 and 0.
Zero sorry; #print axioms at the end.
-/
import Mathlib

open Finset BigOperators

namespace E53

/-- 1. Probability that two independent draws from `p` coincide. -/
theorem pair_agreement {K : ℕ} (p : Fin K → ℝ) :
    (∑ i, ∑ j, p i * p j * (if i = j then (1:ℝ) else 0)) = ∑ i, p i ^ 2 := by
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.sum_eq_single i]
  · simp [sq]
  · intro j _ hji; simp [Ne.symm hji]
  · intro h; exact absurd (Finset.mem_univ i) h

/-- 2. Normalized Brier uncertainty term of the truthful forecast. -/
theorem brier_uncertainty {K : ℕ} (p : Fin K → ℝ) (hsum : ∑ j, p j = 1) :
    (∑ y, p y * ((1/2:ℝ) * ∑ j, (p j - (if j = y then (1:ℝ) else 0)) ^ 2))
      = (1/2:ℝ) * (1 - ∑ j, p j ^ 2) := by
  have expand : ∀ y, (∑ j, (p j - (if j = y then (1:ℝ) else 0)) ^ 2)
      = (∑ j, p j ^ 2) - 2 * p y + 1 := by
    intro y
    have h1 : ∀ j, (p j - (if j = y then (1:ℝ) else 0)) ^ 2
        = p j ^ 2 - 2 * p j * (if j = y then (1:ℝ) else 0) + (if j = y then (1:ℝ) else 0) := by
      intro j; split_ifs <;> ring
    simp only [h1, Finset.sum_add_distrib, Finset.sum_sub_distrib]
    simp [Finset.sum_ite_eq']
  simp only [expand]
  set S := ∑ j, p j ^ 2 with hS
  have hy : ∀ y, p y * ((1/2:ℝ) * (S - 2 * p y + 1)) = ((1/2:ℝ) * S + 1/2) * p y - p y ^ 2 := by
    intro y; ring
  rw [Finset.sum_congr rfl (fun y _ => hy y), Finset.sum_sub_distrib, ← Finset.mul_sum, hsum]
  ring

/-- 3. One cumulative threshold of RPS: the truthful forecast `P` of a Bernoulli(`P`) indicator has expected squared
error `P (1 - P)`; RPS sums this over thresholds, so its uncertainty term is `sum_j P_j (1 - P_j) / (K - 1)`. -/
theorem threshold_uncertainty (P : ℝ) : P * (P - 1) ^ 2 + (1 - P) * (P - 0) ^ 2 = P * (1 - P) := by
  ring

/-- 4. Witness: population A has every item at (1/2, 1/2); population B has half the items at (1, 0) and half at (0, 1).
A single rating from a random item has the same law (1/2, 1/2) in both, yet the mean Brier uncertainty terms differ
(1/4 versus 0), so one rating per item cannot identify the term. -/
theorem one_rating_witness :
    let unc : ℝ → ℝ := fun a => (1/2:ℝ) * (1 - (a ^ 2 + (1 - a) ^ 2))
    -- single-rating law of class 0: A gives 1/2; B gives (1/2)*1 + (1/2)*0
    ((1/2:ℝ) = (1/2) * 1 + (1/2) * 0) ∧
    -- mean uncertainty terms: A = 1/4, B = 0, and they differ
    unc (1/2) = 1/4 ∧ ((1/2) * unc 1 + (1/2) * unc 0 = 0) ∧ unc (1/2) ≠ (1/2) * unc 1 + (1/2) * unc 0 := by
  intro unc
  refine ⟨by norm_num, ?_, ?_, ?_⟩ <;> simp only [unc] <;> norm_num

end E53

#print axioms E53.pair_agreement
#print axioms E53.brier_uncertainty
#print axioms E53.threshold_uncertainty
#print axioms E53.one_rating_witness
