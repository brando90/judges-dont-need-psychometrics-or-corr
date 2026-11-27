import GTheory
/-
  Reviewer checks on experiment 40's GTheory.lean (core Lean 4.36.0, no Mathlib).

  1. `cell11_infeasible`: the shared 2x2 "G-study" table assigns cell (1,1) mean 1/4 and
     within-cell variance 5/4. No rating law on the witness's own grid {0,1,2,3} can have both:
     any nonnegative law with mean 1/4 has variance at most 11/16 (Bhatia-Davis). General over
     every law p : Fin 4 -> Rat with p 1, p 2 >= 0; not an instance.
  2. `rps4Terms_eq_rps`: the K = 4 refinement lemma `rps_refines_middle_cut` is stated about a
     separately written `rps4Terms`, not about the `rps` used in `rps_strictly_proper`.
     Here we check that the two agree, so the refinement lemma does speak about `rps 4`.
  3. `cellMean_ignores_laws`: the G-components are closed terms: they do not take the two
     witness laws as arguments, so "same G coefficients for both judges" holds by construction.
-/

theorem cell11_infeasible (p : Fin 4 → Rat) (h1 : 0 ≤ p 1) (h2 : 0 ≤ p 2)
    (hm : 4 * mean4 p = 1) (hv : 16 * second4 p = 21) : False := by
  -- `4 * mean = 1` is mean 1/4; `16 * E[Y^2] = 21` is variance 21/16 - 1/16 = 5/4 at that mean.
  unfold mean4 at hm
  unfold second4 at hv
  grind

/-- The same bound in the positive form: a nonnegative law on {0,1,2,3} with mean 1/4 has
`E[Y^2] ≤ 3/4`, hence variance at most 3/4 - 1/16 = 11/16 < 5/4. -/
theorem cell11_second_moment_bound (p : Fin 4 → Rat) (h1 : 0 ≤ p 1) (h2 : 0 ≤ p 2)
    (hm : 4 * mean4 p = 1) : 4 * second4 p ≤ 3 := by
  unfold mean4 at hm
  unfold second4
  grind

/-- Cell (1,1) of the shared table really has mean 1/4 and the stated variance 5/4. -/
theorem cell11_table_values : cellMean 1 1 = 1 / 4 ∧ sigmaRes2 = 5 / 4 :=
  ⟨cellMean_11, rfl⟩

theorem rps4Terms_eq_rps (q : Fin 4 → Rat) (y : Fin 4) : rps 4 q y = rps4Terms q y := by
  have m1 : mass 4 1 q = q 0 := by simp [mass]; grind
  have m2 : mass 4 2 q = q 0 + q 1 := by simp [mass]; grind
  have m3 : mass 4 3 q = q 0 + q 1 + q 2 := by simp [mass]; grind
  simp only [rps, rpsTerm, rps4Terms, sumF]
  simp [m1, m2, m3]
  grind

/-- The components are constants: they are equal for ANY two laws, which is why the witness's
"same G coefficients" needs no hypothesis about pStar or qBad. -/
theorem cellMean_ignores_laws (_a _b : Fin 4 → Rat) :
    phiCoeff sigmaC2 sigmaR2 sigmaRes2 = phiCoeff sigmaC2 sigmaR2 sigmaRes2 := rfl

#print axioms cell11_infeasible
#print axioms cell11_second_moment_bound
#print axioms rps4Terms_eq_rps
#print axioms cell11_table_values
