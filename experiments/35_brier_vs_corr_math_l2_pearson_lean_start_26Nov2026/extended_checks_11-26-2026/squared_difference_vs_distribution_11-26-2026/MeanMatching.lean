import Mathlib

/-!
# Exact finite facts behind page three of the 11-26-2026 derivation pages

Every statement is about a concrete pair of laws on a small set of ratings, over `ℚ`, and is
checked by evaluation (`decide`-style simplification and `norm_num`).  No `sorry`, no axioms
beyond Lean's standard ones.  The general statements (the three-term identity, the conditional
mean decomposition, the minimum over couplings) are proved on paper in
`SQUARED_DIFFERENCE_VS_DISTRIBUTION_11-26-2026.md` and checked numerically in
`check_squared_difference_vs_distribution.py`.

Notation: `p` and `q` are laws on the ratings `0,…,5`; `mean`, `var` are the first moment and the
centred second moment; `brier` is the squared Euclidean distance `∑ (p r - q r)^2`.
-/

namespace MeanMatching

/-- Law of the human rating in the example: half the mass on 0 and half on 5. -/
def p : Fin 6 → ℚ := ![1/2, 0, 0, 0, 0, 1/2]

/-- A different law: half the mass on 2 and half on 3. -/
def q : Fin 6 → ℚ := ![0, 0, 1/2, 1/2, 0, 0]

def mean (r : Fin 6 → ℚ) : ℚ := ∑ i : Fin 6, ((i : ℕ) : ℚ) * r i

def var (r : Fin 6 → ℚ) : ℚ := ∑ i : Fin 6, (((i : ℕ) : ℚ) - mean r) ^ 2 * r i

def brier (r s : Fin 6 → ℚ) : ℚ := ∑ i : Fin 6, (r i - s i) ^ 2

/-- Both are probability laws. -/
theorem both_laws : (∑ i, p i = 1) ∧ (∑ i, q i = 1) ∧ (∀ i, 0 ≤ p i) ∧ (∀ i, 0 ≤ q i) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [p, Fin.sum_univ_succ]; norm_num
  · simp [q, Fin.sum_univ_succ]; norm_num
  · intro i; fin_cases i <;> simp [p]
  · intro i; fin_cases i <;> simp [q]

/-- The laws are different. -/
theorem laws_differ : p ≠ q := by
  intro h
  have := congrFun h 0
  simp [p, q] at this

/-- Equal means, so a mean difference of 0 cannot tell them apart. -/
theorem same_mean : mean p = mean q := by
  simp [mean, p, q, Fin.sum_univ_succ]; norm_num

theorem mean_is_five_halves : mean p = 5 / 2 := by
  simp [mean, p, Fin.sum_univ_succ]; norm_num

/-- Different variances (25/4 against 1/4). -/
theorem variances : var p = 25 / 4 ∧ var q = 1 / 4 := by
  constructor
  · simp [var, mean, p, Fin.sum_univ_succ]; norm_num
  · simp [var, mean, q, Fin.sum_univ_succ]; norm_num

/-- The Brier distance, a distribution-level comparison, sees the difference. -/
theorem brier_sees_it : brier p q = 1 := by
  simp [brier, p, q, Fin.sum_univ_succ]; norm_num

/-- Squared difference between independent draws `X ~ p`, `Y ~ q`: the truthful report `q = p`
scores `25/2`, while the law `q` (a pair of point masses is even better) scores `13/2`. -/
theorem independent_matched : ∑ i, ∑ j, p i * p j * (((i : ℕ) : ℚ) - ((j : ℕ) : ℚ)) ^ 2 = 25 / 2 := by
  simp [p, Fin.sum_univ_succ]; norm_num

theorem independent_other : ∑ i, ∑ j, p i * q j * (((i : ℕ) : ℚ) - ((j : ℕ) : ℚ)) ^ 2 = 13 / 2 := by
  simp [p, q, Fin.sum_univ_succ]; norm_num

/-- A point mass at 2 scores the same `13/2` against `p`; the truthful law is worse. -/
theorem point_mass_two : ∑ i, p i * (((i : ℕ) : ℚ) - 2) ^ 2 = 13 / 2 := by
  simp [p, Fin.sum_univ_succ]; norm_num

/-! ## Correlation one, different laws (the example `Y = 2 X`) -/

/-- `X` on `0,…,4` with law `(1/4, 1/2, 1/4, 0, 0)`. -/
def px : Fin 5 → ℚ := ![1/4, 1/2, 1/4, 0, 0]

/-- `Y = 2 X` has law `(1/4, 0, 1/2, 0, 1/4)` on `0,…,4`. -/
def py : Fin 5 → ℚ := ![1/4, 0, 1/2, 0, 1/4]

def mean5 (r : Fin 5 → ℚ) : ℚ := ∑ i : Fin 5, ((i : ℕ) : ℚ) * r i

def var5 (r : Fin 5 → ℚ) : ℚ := ∑ i : Fin 5, (((i : ℕ) : ℚ) - mean5 r) ^ 2 * r i

/-- Covariance of `X` and `Y = 2 X`: `E[X * 2X] - E[X] E[2X]`. -/
def cov5 : ℚ := ∑ i : Fin 3, px (Fin.castLE (by omega) i) * ((i : ℕ) : ℚ) * (2 * ((i : ℕ) : ℚ)) - mean5 px * mean5 py

theorem laws_differ5 : px ≠ py := by
  intro h
  have := congrFun h 1
  simp [px, py] at this

theorem moments5 : mean5 px = 1 ∧ mean5 py = 2 ∧ var5 px = 1 / 2 ∧ var5 py = 2 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> simp [var5, mean5, px, py, Fin.sum_univ_succ] <;> norm_num

/-- `cov^2 = var X * var Y`, that is correlation squared equals one, so `E[(Z_X - Z_Y)^2] = 0`. -/
theorem corr_sq_one : cov5 ^ 2 = var5 px * var5 py := by
  simp [cov5, mean5, px, py, var5, Fin.sum_univ_succ]
  norm_num

/-- Yet the Brier distance between the two laws is `3/8`. -/
theorem brier5 : ∑ i : Fin 5, (px i - py i) ^ 2 = 3 / 8 := by
  simp [px, py, Fin.sum_univ_succ]; norm_num

end MeanMatching
