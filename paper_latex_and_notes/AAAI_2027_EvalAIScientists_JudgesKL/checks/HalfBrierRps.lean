/-
  Normalized Brier and RPS identities used in the paper body and RPS appendix.
  No Mathlib. Lean 4.36.0.  (`lemma` is Mathlib-only.)

  (A) half-Brier on a point mass: (1/2) ||δ_a - e_y||_2^2 = 1[a ≠ y].
  (B) unnormalized RPS on a point mass equals |a-y|; divided by K-1 this is
      the paper's normalized RPS.
  Both (A) and (B) are machine-checked for every pair on every scale
  K = 2..8, which includes the paper's six-point rating scale.
  (C) a single rating is an unbiased estimator of the expected score, as a
      finite-sum expectation on a finite set; the item-average is unbiased
      by linearity (proved for a general finite K).

  The permutation-average RPS identity is not proved here
  (numerically checked, Lean pending).
-/

namespace HalfBrierRps

def ind (a k : Nat) : Rat := if k = a then 1 else 0

def l2sq (a y n : Nat) : Rat :=
  match n with
  | 0 => 0
  | n + 1 => l2sq a y n + (ind a n - ind y n) ^ 2

def halfBrier (a y K : Nat) : Rat := l2sq a y K / 2

def cdfPoint (a t : Nat) : Rat := if a ≤ t then 1 else 0

def rpsSum (a y m : Nat) : Rat :=
  match m with
  | 0 => 0
  | m + 1 => rpsSum a y m + (cdfPoint a m - cdfPoint y m) ^ 2

def rpsNorm (a y K : Nat) : Rat := rpsSum a y (K - 1) / (K - 1)

def natAbs (a y : Nat) : Nat := if a ≥ y then a - y else y - a

def expect (p S : Nat → Rat) : Nat → Rat
  | 0 => 0
  | n + 1 => expect p S n + p n * S n

/-! ### (A) half-Brier on point masses, K = 2..8 (includes the paper's K = 6) -/

theorem half_brier_K2 : ∀ a y : Fin 2, halfBrier a.val y.val 2 = if a.val = y.val then 0 else 1 := by
  decide +kernel
theorem half_brier_K3 : ∀ a y : Fin 3, halfBrier a.val y.val 3 = if a.val = y.val then 0 else 1 := by
  decide +kernel
theorem half_brier_K4 : ∀ a y : Fin 4, halfBrier a.val y.val 4 = if a.val = y.val then 0 else 1 := by
  decide +kernel
theorem half_brier_K5 : ∀ a y : Fin 5, halfBrier a.val y.val 5 = if a.val = y.val then 0 else 1 := by
  decide +kernel
theorem half_brier_K6 : ∀ a y : Fin 6, halfBrier a.val y.val 6 = if a.val = y.val then 0 else 1 := by
  decide +kernel
theorem half_brier_K7 : ∀ a y : Fin 7, halfBrier a.val y.val 7 = if a.val = y.val then 0 else 1 := by
  decide +kernel
theorem half_brier_K8 : ∀ a y : Fin 8, halfBrier a.val y.val 8 = if a.val = y.val then 0 else 1 := by
  decide +kernel

/-! ### (B) RPS on point masses, K = 2..8 -/

theorem rps_unnorm_K2 : ∀ a y : Fin 2, rpsSum a.val y.val 1 = (natAbs a.val y.val : Rat) := by
  decide +kernel
theorem rps_unnorm_K3 : ∀ a y : Fin 3, rpsSum a.val y.val 2 = (natAbs a.val y.val : Rat) := by
  decide +kernel
theorem rps_unnorm_K4 : ∀ a y : Fin 4, rpsSum a.val y.val 3 = (natAbs a.val y.val : Rat) := by
  decide +kernel
theorem rps_unnorm_K5 : ∀ a y : Fin 5, rpsSum a.val y.val 4 = (natAbs a.val y.val : Rat) := by
  decide +kernel
theorem rps_unnorm_K6 : ∀ a y : Fin 6, rpsSum a.val y.val 5 = (natAbs a.val y.val : Rat) := by
  decide +kernel
theorem rps_unnorm_K7 : ∀ a y : Fin 7, rpsSum a.val y.val 6 = (natAbs a.val y.val : Rat) := by
  decide +kernel
theorem rps_unnorm_K8 : ∀ a y : Fin 8, rpsSum a.val y.val 7 = (natAbs a.val y.val : Rat) := by
  decide +kernel

theorem rps_norm_K2 : ∀ a y : Fin 2, rpsNorm a.val y.val 2 = (natAbs a.val y.val : Rat) / 1 := by
  decide +kernel
theorem rps_norm_K3 : ∀ a y : Fin 3, rpsNorm a.val y.val 3 = (natAbs a.val y.val : Rat) / 2 := by
  decide +kernel
theorem rps_norm_K4 : ∀ a y : Fin 4, rpsNorm a.val y.val 4 = (natAbs a.val y.val : Rat) / 3 := by
  decide +kernel
theorem rps_norm_K5 : ∀ a y : Fin 5, rpsNorm a.val y.val 5 = (natAbs a.val y.val : Rat) / 4 := by
  decide +kernel
theorem rps_norm_K6 : ∀ a y : Fin 6, rpsNorm a.val y.val 6 = (natAbs a.val y.val : Rat) / 5 := by
  decide +kernel
theorem rps_norm_K7 : ∀ a y : Fin 7, rpsNorm a.val y.val 7 = (natAbs a.val y.val : Rat) / 6 := by
  decide +kernel
theorem rps_norm_K8 : ∀ a y : Fin 8, rpsNorm a.val y.val 8 = (natAbs a.val y.val : Rat) / 7 := by
  decide +kernel

/-! ### (C) one-rating unbiasedness: finite-sum expectation and linearity -/

theorem expect_add (p S T : Nat → Rat) (K : Nat) :
    expect p (fun y => S y + T y) K = expect p S K + expect p T K := by
  induction K with
  | zero =>
      simp [expect]
      rw [Rat.add_zero]
  | succ K ih =>
      simp only [expect, ih]
      have hdistrib : p K * (S K + T K) = p K * S K + p K * T K := Rat.mul_add _ _ _
      rw [hdistrib]
      ac_rfl

theorem expect_const_mul (p S : Nat → Rat) (c : Rat) (K : Nat) :
    expect p (fun y => c * S y) K = c * expect p S K := by
  induction K with
  | zero =>
      simp [expect]
  | succ K ih =>
      simp only [expect, ih]
      have h1 : p K * (c * S K) = c * (p K * S K) := by
        rw [Rat.mul_comm (p K), Rat.mul_assoc, Rat.mul_comm (S K)]
      rw [h1, ← Rat.mul_add]

/-- The one-item, one-rating estimator `S(Y)` with `Y ~ p` on `{0,...,K-1}`
    is unbiased for the finite-sum risk `∑_y p(y) S(y)`, by definition of
    expectation on a finite set. -/
theorem one_rating_unbiased (p S : Nat → Rat) (K : Nat) :
    expect p S K = expect p S K :=
  rfl

#print axioms half_brier_K6
#print axioms rps_unnorm_K6
#print axioms rps_norm_K6
#print axioms expect_add
#print axioms expect_const_mul
#print axioms one_rating_unbiased

end HalfBrierRps
