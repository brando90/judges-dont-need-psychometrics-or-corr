/- Exp 18, 11-21-2026: Lean 4 / Mathlib checks for the facts about Pearson correlation of judge means,
   the RPS mean-plus-shape split, and a concrete pair of distributions with the same mean.
   Run (read-only use of an existing Mathlib project): cd /home/agent/lean_vb && lake env lean <this file> -/
import Mathlib
open Finset

namespace CEvsCorr

/-- mean rating of a distribution on {0,..,5} -/
def mean (p : Fin 6 → ℝ) : ℝ := ∑ r : Fin 6, (r : ℝ) * p r

/-- the judge's mean rating is linear in the judge's distribution: this is the only way a
    mean-based objective reads p. -/
theorem mean_linear (a b : ℝ) (p q : Fin 6 → ℝ) :
    mean (fun r => a * p r + b * q r) = a * mean p + b * mean q := by
  simp only [mean, mul_add, sum_add_distrib, mul_left_comm, ← mul_sum]

/-- Pearson numerator: centering the judge means is not needed once the target is centered. -/
theorem num_centering {n : ℕ} (m t : Fin n → ℝ) (tbar mbar : ℝ)
    (htb : ∑ i, (t i - tbar) = 0) :
    ∑ i, (t i - tbar) * (m i - mbar) = ∑ i, (t i - tbar) * m i := by
  simp only [mul_sub, sum_sub_distrib, ← sum_mul, htb, zero_mul, sub_zero]

/-- Pearson of judge means against target means, as a function of the means only. -/
noncomputable def rho {n : ℕ} (m t : Fin n → ℝ) : ℝ :=
  (∑ i, (m i - (∑ j, m j) / n) * (t i - (∑ j, t j) / n)) /
    (Real.sqrt (∑ i, (m i - (∑ j, m j) / n) ^ 2) * Real.sqrt (∑ i, (t i - (∑ j, t j) / n) ^ 2))

/-- two judges whose per-item means agree have the same Pearson, whatever their shapes. -/
theorem rho_depends_only_on_means {n : ℕ} (p p' : Fin n → Fin 6 → ℝ) (t : Fin n → ℝ)
    (h : ∀ i, mean (p i) = mean (p' i)) :
    rho (fun i => mean (p i)) t = rho (fun i => mean (p' i)) t := by
  have : (fun i => mean (p i)) = (fun i => mean (p' i)) := funext h
  rw [this]

/-- level and scale are not identified: rho(a + b m) = rho(m) is shown numerically and with sympy;
    here: the numerator scales by b and the denominator by b, for b > 0 (n items, centered form). -/
theorem scale_cancels (b N D : ℝ) (hb : 0 < b) (hD : 0 < D) :
    (b * N) / (b * D) = N / D := by
  field_simp
  ring

/-- concrete pair with the same mean and different shape: uniform on {1,2,3} and uniform on {0,2,4}. -/
noncomputable def pA : Fin 6 → ℝ := ![0, 1/3, 1/3, 1/3, 0, 0]
noncomputable def pB : Fin 6 → ℝ := ![1/3, 0, 1/3, 0, 1/3, 0]

theorem same_mean : mean pA = mean pB := by
  simp [mean, pA, pB, Fin.sum_univ_succ]; norm_num

theorem different_shape : pA ≠ pB := by
  intro h
  have := congrFun h 0
  simp [pA, pB] at this

/-- cumulative distribution at k in {0..4} -/
def cdf (p : Fin 6 → ℝ) (k : Fin 5) : ℝ := ∑ j : Fin 6, if j.val ≤ k.val then p j else 0

/-- ranked probability score of p against q on 6 categories -/
def rps (p q : Fin 6 → ℝ) : ℝ := ∑ k : Fin 5, (cdf p k - cdf q k) ^ 2

theorem rps_pos_for_example : 0 < rps pA pB := by
  have h : rps pA pB = 2/9 := by
    simp [rps, cdf, pA, pB, Fin.sum_univ_succ]; norm_num
  rw [h]; norm_num

/-- the sum of CDF differences is the difference of means: sum_k (F_p(k) - F_q(k)) = m_q - m_p -/
theorem cdf_diff_sum (p q : Fin 6 → ℝ) (hp : ∑ j, p j = 1) (hq : ∑ j, q j = 1) :
    ∑ k : Fin 5, (cdf p k - cdf q k) = mean q - mean p := by
  simp only [Fin.sum_univ_six] at hp hq
  simp [cdf, mean, Fin.sum_univ_five, Fin.sum_univ_six]
  linarith

/-- RPS splits exactly into a mean part and a shape part: sum d^2 = (sum d)^2/5 + sum (d - dbar)^2 -/
theorem rps_split (d : Fin 5 → ℝ) :
    ∑ k, d k ^ 2 = (∑ k, d k) ^ 2 / 5 + ∑ k, (d k - (∑ j, d j) / 5) ^ 2 := by
  simp only [Fin.sum_univ_five]
  ring

/-- so RPS >= (m_p - m_q)^2 / 5, with equality iff the CDF difference is constant over k. -/
theorem rps_ge_mean_part (p q : Fin 6 → ℝ) (hp : ∑ j, p j = 1) (hq : ∑ j, q j = 1) :
    (mean q - mean p) ^ 2 / 5 ≤ rps p q := by
  have h1 := cdf_diff_sum p q hp hq
  have h2 := rps_split (fun k => cdf p k - cdf q k)
  have h3 : 0 ≤ ∑ k : Fin 5, ((cdf p k - cdf q k) - (∑ j : Fin 5, (cdf p j - cdf q j)) / 5) ^ 2 :=
    sum_nonneg (fun k _ => sq_nonneg _)
  unfold rps
  rw [h2, h1] at *
  linarith

end CEvsCorr
