import Mathlib

/-!
# Allocation of a rating budget: closed forms and optima (Lean 4 + Mathlib)

Part 1 checks, as exact algebra over the reals, the Study B closed forms that Experiment 39's own
QA file labels "numerics only, not a Lean proof": the absolute-error variance of the three budget-16
designs 8x2, 4x4, 2x8 under the interpolated variance components v(t) = (1-t) C1 + t C2, the two
crossovers t = 4/15 and t = 11/15, and the fact that the protocol's "equal at t = 1/2" is false.

Part 2 states what a decision study does at a fixed budget: a per-item criterion always wants more
raters; a grand-mean criterion always wants more items (when the between-item variance is positive).

Part 3 is the cost-weighted optimum for the variance of a clustered mean loss, (a + b/k)(c + d k)
with a = between-item variance, b = within-item variance, c = cost per item, d = cost per rating:
the optimum is k0 with b c = a d k0^2, and it is the unique minimizer. With free items (c = 0) one
rating per item is optimal.

Part 4 is the D-study reliability threshold: item-mean reliability s / (s + w / k) reaches r exactly
when k >= r/(1-r) * (w/s).

What is NOT proved here: that the variance of a mean of N item means, each the mean of k conditionally
independent ratings, equals (sigma_c^2 + sigma_w^2 / k) / N (a standard fact; checked by simulation in
the Python script) and that REML recovers the components (cited, simulated in Experiment 39).
-/

namespace AllocationAndClosedForms

structure Comp where
  u : ℝ
  a : ℝ
  b : ℝ
  ua : ℝ
  ub : ℝ
  ab : ℝ
  e : ℝ

noncomputable def C1 : Comp := ⟨1, 3/5, 1/10, 3/5, 1/10, 1/10, 1/2⟩
noncomputable def C2 : Comp := ⟨1, 1/10, 3/5, 1/10, 3/5, 1/10, 1/2⟩

/-- v(t) = (1-t) C1 + t C2, componentwise. -/
noncomputable def mix (t : ℝ) : Comp :=
  ⟨(1 - t) * C1.u + t * C2.u, (1 - t) * C1.a + t * C2.a, (1 - t) * C1.b + t * C2.b,
   (1 - t) * C1.ua + t * C2.ua, (1 - t) * C1.ub + t * C2.ub, (1 - t) * C1.ab + t * C2.ab,
   (1 - t) * C1.e + t * C2.e⟩

/-- Absolute-error variance of the mean over na levels of facet a and nb levels of facet b
(the formula of `d_study` in Experiment 39's src/gtheory_lib.py). -/
noncomputable def vDelta (v : Comp) (na nb : ℝ) : ℝ :=
  (v.ua / na + v.ub / nb + v.e / (na * nb)) + v.a / na + v.b / nb + v.ab / (na * nb)

theorem vDelta_8x2 (t : ℝ) : vDelta (mix t) 8 2 = 23 / 80 + 3 / 8 * t := by
  simp only [vDelta, mix, C1, C2]; ring

theorem vDelta_4x4 (t : ℝ) : vDelta (mix t) 4 4 = 31 / 80 := by
  simp only [vDelta, mix, C1, C2]; ring

theorem vDelta_2x8 (t : ℝ) : vDelta (mix t) 2 8 = 53 / 80 - 3 / 8 * t := by
  simp only [vDelta, mix, C1, C2]; ring

theorem crossover_low (t : ℝ) : vDelta (mix t) 8 2 = vDelta (mix t) 4 4 ↔ t = 4 / 15 := by
  rw [vDelta_8x2, vDelta_4x4]; constructor <;> intro h <;> linarith

theorem crossover_high (t : ℝ) : vDelta (mix t) 2 8 = vDelta (mix t) 4 4 ↔ t = 11 / 15 := by
  rw [vDelta_2x8, vDelta_4x4]; constructor <;> intro h <;> linarith

theorem unique_optimum_middle (t : ℝ) (h1 : 4 / 15 < t) (h2 : t < 11 / 15) :
    vDelta (mix t) 4 4 < vDelta (mix t) 8 2 ∧ vDelta (mix t) 4 4 < vDelta (mix t) 2 8 := by
  rw [vDelta_8x2, vDelta_4x4, vDelta_2x8]; constructor <;> linarith

theorem optimum_low (t : ℝ) (h : t < 4 / 15) :
    vDelta (mix t) 8 2 < vDelta (mix t) 4 4 ∧ vDelta (mix t) 8 2 < vDelta (mix t) 2 8 := by
  rw [vDelta_8x2, vDelta_4x4, vDelta_2x8]; constructor <;> linarith

theorem optimum_high (t : ℝ) (h : 11 / 15 < t) :
    vDelta (mix t) 2 8 < vDelta (mix t) 4 4 ∧ vDelta (mix t) 2 8 < vDelta (mix t) 8 2 := by
  rw [vDelta_8x2, vDelta_4x4, vDelta_2x8]; constructor <;> linarith

/-- The protocol's claim that the three designs have equal variance at t = 1/2 is false:
8x2 and 2x8 tie each other at 19/40, and 4x4 is strictly better at 31/80. -/
theorem not_a_three_way_tie_at_half :
    vDelta (mix (1 / 2)) 8 2 = 19 / 40 ∧ vDelta (mix (1 / 2)) 2 8 = 19 / 40 ∧
    vDelta (mix (1 / 2)) 4 4 = 31 / 80 ∧
    vDelta (mix (1 / 2)) 4 4 < vDelta (mix (1 / 2)) 8 2 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [vDelta_8x2]; norm_num
  · rw [vDelta_2x8]; norm_num
  · rw [vDelta_4x4]
  · rw [vDelta_4x4, vDelta_8x2]; norm_num

/-! ## Part 2. A fixed budget B = n k -/

/-- Per-item precision: the residual variance e / k falls strictly as the number of raters k grows. -/
theorem per_item_decreasing {e k1 k2 : ℝ} (he : 0 < e) (hk1 : 0 < k1) (hk12 : k1 < k2) :
    e / k2 < e / k1 :=
  div_lt_div_of_pos_left he hk1 hk12

/-- At a fixed budget B = n k, the grand-mean variance u / n + e / (n k) equals u / n + e / B. -/
theorem grand_mean_var_eq {u e n k B : ℝ} (hB : n * k = B) :
    u / n + e / (n * k) = u / n + e / B := by
  rw [hB]

/-- Grand-mean precision at a fixed budget: more items is strictly better when u > 0. -/
theorem grand_mean_more_items {u e B n1 n2 : ℝ} (hu : 0 < u) (hn1 : 0 < n1) (h12 : n1 < n2) :
    u / n2 + e / B < u / n1 + e / B := by
  have := div_lt_div_of_pos_left hu hn1 h12
  linarith

/-! ## Part 3. Cost-weighted optimum of (a + b/k)(c + d k) -/

theorem cost_identity {a b c d k0 k : ℝ} (hc : 0 < c) (_hk0 : 0 < k0) (hk : 0 < k)
    (hopt : b * c = a * d * k0 ^ 2) :
    (a + b / k) * (c + d * k) - (a + b / k0) * (c + d * k0) = a * d * (k - k0) ^ 2 / k := by
  have hb : b = a * d * k0 ^ 2 / c := by
    field_simp
    linarith
  subst hb
  field_simp
  ring

theorem cost_optimal {a b c d k0 k : ℝ} (ha : 0 ≤ a) (hd : 0 ≤ d) (hc : 0 < c) (hk0 : 0 < k0)
    (hk : 0 < k) (hopt : b * c = a * d * k0 ^ 2) :
    (a + b / k0) * (c + d * k0) ≤ (a + b / k) * (c + d * k) := by
  have h := cost_identity hc hk0 hk hopt
  have : 0 ≤ a * d * (k - k0) ^ 2 / k := by positivity
  linarith

theorem cost_optimal_unique {a b c d k0 k : ℝ} (ha : 0 < a) (hd : 0 < d) (hc : 0 < c)
    (hk0 : 0 < k0) (hk : 0 < k) (hopt : b * c = a * d * k0 ^ 2) :
    (a + b / k) * (c + d * k) = (a + b / k0) * (c + d * k0) ↔ k = k0 := by
  have h := cost_identity hc hk0 hk hopt
  constructor
  · intro heq
    have h0 : a * d * (k - k0) ^ 2 / k = 0 := by linarith
    have hpos : 0 < a * d := mul_pos ha hd
    have h1 : a * d * (k - k0) ^ 2 = 0 := by
      rcases div_eq_zero_iff.mp h0 with h2 | h2
      · exact h2
      · exact absurd h2 hk.ne'
    have h3 : (k - k0) ^ 2 = 0 := by
      rcases mul_eq_zero.mp h1 with h2 | h2
      · exact absurd h2 hpos.ne'
      · exact h2
    have h4 : k - k0 = 0 := by simpa using h3
    linarith
  · intro hk'
    rw [hk']

/-- Free items (cost per item c = 0): one rating per item is optimal among k >= 1. -/
theorem free_items_one_rating_optimal {a b d k : ℝ} (ha : 0 ≤ a) (hd : 0 < d)
    (hk : 1 ≤ k) :
    (a + b / 1) * (0 + d * 1) ≤ (a + b / k) * (0 + d * k) := by
  have hk0 : 0 < k := by linarith
  have h1 : (a + b / k) * (0 + d * k) = a * d * k + b * d := by
    field_simp
    ring
  have h2 : (a + b / 1) * (0 + d * 1) = a * d + b * d := by ring
  rw [h1, h2]
  nlinarith [mul_nonneg (mul_nonneg ha hd.le) (sub_nonneg.mpr hk)]

/-! ## Part 4. The D-study reliability threshold -/

/-- Item-mean reliability s / (s + w / k) reaches r (0 < r < 1) exactly when k >= r/(1-r) * (w/s). -/
theorem reliability_iff {s w r k : ℝ} (hs : 0 < s) (hw : 0 < w) (hr1 : r < 1) (hk : 0 < k) :
    r ≤ s / (s + w / k) ↔ r / (1 - r) * (w / s) ≤ k := by
  have h1r : 0 < 1 - r := by linarith
  have hden : 0 < s + w / k := by positivity
  rw [le_div_iff₀ hden]
  have e1 : r * (s + w / k) = r * s + r * w / k := by ring
  constructor
  · intro h
    have h2 : r * w / k ≤ s - r * s := by linarith
    have h3 : r * w ≤ (s - r * s) * k := by
      have := (div_le_iff₀ hk).mp h2
      linarith
    rw [div_mul_div_comm, div_le_iff₀ (mul_pos h1r hs)]
    nlinarith
  · intro h
    rw [div_mul_div_comm, div_le_iff₀ (mul_pos h1r hs)] at h
    have h3 : r * w ≤ (s - r * s) * k := by nlinarith
    have h2 : r * w / k ≤ s - r * s := by
      rw [div_le_iff₀ hk]; linarith
    linarith

end AllocationAndClosedForms

open AllocationAndClosedForms in
#print axioms vDelta_8x2
open AllocationAndClosedForms in
#print axioms vDelta_4x4
open AllocationAndClosedForms in
#print axioms vDelta_2x8
open AllocationAndClosedForms in
#print axioms crossover_low
open AllocationAndClosedForms in
#print axioms crossover_high
open AllocationAndClosedForms in
#print axioms unique_optimum_middle
open AllocationAndClosedForms in
#print axioms optimum_low
open AllocationAndClosedForms in
#print axioms optimum_high
open AllocationAndClosedForms in
#print axioms not_a_three_way_tie_at_half
open AllocationAndClosedForms in
#print axioms per_item_decreasing
open AllocationAndClosedForms in
#print axioms grand_mean_var_eq
open AllocationAndClosedForms in
#print axioms grand_mean_more_items
open AllocationAndClosedForms in
#print axioms cost_identity
open AllocationAndClosedForms in
#print axioms cost_optimal
open AllocationAndClosedForms in
#print axioms cost_optimal_unique
open AllocationAndClosedForms in
#print axioms free_items_one_rating_optimal
open AllocationAndClosedForms in
#print axioms reliability_iff
