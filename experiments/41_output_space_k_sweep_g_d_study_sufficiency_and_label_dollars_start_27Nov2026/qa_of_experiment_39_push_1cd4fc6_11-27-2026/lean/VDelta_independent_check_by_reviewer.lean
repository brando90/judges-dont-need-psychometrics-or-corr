/-
  Independent review check of Experiment 39 Study B closed forms (core Lean 4, over Rat).
  Absolute-error variance of the two-facet crossed random D-study (objects x a x b):
    vDelta = (a + ua)/na + (b + ub)/nb + (ab + e)/(na*nb).
  Components v(t) = (1-t) C1 + t C2 with
    C1 = (u,a,b,ua,ub,ab,e) = (1, 3/5, 1/10, 3/5, 1/10, 1/10, 1/2),
    C2 = (1, 1/10, 3/5, 1/10, 3/5, 1/10, 1/2).
-/

def vDelta (a b ua ub ab e na nb : Rat) : Rat :=
  ua / na + ub / nb + e / (na * nb) + a / na + b / nb + ab / (na * nb)

def cA (t : Rat) : Rat := (1 - t) * (3/5) + t * (1/10)
def cB (t : Rat) : Rat := (1 - t) * (1/10) + t * (3/5)

def v (t na nb : Rat) : Rat := vDelta (cA t) (cB t) (cA t) (cB t) (1/10) (1/2) na nb

theorem v_8x2 (t : Rat) : v t 8 2 = 23/80 + 3/8 * t := by
  unfold v vDelta cA cB; grind

theorem v_4x4 (t : Rat) : v t 4 4 = 31/80 := by
  unfold v vDelta cA cB; grind

theorem v_2x8 (t : Rat) : v t 2 8 = 53/80 - 3/8 * t := by
  unfold v vDelta cA cB; grind

theorem cross_lo : v (4/15) 8 2 = v (4/15) 4 4 := by
  rw [v_8x2, v_4x4]; grind

theorem cross_hi : v (11/15) 4 4 = v (11/15) 2 8 := by
  rw [v_4x4, v_2x8]; grind

/-- On the open interval (4/15, 11/15), 4x4 is strictly best of the three designs. -/
theorem mid_unique (t : Rat) (h1 : 4/15 < t) (h2 : t < 11/15) :
    v t 4 4 < v t 8 2 ∧ v t 4 4 < v t 2 8 := by
  rw [v_8x2, v_4x4, v_2x8]; constructor <;> grind

/-- At t = 1/2 the two unequal designs tie with each other, not with 4x4. -/
theorem half_values : v (1/2) 8 2 = 19/40 ∧ v (1/2) 4 4 = 31/80 ∧ v (1/2) 2 8 = 19/40 := by
  rw [v_8x2, v_4x4, v_2x8]; grind

/-- Outside [4/15, 11/15] the oracle is 8x2 (t < 4/15) or 2x8 (t > 11/15). -/
theorem low_8x2 (t : Rat) (h : t < 4/15) : v t 8 2 < v t 4 4 := by
  rw [v_8x2, v_4x4]; grind
theorem low_8x2_vs_2x8 (t : Rat) (h : t < 1/2) : v t 8 2 < v t 2 8 := by
  rw [v_8x2, v_2x8]; grind
theorem high_2x8 (t : Rat) (h : 11/15 < t) : v t 2 8 < v t 4 4 := by
  rw [v_2x8, v_4x4]; grind

/-- One-facet nested D-study at fixed budget B = ni*nr:
    grand-mean error variance vu/ni + ve/B is strictly decreasing in ni when vu > 0. -/
theorem grand_mean_more_items (vu ve B n1 n2 : Rat) (hvu : 0 < vu) (h1 : 0 < n1) (h12 : n1 < n2) :
    vu / n2 + ve / B < vu / n1 + ve / B := by
  have h2 : 0 < n2 := by grind
  have hinv2 : n2 * n2⁻¹ = 1 := Rat.mul_inv_cancel n2 (by grind)
  have hpos : 0 < vu / n2 := by
    rw [Rat.div_def]; exact Rat.mul_pos hvu (Rat.inv_pos.mpr h2)
  have hlt : vu / n2 * n1 < vu / n2 * n2 := Rat.mul_lt_mul_of_pos_left h12 hpos
  have hcancel : vu / n2 * n2 = vu := by rw [Rat.div_def]; grind
  have key : vu / n2 < vu / n1 := (Rat.lt_div_iff h1).mpr (by grind)
  grind

/-- Per-item (nested) D-study: ve/nr is strictly decreasing in nr when ve > 0 (U1 = max raters). -/
theorem item_mean_more_raters (ve r1 r2 : Rat) (hve : 0 < ve) (h1 : 0 < r1) (h12 : r1 < r2) :
    ve / r2 < ve / r1 := by
  have h2 : 0 < r2 := by grind
  have hinv2 : r2 * r2⁻¹ = 1 := Rat.mul_inv_cancel r2 (by grind)
  have hpos : 0 < ve / r2 := by
    rw [Rat.div_def]; exact Rat.mul_pos hve (Rat.inv_pos.mpr h2)
  have hlt : ve / r2 * r1 < ve / r2 * r2 := Rat.mul_lt_mul_of_pos_left h12 hpos
  have hcancel : ve / r2 * r2 = ve := by rw [Rat.div_def]; grind
  exact (Rat.lt_div_iff h1).mpr (by grind)

/-- Concrete instance showing the hypotheses can hold. -/
example : (1 : Rat) / 64 + 1 / 128 < 1 / 32 + 1 / 128 := grand_mean_more_items 1 1 128 32 64 (by decide) (by decide) (by decide)

#print axioms v_8x2
#print axioms v_4x4
#print axioms v_2x8
#print axioms cross_lo
#print axioms cross_hi
#print axioms mid_unique
#print axioms half_values
#print axioms low_8x2
#print axioms low_8x2_vs_2x8
#print axioms high_2x8
#print axioms grand_mean_more_items
#print axioms item_mean_more_raters
