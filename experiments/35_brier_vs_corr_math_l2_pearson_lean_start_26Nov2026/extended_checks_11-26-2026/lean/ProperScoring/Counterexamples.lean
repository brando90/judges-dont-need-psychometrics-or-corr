/-
Exact counterexamples (experiment 35, checks N19 and N20). No `sorry`, no new axioms.
Binary items: forecast probability `m i` of rating 1, true probability `t i`.
The per-item expected score is the two-category Brier score `expBrier` of `Basic.lean`.
-/
import ProperScoring.Basic

open Finset BigOperators

namespace ProperScoring

/-- Expected two-category Brier score of forecast `m` under truth `t`. -/
noncomputable def binBrier (m t : ℝ) : ℝ := expBrier (ι := Fin 2) ![1 - t, t] ![1 - m, m]

lemma binBrier_eq (m t : ℝ) : binBrier m t = 2 * ((m - t) ^ 2 + t * (1 - t)) := by
  simp [binBrier, expBrier, brier, Fin.sum_univ_two]
  ring

/-! ### N19: calibrated judge A = t and squashed judge B = 1/2 + (t - 1/2)/10 -/

noncomputable def t5 : Fin 5 → ℝ := ![1/10, 3/10, 1/2, 7/10, 9/10]
noncomputable def w5 : Fin 5 → ℝ := fun _ => 1/5
noncomputable def B5 : Fin 5 → ℝ := fun i => (1/10) * t5 i + 9/20

lemma w5_sum : ∑ i, w5 i = 1 := by simp [w5, Fin.sum_univ_five]; try norm_num

lemma var_t5_pos : 0 < var w5 t5 := by
  simp [var, cov, wmean, w5, t5, Fin.sum_univ_five]; norm_num

theorem N19_equal_corr : corr w5 t5 t5 = 1 ∧ corr w5 B5 t5 = 1 := by
  refine ⟨corr_self w5 t5 var_t5_pos, ?_⟩
  have h := corr_affine_invariant w5 t5 t5 w5_sum (1/10) (9/20) (by norm_num)
  have hB : B5 = fun i => (1/10) * t5 i + 9/20 := rfl
  rw [hB, h, corr_self w5 t5 var_t5_pos]

theorem N19_different_brier :
    ∑ i, w5 i * binBrier (t5 i) (t5 i) = 17/50 ∧
    ∑ i, w5 i * binBrier (B5 i) (t5 i) = 587/1250 := by
  simp only [binBrier_eq]
  constructor <;> (simp [w5, t5, B5, Fin.sum_univ_five]; norm_num)

/-! ### N20: ordering reversal. Judge C has correlation 1, judge D strictly less,
yet C has strictly larger expected Brier score. -/

noncomputable def t4 : Fin 4 → ℝ := ![1/5, 2/5, 3/5, 4/5]
noncomputable def w4 : Fin 4 → ℝ := fun _ => 1/4
noncomputable def C4 : Fin 4 → ℝ := fun i => (1/10) * t4 i + 9/20
noncomputable def D4 : Fin 4 → ℝ := ![1/4, 7/20, 13/20, 3/4]

lemma w4_sum : ∑ i, w4 i = 1 := by simp [w4, Fin.sum_univ_four]; try norm_num

lemma var_t4_pos : 0 < var w4 t4 := by
  simp [var, cov, wmean, w4, t4, Fin.sum_univ_four]; norm_num

theorem N20_corr_C : corr w4 C4 t4 = 1 := by
  have h := corr_affine_invariant w4 t4 t4 w4_sum (1/10) (9/20) (by norm_num)
  have hC : C4 = fun i => (1/10) * t4 i + 9/20 := rfl
  rw [hC, h, corr_self w4 t4 var_t4_pos]

lemma D4_cov : cov w4 D4 t4 = 9/200 := by
  simp [cov, wmean, w4, t4, D4, Fin.sum_univ_four]; norm_num
lemma D4_var : var w4 D4 = 17/400 := by
  simp [var, cov, wmean, w4, D4, Fin.sum_univ_four]; norm_num
lemma t4_var : var w4 t4 = 1/20 := by
  simp [var, cov, wmean, w4, t4, Fin.sum_univ_four]; norm_num

theorem N20_corr_D_lt_one : corr w4 D4 t4 < 1 := by
  unfold corr
  rw [D4_cov, D4_var, t4_var]
  have hpos : (0:ℝ) < Real.sqrt (17/400 * (1/20)) := Real.sqrt_pos.mpr (by norm_num)
  rw [div_lt_one hpos]
  rw [Real.lt_sqrt (by norm_num)]
  norm_num

theorem N20_brier_reversal :
    ∑ i, w4 i * binBrier (C4 i) (t4 i) = 481/1000 ∧
    ∑ i, w4 i * binBrier (D4 i) (t4 i) = 81/200 ∧
    ∑ i, w4 i * binBrier (D4 i) (t4 i) < ∑ i, w4 i * binBrier (C4 i) (t4 i) := by
  simp only [binBrier_eq]
  refine ⟨?_, ?_, ?_⟩ <;> (simp [w4, t4, C4, D4, Fin.sum_univ_four]; norm_num)

end ProperScoring
