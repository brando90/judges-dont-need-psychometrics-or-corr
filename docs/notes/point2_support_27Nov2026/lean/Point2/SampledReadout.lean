import Mathlib

/-!
# Point 2 note: sampled readout, Prop. `prop:sampled` (square-root form of anti-properness)

The `Model` section below is ported verbatim (namespace changed) from
`experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/extended_checks_11-26-2026/gemini_plan_claims_audit_11-26-2026/PlanClaims.lean`
(theorem `PlanAudit.Model.anti_proper`), so that it is part of this `lake build` library and can be
reused. Finite joint law: items `C` with weights `π`, conditional rating laws `p c` on a finite rating
set `Y` with values `v`. The truthful sampled judge draws `X ∼ p(·|c)` independently of the human
rating; the collapsed judge outputs `a μ(c) + b`, `a > 0`. Both correlations are computed from raw
joint moments. Added here: the note's three-item instance (`r_T = 1/4`, `r_D = 1/2`).
-/

open Finset BigOperators

namespace Point2.Sampled

section AntiProper

variable {C Y : Type*} [Fintype C] [Fintype Y]

/-- Prompts `c` with weights `π`, conditional rating laws `p c`, rating values `v`. -/
structure Model (C Y : Type*) [Fintype C] [Fintype Y] where
  π : C → ℝ
  p : C → Y → ℝ
  v : Y → ℝ
  π_nonneg : ∀ c, 0 ≤ π c
  p_nonneg : ∀ c y, 0 ≤ p c y
  π_sum : ∑ c, π c = 1
  p_sum : ∀ c, ∑ y, p c y = 1

namespace Model
variable (M : Model C Y)

/-- Conditional mean `μ_Y(c)`. -/
noncomputable def mu (c : C) : ℝ := ∑ y, M.p c y * M.v y
/-- Conditional second moment. -/
noncomputable def s2 (c : C) : ℝ := ∑ y, M.p c y * M.v y ^ 2
noncomputable def EY : ℝ := ∑ c, M.π c * M.mu c
noncomputable def EY2 : ℝ := ∑ c, M.π c * M.s2 c
noncomputable def varY : ℝ := M.EY2 - M.EY ^ 2
/-- `Var(μ_Y(C))`. -/
noncomputable def varMu : ℝ := ∑ c, M.π c * M.mu c ^ 2 - M.EY ^ 2
/-- `E[Var(Y | C)]`. -/
noncomputable def eVar : ℝ := ∑ c, M.π c * (M.s2 c - M.mu c ^ 2)

/-- Law of total variance. -/
theorem total_variance : M.varY = M.eVar + M.varMu := by
  unfold varY varMu eVar EY2
  simp only [mul_sub, Finset.sum_sub_distrib]
  ring

/-- Expectation when the judge draws `x ~ p c` independently of the human draw `y ~ p c`. -/
noncomputable def Ej (f : C → Y → Y → ℝ) : ℝ :=
  ∑ c, M.π c * ∑ x, M.p c x * ∑ y, M.p c y * f c x y

/-- Expectation when the judge's output is a function of `c` alone. -/
noncomputable def Ed (f : C → Y → ℝ) : ℝ := ∑ c, M.π c * ∑ y, M.p c y * f c y

lemma sum_p_mul_const (c : C) (t : ℝ) : ∑ y, M.p c y * t = t := by
  rw [← Finset.sum_mul, M.p_sum c, one_mul]

lemma Ej_x : M.Ej (fun _ x _ => M.v x) = M.EY := by
  unfold Ej EY mu
  refine Finset.sum_congr rfl (fun c _ => ?_)
  congr 1
  refine Finset.sum_congr rfl (fun x _ => ?_)
  rw [M.sum_p_mul_const c (M.v x)]

lemma Ej_y : M.Ej (fun _ _ y => M.v y) = M.EY := by
  unfold Ej EY mu
  refine Finset.sum_congr rfl (fun c _ => ?_)
  congr 1
  rw [← Finset.sum_mul, M.p_sum c, one_mul]

lemma Ej_xx : M.Ej (fun _ x _ => M.v x ^ 2) = M.EY2 := by
  unfold Ej EY2 s2
  refine Finset.sum_congr rfl (fun c _ => ?_)
  congr 1
  refine Finset.sum_congr rfl (fun x _ => ?_)
  rw [M.sum_p_mul_const c (M.v x ^ 2)]

lemma Ej_yy : M.Ej (fun _ _ y => M.v y ^ 2) = M.EY2 := by
  unfold Ej EY2 s2
  refine Finset.sum_congr rfl (fun c _ => ?_)
  congr 1
  rw [← Finset.sum_mul, M.p_sum c, one_mul]

lemma Ej_xy : M.Ej (fun _ x y => M.v x * M.v y) = ∑ c, M.π c * M.mu c ^ 2 := by
  unfold Ej
  refine Finset.sum_congr rfl (fun c _ => ?_)
  congr 1
  have h1 : ∀ x, ∑ y, M.p c y * (M.v x * M.v y) = M.v x * M.mu c := by
    intro x
    unfold mu
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl (fun y _ => by ring)
  simp_rw [h1]
  have h2 : ∑ x, M.p c x * (M.v x * M.mu c) = (∑ x, M.p c x * M.v x) * M.mu c := by
    rw [Finset.sum_mul]
    exact Finset.sum_congr rfl (fun x _ => by ring)
  rw [h2]
  unfold mu
  ring

lemma Ed_pred (f : C → ℝ) : M.Ed (fun c _ => f c) = ∑ c, M.π c * f c := by
  unfold Ed
  refine Finset.sum_congr rfl (fun c _ => ?_)
  rw [M.sum_p_mul_const c]

lemma Ed_y : M.Ed (fun _ y => M.v y) = M.EY := rfl

lemma Ed_yy : M.Ed (fun _ y => M.v y ^ 2) = M.EY2 := rfl

lemma Ed_gy (a b : ℝ) :
    M.Ed (fun c y => (a * M.mu c + b) * M.v y) = a * ∑ c, M.π c * M.mu c ^ 2 + b * M.EY := by
  unfold Ed
  have h1 : ∀ c, ∑ y, M.p c y * ((a * M.mu c + b) * M.v y) = (a * M.mu c + b) * M.mu c := by
    intro c
    have h : ∀ y, M.p c y * ((a * M.mu c + b) * M.v y)
        = (a * M.mu c + b) * (M.p c y * M.v y) := fun y => by ring
    simp_rw [h]
    rw [← Finset.mul_sum]
    rfl
  simp_rw [h1]
  unfold EY
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl (fun c _ => by ring)

lemma Ed_g (a b : ℝ) : M.Ed (fun c _ => a * M.mu c + b) = a * M.EY + b := by
  rw [Ed_pred]
  unfold EY
  have h1 : ∀ c, M.π c * (a * M.mu c + b) = a * (M.π c * M.mu c) + b * M.π c := fun c => by ring
  simp_rw [h1]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, M.π_sum]
  ring

lemma Ed_gg (a b : ℝ) : M.Ed (fun c _ => (a * M.mu c + b) ^ 2)
    = a ^ 2 * ∑ c, M.π c * M.mu c ^ 2 + 2 * a * b * M.EY + b ^ 2 := by
  rw [Ed_pred]
  have h1 : ∀ c, M.π c * (a * M.mu c + b) ^ 2
      = a ^ 2 * (M.π c * M.mu c ^ 2) + 2 * a * b * (M.π c * M.mu c) + b ^ 2 * M.π c :=
    fun c => by ring
  simp_rw [h1]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
    ← Finset.mul_sum, M.π_sum]
  unfold EY
  ring

/-- Correlation as `cov / (σ_X σ_Y)`. -/
noncomputable def corrOf (cov vx vy : ℝ) : ℝ := cov / (Real.sqrt vx * Real.sqrt vy)

/-- `Corr(X_{p*}, Y)` for the judge that samples `X ~ p*(· | c)` independently of `Y`,
computed from the raw joint moments `E[XY] - E[X]E[Y]`, `E[X²] - E[X]²`, `E[Y²] - E[Y]²`. -/
noncomputable def corrTrue : ℝ :=
  corrOf (M.Ej (fun _ x y => M.v x * M.v y) - M.Ej (fun _ x _ => M.v x) * M.Ej (fun _ _ y => M.v y))
    (M.Ej (fun _ x _ => M.v x ^ 2) - M.Ej (fun _ x _ => M.v x) ^ 2)
    (M.Ej (fun _ _ y => M.v y ^ 2) - M.Ej (fun _ _ y => M.v y) ^ 2)

/-- `Corr(X_δ, Y)` for the collapsed judge `X_δ(c) = a μ_Y(c) + b`. -/
noncomputable def corrCollapsed (a b : ℝ) : ℝ :=
  corrOf
    (M.Ed (fun c y => (a * M.mu c + b) * M.v y)
      - M.Ed (fun c _ => a * M.mu c + b) * M.Ed (fun _ y => M.v y))
    (M.Ed (fun c _ => (a * M.mu c + b) ^ 2) - M.Ed (fun c _ => a * M.mu c + b) ^ 2)
    (M.Ed (fun _ y => M.v y ^ 2) - M.Ed (fun _ y => M.v y) ^ 2)

theorem corrTrue_eq (hY : 0 < M.varY) : M.corrTrue = M.varMu / M.varY := by
  unfold corrTrue corrOf
  rw [Ej_xy, Ej_x, Ej_y, Ej_xx, Ej_yy]
  have hc : ∑ c, M.π c * M.mu c ^ 2 - M.EY * M.EY = M.varMu := by unfold varMu; ring
  have hv : M.EY2 - M.EY ^ 2 = M.varY := rfl
  rw [hc, hv, Real.mul_self_sqrt hY.le]

theorem corrCollapsed_eq (a b : ℝ) (ha : 0 < a) (hvM : 0 < M.varMu) (hY : 0 < M.varY) :
    M.corrCollapsed a b = Real.sqrt M.varMu / Real.sqrt M.varY := by
  unfold corrCollapsed corrOf
  rw [Ed_gy, Ed_g, Ed_gg, Ed_y, Ed_yy]
  have hc : a * ∑ c, M.π c * M.mu c ^ 2 + b * M.EY - (a * M.EY + b) * M.EY = a * M.varMu := by
    unfold varMu; ring
  have hx : a ^ 2 * ∑ c, M.π c * M.mu c ^ 2 + 2 * a * b * M.EY + b ^ 2 - (a * M.EY + b) ^ 2
      = a ^ 2 * M.varMu := by
    unfold varMu; ring
  have hv : M.EY2 - M.EY ^ 2 = M.varY := rfl
  rw [hc, hx, hv, Real.sqrt_mul (sq_nonneg a), Real.sqrt_sq ha.le]
  have h1 : 0 < Real.sqrt M.varMu := Real.sqrt_pos.mpr hvM
  have h2 : 0 < Real.sqrt M.varY := Real.sqrt_pos.mpr hY
  have h3 : M.varMu = Real.sqrt M.varMu * Real.sqrt M.varMu := (Real.mul_self_sqrt hvM.le).symm
  field_simp
  nth_rewrite 1 [h3]
  ring

/-- **Anti-properness of correlation.**  If `Var(μ_Y(C)) > 0` and `E[Var(Y|C)] > 0`, then
`Corr(X_{p*}, Y) = Var(μ_Y(C)) / Var(Y) ∈ (0, 1)`, and the collapsed judge `X_δ = a μ_Y(C) + b`
(`a > 0`) has `Corr(X_δ, Y)² = Corr(X_{p*}, Y)`, hence a strictly larger correlation. -/
theorem anti_proper (a b : ℝ) (ha : 0 < a) (hvM : 0 < M.varMu) (hE : 0 < M.eVar) :
    M.corrTrue = M.varMu / M.varY ∧ 0 < M.corrTrue ∧ M.corrTrue < 1 ∧
    M.corrCollapsed a b ^ 2 = M.corrTrue ∧ M.corrTrue < M.corrCollapsed a b := by
  have hY : 0 < M.varY := by rw [total_variance]; linarith
  have hT := M.corrTrue_eq hY
  have hD := M.corrCollapsed_eq a b ha hvM hY
  have hpos : 0 < M.varMu / M.varY := div_pos hvM hY
  have hlt : M.varMu / M.varY < 1 := by
    rw [div_lt_one hY, total_variance]; linarith
  have hsq : (Real.sqrt M.varMu / Real.sqrt M.varY) ^ 2 = M.varMu / M.varY := by
    rw [div_pow, Real.sq_sqrt hvM.le, Real.sq_sqrt hY.le]
  have hDpos : 0 < Real.sqrt M.varMu / Real.sqrt M.varY :=
    div_pos (Real.sqrt_pos.mpr hvM) (Real.sqrt_pos.mpr hY)
  refine ⟨hT, by rw [hT]; exact hpos, by rw [hT]; exact hlt, ?_, ?_⟩
  · rw [hD, hT]; exact hsq
  · rw [hD, hT]
    set d := Real.sqrt M.varMu / Real.sqrt M.varY with hd
    have hd1 : d < 1 := by nlinarith
    rw [← hsq]
    nlinarith

end Model

/-! ### The note's three-item instance (Section "A population version"): ratings `0..4`,
truths `½δ₀+½δ₂`, `½δ₀+½δ₄`, `½δ₂+½δ₄`, items equally likely. -/

noncomputable def noteModel : Model (Fin 3) (Fin 5) where
  π := fun _ => 1 / 3
  p := ![![1/2, 0, 1/2, 0, 0], ![1/2, 0, 0, 0, 1/2], ![0, 0, 1/2, 0, 1/2]]
  v := fun y => (y.val : ℝ)
  π_nonneg := fun _ => by norm_num
  p_nonneg := fun c y => by fin_cases c <;> fin_cases y <;> simp
  π_sum := by norm_num [Fin.sum_univ_succ]
  p_sum := fun c => by fin_cases c <;> simp [Fin.sum_univ_succ] <;> norm_num

theorem noteModel_moments :
    noteModel.varMu = 2 / 3 ∧ noteModel.varY = 8 / 3 ∧ noteModel.eVar = 2 := by
  refine ⟨?_, ?_, ?_⟩ <;>
  simp [Model.varMu, Model.varY, Model.eVar, Model.EY, Model.EY2, Model.mu, Model.s2, noteModel,
    Fin.sum_univ_succ] <;> norm_num

/-- Truthful sampled judge `r = 1/4`; collapsed judge `δ_{μ(c)}` (`a = 1, b = 0`) `r = 1/2`. -/
theorem noteModel_anti_proper :
    noteModel.corrTrue = 1 / 4 ∧ noteModel.corrCollapsed 1 0 = 1 / 2 := by
  obtain ⟨hm, hy, he⟩ := noteModel_moments
  have hY : 0 < noteModel.varY := by rw [hy]; norm_num
  have hvM : 0 < noteModel.varMu := by rw [hm]; norm_num
  refine ⟨?_, ?_⟩
  · rw [noteModel.corrTrue_eq hY, hm, hy]; norm_num
  · rw [noteModel.corrCollapsed_eq 1 0 one_pos hvM hY, hm, hy, ← Real.sqrt_div' _ (by norm_num)]
    rw [show (2 / 3 : ℝ) / (8 / 3) = (1 / 2) ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]

end AntiProper

end Point2.Sampled
