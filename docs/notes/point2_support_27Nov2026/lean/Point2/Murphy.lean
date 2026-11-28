import Point2.Moments

/-!
# Point 2 note: conditional expectation on a finite law, the Murphy decomposition
(Prop. `prop:murphy`), and the expected-score readout (Prop. `prop:expected`).

`key : ι → β` is the information conditioned on (the score value `S`, the forecast vector, or the
item `C`). `condE key Y i = E[Y | key = key i]`, computed fiberwise; on a zero-weight fiber it is 0
(any value works there, since those points carry no weight).
-/

open Finset BigOperators

namespace Point2
namespace FinLaw

set_option linter.unusedSectionVars false

variable {ι β : Type*} [Fintype ι] [DecidableEq β] (P : FinLaw ι)

def fibW (key : ι → β) (b : β) : ℝ := ∑ j ∈ univ.filter (fun j => key j = b), P.w j
def fibWY (key : ι → β) (Y : ι → ℝ) (b : β) : ℝ :=
  ∑ j ∈ univ.filter (fun j => key j = b), P.w j * Y j
/-- `E[Y | key = b]` on the fiber `b`. -/
noncomputable def condVal (key : ι → β) (Y : ι → ℝ) (b : β) : ℝ := P.fibWY key Y b / P.fibW key b
/-- `E[Y | key](i)`. -/
noncomputable def condE (key : ι → β) (Y : ι → ℝ) (i : ι) : ℝ := P.condVal key Y (key i)

lemma fib_balance (key : ι → β) (Y : ι → ℝ) (b : β) :
    P.fibWY key Y b - P.condVal key Y b * P.fibW key b = 0 := by
  unfold condVal
  by_cases h0 : P.fibW key b = 0
  · have hz : ∀ j ∈ univ.filter (fun j => key j = b), P.w j = 0 :=
      (sum_eq_zero_iff_of_nonneg (fun j _ => P.nonneg j)).mp h0
    have : P.fibWY key Y b = 0 := sum_eq_zero (fun j hj => by rw [hz j hj, zero_mul])
    rw [this, h0]; simp
  · rw [div_mul_cancel₀ _ h0, sub_self]

/-- Orthogonality: `E[(Y - E[Y|key]) h(key)] = 0` for every `h`. -/
theorem orth (key : ι → β) (Y : ι → ℝ) (h : β → ℝ) :
    P.E (fun i => (Y i - P.condE key Y i) * h (key i)) = 0 := by
  unfold E
  rw [← sum_fiberwise_of_maps_to (s := univ) (t := univ.image key) (g := key)
    (fun i _ => mem_image_of_mem key (mem_univ i))]
  apply sum_eq_zero
  intro b _
  have hpt : ∀ i ∈ univ.filter (fun i => key i = b),
      P.w i * ((Y i - P.condE key Y i) * h (key i))
        = h b * (P.w i * Y i) - h b * P.condVal key Y b * P.w i := by
    intro i hi
    rw [mem_filter] at hi
    simp only [condE, hi.2]; ring
  rw [sum_congr rfl hpt, sum_sub_distrib, ← mul_sum, ← mul_sum]
  have := P.fib_balance key Y b
  unfold fibWY fibW at this
  linear_combination h b * this

lemma E_condE (key : ι → β) (Y : ι → ℝ) : P.E (P.condE key Y) = P.E Y := by
  have := P.orth key Y (fun _ => 1)
  simp only [mul_one] at this
  rw [E_sub] at this; linarith

/-- **Prop. `prop:murphy`** (finite law, any score `S` that is a function of `key`; in particular
`key = S`): `E(Y - S)² = Var Y - Var(g) + E(g - S)²` with `g = E[Y | key]`
(uncertainty − resolution + reliability). -/
theorem murphy (key : ι → β) (Y S : ι → ℝ) (s : β → ℝ) (hS : ∀ i, S i = s (key i)) :
    P.E (fun i => (Y i - S i) ^ 2)
      = P.var Y - P.var (P.condE key Y) + P.E (fun i => (P.condE key Y i - S i) ^ 2) := by
  set g := P.condE key Y with hg
  have o1 : P.E (fun i => (Y i - g i) * (g i - S i)) = 0 := by
    have := P.orth key Y (fun b => P.condVal key Y b - s b)
    simp only [hS]; exact this
  have o2 : P.E (fun i => (Y i - g i) * g i) = 0 := P.orth key Y (fun b => P.condVal key Y b)
  have o3 : P.E g = P.E Y := P.E_condE key Y
  unfold var cov
  rw [o3]
  unfold E at *
  rw [show ∑ i, P.w i * (Y i - S i) ^ 2 = ∑ i, (P.w i * (Y i * Y i) - P.w i * (g i * g i)
      - 2 * (P.w i * ((Y i - g i) * g i)) + 2 * (P.w i * ((Y i - g i) * (g i - S i)))
      + P.w i * (g i - S i) ^ 2) from sum_congr rfl (fun i _ => by ring)]
  simp only [sum_add_distrib, sum_sub_distrib, ← mul_sum]
  rw [o1, o2]; ring

/-- Reliability is zero iff the score is calibrated on the support. -/
theorem reliability_zero_iff (key : ι → β) (Y S : ι → ℝ) :
    P.E (fun i => (P.condE key Y i - S i) ^ 2) = 0 ↔ ∀ i, P.w i ≠ 0 → P.condE key Y i = S i := by
  unfold E
  rw [sum_eq_zero_iff_of_nonneg (fun i _ => mul_nonneg (P.nonneg i) (sq_nonneg _))]
  constructor
  · intro h i hi
    have := h i (mem_univ i)
    rcases mul_eq_zero.mp this with h1 | h1
    · exact absurd h1 hi
    · exact sub_eq_zero.mp (pow_eq_zero_iff (n := 2) (by norm_num) |>.mp h1)
  · intro h i _
    by_cases hi : P.w i = 0
    · rw [hi, zero_mul]
    · rw [h i hi, sub_self]; ring

/-- Scalar-score form of Prop. `prop:murphy` with `g(s) = E[Y | S = s]`. -/
theorem murphy_scalar (Y S : ι → ℝ) :
    P.E (fun i => (Y i - S i) ^ 2)
      = P.var Y - P.var (P.condE S Y) + P.E (fun i => (P.condE S Y i - S i) ^ 2) :=
  P.murphy S Y S id (fun _ => rfl)

/-- Calibration curve is unchanged by an injective affine relabelling of the score (`b ≠ 0`). -/
theorem condE_affine (Y S : ι → ℝ) (a b : ℝ) (hb : b ≠ 0) :
    P.condE (fun i => a + b * S i) Y = P.condE S Y := by
  funext i
  unfold condE condVal fibWY fibW
  have : ∀ j, (a + b * S j = a + b * S i) ↔ (S j = S i) := fun j => by
    constructor
    · intro h; have := mul_left_cancel₀ hb (add_left_cancel h); exact this
    · intro h; rw [h]
  simp only [this]

/-- **Prop. `prop:murphy`, second display.** With the best affine refit `S' = a* + b* S`
(`Var S > 0`, `Var Y > 0`): `r(S,Y)² Var Y = Var(g(S)) - E(g(S) - S')²`. -/
theorem r_sq_resolution (Y S : ι → ℝ) (hS : 0 < P.var S) (hY : 0 < P.var Y) :
    P.corr S Y ^ 2 * P.var Y
      = P.var (P.condE S Y)
        - P.E (fun i => (P.condE S Y i
            - ((P.E Y - P.cov S Y / P.var S * P.E S) + P.cov S Y / P.var S * S i)) ^ 2) := by
  have hm := P.murphy S Y (fun i => (P.E Y - P.cov S Y / P.var S * P.E S) + P.cov S Y / P.var S * S i)
    (fun x => (P.E Y - P.cov S Y / P.var S * P.E S) + P.cov S Y / P.var S * x) (fun _ => rfl)
  have ho := P.mse_affine_opt S Y hS
  have e : (fun i => (Y i - ((P.E Y - P.cov S Y / P.var S * P.E S) + P.cov S Y / P.var S * S i)) ^ 2)
      = (fun i => (Y i - (P.E Y - P.cov S Y / P.var S * P.E S) - P.cov S Y / P.var S * S i) ^ 2) := by
    funext i; ring
  rw [e, ho] at hm
  rw [P.corr_sq S Y hS hY]
  have h2 : P.cov S Y ^ 2 / (P.var S * P.var Y) * P.var Y = P.cov S Y ^ 2 / P.var S := by
    field_simp
  rw [h2]; linarith

/-! ### Multiclass Brier: Murphy decomposition with finite forecast support. -/

lemma E_sum {κ : Type*} [Fintype κ] (f : κ → ι → ℝ) :
    P.E (fun i => ∑ k, f k i) = ∑ k, P.E (f k) := by
  unfold E; simp only [mul_sum]; exact sum_comm

/-- Vector forecast `f i : κ → ℝ`, one-hot outcome coordinates `Y k i`, Brier score
`∑ₖ (Yₖ - fₖ)²`, conditioning on the forecast vector (finite support since `ι` is finite):
`E Brier = ∑ₖ [Var Yₖ - Var gₖ + E(gₖ - fₖ)²]` with `gₖ = E[Yₖ | f]`. -/
theorem brier_murphy {κ : Type*} [Fintype κ] [DecidableEq (κ → ℝ)]
    (f : ι → κ → ℝ) (Y : κ → ι → ℝ) :
    P.E (fun i => ∑ k, (Y k i - f i k) ^ 2)
      = ∑ k, (P.var (Y k) - P.var (P.condE f (Y k))
          + P.E (fun i => (P.condE f (Y k) i - f i k) ^ 2)) := by
  rw [P.E_sum (fun k i => (Y k i - f i k) ^ 2)]
  exact sum_congr rfl (fun k _ => P.murphy f (Y k) (fun i => f i k) (fun v => v k) (fun _ => rfl))

theorem brier_reliability_zero_iff {κ : Type*} [Fintype κ] [DecidableEq (κ → ℝ)]
    (f : ι → κ → ℝ) (Y : κ → ι → ℝ) :
    ∑ k, P.E (fun i => (P.condE f (Y k) i - f i k) ^ 2) = 0
      ↔ ∀ i, P.w i ≠ 0 → ∀ k, P.condE f (Y k) i = f i k := by
  rw [sum_eq_zero_iff_of_nonneg (fun k _ => P.E_nonneg _ (fun i => sq_nonneg _))]
  constructor
  · intro h i hi k
    exact (P.reliability_zero_iff f (Y k) (fun i => f i k)).mp (h k (mem_univ k)) i hi
  · intro h k _
    exact (P.reliability_zero_iff f (Y k) (fun i => f i k)).mpr (fun i hi => h i hi k)

/-! ### Prop. `prop:expected`: expected-score readout. `key` is the item `C`, `μ = E[Y | C]`,
and the readout `s` is a function of the item. -/

lemma cov_readout (key : ι → β) (Y : ι → ℝ) (s : ι → ℝ) (s' : β → ℝ) (hs : ∀ i, s i = s' (key i)) :
    P.cov s Y = P.cov s (P.condE key Y) := by
  have o := P.orth key Y s'
  simp only [← hs] at o
  unfold cov
  rw [P.E_condE key Y]
  have : P.E (fun i => s i * Y i) = P.E (fun i => s i * P.condE key Y i) := by
    have e : (fun i => (Y i - P.condE key Y i) * s i)
        = fun i => s i * Y i - s i * P.condE key Y i := by funext i; ring
    rw [e, E_sub] at o; linarith
  rw [this]

/-- Cauchy–Schwarz for the finite-law covariance. -/
lemma cov_sq_le (f g : ι → ℝ) : P.cov f g ^ 2 ≤ P.var f * P.var g := by
  rcases (P.var_nonneg f).lt_or_eq with hf | hf
  · have h0 := P.mse_affine_opt f g hf
    have h1 := P.E_nonneg (fun i => (g i - (P.E g - P.cov f g / P.var f * P.E f)
      - P.cov f g / P.var f * f i) ^ 2) (fun i => sq_nonneg _)
    rw [h0] at h1
    have : P.cov f g ^ 2 / P.var f ≤ P.var g := by linarith
    rw [div_le_iff₀ hf] at this; linarith
  · -- Var f = 0 forces Cov f g = 0
    have hv : ∀ t : ℝ, 0 ≤ P.var g - 2 * t * P.cov f g := by
      intro t
      have := P.E_nonneg (fun i => (g i - (P.E g - t * P.E f) - t * f i) ^ 2) (fun i => sq_nonneg _)
      rw [P.mse_affine, ← hf] at this; nlinarith [this]
    have hc : P.cov f g = 0 := by
      by_contra hc
      have e : 2 * ((P.var g + 1) / (2 * P.cov f g)) * P.cov f g = P.var g + 1 := by
        field_simp
      linarith [hv ((P.var g + 1) / (2 * P.cov f g)), e]
    rw [hc, ← hf]; simp

/-- **Prop. `prop:expected`, identity and bound.** For an item-level readout `s` with
`Var s > 0`, `Var μ > 0`, `Var Y > 0`: `r(s,Y) = r(s,μ) · √(Var μ / Var Y)` and
`r(s,Y) ≤ √(Var μ / Var Y) = r(μ,Y)`. -/
theorem expected_readout (key : ι → β) (Y s : ι → ℝ) (s' : β → ℝ) (hs : ∀ i, s i = s' (key i))
    (hsv : 0 < P.var s) (hmu : 0 < P.var (P.condE key Y)) (hY : 0 < P.var Y) :
    P.corr s Y = P.corr s (P.condE key Y) * Real.sqrt (P.var (P.condE key Y) / P.var Y) ∧
    P.corr (P.condE key Y) Y = Real.sqrt (P.var (P.condE key Y) / P.var Y) ∧
    P.corr s Y ≤ P.corr (P.condE key Y) Y := by
  set μ := P.condE key Y
  have hcov : P.cov s Y = P.cov s μ := P.cov_readout key Y s s' hs
  have hcovμ : P.cov μ Y = P.var μ := by
    rw [P.cov_readout key Y μ (P.condVal key Y) (fun _ => rfl)]; rfl
  have sμ := Real.sqrt_pos.mpr hmu
  have sY := Real.sqrt_pos.mpr hY
  have ss := Real.sqrt_pos.mpr hsv
  have e1 : P.corr s Y = P.corr s μ * Real.sqrt (P.var μ / P.var Y) := by
    unfold corr; rw [hcov, Real.sqrt_div hmu.le]; field_simp
  have e2 : P.corr μ Y = Real.sqrt (P.var μ / P.var Y) := by
    unfold corr; rw [hcovμ, Real.sqrt_div hmu.le]
    have : P.var μ = Real.sqrt (P.var μ) * Real.sqrt (P.var μ) := (Real.mul_self_sqrt hmu.le).symm
    field_simp; nth_rewrite 1 [this]; ring
  refine ⟨e1, e2, ?_⟩
  rw [e1, e2]
  have hle : P.corr s μ ≤ 1 := by
    unfold corr
    rw [div_le_one (by positivity)]
    have cs := P.cov_sq_le s μ
    have : P.cov s μ ≤ |P.cov s μ| := le_abs_self _
    have h2 : |P.cov s μ| ≤ Real.sqrt (P.var s) * Real.sqrt (P.var μ) := by
      rw [← Real.sqrt_mul hsv.le, ← Real.sqrt_sq_eq_abs]
      exact Real.sqrt_le_sqrt cs
    linarith
  have := Real.sqrt_nonneg (P.var μ / P.var Y)
  nlinarith

/-- Prop. `prop:expected`, equality case (`if` direction): a positive affine image of `μ` attains
the maximum. -/
theorem expected_readout_attained (key : ι → β) (Y : ι → ℝ) (a b : ℝ) (ha : 0 < a) :
    P.corr (fun i => a * P.condE key Y i + b) Y = P.corr (P.condE key Y) Y :=
  P.pearson_affine_pos _ Y a b ha

/-- Prop. `prop:expected`, equality case (`only if` direction): if the readout attains
`r(s,Y) = r(μ,Y)`, then `μ = a* + b* s` on the support, with `b* = Cov(s,μ)/Var s > 0`. -/
theorem expected_readout_equality (key : ι → β) (Y s : ι → ℝ) (s' : β → ℝ)
    (hs : ∀ i, s i = s' (key i))
    (hsv : 0 < P.var s) (hmu : 0 < P.var (P.condE key Y)) (hY : 0 < P.var Y)
    (heq : P.corr s Y = P.corr (P.condE key Y) Y) :
    0 < P.cov s (P.condE key Y) / P.var s ∧
    ∀ i, P.w i ≠ 0 → P.condE key Y i = (P.E (P.condE key Y)
        - P.cov s (P.condE key Y) / P.var s * P.E s) + P.cov s (P.condE key Y) / P.var s * s i := by
  set μ := P.condE key Y
  obtain ⟨e1, e2, _⟩ := P.expected_readout key Y s s' hs hsv hmu hY
  have hpos := Real.sqrt_pos.mpr (div_pos hmu hY)
  have hr1 : P.corr s μ = 1 := by
    rw [e1, e2] at heq
    have := mul_right_cancel₀ hpos.ne' (heq.trans (one_mul _).symm); exact this
  have ss := Real.sqrt_pos.mpr hsv
  have sμ := Real.sqrt_pos.mpr hmu
  have hc : P.cov s μ = Real.sqrt (P.var s) * Real.sqrt (P.var μ) := by
    unfold corr at hr1; rw [div_eq_one_iff_eq (by positivity)] at hr1; exact hr1
  have hc2 : P.cov s μ ^ 2 = P.var s * P.var μ := by
    rw [hc, mul_pow, Real.sq_sqrt hsv.le, Real.sq_sqrt hmu.le]
  refine ⟨div_pos (by rw [hc]; positivity) hsv, ?_⟩
  have h0 := P.mse_affine_opt s μ hsv
  have hz : P.var μ - P.cov s μ ^ 2 / P.var s = 0 := by rw [hc2]; field_simp; ring
  rw [hz] at h0
  unfold E at h0
  rw [sum_eq_zero_iff_of_nonneg (fun i _ => mul_nonneg (P.nonneg i) (sq_nonneg _))] at h0
  intro i hi
  have := h0 i (mem_univ i)
  rcases mul_eq_zero.mp this with h1 | h1
  · exact absurd h1 hi
  · have := pow_eq_zero_iff (n := 2) (by norm_num) |>.mp h1
    unfold E; linarith

end FinLaw
end Point2
