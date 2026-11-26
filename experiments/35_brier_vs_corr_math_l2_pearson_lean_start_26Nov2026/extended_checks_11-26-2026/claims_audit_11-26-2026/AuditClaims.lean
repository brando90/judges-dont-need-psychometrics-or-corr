import Mathlib

open Finset BigOperators

namespace ClaimsAudit

/-! ## Part 1. The identities written on pages 1 and 2 (finite joint law).

`w : Ω → ℝ` are the weights of a finite joint law of `(X, Y)` (`∑ w = 1`). `E`, `mu`, `var`, `cov`
are the expectation, mean, variance and covariance under `w`. -/

section Identities
variable {Ω : Type*} [Fintype Ω]

noncomputable def E (w f : Ω → ℝ) : ℝ := ∑ ω, w ω * f ω
noncomputable def mu (w X : Ω → ℝ) : ℝ := E w X
noncomputable def var (w X : Ω → ℝ) : ℝ := E w (fun ω => (X ω - mu w X) ^ 2)
noncomputable def cov (w X Y : Ω → ℝ) : ℝ := E w (fun ω => (X ω - mu w X) * (Y ω - mu w Y))
noncomputable def sd (w X : Ω → ℝ) : ℝ := Real.sqrt (var w X)
/-- The z-score of the pages: `(X - μ_X) / σ_X`. -/
noncomputable def Z (w X : Ω → ℝ) : Ω → ℝ := fun ω => (X ω - mu w X) / sd w X
/-- Correlation as `cov / (σ_X σ_Y)`. -/
noncomputable def rho (w X Y : Ω → ℝ) : ℝ := cov w X Y / (sd w X * sd w Y)

lemma E_const_mul (w f : Ω → ℝ) (c : ℝ) : E w (fun ω => c * f ω) = c * E w f := by
  unfold E; rw [Finset.mul_sum]; exact Finset.sum_congr rfl (fun ω _ => by ring)

lemma E_add (w f g : Ω → ℝ) : E w (fun ω => f ω + g ω) = E w f + E w g := by
  unfold E; rw [← Finset.sum_add_distrib]; exact Finset.sum_congr rfl (fun ω _ => by ring)

lemma E_sub (w f g : Ω → ℝ) : E w (fun ω => f ω - g ω) = E w f - E w g := by
  unfold E; rw [← Finset.sum_sub_distrib]; exact Finset.sum_congr rfl (fun ω _ => by ring)

lemma E_one (w : Ω → ℝ) (hw : ∑ ω, w ω = 1) (c : ℝ) : E w (fun _ => c) = c := by
  unfold E; rw [← Finset.sum_mul, hw]; ring

/-- Page 1, right margin: `Cov(X,Y) = E[XY] - E[X] E[Y]`. -/
theorem cov_eq_raw (w X Y : Ω → ℝ) (hw : ∑ ω, w ω = 1) :
    cov w X Y = E w (fun ω => X ω * Y ω) - mu w X * mu w Y := by
  have h : ∀ ω, (X ω - mu w X) * (Y ω - mu w Y)
      = X ω * Y ω - mu w Y * X ω - mu w X * Y ω + mu w X * mu w Y := by intro ω; ring
  unfold cov
  simp only [h, E_add, E_sub, E_const_mul, E_one w hw]
  unfold mu; ring

/-- Page 1 and page 2: `Var[X] = E[X²] - μ² = E[X²] - E[X]²`. -/
theorem var_eq_raw (w X : Ω → ℝ) (hw : ∑ ω, w ω = 1) :
    var w X = E w (fun ω => X ω ^ 2) - mu w X ^ 2 := by
  have : var w X = cov w X X := by unfold var cov; simp [sq]
  rw [this, cov_eq_raw w X X hw]; simp [sq]

/-- Page 1, small notes above the expectations: `E[X²] = σ_X² + μ_X²`. -/
theorem E_sq (w X : Ω → ℝ) (hw : ∑ ω, w ω = 1) :
    E w (fun ω => X ω ^ 2) = var w X + mu w X ^ 2 := by
  rw [var_eq_raw w X hw]; ring

/-- Page 1: `E[XY] = cov(X,Y) + μ_X μ_Y`. -/
theorem E_mul (w X Y : Ω → ℝ) (hw : ∑ ω, w ω = 1) :
    E w (fun ω => X ω * Y ω) = cov w X Y + mu w X * mu w Y := by
  rw [cov_eq_raw w X Y hw]; ring

/-- Page 1: `E[(X-Y)²] = E[X²] + E[Y²] - 2E[XY]`. -/
theorem sqdiff_expand (w X Y : Ω → ℝ) :
    E w (fun ω => (X ω - Y ω) ^ 2)
      = E w (fun ω => X ω ^ 2) + E w (fun ω => Y ω ^ 2) - 2 * E w (fun ω => X ω * Y ω) := by
  have h : ∀ ω, (X ω - Y ω) ^ 2 = X ω ^ 2 + Y ω ^ 2 - 2 * (X ω * Y ω) := by intro ω; ring
  simp only [h, E_add, E_sub, E_const_mul]

/-- Page 1, boxed: `E[(X-Y)²] = σ_X² + σ_Y² - 2cov + μ_X² + μ_Y² - 2 μ_X μ_Y`. -/
theorem boxed_raw (w X Y : Ω → ℝ) (hw : ∑ ω, w ω = 1) :
    E w (fun ω => (X ω - Y ω) ^ 2)
      = var w X + var w Y - 2 * cov w X Y + mu w X ^ 2 + mu w Y ^ 2 - 2 * mu w X * mu w Y := by
  rw [sqdiff_expand, E_sq w X hw, E_sq w Y hw, E_mul w X Y hw]; ring

/-- The same, grouped: location gap, variances and covariance. -/
theorem raw_three_levels (w X Y : Ω → ℝ) (hw : ∑ ω, w ω = 1) :
    E w (fun ω => (X ω - Y ω) ^ 2)
      = (mu w X - mu w Y) ^ 2 + (var w X + var w Y - 2 * cov w X Y) := by
  rw [boxed_raw w X Y hw]; ring

/-- Centered variables (the new meaning of the tilde after the PS):
`E[(X̃ - Ỹ)²] = σ_X² + σ_Y² - 2cov`, which is *not* `2(1-ρ)` unless the variances are one. -/
theorem centered_sqdiff (w X Y : Ω → ℝ) :
    E w (fun ω => ((X ω - mu w X) - (Y ω - mu w Y)) ^ 2)
      = var w X + var w Y - 2 * cov w X Y := by
  have h : ∀ ω, ((X ω - mu w X) - (Y ω - mu w Y)) ^ 2
      = (X ω - mu w X) ^ 2 + (Y ω - mu w Y) ^ 2 - 2 * ((X ω - mu w X) * (Y ω - mu w Y)) := by
    intro ω; ring
  simp only [h, E_add, E_sub, E_const_mul]
  unfold var cov; ring

/-! ### Standardized variables (page 1 boxed result, page 2 lines) -/

lemma sd_sq (w X : Ω → ℝ) (hx : 0 < var w X) : sd w X ^ 2 = var w X := by
  unfold sd; exact Real.sq_sqrt hx.le

lemma sd_pos (w X : Ω → ℝ) (hx : 0 < var w X) : 0 < sd w X := by
  unfold sd; exact Real.sqrt_pos.mpr hx

/-- Page 2: `E[X̃] = E[X-μ]/σ = 0`, the z-score has mean zero. -/
theorem E_Z (w X : Ω → ℝ) (hw : ∑ ω, w ω = 1) : E w (Z w X) = 0 := by
  have h : ∀ ω, Z w X ω = (sd w X)⁻¹ * (X ω - mu w X) := by intro ω; unfold Z; ring
  have : E w (Z w X) = E w (fun ω => (sd w X)⁻¹ * (X ω - mu w X)) := by
    rw [show Z w X = fun ω => (sd w X)⁻¹ * (X ω - mu w X) from funext h]
  rw [this, E_const_mul, E_sub, E_one w hw]
  have : E w X = mu w X := rfl
  rw [this]; ring

/-- Page 2: `E[X̃²] = E[(X-μ)²]/σ² = Var[X]/Var[X] = 1`. -/
theorem E_Z_sq (w X : Ω → ℝ) (hx : 0 < var w X) : E w (fun ω => Z w X ω ^ 2) = 1 := by
  have h : ∀ ω, Z w X ω ^ 2 = (sd w X ^ 2)⁻¹ * (X ω - mu w X) ^ 2 := by
    intro ω; unfold Z; rw [div_pow]; ring
  have : E w (fun ω => Z w X ω ^ 2) = E w (fun ω => (sd w X ^ 2)⁻¹ * (X ω - mu w X) ^ 2) := by
    rw [show (fun ω => Z w X ω ^ 2) = fun ω => (sd w X ^ 2)⁻¹ * (X ω - mu w X) ^ 2 from funext h]
  rw [this, E_const_mul, sd_sq w X hx]
  have : E w (fun ω => (X ω - mu w X) ^ 2) = var w X := rfl
  rw [this]; field_simp

/-- Page 1: `Corr(X,Y) = E[Z_X Z_Y]`. -/
theorem E_ZZ (w X Y : Ω → ℝ) :
    E w (fun ω => Z w X ω * Z w Y ω) = rho w X Y := by
  have h : ∀ ω, Z w X ω * Z w Y ω
      = (sd w X * sd w Y)⁻¹ * ((X ω - mu w X) * (Y ω - mu w Y)) := by
    intro ω; unfold Z; rw [mul_inv]; ring
  have : E w (fun ω => Z w X ω * Z w Y ω)
      = E w (fun ω => (sd w X * sd w Y)⁻¹ * ((X ω - mu w X) * (Y ω - mu w Y))) := by
    rw [show (fun ω => Z w X ω * Z w Y ω)
        = fun ω => (sd w X * sd w Y)⁻¹ * ((X ω - mu w X) * (Y ω - mu w Y)) from funext h]
  rw [this, E_const_mul]
  have : E w (fun ω => (X ω - mu w X) * (Y ω - mu w Y)) = cov w X Y := rfl
  rw [this]; unfold rho; ring

/-- Page 1, boxed: `E[(Z_X - Z_Y)²] = 2(1 - corr)`. -/
theorem std_sqdiff (w X Y : Ω → ℝ)
    (hx : 0 < var w X) (hy : 0 < var w Y) :
    E w (fun ω => (Z w X ω - Z w Y ω) ^ 2) = 2 * (1 - rho w X Y) := by
  rw [sqdiff_expand, E_Z_sq w X hx, E_Z_sq w Y hy, E_ZZ]; ring

/-- Page 2, first expansion line as written: the cross term has coefficient 1, not 2.
Its value is `2 - ρ`, so it agrees with the correct `2(1 - ρ)` only when `ρ = 0`. -/
theorem slip_missing_two (w X Y : Ω → ℝ)
    (hx : 0 < var w X) (hy : 0 < var w Y) :
    E w (fun ω => Z w X ω ^ 2 - Z w X ω * Z w Y ω + Z w Y ω ^ 2) = 2 - rho w X Y := by
  have h : ∀ ω, Z w X ω ^ 2 - Z w X ω * Z w Y ω + Z w Y ω ^ 2
      = (Z w X ω ^ 2 + Z w Y ω ^ 2) - Z w X ω * Z w Y ω := by intro ω; ring
  simp only [h, E_add, E_sub]
  rw [E_Z_sq w X hx, E_Z_sq w Y hy, E_ZZ]; ring

theorem slip_agrees_iff (w X Y : Ω → ℝ)
    (hx : 0 < var w X) (hy : 0 < var w Y) :
    E w (fun ω => Z w X ω ^ 2 - Z w X ω * Z w Y ω + Z w Y ω ^ 2) = 2 * (1 - rho w X Y)
      ↔ rho w X Y = 0 := by
  rw [slip_missing_two w X Y hx hy]; constructor <;> intro h <;> linarith

/-- The variance of the centered variable equals the variance of `X`. -/
lemma var_centered (w X : Ω → ℝ) (hw : ∑ ω, w ω = 1) :
    var w (fun ω => X ω - mu w X) = var w X := by
  have hmu : mu w (fun ω => X ω - mu w X) = 0 := by
    unfold mu; rw [E_sub, E_one w hw]; ring
  unfold var; rw [hmu]; simp

/-- Page 2, last lines: `Var[X̃] = Var[X]/σ² = 1`. -/
theorem var_Z (w X : Ω → ℝ) (hw : ∑ ω, w ω = 1) (hx : 0 < var w X) : var w (Z w X) = 1 := by
  have hmu : mu w (Z w X) = 0 := E_Z w X hw
  unfold var; rw [hmu]; simpa using E_Z_sq w X hx

/-- Page 2, the scribbled-out line `Var[X̃] = Var[X-μ] / sqrt(Var[X̃])`: its right-hand side is
`Var[X]`, not `1`, so unless `Var[X] = 1` the line is false (the divisor must be `σ² = Var[X]`). -/
theorem scribbled_line_false (w X : Ω → ℝ) (hw : ∑ ω, w ω = 1) (hx : 0 < var w X)
    (h1 : var w X ≠ 1) :
    var w (Z w X) ≠ var w (fun ω => X ω - mu w X) / Real.sqrt (var w (Z w X)) := by
  rw [var_Z w X hw hx, var_centered w X hw]; simpa using h1.symm

/-- Page 1 and the three-term identity (the exact form used in the note): location gap,
scale gap, coupling term. -/
theorem three_term (w X Y : Ω → ℝ) (hw : ∑ ω, w ω = 1)
    (hx : 0 < var w X) (hy : 0 < var w Y) :
    E w (fun ω => (X ω - Y ω) ^ 2)
      = (mu w X - mu w Y) ^ 2 + (sd w X - sd w Y) ^ 2
        + 2 * sd w X * sd w Y * (1 - rho w X Y) := by
  have px := sd_pos w X hx
  have py := sd_pos w Y hy
  have sx := sd_sq w X hx
  have sy := sd_sq w Y hy
  have hc : cov w X Y = rho w X Y * (sd w X * sd w Y) := by
    unfold rho; field_simp
  rw [boxed_raw w X Y hw, ← sx, ← sy, hc]; ring

end Identities

/-! ## Part 2. Page 3: how many pmfs share a mean?

`momMap K j` sends a signed measure `q` on the ratings `0,…,K-1` to its first `j+1` raw moments
(`k = 0` is the total mass, `k = 1` the mean times mass, and so on). -/

section DoF

noncomputable def momMat (K j : ℕ) : Matrix (Fin (j+1)) (Fin K) ℝ :=
  Matrix.of fun k i => ((i : ℕ) : ℝ) ^ (k : ℕ)

noncomputable def momMap (K j : ℕ) : (Fin K → ℝ) →ₗ[ℝ] (Fin (j+1) → ℝ) :=
  Matrix.mulVecLin (momMat K j)

lemma momMap_apply (K j : ℕ) (q : Fin K → ℝ) (k : Fin (j+1)) :
    momMap K j q k = ∑ i : Fin K, ((i : ℕ) : ℝ) ^ (k : ℕ) * q i := by
  simp [momMap, momMat, Matrix.mulVec, dotProduct]

/-- The first `j+1` moment conditions are linearly independent when `j + 1 ≤ K`: the map is onto
(a Vandermonde system on the first `j+1` ratings can be solved). -/
theorem momMap_surjective (K j : ℕ) (h : j + 1 ≤ K) : Function.Surjective (momMap K j) := by
  intro t
  set e : Fin (j+1) → Fin K := Fin.castLE h with he_def
  have he : Function.Injective e := Fin.castLE_injective h
  set v : Fin (j+1) → ℝ := fun a => (((e a : Fin K) : ℕ) : ℝ) with hv_def
  have hv : Function.Injective v := by
    intro a b hab
    have h1 : (((e a : Fin K) : ℕ) : ℝ) = (((e b : Fin K) : ℕ) : ℝ) := hab
    have h2 : ((e a : Fin K) : ℕ) = ((e b : Fin K) : ℕ) := by exact_mod_cast h1
    exact he (Fin.ext h2)
  have hdet : (Matrix.vandermonde v).det ≠ 0 := Matrix.det_vandermonde_ne_zero_iff.mpr hv
  set M' : Matrix (Fin (j+1)) (Fin (j+1)) ℝ := (Matrix.vandermonde v).transpose with hM'
  have hdet' : IsUnit M'.det := by
    rw [hM', Matrix.det_transpose]; exact isUnit_iff_ne_zero.mpr hdet
  obtain ⟨q', hq'⟩ : ∃ q' : Fin (j+1) → ℝ, Matrix.mulVec M' q' = t :=
    ⟨Matrix.mulVec M'⁻¹ t, by rw [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv _ hdet', Matrix.one_mulVec]⟩
  refine ⟨∑ a, q' a • (Pi.single (e a) (1 : ℝ) : Fin K → ℝ), ?_⟩
  funext k
  have hsingle : ∀ (i : Fin K), momMap K j (Pi.single i 1) k = ((i : ℕ) : ℝ) ^ (k : ℕ) := by
    intro i; rw [momMap_apply]; simp [Pi.single_apply]
  rw [map_sum, Finset.sum_apply]
  simp only [map_smul, Pi.smul_apply, smul_eq_mul, hsingle]
  have hk := congrFun hq' k
  simp only [Matrix.mulVec, dotProduct, hM', Matrix.transpose_apply, Matrix.vandermonde_apply] at hk
  rw [← hk]
  refine Finset.sum_congr rfl (fun a _ => ?_)
  simp only [hv_def]; ring

/-- Normalization plus the first `j` moments leave `K - 1 - j` free directions. -/
theorem finrank_ker_momMap (K j : ℕ) (h : j + 1 ≤ K) :
    Module.finrank ℝ (LinearMap.ker (momMap K j)) = K - (j + 1) := by
  have hr := LinearMap.finrank_range_add_finrank_ker (momMap K j)
  have hrange : LinearMap.range (momMap K j) = ⊤ :=
    LinearMap.range_eq_top.mpr (momMap_surjective K j h)
  rw [hrange, finrank_top] at hr
  simp only [Module.finrank_fin_fun] at hr
  omega

/-- Page 3, "6 degrees of freedom, 1 equation": for six ratings, normalization and the mean leave
**four** free directions (not six, not five). -/
theorem four_free_directions : Module.finrank ℝ (LinearMap.ker (momMap 6 1)) = 4 := by
  have := finrank_ker_momMap 6 1 (by norm_num)
  simpa using this

/-- Matching the mean and the variance too still leaves three. -/
theorem three_after_variance : Module.finrank ℝ (LinearMap.ker (momMap 6 2)) = 3 := by
  have := finrank_ker_momMap 6 2 (by norm_num)
  simpa using this

/-- Five moments pin a six-point law. -/
theorem none_after_five_moments : Module.finrank ℝ (LinearMap.ker (momMap 6 5)) = 0 := by
  have := finrank_ker_momMap 6 5 (by norm_num)
  simpa using this

/-- With two ratings the mean (with normalization) fixes the law: the kernel is trivial. -/
theorem two_ratings_mean_is_distribution : Module.finrank ℝ (LinearMap.ker (momMap 2 1)) = 0 := by
  have := finrank_ker_momMap 2 1 (by norm_num)
  simpa using this

/-- The uniform pmf on six ratings has total mass one and mean `5/2`, and is strictly positive,
so the set of pmfs with mean `5/2` contains an interior point; near it the set is exactly this
point plus a four-dimensional space. -/
theorem uniform_in_the_set : momMap 6 1 (fun _ => (1 : ℝ) / 6) = ![1, 5 / 2] := by
  funext k
  fin_cases k
  · simp [momMap_apply]
  · simp [momMap_apply, Fin.sum_univ_succ]; norm_num

end DoF

/-! ## Part 3. A fixed pairing identifies only the conditional mean (page 3, "squared difference").

Items `z`, rating values `v x`, joint weights `w z x ≥ 0` (item `z` has rating `x` with weight
`w z x`).  A judge reports one number `f z` per item. -/

section CondMean
variable {Z X : Type*} [Fintype Z] [Fintype X]

noncomputable def m (w : Z → X → ℝ) (z : Z) : ℝ := ∑ x, w z x
noncomputable def cm (w : Z → X → ℝ) (v : X → ℝ) (z : Z) : ℝ := (∑ x, w z x * v x) / m w z
noncomputable def cvar (w : Z → X → ℝ) (v : X → ℝ) (z : Z) : ℝ :=
  (∑ x, w z x * (v x - cm w v z) ^ 2) / m w z
/-- Expected squared difference between the human rating `v x` and the judge's number `f z`. -/
noncomputable def risk2 (w : Z → X → ℝ) (v : X → ℝ) (f : Z → ℝ) : ℝ :=
  ∑ z, ∑ x, w z x * (v x - f z) ^ 2

omit [Fintype Z] in
/-- One item: `∑ₓ w (v - f)² = ∑ₓ w (v - c)² + m (c - f)²` with `c` the conditional mean. -/
lemma item_decomp (w : Z → X → ℝ) (v : X → ℝ) (z : Z) (hz : m w z ≠ 0) (f : ℝ) :
    ∑ x, w z x * (v x - f) ^ 2
      = m w z * cvar w v z + m w z * (cm w v z - f) ^ 2 := by
  set c := cm w v z with hc
  have hcm : c * m w z = ∑ x, w z x * v x := by
    rw [hc]; unfold cm; field_simp
  have hcv : m w z * cvar w v z = ∑ x, w z x * (v x - c) ^ 2 := by
    unfold cvar; rw [← hc]; field_simp
  rw [hcv]
  have h : ∀ x, w z x * (v x - f) ^ 2
      = w z x * (v x - c) ^ 2 + (2 * (c - f) * (w z x * v x) - 2 * (c - f) * c * w z x
          + (c - f) ^ 2 * w z x) := by intro x; ring
  simp only [h, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
  have hm : ∑ x, w z x = m w z := rfl
  rw [hm, ← hcm]; ring

/-- R4: `E[(X - f(Z))²] = E[Var(X | Z)] + E[(E[X | Z] - f(Z))²]`. -/
theorem fixed_pairing_decomposition (w : Z → X → ℝ) (v : X → ℝ) (hm : ∀ z, m w z ≠ 0)
    (f : Z → ℝ) :
    risk2 w v f = ∑ z, m w z * cvar w v z + ∑ z, m w z * (cm w v z - f z) ^ 2 := by
  unfold risk2
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl (fun z _ => item_decomp w v z (hm z) (f z))

/-- The conditional mean is the unique minimizer of the expected squared difference, and the
minimum is `E[Var(X | Z)]`. -/
theorem condmean_unique_minimizer (w : Z → X → ℝ) (v : X → ℝ) (hm : ∀ z, 0 < m w z)
    (f : Z → ℝ) :
    risk2 w v (cm w v) ≤ risk2 w v f
      ∧ (risk2 w v f = risk2 w v (cm w v) ↔ f = cm w v) := by
  have hne : ∀ z, m w z ≠ 0 := fun z => (hm z).ne'
  have h1 := fixed_pairing_decomposition w v hne f
  have h2 := fixed_pairing_decomposition w v hne (cm w v)
  have hnn : 0 ≤ ∑ z, m w z * (cm w v z - f z) ^ 2 :=
    Finset.sum_nonneg (fun z _ => mul_nonneg (hm z).le (sq_nonneg _))
  have h0 : ∑ z, m w z * (cm w v z - cm w v z) ^ 2 = 0 := by simp
  refine ⟨by linarith, ?_⟩
  constructor
  · intro heq
    have hz : ∑ z, m w z * (cm w v z - f z) ^ 2 = 0 := by linarith
    rw [Finset.sum_eq_zero_iff_of_nonneg (fun z _ => mul_nonneg (hm z).le (sq_nonneg _))] at hz
    funext z
    have := hz z (Finset.mem_univ z)
    have h3 : (cm w v z - f z) ^ 2 = 0 := by
      rcases mul_eq_zero.mp this with h | h
      · exact absurd h (hm z).ne'
      · exact h
    have : cm w v z - f z = 0 := by simpa using h3
    linarith
  · intro hf; rw [hf]

/-- Two different rating laws with the same conditional mean have the same minimizer: the set of
minimizers is `{cm}` and `cm` is the only thing about the law that matters. -/
theorem same_conditional_mean_same_minimizer (w w' : Z → X → ℝ) (v : X → ℝ)
    (hm : ∀ z, 0 < m w z) (hm' : ∀ z, 0 < m w' z) (hcm : cm w v = cm w' v) (f : Z → ℝ) :
    risk2 w v f = risk2 w v (cm w v) ↔ risk2 w' v f = risk2 w' v (cm w' v) := by
  rw [(condmean_unique_minimizer w v hm f).2, (condmean_unique_minimizer w' v hm' f).2, hcm]

end CondMean

/-! ### The two coins of the note, as a one-item pairing -/

section Coins
noncomputable def coinP : Unit → Fin 6 → ℝ := fun _ => ![1/2, 0, 0, 0, 0, 1/2]
noncomputable def coinQ : Unit → Fin 6 → ℝ := fun _ => ![0, 0, 1/2, 1/2, 0, 0]
noncomputable def rating : Fin 6 → ℝ := fun x => ((x : ℕ) : ℝ)

theorem coins_same_conditional_mean :
    cm coinP rating () = 5 / 2 ∧ cm coinQ rating () = 5 / 2 := by
  constructor <;> simp [cm, m, coinP, coinQ, rating, Fin.sum_univ_succ] <;> norm_num

theorem coins_irreducible_error :
    cvar coinP rating () = 25 / 4 ∧ cvar coinQ rating () = 1 / 4 := by
  constructor <;> simp [cvar, cm, m, coinP, coinQ, rating, Fin.sum_univ_succ] <;> norm_num

end Coins


/-! ## Part 4. Couplings and independent draws (page 3, "squared difference").

Part 4a: whatever the coupling, a squared difference sees the gap in means and the gap in
standard deviations, and it is zero only if the two laws are equal. -/

section Coupling
variable {Ω : Type*} [Fintype Ω]

/-- Cauchy–Schwarz for the weighted covariance. -/
theorem cov_sq_le (w X Y : Ω → ℝ) (hw0 : ∀ ω, 0 ≤ w ω) :
    cov w X Y ^ 2 ≤ var w X * var w Y := by
  have hq : ∀ t : ℝ, 0 ≤ var w X * (t * t) + (-2 * cov w X Y) * t + var w Y := by
    intro t
    have h : E w (fun ω => (t * (X ω - mu w X) - (Y ω - mu w Y)) ^ 2)
        = t ^ 2 * var w X - 2 * t * cov w X Y + var w Y := by
      have e : ∀ ω, (t * (X ω - mu w X) - (Y ω - mu w Y)) ^ 2
          = t ^ 2 * (X ω - mu w X) ^ 2 + (Y ω - mu w Y) ^ 2
            - 2 * t * ((X ω - mu w X) * (Y ω - mu w Y)) := by intro ω; ring
      simp only [e, E_add, E_sub, E_const_mul]
      unfold var cov; ring
    have hnn : 0 ≤ E w (fun ω => (t * (X ω - mu w X) - (Y ω - mu w Y)) ^ 2) :=
      Finset.sum_nonneg (fun ω _ => mul_nonneg (hw0 ω) (sq_nonneg _))
    nlinarith [h, hnn]
  have := discrim_le_zero hq
  unfold discrim at this
  nlinarith [this]

lemma var_nonneg' (w X : Ω → ℝ) (hw0 : ∀ ω, 0 ≤ w ω) : 0 ≤ var w X :=
  Finset.sum_nonneg (fun ω _ => mul_nonneg (hw0 ω) (sq_nonneg _))

/-- For every coupling, `E[(X-Y)²] ≥ (μ_X-μ_Y)² + (σ_X-σ_Y)²`. -/
theorem sqdiff_lower_bound (w X Y : Ω → ℝ) (hw : ∑ ω, w ω = 1) (hw0 : ∀ ω, 0 ≤ w ω) :
    (mu w X - mu w Y) ^ 2 + (sd w X - sd w Y) ^ 2 ≤ E w (fun ω => (X ω - Y ω) ^ 2) := by
  have hx := var_nonneg' w X hw0
  have hy := var_nonneg' w Y hw0
  have hcs : cov w X Y ≤ sd w X * sd w Y := by
    have h1 : |cov w X Y| ≤ Real.sqrt (var w X * var w Y) :=
      Real.abs_le_sqrt (cov_sq_le w X Y hw0)
    have h2 : Real.sqrt (var w X * var w Y) = sd w X * sd w Y := by
      unfold sd; rw [Real.sqrt_mul hx]
    rw [h2] at h1
    exact (le_abs_self _).trans h1
  have sx : sd w X ^ 2 = var w X := by unfold sd; exact Real.sq_sqrt hx
  have sy : sd w Y ^ 2 = var w Y := by unfold sd; exact Real.sq_sqrt hy
  rw [boxed_raw w X Y hw, ← sx, ← sy]
  nlinarith [hcs]

/-- If the squared difference is zero, then the two laws (distributions of `X` and of `Y`) are
equal: on every value, the mass of `X` equals the mass of `Y`. -/
theorem zero_sqdiff_forces_equal_laws (w X Y : Ω → ℝ) (hw0 : ∀ ω, 0 ≤ w ω)
    (h : E w (fun ω => (X ω - Y ω) ^ 2) = 0) (a : ℝ) :
    ∑ ω, (if X ω = a then w ω else 0) = ∑ ω, (if Y ω = a then w ω else 0) := by
  classical
  have h0 : ∀ ω ∈ (Finset.univ : Finset Ω), w ω * (X ω - Y ω) ^ 2 = 0 :=
    (Finset.sum_eq_zero_iff_of_nonneg (fun ω _ => mul_nonneg (hw0 ω) (sq_nonneg _))).mp h
  refine Finset.sum_congr rfl (fun ω hω => ?_)
  have hz := h0 ω hω
  rcases mul_eq_zero.mp hz with hw1 | hsq
  · simp [hw1]
  · have : X ω - Y ω = 0 := by simpa using hsq
    have hxy : X ω = Y ω := by linarith
    simp [hxy]

end Coupling

/-! Part 4b: a stochastic judge that draws its rating independently of the human's. -/

section Independent
variable {n : ℕ}

/-- `E[(X - Y)²]` for `X ~ p`, `Y ~ q` independent, on the ratings `0,…,n`. -/
noncomputable def indep (p q : Fin (n+1) → ℝ) : ℝ :=
  ∑ i, ∑ j, p i * q j * (((i : ℕ) : ℝ) - ((j : ℕ) : ℝ)) ^ 2

noncomputable def pm (p : Fin (n+1) → ℝ) : ℝ := ∑ i, p i * ((i : ℕ) : ℝ)
noncomputable def pv (p : Fin (n+1) → ℝ) : ℝ := ∑ i, p i * (((i : ℕ) : ℝ) - pm p) ^ 2

/-- `E[(X-Y)²] = σ_p² + σ_q² + (μ_p - μ_q)²` for independent draws. -/
theorem indep_formula (p q : Fin (n+1) → ℝ) (hp : ∑ i, p i = 1) (hq : ∑ j, q j = 1) :
    indep p q = pv p + pv q + (pm p - pm q) ^ 2 := by
  have e1 : ∀ i j : Fin (n+1), p i * q j * (((i : ℕ) : ℝ) - ((j : ℕ) : ℝ)) ^ 2
      = (p i * ((i : ℕ) : ℝ) ^ 2) * q j + p i * (q j * ((j : ℕ) : ℝ) ^ 2)
        - 2 * ((p i * ((i : ℕ) : ℝ)) * (q j * ((j : ℕ) : ℝ))) := by intro i j; ring
  unfold indep
  simp only [e1, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum,
    ← Finset.sum_mul, hp, hq]
  have e2 : pv p = ∑ i, p i * ((i : ℕ) : ℝ) ^ 2 - pm p ^ 2 := by
    unfold pv
    have : ∀ i : Fin (n+1), p i * (((i : ℕ) : ℝ) - pm p) ^ 2
        = p i * ((i : ℕ) : ℝ) ^ 2 - 2 * pm p * (p i * ((i : ℕ) : ℝ)) + pm p ^ 2 * p i := by
      intro i; ring
    simp only [this, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, hp]
    unfold pm; ring
  have e3 : pv q = ∑ j, q j * ((j : ℕ) : ℝ) ^ 2 - pm q ^ 2 := by
    unfold pv
    have : ∀ i : Fin (n+1), q i * (((i : ℕ) : ℝ) - pm q) ^ 2
        = q i * ((i : ℕ) : ℝ) ^ 2 - 2 * pm q * (q i * ((i : ℕ) : ℝ)) + pm q ^ 2 * q i := by
      intro i; ring
    simp only [this, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, hq]
    unfold pm; ring
  rw [e2, e3]; unfold pm; ring

/-- A point mass at rating `y`. -/
noncomputable def ptmass (y : Fin (n+1)) : Fin (n+1) → ℝ := Pi.single y 1

lemma ptmass_sum (y : Fin (n+1)) : ∑ j, ptmass y j = 1 := by simp [ptmass]

/-- Against a point mass at `y`: `E[(X-y)²] = σ_p² + (y - μ_p)²`. -/
theorem indep_point (p : Fin (n+1) → ℝ) (hp : ∑ i, p i = 1) (y : Fin (n+1)) :
    indep p (ptmass y) = pv p + (((y : ℕ) : ℝ) - pm p) ^ 2 := by
  rw [indep_formula p (ptmass y) hp (ptmass_sum y)]
  have : pm (ptmass y) = ((y : ℕ) : ℝ) := by simp [pm, ptmass, Pi.single_apply]
  have hv : pv (ptmass y) = 0 := by
    unfold pv; rw [this]; simp [ptmass, Pi.single_apply]
  rw [hv, this]; ring

/-- The truthful report `q = p` scores `2σ_p²`. -/
theorem indep_truthful (p : Fin (n+1) → ℝ) (hp : ∑ i, p i = 1) : indep p p = 2 * pv p := by
  rw [indep_formula p p hp hp]; ring

/-- Improper: if some rating that carries mass is farther from the mean than another rating
`y0` is, then a point mass at the rating nearest the mean beats the truthful report. -/
theorem truthful_not_optimal (p : Fin (n+1) → ℝ) (hp : ∑ i, p i = 1) (hp0 : ∀ i, 0 ≤ p i)
    (i0 y1 : Fin (n+1)) (hi0 : 0 < p i0)
    (hfar : (((y1 : ℕ) : ℝ) - pm p) ^ 2 < (((i0 : ℕ) : ℝ) - pm p) ^ 2) :
    ∃ y : Fin (n+1), indep p (ptmass y) < indep p p := by
  obtain ⟨y, -, hy⟩ := Finset.exists_min_image (Finset.univ : Finset (Fin (n+1)))
    (fun j => (((j : ℕ) : ℝ) - pm p) ^ 2) ⟨0, Finset.mem_univ _⟩
  refine ⟨y, ?_⟩
  rw [indep_point p hp y, indep_truthful p hp]
  have hlt : (((y : ℕ) : ℝ) - pm p) ^ 2 < (((i0 : ℕ) : ℝ) - pm p) ^ 2 :=
    lt_of_le_of_lt (hy y1 (Finset.mem_univ _)) hfar
  have hsum : (((y : ℕ) : ℝ) - pm p) ^ 2 = ∑ i, p i * (((y : ℕ) : ℝ) - pm p) ^ 2 := by
    rw [← Finset.sum_mul, hp]; ring
  have hstrict : ∑ i, p i * (((y : ℕ) : ℝ) - pm p) ^ 2 < pv p := by
    unfold pv
    refine Finset.sum_lt_sum (fun i _ => ?_) ⟨i0, Finset.mem_univ _, ?_⟩
    · exact mul_le_mul_of_nonneg_left (hy i (Finset.mem_univ _)) (hp0 i)
    · exact mul_lt_mul_of_pos_left hlt hi0
  nlinarith [hsum, hstrict]

end Independent



/-! ## Part 5. The 6.437 margin notes.

(a) Log-loss is proper (Gibbs' inequality). (b) "Log is one-to-one in its input" is not the reason:
the score `q x` is one-to-one in `q x` and is not proper. (c) The Bayes rule of the top-right
margin minimizes the posterior expected cost one observation at a time. -/

section LogLoss
variable {K : ℕ}

/-- Gibbs, strict version: among strictly positive pmfs, the expected log score is maximized at
the truth and only there. -/
theorem gibbs (p q : Fin K → ℝ) (hp : ∀ i, 0 < p i) (hq : ∀ i, 0 < q i)
    (hp1 : ∑ i, p i = 1) (hq1 : ∑ i, q i = 1) :
    ∑ i, p i * Real.log (q i) ≤ ∑ i, p i * Real.log (p i)
      ∧ (∑ i, p i * Real.log (q i) = ∑ i, p i * Real.log (p i) → q = p) := by
  have key : ∀ i, p i * Real.log (q i) - p i * Real.log (p i) ≤ q i - p i := by
    intro i
    have hpi : p i ≠ 0 := (hp i).ne'
    have hr : 0 < q i / p i := div_pos (hq i) (hp i)
    have h1 : Real.log (q i / p i) ≤ q i / p i - 1 := Real.log_le_sub_one_of_pos hr
    have h2 : p i * Real.log (q i / p i) ≤ p i * (q i / p i - 1) :=
      mul_le_mul_of_nonneg_left h1 (hp i).le
    have h3 : p i * (q i / p i - 1) = q i - p i := by field_simp
    have h4 : Real.log (q i / p i) = Real.log (q i) - Real.log (p i) :=
      Real.log_div (hq i).ne' (hp i).ne'
    rw [h4] at h2
    linarith
  have hsum : ∑ i, (q i - p i) = 0 := by rw [Finset.sum_sub_distrib, hq1, hp1]; ring
  have hdiff : ∑ i, (p i * Real.log (q i) - p i * Real.log (p i))
      = ∑ i, p i * Real.log (q i) - ∑ i, p i * Real.log (p i) := by
    rw [Finset.sum_sub_distrib]
  refine ⟨?_, ?_⟩
  · have := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => key i)
    linarith
  · intro heq
    have hd : ∀ i ∈ (Finset.univ : Finset (Fin K)),
        0 ≤ (q i - p i) - (p i * Real.log (q i) - p i * Real.log (p i)) :=
      fun i _ => by linarith [key i]
    have hz : ∑ i, ((q i - p i) - (p i * Real.log (q i) - p i * Real.log (p i))) = 0 := by
      rw [Finset.sum_sub_distrib, hsum, hdiff]; linarith
    rw [Finset.sum_eq_zero_iff_of_nonneg hd] at hz
    funext i
    have hi := hz i (Finset.mem_univ i)
    by_contra hne
    have hpi : p i ≠ 0 := (hp i).ne'
    have hr : 0 < q i / p i := div_pos (hq i) (hp i)
    have hne1 : q i / p i ≠ 1 := by
      intro h1; apply hne; exact (div_eq_one_iff_eq hpi).mp h1
    have hlt : Real.log (q i / p i) < q i / p i - 1 := Real.log_lt_sub_one_of_pos hr hne1
    have h2 : p i * Real.log (q i / p i) < p i * (q i / p i - 1) :=
      mul_lt_mul_of_pos_left hlt (hp i)
    have h3 : p i * (q i / p i - 1) = q i - p i := by field_simp
    have h4 : Real.log (q i / p i) = Real.log (q i) - Real.log (p i) :=
      Real.log_div (hq i).ne' (hp i).ne'
    rw [h4] at h2
    linarith

end LogLoss

section Counterexamples

/-- The score `q x` (a one-to-one function of `q x`): with truth `p = (3/5, 2/5)`, reporting the
truth earns `13/25` in expectation, while the point mass `(1, 0)` earns `3/5 = 15/25`.
So a score that is a one-to-one function of the probability assigned to the outcome need not be
proper. -/
theorem linear_score_improper :
    (∑ i : Fin 2, (![3/5, 2/5] : Fin 2 → ℝ) i * (![3/5, 2/5] : Fin 2 → ℝ) i)
      < ∑ i : Fin 2, (![3/5, 2/5] : Fin 2 → ℝ) i * (![1, 0] : Fin 2 → ℝ) i := by
  simp [Fin.sum_univ_succ]; norm_num

/-- Same for the score `q x ^ 2`, also one-to-one on `[0, 1]`. -/
theorem square_score_improper :
    (∑ i : Fin 2, (![3/5, 2/5] : Fin 2 → ℝ) i * ((![3/5, 2/5] : Fin 2 → ℝ) i) ^ 2)
      < ∑ i : Fin 2, (![3/5, 2/5] : Fin 2 → ℝ) i * ((![1, 0] : Fin 2 → ℝ) i) ^ 2 := by
  simp [Fin.sum_univ_succ]; norm_num

end Counterexamples

section Bayes
variable {Y H A : Type*} [Fintype Y] [Fintype H] [Fintype A] [Nonempty A]

/-- Top-right margin of the 6.437 page: choosing `Â(y) = argmin_a E_{H | y}[C(a, H)]` for each
observation `y` minimizes the expected cost over all decision rules. -/
theorem bayes_rule_optimal (py : Y → ℝ) (hpy : ∀ y, 0 ≤ py y) (post : Y → H → ℝ) (C : A → H → ℝ) :
    ∃ rule : Y → A, ∀ other : Y → A,
      ∑ y, py y * ∑ h, post y h * C (rule y) h ≤ ∑ y, py y * ∑ h, post y h * C (other y) h := by
  have hmin : ∀ y, ∃ a : A, ∀ b : A, ∑ h, post y h * C a h ≤ ∑ h, post y h * C b h := by
    intro y
    obtain ⟨a, -, ha⟩ := Finset.exists_min_image (Finset.univ : Finset A)
      (fun a => ∑ h, post y h * C a h) Finset.univ_nonempty
    exact ⟨a, fun b => ha b (Finset.mem_univ _)⟩
  choose rule hrule using hmin
  exact ⟨rule, fun other =>
    Finset.sum_le_sum (fun y _ => mul_le_mul_of_nonneg_left (hrule y (other y)) (hpy y))⟩

end Bayes

section Pushforward
variable {Ω : Type*} [Fintype Ω]

/-- Page 2, "centering doesn't really chg pmf of a 1-1 invertible op RV": under an injective
map the probability masses are carried along unchanged (the support points are relabelled). -/
theorem masses_preserved_under_injective (w X : Ω → ℝ) (T : ℝ → ℝ) (hT : Function.Injective T)
    (a : ℝ) :
    ∑ ω, (if T (X ω) = T a then w ω else 0) = ∑ ω, (if X ω = a then w ω else 0) := by
  classical
  exact Finset.sum_congr rfl (fun ω _ => by simp [hT.eq_iff])

/-- If `X = Y` then the squared difference is zero. -/
theorem sqdiff_self (w X : Ω → ℝ) : E w (fun ω => (X ω - X ω) ^ 2) = 0 := by
  simp [E]

end Pushforward


/-! ## Part 6. Non-vacuity: the hypotheses are satisfiable, and the page-2 slip really differs.

A theorem about all laws could in principle be empty if its hypotheses never held.  Here is one
explicit three-point law (weights `1/4, 1/2, 1/4`) on which every hypothesis used above holds. -/

section Instance

noncomputable def w3 : Fin 3 → ℝ := ![1/4, 1/2, 1/4]
noncomputable def X3 : Fin 3 → ℝ := ![0, 1, 2]
noncomputable def Y3 : Fin 3 → ℝ := ![0, 0, 1]

lemma w3_sum : ∑ i, w3 i = 1 := by
  simp [w3, Fin.sum_univ_three]; norm_num

lemma mu_X3 : mu w3 X3 = 1 := by
  unfold mu E; simp [w3, X3, Fin.sum_univ_three]; norm_num

lemma mu_Y3 : mu w3 Y3 = 1/4 := by
  unfold mu E; simp [w3, Y3, Fin.sum_univ_three]

lemma var_X3 : var w3 X3 = 1/2 := by
  unfold var; rw [mu_X3]; unfold E; simp [w3, X3, Fin.sum_univ_three]; norm_num

lemma var_Y3 : var w3 Y3 = 3/16 := by
  unfold var; rw [mu_Y3]; unfold E; simp [w3, Y3, Fin.sum_univ_three]; norm_num

lemma cov_XY3 : cov w3 X3 Y3 = 1/4 := by
  unfold cov; rw [mu_X3, mu_Y3]; unfold E; simp [w3, X3, Y3, Fin.sum_univ_three]; norm_num

/-- Every hypothesis used by the theorems above holds on this law, and its correlation is not 0. -/
theorem instance_hypotheses :
    (∑ i, w3 i = 1) ∧ 0 < var w3 X3 ∧ 0 < var w3 Y3 ∧ var w3 X3 ≠ 1 ∧ rho w3 X3 Y3 ≠ 0 := by
  have hx : 0 < var w3 X3 := by rw [var_X3]; norm_num
  have hy : 0 < var w3 Y3 := by rw [var_Y3]; norm_num
  refine ⟨w3_sum, hx, hy, ?_, ?_⟩
  · rw [var_X3]; norm_num
  · unfold rho
    rw [cov_XY3]
    exact div_ne_zero (by norm_num) (mul_pos (sd_pos w3 X3 hx) (sd_pos w3 Y3 hy)).ne'

/-- On this law the page-2 first expansion line is not `2(1 - ρ)`. -/
theorem instance_slip_differs :
    E w3 (fun ω => Z w3 X3 ω ^ 2 - Z w3 X3 ω * Z w3 Y3 ω + Z w3 Y3 ω ^ 2)
      ≠ 2 * (1 - rho w3 X3 Y3) := by
  obtain ⟨_, hx, hy, _, hr⟩ := instance_hypotheses
  intro h
  exact hr ((slip_agrees_iff w3 X3 Y3 hx hy).mp h)

/-- On this law the scribbled-out line is false. -/
theorem instance_scribbled_false :
    var w3 (Z w3 X3) ≠ var w3 (fun ω => X3 ω - mu w3 X3) / Real.sqrt (var w3 (Z w3 X3)) := by
  obtain ⟨hw, hx, _, h1, _⟩ := instance_hypotheses
  exact scribbled_line_false w3 X3 hw hx h1

end Instance


end ClaimsAudit
