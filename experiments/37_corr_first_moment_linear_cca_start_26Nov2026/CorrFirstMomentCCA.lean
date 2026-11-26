set_option linter.unusedSimpArgs false
/-
  CorrFirstMomentCCA.lean — core Lean 4 only (no Mathlib), over `Rat`.

  What Pearson correlation is, once a judge is independent of the human given the prompt,
  and what canonical correlation analysis (CCA) adds.

  1. Conditional independence factors the cross moment, so
     Cov(X, Y) = Cov_c(μ_X, μ_Y). The law of total variance puts conditional
     spread only in the denominator.
  2. A symmetric quadratic, Y = X^2 on {-1, 0, 1}, has Cov = 0. Adding a
     constant does not change covariance. Across prompts, Pearson is a
     standardized linear second-moment statistic.
  3. An invertible affine map of a vector view does not change the attainable
     correlations: uᵀ(Ax + b) = (Aᵀu)ᵀx + uᵀb, and Aᵀ is bijective when A is.
  4. A diagonal two-aspect witness: collapsing conditional variance strictly
     raises both squared canonical correlations, and the point-mass forecast
     is strictly worse on Brier and on the ranked probability score, and it
     puts zero mass on a class the human still uses, so cross-entropy is not finite.
-/

def sumF : (n : Nat) → (Fin n → Rat) → Rat
  | 0, _ => 0
  | n+1, f => f 0 + sumF n (fun i => f i.succ)

theorem sumF_add (n : Nat) (f g : Fin n → Rat) :
    sumF n (fun i => f i + g i) = sumF n f + sumF n g := by
  induction n with
  | zero => simp only [sumF]; grind
  | succ n ih => simp only [sumF]; rw [ih (fun i => f i.succ) (fun i => g i.succ)]; grind

theorem sumF_sub (n : Nat) (f g : Fin n → Rat) :
    sumF n (fun i => f i - g i) = sumF n f - sumF n g := by
  induction n with
  | zero => simp only [sumF]; grind
  | succ n ih => simp only [sumF]; rw [ih (fun i => f i.succ) (fun i => g i.succ)]; grind

theorem sumF_mul_left (n : Nat) (c : Rat) (f : Fin n → Rat) :
    sumF n (fun i => c * f i) = c * sumF n f := by
  induction n with
  | zero => simp only [sumF]; grind
  | succ n ih => simp only [sumF]; rw [ih (fun i => f i.succ)]; grind

theorem sumF_congr (n : Nat) (f g : Fin n → Rat) (h : ∀ i, f i = g i) :
    sumF n f = sumF n g := by
  have : f = g := funext h
  rw [this]

theorem sq_nonneg' (a : Rat) : 0 ≤ a * a := by
  cases Rat.le_total (a := 0) (b := a) with
  | inl h => exact Rat.mul_nonneg h h
  | inr h =>
    have h' : 0 ≤ -a := by grind
    have := Rat.mul_nonneg h' h'
    grind

theorem sq_eq_zero' (a : Rat) (h : a * a = 0) : a = 0 := by
  cases Rat.mul_eq_zero.mp h with
  | inl h => exact h
  | inr h => exact h

/-! ## Sample covariance, shift, and scale -/

theorem sumF_const (n : Nat) (c : Rat) : sumF n (fun _ => c) = (n : Rat) * c := by
  induction n with
  | zero => simp only [sumF]; grind
  | succ n ih => simp only [sumF]; rw [ih]; grind

def mean (n : Nat) (x : Fin n → Rat) : Rat := sumF n x / n

def cov (n : Nat) (x y : Fin n → Rat) : Rat :=
  sumF n (fun i => (x i - mean n x) * (y i - mean n y)) / n

theorem cov_raw (n : Nat) (hn : (n : Rat) ≠ 0) (x y : Fin n → Rat) :
    cov n x y = (sumF n (fun i => x i * y i) - sumF n x * sumF n y / n) / n := by
  unfold cov mean
  generalize ha : sumF n x / n = a
  generalize hb : sumF n y / n = b
  have hpt : ∀ i, (x i - a) * (y i - b) = x i * y i + ((-b) * x i + ((-a) * y i + a * b)) := by
    intro i; grind
  rw [sumF_congr n _ _ hpt, sumF_add, sumF_add, sumF_add, sumF_mul_left, sumF_mul_left, sumF_const]
  have hx : sumF n x = a * n := by rw [← ha]; grind
  have hy : sumF n y = b * n := by rw [← hb]; grind
  rw [hx, hy]; grind

/-- Adding a constant deletes it from the covariance. Pearson does not see a global first moment. -/
theorem cov_add_const (n : Nat) (hn : (n : Rat) ≠ 0) (x y : Fin n → Rat) (b : Rat) :
    cov n (fun i => x i + b) y = cov n x y := by
  rw [cov_raw n hn, cov_raw n hn]
  have h1 : sumF n (fun i => (x i + b) * y i) = sumF n (fun i => x i * y i) + b * sumF n y := by
    have hpt : ∀ i, (x i + b) * y i = x i * y i + b * y i := by intro i; grind
    rw [sumF_congr n _ _ hpt, sumF_add, sumF_mul_left]
  have h2 : sumF n (fun i => x i + b) = sumF n x + (n : Rat) * b := by
    rw [sumF_add, sumF_const]
  rw [h1, h2]; grind

/-- The same constant shift leaves the variance unchanged, so a correlation that was defined stays defined. -/
theorem var_add_const (n : Nat) (hn : (n : Rat) ≠ 0) (x : Fin n → Rat) (b : Rat) :
    cov n (fun i => x i + b) (fun i => x i + b) = cov n x x := by
  rw [cov_raw n hn, cov_raw n hn]
  have h1 : sumF n (fun i => (x i + b) * (x i + b))
      = sumF n (fun i => x i * x i) + (2 * b) * sumF n x + (n : Rat) * (b * b) := by
    have hpt : ∀ i, (x i + b) * (x i + b) = x i * x i + ((2 * b) * x i + b * b) := by
      intro i; grind
    rw [sumF_congr n _ _ hpt, sumF_add, sumF_add, sumF_mul_left, sumF_const]; grind
  have h2 : sumF n (fun i => x i + b) = sumF n x + (n : Rat) * b := by
    rw [sumF_add, sumF_const]
  rw [h1, h2]; grind

theorem cov_scale (n : Nat) (hn : (n : Rat) ≠ 0) (x y : Fin n → Rat) (a : Rat) :
    cov n (fun i => a * x i) y = a * cov n x y := by
  rw [cov_raw n hn, cov_raw n hn]
  have h1 : sumF n (fun i => (a * x i) * y i) = a * sumF n (fun i => x i * y i) := by
    have hpt : ∀ i, (a * x i) * y i = a * (x i * y i) := by intro i; grind
    rw [sumF_congr n _ _ hpt, sumF_mul_left]
  have h2 : sumF n (fun i => a * x i) = a * sumF n x := sumF_mul_left n a x
  rw [h1, h2]; grind

theorem var_scale (n : Nat) (hn : (n : Rat) ≠ 0) (x : Fin n → Rat) (a : Rat) :
    cov n (fun i => a * x i) (fun i => a * x i) = (a * a) * cov n x x := by
  rw [cov_raw n hn, cov_raw n hn]
  have h1 : sumF n (fun i => (a * x i) * (a * x i)) = (a * a) * sumF n (fun i => x i * x i) := by
    have hpt : ∀ i, (a * x i) * (a * x i) = (a * a) * (x i * x i) := by intro i; grind
    rw [sumF_congr n _ _ hpt, sumF_mul_left]
  have h2 : sumF n (fun i => a * x i) = a * sumF n x := sumF_mul_left n a x
  rw [h1, h2]; grind

/-- Squared correlation is unchanged by a nonzero rescaling and by a shift.
    The signed correlation flips when `a < 0`. The absolute correlation does not. -/
theorem sqCorr_affine (n : Nat) (hn : (n : Rat) ≠ 0) (x y : Fin n → Rat) (a b : Rat) (ha : a ≠ 0)
    (hx : cov n x x ≠ 0) (hy : cov n y y ≠ 0) :
    (cov n (fun i => a * x i + b) y * cov n (fun i => a * x i + b) y)
        / (cov n (fun i => a * x i + b) (fun i => a * x i + b) * cov n y y)
      = (cov n x y * cov n x y) / (cov n x x * cov n y y) := by
  have hshift : cov n (fun i => a * x i + b) y = cov n (fun i => a * x i) y :=
    cov_add_const n hn (fun i => a * x i) y b
  have hvar : cov n (fun i => a * x i + b) (fun i => a * x i + b)
      = cov n (fun i => a * x i) (fun i => a * x i) :=
    var_add_const n hn (fun i => a * x i) b
  rw [hshift, hvar, cov_scale n hn x y a, var_scale n hn x a]
  have haa : a * a ≠ 0 := by
    intro h
    cases Rat.mul_eq_zero.mp h with
    | inl h0 => exact ha h0
    | inr h0 => exact ha h0
  grind

theorem sign_flip_cov (n : Nat) (hn : (n : Rat) ≠ 0) (x y : Fin n → Rat) :
    cov n (fun i => (-1) * x i) y = (-1) * cov n x y :=
  cov_scale n hn x y (-1)

/-! ## Conditional independence: the cross moment is the product of the conditional means -/

/-- Under `X ⊥ Y | weights`, the joint cross moment equals `(∑ px x) (∑ py y)`. -/
theorem expect_indep_product (n m : Nat) (px : Fin n → Rat) (py : Fin m → Rat)
    (xv : Fin n → Rat) (yv : Fin m → Rat) :
    sumF n (fun i => sumF m (fun j => (px i * py j) * (xv i * yv j)))
      = (sumF n (fun i => px i * xv i)) * (sumF m (fun j => py j * yv j)) := by
  have hinner : ∀ i, sumF m (fun j => (px i * py j) * (xv i * yv j))
      = (px i * xv i) * sumF m (fun j => py j * yv j) := by
    intro i
    have hpt : ∀ j, (px i * py j) * (xv i * yv j) = (px i * xv i) * (py j * yv j) := by
      intro j; grind
    rw [sumF_congr m _ _ hpt, sumF_mul_left]
  have houter : ∀ i, (px i * xv i) * sumF m (fun j => py j * yv j)
      = (sumF m (fun j => py j * yv j)) * (px i * xv i) := by
    intro i; grind
  rw [sumF_congr n _ _ hinner, sumF_congr n _ _ houter, sumF_mul_left]; grind

/-- Mixing conditionally independent prompts replaces the joint cross moment by the product of the conditional means. -/
theorem mixture_cross_moment (C n m : Nat) (pc : Fin C → Rat)
    (px : Fin C → Fin n → Rat) (py : Fin C → Fin m → Rat)
    (xv : Fin n → Rat) (yv : Fin m → Rat) :
    sumF C (fun c => pc c * sumF n (fun i => sumF m (fun j => (px c i * py c j) * (xv i * yv j))))
      = sumF C (fun c => pc c *
          ((sumF n (fun i => px c i * xv i)) * (sumF m (fun j => py c j * yv j)))) := by
  apply sumF_congr
  intro c
  rw [expect_indep_product]

theorem marginal_of_independent_y (n m : Nat) (px : Fin n → Rat) (py : Fin m → Rat)
    (xv : Fin n → Rat) (hpy : sumF m py = 1) :
    sumF n (fun i => sumF m (fun j => (px i * py j) * xv i))
      = sumF n (fun i => px i * xv i) := by
  have hinner : ∀ i, sumF m (fun j => (px i * py j) * xv i) = px i * xv i := by
    intro i
    have hpt : ∀ j, (px i * py j) * xv i = (px i * xv i) * py j := by intro j; grind
    rw [sumF_congr m _ _ hpt, sumF_mul_left, hpy]; grind
  exact sumF_congr n _ _ hinner

/-- **Law of total covariance**, finite mixture, conditional independence.
The numerator of Pearson correlation is the covariance of the conditional means.
Nothing inside a single prompt survives, because that within-prompt cross moment was replaced by `μ_X(c) μ_Y(c)`. -/
theorem law_total_covariance (C n m : Nat) (pc : Fin C → Rat)
    (px : Fin C → Fin n → Rat) (py : Fin C → Fin m → Rat)
    (xv : Fin n → Rat) (yv : Fin m → Rat) (hpy : ∀ c, sumF m (py c) = 1) (hpx : ∀ c, sumF n (px c) = 1) :
    let muX : Fin C → Rat := fun c => sumF n (fun i => px c i * xv i)
    let muY : Fin C → Rat := fun c => sumF m (fun j => py c j * yv j)
    let cross := sumF C (fun c => pc c * sumF n (fun i => sumF m (fun j =>
        (px c i * py c j) * (xv i * yv j))))
    let ex := sumF C (fun c => pc c * sumF n (fun i => sumF m (fun j => (px c i * py c j) * xv i)))
    let ey := sumF C (fun c => pc c * sumF n (fun i => sumF m (fun j => (px c i * py c j) * yv j)))
    cross - ex * ey
      = sumF C (fun c => pc c * (muX c * muY c))
        - (sumF C (fun c => pc c * muX c)) * (sumF C (fun c => pc c * muY c)) := by
  intro muX muY cross ex ey
  have hcross : cross = sumF C (fun c => pc c * (muX c * muY c)) := by
    simpa [cross, muX, muY] using mixture_cross_moment C n m pc px py xv yv
  have hex : ex = sumF C (fun c => pc c * muX c) := by
    have hpt : ∀ c, sumF n (fun i => sumF m (fun j => (px c i * py c j) * xv i)) = muX c := by
      intro c
      simpa [muX] using marginal_of_independent_y n m (px c) (py c) xv (hpy c)
    simp only [ex, muX]
    apply sumF_congr
    intro c
    rw [hpt c]
  have hey : ey = sumF C (fun c => pc c * muY c) := by
    have hpt : ∀ c, sumF m (fun j => sumF n (fun i => (py c j * px c i) * yv j)) = muY c := by
      intro c
      simpa [muY] using marginal_of_independent_y m n (py c) (px c) yv (hpx c)
    -- The joint above is written with `px` outermost. Swap the finite sums.
    have hswap : ∀ c, sumF n (fun i => sumF m (fun j => (px c i * py c j) * yv j))
        = sumF m (fun j => sumF n (fun i => (py c j * px c i) * yv j)) := by
      intro c
      -- both sides equal muY after the product theorem, so they equal each other
      have hL : sumF n (fun i => sumF m (fun j => (px c i * py c j) * yv j))
          = (sumF n (fun i => px c i * 1)) * (sumF m (fun j => py c j * yv j)) := by
        have hspec := expect_indep_product n m (px c) (py c) (fun _ => (1 : Rat)) yv
        simpa using hspec
      have hR : sumF m (fun j => sumF n (fun i => (py c j * px c i) * yv j))
          = muY c := hpt c
      have hL2 : sumF n (fun i => sumF m (fun j => (px c i * py c j) * yv j)) = muY c := by
        rw [hL]
        have hone : sumF n (fun i => px c i * 1) = sumF n (px c) := by
          apply sumF_congr; intro i; grind
        rw [hone, hpx c]; simp [muY]
      rw [hL2, hR]
    simp only [ey]
    apply sumF_congr
    intro c
    rw [hswap c, hpt c]
  rw [hcross, hex, hey]

/-- **Law of total variance.** `σ_X² = Var_c(μ_X) + E_c[Var(X | c)]`.
Conditional spread is not in the covariance numerator. It is added in the denominator. -/
theorem law_total_variance (C : Nat) (pc second mu : Fin C → Rat) :
    sumF C (fun c => pc c * second c) - (sumF C (fun c => pc c * mu c)) * (sumF C (fun c => pc c * mu c))
      = sumF C (fun c => pc c * (second c - mu c * mu c))
        + (sumF C (fun c => pc c * (mu c * mu c))
            - (sumF C (fun c => pc c * mu c)) * (sumF C (fun c => pc c * mu c))) := by
  have hsplit : ∀ c, pc c * second c = pc c * (second c - mu c * mu c) + pc c * (mu c * mu c) := by
    intro c; grind
  have hsum : sumF C (fun c => pc c * second c)
      = sumF C (fun c => pc c * (second c - mu c * mu c))
        + sumF C (fun c => pc c * (mu c * mu c)) := by
    have hpt : ∀ c, pc c * second c
        = pc c * (second c - mu c * mu c) + pc c * (mu c * mu c) := hsplit
    rw [sumF_congr C _ _ hpt, sumF_add]
  rw [hsum]; grind

theorem rat_mul_lt_left (a b c : Rat) (ha : 0 < a) (hbc : b < c) : a * b < a * c := by
  have hpos : 0 < c - b := by grind
  have hmul : 0 < a * (c - b) := Rat.mul_pos ha hpos
  grind

theorem div_neg (n d : Rat) (hn : n < 0) (hd : 0 < d) : n / d < 0 := by
  cases Classical.em (n / d < 0) with
  | inl h => exact h
  | inr hnot =>
    have hd0 : 0 ≤ d := by grind
    cases Rat.le_total (a := n / d) (b := 0) with
    | inl hle =>
      have heq : (n / d) * d = n := by grind
      grind
    | inr hge =>
      have hmul : 0 ≤ (n / d) * d := Rat.mul_nonneg hge hd0
      have heq : (n / d) * d = n := by grind
      grind

theorem div_lt_one_of_lt (a b : Rat) (hb : 0 < b) (h : a < b) : a / b < 1 := by
  have hneg : a - b < 0 := by grind
  have hdiv : (a - b) / b < 0 := div_neg (a - b) b hneg hb
  grind

theorem div_pos_of_pos (a b : Rat) (ha : 0 < a) (hb : 0 < b) : 0 < a / b := by
  have hneg : -a < 0 := by grind
  have hdiv : (-a) / b < 0 := div_neg (-a) b hneg hb
  grind

/-- Squared correlation with conditional spread equals the collapsed squared correlation
times `Var(μ) / (Var(μ) + E[σ²])`, which is strictly less than 1. -/
theorem spread_sqcorr_scales (cNum vMu eVar vY : Rat)
    (h1 : (vMu + eVar) * vY ≠ 0) (h2 : vMu * vY ≠ 0) (h3 : vMu + eVar ≠ 0) :
    cNum / ((vMu + eVar) * vY) = (cNum / (vMu * vY)) * (vMu / (vMu + eVar)) := by
  grind

theorem spread_factor_lt_one (vMu eVar : Rat) (hm : 0 < vMu) (he : 0 < eVar) :
    vMu / (vMu + eVar) < 1 := by
  have hlt : vMu < vMu + eVar := by grind
  have hpos : 0 < vMu + eVar := by grind
  exact div_lt_one_of_lt vMu (vMu + eVar) hpos hlt

/-- A larger positive conditional-variance term strictly shrinks a positive squared correlation. -/
theorem spread_shrinks_sqcorr (cNum vMu eVar vY : Rat)
    (hc : 0 < cNum) (hm : 0 < vMu) (he : 0 < eVar) (hy : 0 < vY) :
    cNum / ((vMu + eVar) * vY) < cNum / (vMu * vY) := by
  have hsmall : 0 < vMu * vY := Rat.mul_pos hm hy
  have hsum : 0 < vMu + eVar := by grind
  have hbig : 0 < (vMu + eVar) * vY := Rat.mul_pos hsum hy
  have h1 : (vMu + eVar) * vY ≠ 0 := by intro h; grind
  have h2 : vMu * vY ≠ 0 := by intro h; grind
  have h3 : vMu + eVar ≠ 0 := by intro h; grind
  have hid := spread_sqcorr_scales cNum vMu eVar vY h1 h2 h3
  have hfac : vMu / (vMu + eVar) < 1 := spread_factor_lt_one vMu eVar hm he
  have hfacpos : 0 < vMu / (vMu + eVar) := div_pos_of_pos vMu (vMu + eVar) hm hsum
  have hbase : 0 < cNum / (vMu * vY) := div_pos_of_pos cNum (vMu * vY) hc hsmall
  have hmul : (cNum / (vMu * vY)) * (vMu / (vMu + eVar)) < (cNum / (vMu * vY)) * 1 :=
    rat_mul_lt_left (cNum / (vMu * vY)) (vMu / (vMu + eVar)) 1 hbase hfac
  have hmul' : (cNum / (vMu * vY)) * (vMu / (vMu + eVar)) < cNum / (vMu * vY) := by
    simpa using hmul
  rw [← hid] at hmul'
  exact hmul'

/-! ## Quadratic witness: deterministic and uncorrelated -/

def xQuad : Fin 3 → Rat
  | 0 => -1
  | 1 => 0
  | 2 => 1

def yQuad : Fin 3 → Rat
  | 0 => 1
  | 1 => 0
  | 2 => 1

theorem yQuad_eq_sq : yQuad 0 = xQuad 0 * xQuad 0 ∧ yQuad 1 = xQuad 1 * xQuad 1 ∧ yQuad 2 = xQuad 2 * xQuad 2 := by
  simp [xQuad, yQuad]; grind

theorem quadratic_cov_zero :
    sumF 3 (fun i => xQuad i * yQuad i) / 3
      - (sumF 3 xQuad / 3) * (sumF 3 yQuad / 3) = 0 := by
  simp [sumF, xQuad, yQuad]; grind

theorem quadratic_vars_positive :
    sumF 3 (fun i => xQuad i * xQuad i) / 3 - (sumF 3 xQuad / 3) * (sumF 3 xQuad / 3) = (2 : Rat) / 3 ∧
    sumF 3 (fun i => yQuad i * yQuad i) / 3 - (sumF 3 yQuad / 3) * (sumF 3 yQuad / 3) = (2 : Rat) / 9 := by
  simp [sumF, xQuad, yQuad]; grind

/-- The joint is not the product of the marginals: `P(X = 0) P(Y = 1) = 2/9`, but that atom has mass 0. -/
theorem quadratic_not_independent :
    ((1 : Rat) / 3) * (2 / 3) = 2 / 9 ∧ (2 : Rat) / 9 ≠ 0 := by
  grind

theorem quadratic_shift_still_zero (b : Rat) :
    cov 3 (fun i => xQuad i + b) yQuad = cov 3 xQuad yQuad := by
  have hn : ((3 : Rat) ≠ 0) := by grind
  exact cov_add_const 3 hn xQuad yQuad b

theorem quadratic_sample_cov_matches_zero :
    cov 3 xQuad yQuad = 0 := by
  have hn : ((3 : Rat) ≠ 0) := by grind
  rw [cov_raw 3 hn]
  simp [sumF, xQuad, yQuad]; grind

/-! ## Concrete two-prompt instance of the total-covariance identity -/

def pLo : Fin 2 → Rat
  | 0 => 3 / 4
  | 1 => 1 / 4

def pHi : Fin 2 → Rat
  | 0 => 1 / 4
  | 1 => 3 / 4

def bit : Fin 2 → Rat
  | 0 => 0
  | 1 => 1

theorem instance_low_cross :
    sumF 2 (fun i => sumF 2 (fun j => (pLo i * pLo j) * (bit i * bit j))) = (1 : Rat) / 16 := by
  simp [sumF, pLo, bit]; grind

theorem instance_low_mean :
    sumF 2 (fun i => pLo i * bit i) = (1 : Rat) / 4 := by
  simp [sumF, pLo, bit]; grind

theorem instance_high_mean :
    sumF 2 (fun i => pHi i * bit i) = (3 : Rat) / 4 := by
  simp [sumF, pHi, bit]; grind

/-- Equal mixture of the two prompts. Covariance of the conditionally independent copies is `1/16`,
    which is also the covariance of the conditional means `1/4` and `3/4`. -/
theorem instance_total_cov :
    ((1 : Rat) / 2) * (1 / 16) + (1 / 2) * ((3 / 4) * (3 / 4))
      - ((1 / 2) * (1 / 4) + (1 / 2) * (3 / 4)) * ((1 / 2) * (1 / 4) + (1 / 2) * (3 / 4))
      = (1 : Rat) / 16 := by
  grind

theorem instance_variance_split :
    let eVar : Rat := (1 / 2) * ((1 / 4) * (3 / 4)) + (1 / 2) * ((3 / 4) * (1 / 4))
    let varMu : Rat := (1 / 2) * ((1 / 4) * (1 / 4)) + (1 / 2) * ((3 / 4) * (3 / 4)) - ((1 : Rat) / 2) * (1 / 2)
    eVar = 3 / 16 ∧ varMu = 1 / 16 ∧ eVar + varMu = 1 / 4 := by
  grind

/-! ## CCA: invertible affine reparameterization of one view -/

structure Vec2 where
  x : Rat
  y : Rat

def vdot (u v : Vec2) : Rat := u.x * v.x + u.y * v.y

def vadd (u v : Vec2) : Vec2 := ⟨u.x + v.x, u.y + v.y⟩

/-- Matrix `[[a, b], [c, d]]` applied on the left to a column. -/
def mapply (a b c d : Rat) (v : Vec2) : Vec2 :=
  ⟨a * v.x + b * v.y, c * v.x + d * v.y⟩

/-- `Aᵀ`. -/
def mapplyT (a b c d : Rat) (u : Vec2) : Vec2 :=
  ⟨a * u.x + c * u.y, b * u.x + d * u.y⟩

theorem dot_mapply (a b c d : Rat) (u v : Vec2) :
    vdot u (mapply a b c d v) = vdot (mapplyT a b c d u) v := by
  unfold vdot mapply mapplyT
  grind

theorem dot_vadd (u v w : Vec2) : vdot u (vadd v w) = vdot u v + vdot u w := by
  unfold vdot vadd
  grind

theorem dot_affine (a b c d : Rat) (u v shift : Vec2) :
    vdot u (vadd (mapply a b c d v) shift) = vdot (mapplyT a b c d u) v + vdot u shift := by
  rw [dot_vadd, dot_mapply]

theorem cov_proj_affine (n : Nat) (hn : (n : Rat) ≠ 0) (a b c d : Rat) (u shift : Vec2)
    (xs : Fin n → Vec2) (y : Fin n → Rat) :
    cov n (fun i => vdot u (vadd (mapply a b c d (xs i)) shift)) y
      = cov n (fun i => vdot (mapplyT a b c d u) (xs i)) y := by
  have hpt : ∀ i, vdot u (vadd (mapply a b c d (xs i)) shift)
      = vdot (mapplyT a b c d u) (xs i) + vdot u shift := by
    intro i; exact dot_affine a b c d u (xs i) shift
  have hfun : (fun i => vdot u (vadd (mapply a b c d (xs i)) shift))
      = fun i => vdot (mapplyT a b c d u) (xs i) + vdot u shift := by
    funext i; exact hpt i
  rw [hfun]
  exact cov_add_const n hn (fun i => vdot (mapplyT a b c d u) (xs i)) y (vdot u shift)

theorem var_proj_affine (n : Nat) (hn : (n : Rat) ≠ 0) (a b c d : Rat) (u shift : Vec2)
    (xs : Fin n → Vec2) :
    cov n (fun i => vdot u (vadd (mapply a b c d (xs i)) shift))
        (fun i => vdot u (vadd (mapply a b c d (xs i)) shift))
      = cov n (fun i => vdot (mapplyT a b c d u) (xs i))
          (fun i => vdot (mapplyT a b c d u) (xs i)) := by
  have hpt : ∀ i, vdot u (vadd (mapply a b c d (xs i)) shift)
      = vdot (mapplyT a b c d u) (xs i) + vdot u shift := by
    intro i; exact dot_affine a b c d u (xs i) shift
  have hfun : (fun i => vdot u (vadd (mapply a b c d (xs i)) shift))
      = fun i => vdot (mapplyT a b c d u) (xs i) + vdot u shift := by
    funext i; exact hpt i
  rw [hfun]
  exact var_add_const n hn (fun i => vdot (mapplyT a b c d u) (xs i)) (vdot u shift)

/-- Every coefficient `u'` of the original view is `Aᵀ u` for an explicit `u`, so the sets of attainable projections match. -/
theorem shear_preimage (u' : Vec2) :
    mapplyT 1 1 0 1 ⟨u'.x, u'.y - u'.x⟩ = u' := by
  cases u' with
  | mk ux uy =>
    unfold mapplyT
    grind

theorem sign_flip_preimage (u' : Vec2) :
    mapplyT (-1) 0 0 (-1) ⟨-u'.x, -u'.y⟩ = u' := by
  cases u' with
  | mk ux uy =>
    unfold mapplyT
    grind

theorem swap_preimage (u' : Vec2) :
    mapplyT 0 1 1 0 ⟨u'.y, u'.x⟩ = u' := by
  cases u' with
  | mk ux uy =>
    unfold mapplyT
    grind

theorem mix_preimage (u' : Vec2) :
    mapplyT 2 1 1 2 ⟨(2 * u'.x - u'.y) / 3, (-u'.x + 2 * u'.y) / 3⟩ = u' := by
  cases u' with
  | mk ux uy =>
    unfold mapplyT
    grind

theorem shear_det : (1 : Rat) * 1 - 1 * 0 = 1 := by grind
theorem swap_det : (0 : Rat) * 0 - 1 * 1 = -1 := by grind
theorem mix_det : (2 : Rat) * 2 - 1 * 1 = 3 := by grind
theorem sign_det : (-1 : Rat) * (-1) - 0 * 0 = 1 := by grind

/-! ## CCA anti-properness on a diagonal two-aspect law -/

/-- Squared canonical correlations of the spread law, from the diagonal product
`Σxx⁻¹ Σxy Σyy⁻¹ Σyx` with `Σxx = Σyy = diag(1/4, 1/4)` and `Σxy = diag(1/16, 1/36)`. -/
theorem cca_sq_spread :
    ((4 : Rat) * (1 / 16)) * (4 * (1 / 16)) = 1 / 16 ∧
    ((4 : Rat) * (1 / 36)) * (4 * (1 / 36)) = 1 / 81 := by
  grind

/-- Same product after replacing `Σxx` by `diag(Var μ) = diag(1/16, 1/36)`. Both squared canonical correlations jump. -/
theorem cca_sq_collapsed :
    ((16 : Rat) * (1 / 16)) * (4 * (1 / 16)) = 1 / 4 ∧
    ((36 : Rat) * (1 / 36)) * (4 * (1 / 36)) = 1 / 9 := by
  grind

theorem cca_sq_strictly_increase :
    (1 : Rat) / 16 < 1 / 4 ∧ (1 : Rat) / 81 < 1 / 9 := by
  grind

/-- Binary Brier of the true conditional probability, averaged over the two prompts. -/
theorem brier_truth_aspect1 : (2 : Rat) * ((1 / 4) * (3 / 4)) = 3 / 8 := by grind
theorem brier_truth_aspect2 : (2 : Rat) * ((1 / 3) * (2 / 3)) = 4 / 9 := by grind

/-- Point mass at the nearer class. Aspect 1 predicts 0 on the low prompt and 1 on the high prompt. -/
theorem brier_point_aspect1 : (1 : Rat) / 2 > 3 / 8 := by grind
theorem brier_point_aspect2 : (2 : Rat) / 3 > 4 / 9 := by grind

/-- One-threshold ranked probability score. Truth averages `p(1-p)`. The point mass averages the missed-class probability. -/
theorem rps_truth_aspect1 : (1 : Rat) / 4 * (3 / 4) = 3 / 16 := by grind
theorem rps_point_aspect1 : (1 : Rat) / 4 > 3 / 16 := by grind
theorem rps_truth_aspect2 : (1 : Rat) / 3 * (2 / 3) = 2 / 9 := by grind
theorem rps_point_aspect2 : (1 : Rat) / 3 > 2 / 9 := by grind

def admitsFiniteCE (p q : Rat) : Prop := p = 0 ∨ 0 < q

theorem truth_ce_finite_aspect1 : admitsFiniteCE ((1 : Rat) / 4) (1 / 4) := by
  unfold admitsFiniteCE
  grind

theorem point_ce_not_finite_aspect1 : ¬ admitsFiniteCE ((1 : Rat) / 4) 0 := by
  intro h
  unfold admitsFiniteCE at h
  cases h with
  | inl h => grind
  | inr h => grind

theorem truth_ce_finite_aspect2 : admitsFiniteCE ((1 : Rat) / 3) (1 / 3) := by
  unfold admitsFiniteCE
  grind

theorem point_ce_not_finite_aspect2 : ¬ admitsFiniteCE ((1 : Rat) / 3) 0 := by
  intro h
  unfold admitsFiniteCE at h
  cases h with
  | inl h => grind
  | inr h => grind

/-! The population denominator identity used by `spread_shrinks_sqcorr`, on the first aspect:
`cov² = (1/16)²` is not needed separately; the variance split is `3/16 + 1/16 = 1/4`. -/
theorem aspect1_sqcorr_values :
    (((1 : Rat) / 16) * (1 / 16)) / ((1 / 4) * (1 / 4)) = 1 / 16 ∧
    (((1 : Rat) / 16) * (1 / 16)) / ((1 / 16) * (1 / 4)) = 1 / 4 := by
  grind

theorem aspect2_sqcorr_values :
    (((1 : Rat) / 36) * (1 / 36)) / ((1 / 4) * (1 / 4)) = 1 / 81 ∧
    (((1 : Rat) / 36) * (1 / 36)) / ((1 / 36) * (1 / 4)) = 1 / 9 := by
  grind

#print axioms expect_indep_product
#print axioms mixture_cross_moment
#print axioms law_total_covariance
#print axioms law_total_variance
#print axioms spread_shrinks_sqcorr
#print axioms cov_add_const
#print axioms var_add_const
#print axioms sqCorr_affine
#print axioms sign_flip_cov
#print axioms quadratic_cov_zero
#print axioms quadratic_vars_positive
#print axioms quadratic_not_independent
#print axioms quadratic_shift_still_zero
#print axioms quadratic_sample_cov_matches_zero
#print axioms yQuad_eq_sq
#print axioms instance_total_cov
#print axioms instance_variance_split
#print axioms dot_affine
#print axioms cov_proj_affine
#print axioms var_proj_affine
#print axioms shear_preimage
#print axioms sign_flip_preimage
#print axioms swap_preimage
#print axioms mix_preimage
#print axioms cca_sq_spread
#print axioms cca_sq_collapsed
#print axioms cca_sq_strictly_increase
#print axioms brier_point_aspect1
#print axioms brier_point_aspect2
#print axioms rps_point_aspect1
#print axioms rps_point_aspect2
#print axioms point_ce_not_finite_aspect1
#print axioms point_ce_not_finite_aspect2
#print axioms aspect1_sqcorr_values
#print axioms aspect2_sqcorr_values
