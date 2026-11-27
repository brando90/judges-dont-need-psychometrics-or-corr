set_option linter.unusedSimpArgs false
/-
  BrierCorr.lean — core Lean 4 only (no Mathlib), over `Rat`, zero `sorry`, zero custom axioms.

  1. Brier score is strictly proper (exact decomposition, K classes, `Fin K`).
  2. Scalar L2 loss: E[(Y-c)^2] = Var(Y) + (mu-c)^2, minimized iff c = mu  (no Gaussianity used).
     + binary Brier = 2 * scalar L2.
  3. Correlation facts:
     (a) standardized vectors: (1/n) sum (x-y)^2 = 2 (1 - (1/n) sum x y)
     (b) unstandardized:  MSE = (mu_x-mu_y)^2 + (s_x-s_y)^2 + 2 s_x s_y (1 - rho)
     (c) Pearson rho is invariant under x ↦ a x + b, a > 0.
  Since core Lean has no `sqrt` on `Rat`, standard deviations are carried as
  positive rationals `sx` with `sx*sx = var x` (any positive square root).
-/

/-! ## Finite sums over `Fin n` -/

def sumF : (n : Nat) → (Fin n → Rat) → Rat
  | 0, _ => 0
  | n+1, f => f 0 + sumF n (fun i => f i.succ)

theorem sumF_add (n : Nat) (f g : Fin n → Rat) :
    sumF n (fun i => f i + g i) = sumF n f + sumF n g := by
  induction n with
  | zero => simp only [sumF]; grind
  | succ n ih => simp only [sumF]; rw [ih (fun i => f i.succ) (fun i => g i.succ)]; grind

theorem sumF_mul_left (n : Nat) (c : Rat) (f : Fin n → Rat) :
    sumF n (fun i => c * f i) = c * sumF n f := by
  induction n with
  | zero => simp only [sumF]; grind
  | succ n ih => simp only [sumF]; rw [ih (fun i => f i.succ)]; grind

theorem sumF_const (n : Nat) (c : Rat) : sumF n (fun _ => c) = (n : Rat) * c := by
  induction n with
  | zero => simp only [sumF]; grind
  | succ n ih => simp only [sumF]; rw [ih]; grind

theorem sumF_congr (n : Nat) (f g : Fin n → Rat) (h : ∀ i, f i = g i) :
    sumF n f = sumF n g := by
  have : f = g := funext h
  rw [this]

/-- Kronecker delta picks out one term. -/
theorem sumF_delta (n : Nat) (i : Fin n) (f : Fin n → Rat) :
    sumF n (fun j => (if i = j then (1 : Rat) else 0) * f j) = f i := by
  induction n with
  | zero => exact i.elim0
  | succ n ih =>
    simp only [sumF]
    cases i using Fin.cases with
    | zero =>
      have h0 : sumF n (fun j => (if (0 : Fin (n+1)) = j.succ then (1 : Rat) else 0) * f j.succ)
          = sumF n (fun _ => 0) := by
        apply sumF_congr; intro j
        have : (0 : Fin (n+1)) ≠ j.succ := fun h => Fin.succ_ne_zero j h.symm
        simp [this]
      rw [h0, sumF_const]; grind
    | succ i' =>
      have h1 : sumF n (fun j => (if i'.succ = j.succ then (1 : Rat) else 0) * f j.succ)
          = sumF n (fun j => (if i' = j then (1 : Rat) else 0) * f j.succ) := by
        apply sumF_congr; intro j; simp [Fin.succ_inj]
      rw [h1, ih i' (fun j => f j.succ)]
      have : i'.succ ≠ (0 : Fin (n+1)) := Fin.succ_ne_zero i'
      simp [this]; grind

/-! ## Elementary facts about squares in `Rat` -/

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

theorem sumF_nonneg (n : Nat) (f : Fin n → Rat) (h : ∀ i, 0 ≤ f i) : 0 ≤ sumF n f := by
  induction n with
  | zero => simp only [sumF]; grind
  | succ n ih =>
    simp only [sumF]
    have := ih (fun i => f i.succ) (fun i => h i.succ)
    have := h 0
    grind

theorem sumF_eq_zero_iff (n : Nat) (f : Fin n → Rat) (h : ∀ i, 0 ≤ f i) :
    sumF n f = 0 ↔ ∀ i, f i = 0 := by
  induction n with
  | zero => simp [sumF]
  | succ n ih =>
    simp only [sumF]
    have hrest := sumF_nonneg n (fun i => f i.succ) (fun i => h i.succ)
    have h0 := h 0
    have ih' := ih (fun i => f i.succ) (fun i => h i.succ)
    constructor
    · intro hs
      have hf0 : f 0 = 0 := by grind
      have hr : sumF n (fun i => f i.succ) = 0 := by grind
      have hall := ih'.mp hr
      intro i
      cases i using Fin.cases with
      | zero => exact hf0
      | succ j => exact hall j
    · intro hall
      have hr : sumF n (fun i => f i.succ) = 0 := ih'.mpr (fun i => hall i.succ)
      rw [hr, hall 0]; grind

/-! ## 1. Brier score is strictly proper -/

/-- Brier loss of forecast `q` when the realised class is `i`: ∑_j (q_j - δ_ij)^2. -/
def brier (K : Nat) (q : Fin K → Rat) (i : Fin K) : Rat :=
  sumF K (fun j => (q j - (if i = j then (1 : Rat) else 0)) * (q j - (if i = j then (1 : Rat) else 0)))

/-- Expected Brier loss of forecast `q` under true distribution `p`. -/
def expBrier (K : Nat) (p q : Fin K → Rat) : Rat :=
  sumF K (fun i => p i * brier K q i)

theorem brier_expand (K : Nat) (q : Fin K → Rat) (i : Fin K) :
    brier K q i = sumF K (fun j => q j * q j) + (1 - 2 * q i) := by
  unfold brier
  have hpt : ∀ j, (q j - (if i = j then (1 : Rat) else 0)) * (q j - (if i = j then (1 : Rat) else 0))
      = q j * q j + (if i = j then (1 : Rat) else 0) * (1 - 2 * q j) := by
    intro j; by_cases hij : i = j <;> simp [hij] <;> grind
  rw [sumF_congr K _ _ hpt, sumF_add, sumF_delta K i (fun j => 1 - 2 * q j)]

theorem expBrier_closed (K : Nat) (p q : Fin K → Rat) (hp : sumF K p = 1) :
    expBrier K p q = sumF K (fun j => q j * q j) + 1 - 2 * sumF K (fun i => p i * q i) := by
  unfold expBrier
  have hpt : ∀ i, p i * brier K q i
      = (sumF K (fun j => q j * q j) + 1) * p i + (-2) * (p i * q i) := by
    intro i; rw [brier_expand]; grind
  rw [sumF_congr K _ _ hpt, sumF_add, sumF_mul_left, sumF_mul_left, hp]; grind

/-- **Brier decomposition**: E_p[BS(q)] - E_p[BS(p)] = ‖q - p‖² (needs only ∑ p = 1). -/
theorem brier_decomposition (K : Nat) (p q : Fin K → Rat) (hp : sumF K p = 1) :
    expBrier K p q - expBrier K p p = sumF K (fun j => (q j - p j) * (q j - p j)) := by
  rw [expBrier_closed K p q hp, expBrier_closed K p p hp]
  have hpt : ∀ j, (q j - p j) * (q j - p j)
      = q j * q j + ((-2) * (p j * q j) + p j * p j) := by intro j; grind
  rw [sumF_congr K _ _ hpt, sumF_add, sumF_add, sumF_mul_left]; grind

/-- **Brier is proper**: the truth `p` is a minimiser of expected Brier loss. -/
theorem brier_proper (K : Nat) (p q : Fin K → Rat) (hp : sumF K p = 1) :
    expBrier K p p ≤ expBrier K p q := by
  have h := brier_decomposition K p q hp
  have := sumF_nonneg K (fun j => (q j - p j) * (q j - p j)) (fun j => sq_nonneg' _)
  grind

/-- **Brier is strictly proper**: equality iff `q = p` coordinatewise. -/
theorem brier_strictly_proper (K : Nat) (p q : Fin K → Rat) (hp : sumF K p = 1) :
    expBrier K p q = expBrier K p p ↔ ∀ j, q j = p j := by
  have h := brier_decomposition K p q hp
  have hz := sumF_eq_zero_iff K (fun j => (q j - p j) * (q j - p j)) (fun j => sq_nonneg' _)
  constructor
  · intro he j
    have : sumF K (fun j => (q j - p j) * (q j - p j)) = 0 := by grind
    have := sq_eq_zero' _ (hz.mp this j); grind
  · intro hq
    have : sumF K (fun j => (q j - p j) * (q j - p j)) = 0 :=
      hz.mpr (fun j => by rw [hq j]; grind)
    grind

/-! ## 2. Scalar L2 loss: minimiser is the mean, for ANY distribution -/

/-- E[(Y-c)^2] = E[(Y-mu)^2] + (mu-c)^2, with mu = ∑ p_i y_i and ∑ p_i = 1.
    No Gaussian (or any distributional) assumption. -/
theorem l2_decomposition (n : Nat) (p y : Fin n → Rat) (c : Rat) (hp : sumF n p = 1) :
    sumF n (fun i => p i * ((y i - c) * (y i - c)))
      = sumF n (fun i => p i * ((y i - sumF n (fun k => p k * y k)) * (y i - sumF n (fun k => p k * y k))))
        + (sumF n (fun k => p k * y k) - c) * (sumF n (fun k => p k * y k) - c) := by
  generalize hmu : sumF n (fun k => p k * y k) = mu
  have hpt : ∀ i, p i * ((y i - c) * (y i - c))
      = p i * ((y i - mu) * (y i - mu)) + ((2 * (mu - c)) * (p i * y i)
        + ((mu - c) * (mu - c) - 2 * (mu - c) * mu) * p i) := by intro i; grind
  rw [sumF_congr n _ _ hpt, sumF_add, sumF_add, sumF_mul_left, sumF_mul_left, hmu, hp]; grind

theorem l2_mean_minimizer (n : Nat) (p y : Fin n → Rat) (c : Rat) (hp : sumF n p = 1) :
    let mu := sumF n (fun k => p k * y k)
    sumF n (fun i => p i * ((y i - mu) * (y i - mu))) ≤ sumF n (fun i => p i * ((y i - c) * (y i - c)))
    ∧ (sumF n (fun i => p i * ((y i - c) * (y i - c)))
        = sumF n (fun i => p i * ((y i - mu) * (y i - mu))) ↔ c = mu) := by
  intro mu
  have h := l2_decomposition n p y c hp
  have hs := sq_nonneg' (mu - c)
  refine ⟨by grind, ?_⟩
  constructor
  · intro he
    have : (mu - c) * (mu - c) = 0 := by grind
    have := sq_eq_zero' _ this; grind
  · intro hc; rw [hc]

/-- Binary case: two-class Brier with forecast (q, 1-q) and outcome (y, 1-y)
    is exactly twice the scalar squared error. -/
theorem binary_brier_eq_two_l2 (q y : Rat) :
    (q - y) * (q - y) + ((1 - q) - (1 - y)) * ((1 - q) - (1 - y)) = 2 * ((q - y) * (q - y)) := by
  grind

/-! ## 3. Correlation vs MSE -/

/-- (a) Standardized vectors (∑x_i^2 = ∑y_i^2 = n):  (1/n)∑(x_i-y_i)^2 = 2(1 - (1/n)∑x_i y_i).
    (Zero mean is what makes (1/n)∑ x_i y_i = rho; it is not needed for the algebra.) -/
theorem standardized_mse_corr (n : Nat) (hn : (n : Rat) ≠ 0) (x y : Fin n → Rat)
    (hx2 : sumF n (fun i => x i * x i) = n) (hy2 : sumF n (fun i => y i * y i) = n) :
    sumF n (fun i => (x i - y i) * (x i - y i)) / n
      = 2 * (1 - sumF n (fun i => x i * y i) / n) := by
  have hpt : ∀ i, (x i - y i) * (x i - y i) = x i * x i + ((-2) * (x i * y i) + y i * y i) := by
    intro i; grind
  rw [sumF_congr n _ _ hpt, sumF_add, sumF_add, sumF_mul_left, hx2, hy2]
  grind

def mean (n : Nat) (x : Fin n → Rat) : Rat := sumF n x / n
def cov (n : Nat) (x y : Fin n → Rat) : Rat :=
  sumF n (fun i => (x i - mean n x) * (y i - mean n y)) / n

/-- covariance in raw sums: n * cov = ∑xy - (∑x)(∑y)/n -/
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

/-- (b) Unstandardized MSE decomposition. `sx, sy` are any positive square roots of the
    (population) variances and `rho = cov/(sx*sy)`. -/
theorem mse_decomposition (n : Nat) (hn : (n : Rat) ≠ 0) (x y : Fin n → Rat) (sx sy : Rat)
    (hsx : 0 < sx) (hsy : 0 < sy) (hvx : sx * sx = cov n x x) (hvy : sy * sy = cov n y y) :
    sumF n (fun i => (x i - y i) * (x i - y i)) / n
      = (mean n x - mean n y) * (mean n x - mean n y) + (sx - sy) * (sx - sy)
        + 2 * sx * sy * (1 - cov n x y / (sx * sy)) := by
  rw [cov_raw n hn] at hvx hvy
  rw [cov_raw n hn]
  unfold mean
  have hpt : ∀ i, (x i - y i) * (x i - y i) = x i * x i + ((-2) * (x i * y i) + y i * y i) := by
    intro i; grind
  rw [sumF_congr n _ _ hpt, sumF_add, sumF_add, sumF_mul_left]
  have hsxy : sx * sy ≠ 0 := by
    have := Rat.mul_pos hsx hsy; grind
  grind

/-- the positive square root is unique -/
theorem pos_sqrt_unique (s t : Rat) (hs : 0 < s) (ht : 0 < t) (h : s * s = t * t) : s = t := by
  have h1 : (s - t) * (s + t) = 0 := by grind
  cases Rat.mul_eq_zero.mp h1 with
  | inl h => grind
  | inr h => grind

/-- (c) **Affine invariance of Pearson correlation**: for `a > 0`,
    rho(a x + b, y) = rho(x, y), where `sx'` is the (unique positive) std of `a x + b`. -/
theorem pearson_affine_invariant (n : Nat) (hn : (n : Rat) ≠ 0) (x y : Fin n → Rat)
    (a b : Rat) (ha : 0 < a) (sx sy sx' : Rat)
    (hsx : 0 < sx) (hsy : 0 < sy) (hsx' : 0 < sx')
    (hvx : sx * sx = cov n x x) (hvx' : sx' * sx' = cov n (fun i => a * x i + b) (fun i => a * x i + b)) :
    cov n (fun i => a * x i + b) y / (sx' * sy) = cov n x y / (sx * sy) := by
  -- covariance scales by a, variance by a^2
  have hcov : cov n (fun i => a * x i + b) y = a * cov n x y := by
    rw [cov_raw n hn, cov_raw n hn]
    have h1 : sumF n (fun i => (a * x i + b) * y i) = a * sumF n (fun i => x i * y i) + b * sumF n y := by
      have hpt : ∀ i, (a * x i + b) * y i = a * (x i * y i) + b * y i := by intro i; grind
      rw [sumF_congr n _ _ hpt, sumF_add, sumF_mul_left, sumF_mul_left]
    have h2 : sumF n (fun i => a * x i + b) = a * sumF n x + n * b := by
      rw [sumF_add, sumF_mul_left, sumF_const]
    rw [h1, h2]; grind
  have hvar : cov n (fun i => a * x i + b) (fun i => a * x i + b) = a * a * cov n x x := by
    rw [cov_raw n hn, cov_raw n hn]
    have h1 : sumF n (fun i => (a * x i + b) * (a * x i + b))
        = (a * a) * sumF n (fun i => x i * x i) + (2 * a * b) * sumF n x + n * (b * b) := by
      have hpt : ∀ i, (a * x i + b) * (a * x i + b)
          = (a * a) * (x i * x i) + ((2 * a * b) * x i + b * b) := by intro i; grind
      rw [sumF_congr n _ _ hpt, sumF_add, sumF_add, sumF_mul_left, sumF_mul_left, sumF_const]; grind
    have h2 : sumF n (fun i => a * x i + b) = a * sumF n x + n * b := by
      rw [sumF_add, sumF_mul_left, sumF_const]
    rw [h1, h2]; grind
  have hs : sx' = a * sx := by
    apply pos_sqrt_unique sx' (a * sx) hsx' (Rat.mul_pos ha hsx)
    rw [hvx', hvar, ← hvx]; grind
  rw [hcov, hs]
  have : a ≠ 0 := by grind
  have : sx ≠ 0 := by grind
  have : sy ≠ 0 := by grind
  grind


/-! ## Empirical frequencies: E[‖q_N - p‖²] = (1/N)(1 - ‖p‖²) -/

def finCons {n : Nat} {α : Type} (a : α) (f : Fin n → α) : Fin (n + 1) → α :=
  fun i => if _h : i.val = 0 then a else f ⟨i.val - 1, by omega⟩

theorem finCons_zero {n : Nat} {α : Type} (a : α) (f : Fin n → α) :
    finCons a f 0 = a := by simp [finCons]

theorem finCons_succ {n : Nat} {α : Type} (a : α) (f : Fin n → α) (i : Fin n) :
    finCons a f i.succ = f i := by
  unfold finCons
  have hne : i.succ.val ≠ 0 := by
    have : i.succ.val = i.val + 1 := Fin.val_succ i
    omega
  simp [hne]

def expN {K : Nat} : (n : Nat) → (Fin K → Rat) → ((Fin n → Fin K) → Rat) → Rat
  | 0, _, f => f (fun i => Fin.elim0 i)
  | n+1, p, f => sumF K (fun a => p a * expN n p (fun t => f (finCons a t)))

theorem expN_const {K n : Nat} (p : Fin K → Rat) (c : Rat) (hp : sumF K p = 1) :
    expN n p (fun _ => c) = c := by
  induction n with
  | zero => unfold expN; rfl
  | succ n ih =>
    unfold expN
    have hpt : ∀ a, expN n p (fun t => (fun _ => c) (finCons a t)) = c := by
      intro a
      have : (fun t : Fin n → Fin K => (fun _ => c) (finCons a t)) = fun _ => c := by
        funext t; rfl
      rw [this, ih]
    have hsum : sumF K (fun a => p a * expN n p (fun t => (fun _ => c) (finCons a t)))
        = sumF K (fun a => p a * c) := by
      apply sumF_congr
      intro a; rw [hpt a]
    rw [hsum]
    have hmul : ∀ a, p a * c = c * p a := by intro a; grind
    rw [sumF_congr K _ _ hmul, sumF_mul_left, hp]; grind

theorem expN_add {K n : Nat} (p : Fin K → Rat) (f g : (Fin n → Fin K) → Rat) :
    expN n p (fun s => f s + g s) = expN n p f + expN n p g := by
  induction n with
  | zero => unfold expN; grind
  | succ n ih =>
    unfold expN
    have hpt : ∀ a,
        expN n p (fun t => f (finCons a t) + g (finCons a t))
          = expN n p (fun t => f (finCons a t)) + expN n p (fun t => g (finCons a t)) := by
      intro a
      simpa using ih (fun t => f (finCons a t)) (fun t => g (finCons a t))
    have : sumF K (fun a => p a * expN n p (fun t => f (finCons a t) + g (finCons a t)))
        = sumF K (fun a => p a * (expN n p (fun t => f (finCons a t))
            + expN n p (fun t => g (finCons a t)))) := by
      apply sumF_congr; intro a; rw [hpt a]
    rw [this]
    have hdist : ∀ a, p a * (expN n p (fun t => f (finCons a t))
          + expN n p (fun t => g (finCons a t)))
        = p a * expN n p (fun t => f (finCons a t))
          + p a * expN n p (fun t => g (finCons a t)) := by intro a; grind
    rw [sumF_congr K _ _ hdist, sumF_add]

theorem expN_mul {K n : Nat} (p : Fin K → Rat) (c : Rat) (f : (Fin n → Fin K) → Rat) :
    expN n p (fun s => c * f s) = c * expN n p f := by
  induction n with
  | zero => unfold expN; grind
  | succ n ih =>
    unfold expN
    have hpt : ∀ a, expN n p (fun t => c * f (finCons a t))
        = c * expN n p (fun t => f (finCons a t)) := by
      intro a
      simpa using ih (fun t => f (finCons a t))
    have : sumF K (fun a => p a * expN n p (fun t => c * f (finCons a t)))
        = sumF K (fun a => (c * p a) * expN n p (fun t => f (finCons a t))) := by
      apply sumF_congr; intro a; rw [hpt a]; grind
    rw [this]
    have hord : ∀ a, (c * p a) * expN n p (fun t => f (finCons a t))
        = c * (p a * expN n p (fun t => f (finCons a t))) := by intro a; grind
    rw [sumF_congr K _ _ hord, sumF_mul_left]

def indEq {K : Nat} (j a : Fin K) : Rat := if a = j then 1 else 0

def countCls {K n : Nat} (j : Fin K) (s : Fin n → Fin K) : Rat :=
  sumF n (fun t => indEq j (s t))

theorem indEq_sq {K : Nat} (j a : Fin K) : indEq j a * indEq j a = indEq j a := by
  unfold indEq; by_cases h : a = j <;> simp [h]

theorem count_cons {K n : Nat} (j a : Fin K) (t : Fin n → Fin K) :
    countCls j (finCons a t) = indEq j a + countCls j t := by
  unfold countCls
  simp only [sumF]
  rw [finCons_zero]
  have hpt : ∀ i, indEq j (finCons a t i.succ) = indEq j (t i) := by
    intro i; rw [finCons_succ]
  rw [sumF_congr n _ _ hpt]

theorem exp_ind_pick {K : Nat} (p : Fin K → Rat) (j : Fin K) :
    sumF K (fun a => p a * indEq j a) = p j := by
  classical
  have hpt : ∀ a, p a * indEq j a = (if a = j then p a else 0) := by
    intro a; unfold indEq; by_cases h : a = j <;> simp [h]
  rw [sumF_congr K _ _ hpt]
  -- reuse the Kronecker argument via sumF_delta with f = p, but the indicator is on the left
  have : sumF K (fun a => (if j = a then (1 : Rat) else 0) * p a) = p j := sumF_delta K j p
  have h2 : ∀ a, (if a = j then p a else 0) = (if j = a then (1 : Rat) else 0) * p a := by
    intro a
    by_cases h : a = j
    · simp [h]
    · have hj : ¬ j = a := fun heq => h heq.symm
      simp [h, hj]
  rw [sumF_congr K _ _ h2, this]

theorem exp_count {K n : Nat} (p : Fin K → Rat) (j : Fin K) (hp : sumF K p = 1) :
    expN n p (fun s => countCls j s) = (n : Rat) * p j := by
  induction n with
  | zero =>
    unfold expN countCls sumF
    grind
  | succ n ih =>
    unfold expN
    have htail : ∀ a, expN n p (fun t => countCls j (finCons a t))
        = indEq j a + (n : Rat) * p j := by
      intro a
      have hrewrite : (fun t : Fin n → Fin K => countCls j (finCons a t))
          = fun t : Fin n → Fin K => indEq j a + countCls j t := by
        funext t; rw [count_cons]
      rw [hrewrite, expN_add, expN_const p (indEq j a) hp, ih]
    have : sumF K (fun a => p a * expN n p (fun t => countCls j (finCons a t)))
        = sumF K (fun a => p a * (indEq j a + (n : Rat) * p j)) := by
      apply sumF_congr; intro a; rw [htail a]
    rw [this]
    have hdist : ∀ a, p a * (indEq j a + (n : Rat) * p j)
        = p a * indEq j a + ((n : Rat) * p j) * p a := by intro a; grind
    rw [sumF_congr K _ _ hdist, sumF_add, exp_ind_pick p j, sumF_mul_left, hp]
    grind

theorem exp_count_sq {K n : Nat} (p : Fin K → Rat) (j : Fin K) (hp : sumF K p = 1) :
    expN n p (fun s => countCls j s * countCls j s)
      = (n : Rat) * p j * (1 - p j) + ((n : Rat) * p j) * ((n : Rat) * p j) := by
  induction n with
  | zero =>
    unfold expN countCls sumF
    grind
  | succ n ih =>
    unfold expN
    have hsq : ∀ (a : Fin K) (t : Fin n → Fin K),
        countCls j (finCons a t) * countCls j (finCons a t)
        = indEq j a + countCls j t * countCls j t + 2 * indEq j a * countCls j t := by
      intro a t
      rw [count_cons]
      have hi : indEq j a * indEq j a = indEq j a := indEq_sq j a
      grind
    have htail : ∀ a, expN n p (fun t => countCls j (finCons a t) * countCls j (finCons a t))
        = indEq j a + expN n p (fun t => countCls j t * countCls j t)
          + 2 * indEq j a * expN n p (fun t => countCls j t) := by
      intro a
      have hr : (fun t : Fin n → Fin K => countCls j (finCons a t) * countCls j (finCons a t))
          = fun t : Fin n → Fin K =>
              indEq j a + (countCls j t * countCls j t + 2 * indEq j a * countCls j t) := by
        funext t; have := hsq a t; grind
      rw [hr, expN_add, expN_const p (indEq j a) hp]
      have h2 : expN n p (fun t => countCls j t * countCls j t + 2 * indEq j a * countCls j t)
          = expN n p (fun t => countCls j t * countCls j t)
            + expN n p (fun t => 2 * indEq j a * countCls j t) := expN_add _ _ _
      rw [h2]
      have h3 : expN n p (fun t => (2 * indEq j a) * countCls j t)
          = (2 * indEq j a) * expN n p (fun t => countCls j t) := expN_mul _ _ _
      have h4 : (fun t : Fin n → Fin K => 2 * indEq j a * countCls j t)
          = fun t : Fin n → Fin K => (2 * indEq j a) * countCls j t := by funext t; grind
      rw [h4, h3]
      grind
    have : sumF K (fun a => p a * expN n p (fun t =>
          countCls j (finCons a t) * countCls j (finCons a t)))
        = sumF K (fun a => p a * (indEq j a
            + expN n p (fun t => countCls j t * countCls j t)
            + 2 * indEq j a * ((n : Rat) * p j))) := by
      apply sumF_congr
      intro a
      rw [htail a, exp_count p j hp]
    rw [this, ih]
    have hdist : ∀ a, p a * (indEq j a
          + ((n : Rat) * p j * (1 - p j) + ((n : Rat) * p j) * ((n : Rat) * p j))
          + 2 * indEq j a * ((n : Rat) * p j))
        = p a * indEq j a
          + p a * ((n : Rat) * p j * (1 - p j) + ((n : Rat) * p j) * ((n : Rat) * p j))
          + (2 * ((n : Rat) * p j)) * (p a * indEq j a) := by
      intro a; grind
    rw [sumF_congr K _ _ hdist, sumF_add, sumF_add]
    have hmid : ∀ a, p a * ((n : Rat) * p j * (1 - p j) + ((n : Rat) * p j) * ((n : Rat) * p j))
        = ((n : Rat) * p j * (1 - p j) + ((n : Rat) * p j) * ((n : Rat) * p j)) * p a := by
      intro a; grind
    rw [sumF_congr K _ _ hmid, sumF_mul_left, exp_ind_pick p j, sumF_mul_left, exp_ind_pick p j, hp]
    grind

theorem exp_sumF {K n m : Nat} (p : Fin K → Rat) (hp : sumF K p = 1)
    (g : Fin m → (Fin n → Fin K) → Rat) :
    expN n p (fun s => sumF m (fun j => g j s))
      = sumF m (fun j => expN n p (fun s => g j s)) := by
  induction m with
  | zero =>
    simp only [sumF]
    rw [expN_const p 0 hp]
  | succ m ih =>
    simp only [sumF]
    rw [expN_add]
    have htail := ih (fun j => g j.succ)
    rw [htail]

theorem centered_count_sq {K n : Nat} (p : Fin K → Rat) (j : Fin K)
    (hp : sumF K p = 1) (hn : (n : Rat) ≠ 0) :
    expN n p (fun s => (countCls j s / n - p j) * (countCls j s / n - p j))
      = p j * (1 - p j) / n := by
  have hpoly : ∀ s : Fin n → Fin K,
      (countCls j s / n - p j) * (countCls j s / n - p j)
        = (1 / (n * n)) * (countCls j s * countCls j s)
          + (((-2) * p j / n) * countCls j s + p j * p j) := by
    intro s; grind
  have hfun :
      (fun s : Fin n → Fin K => (countCls j s / n - p j) * (countCls j s / n - p j))
        = fun s => (1 / (n * n)) * (countCls j s * countCls j s)
          + (((-2) * p j / n) * countCls j s + p j * p j) := by
    funext s; exact hpoly s
  rw [hfun, expN_add, expN_add, expN_mul, exp_count_sq p j hp, expN_mul, exp_count p j hp,
    expN_const p (p j * p j) hp]
  grind

theorem sum_bernoulli_var (K : Nat) (p : Fin K → Rat) (hp : sumF K p = 1) :
    sumF K (fun j => p j * (1 - p j)) = 1 - sumF K (fun j => p j * p j) := by
  have hpt : ∀ j, p j * (1 - p j) = p j + (-1) * (p j * p j) := by intro j; grind
  rw [sumF_congr K _ _ hpt, sumF_add, sumF_mul_left, hp]
  grind

/-- **O(1/N) frequency rate.** For `N` i.i.d. draws from `p`, the expected squared
Euclidean error of the empirical frequency is `(1/N) (1 - ‖p‖²)`. -/
theorem empirical_l2_rate {K n : Nat} (p : Fin K → Rat)
    (hp : sumF K p = 1) (hn : (n : Rat) ≠ 0) :
    expN n p (fun s => sumF K (fun j =>
        (countCls j s / n - p j) * (countCls j s / n - p j)))
      = (1 / n) * (1 - sumF K (fun j => p j * p j)) := by
  rw [exp_sumF p hp]
  have hpt : ∀ j, expN n p (fun s => (countCls j s / n - p j) * (countCls j s / n - p j))
      = p j * (1 - p j) / n := fun j => centered_count_sq p j hp hn
  rw [sumF_congr K _ _ hpt]
  have hdiv : ∀ j, p j * (1 - p j) / n = (1 / n) * (p j * (1 - p j)) := by intro j; grind
  rw [sumF_congr K _ _ hdiv, sumF_mul_left, sum_bernoulli_var K p hp]

/-- Excess Brier divergence `D_B(p ‖ q) = ∑_y (p(y) - q(y))²`. -/
def DB (K : Nat) (p q : Fin K → Rat) : Rat :=
  sumF K (fun j => (p j - q j) * (p j - q j))

theorem brier_excess_eq_DB (K : Nat) (p q : Fin K → Rat) (hp : sumF K p = 1) :
    expBrier K p q - expBrier K p p = DB K p q := by
  rw [brier_decomposition K p q hp]
  unfold DB
  apply sumF_congr
  intro j; grind

theorem DB_nonneg (K : Nat) (p q : Fin K → Rat) : 0 ≤ DB K p q := by
  unfold DB
  exact sumF_nonneg K _ (fun j => sq_nonneg' _)

theorem DB_zero_iff (K : Nat) (p q : Fin K → Rat) :
    DB K p q = 0 ↔ ∀ j, p j = q j := by
  unfold DB
  have hz := sumF_eq_zero_iff K (fun j => (p j - q j) * (p j - q j)) (fun j => sq_nonneg' _)
  constructor
  · intro h j
    have := sq_eq_zero' _ ((hz.mp h) j)
    grind
  · intro h
    exact hz.mpr (fun j => by rw [h j]; grind)

theorem one_coord_plugin {K n : Nat} (p : Fin K → Rat) (j : Fin K) (qj : Rat)
    (hp : sumF K p = 1) (hn : (n : Rat) ≠ 0) :
    expN n p (fun s => (countCls j s / n - qj) * (countCls j s / n - qj))
      = (p j - qj) * (p j - qj) + p j * (1 - p j) / n := by
  have hpoly : ∀ s : Fin n → Fin K,
      (countCls j s / n - qj) * (countCls j s / n - qj)
        = (1 / (n * n)) * (countCls j s * countCls j s)
          + (((-2) * qj / n) * countCls j s + qj * qj) := by
    intro s; grind
  have hfun :
      (fun s : Fin n → Fin K => (countCls j s / n - qj) * (countCls j s / n - qj))
        = fun s => (1 / (n * n)) * (countCls j s * countCls j s)
          + (((-2) * qj / n) * countCls j s + qj * qj) := by
    funext s; exact hpoly s
  rw [hfun, expN_add, expN_add, expN_mul, exp_count_sq p j hp, expN_mul, exp_count p j hp,
    expN_const p (qj * qj) hp]
  grind

/-- The plug-in divergence `‖q_N - q‖²` is biased above `D_B(p ‖ q)` by exactly
`(1/N)(1 - ‖p‖²)`, for every fixed forecast `q`. -/
theorem plugin_DB_bias {K n : Nat} (p q : Fin K → Rat)
    (hp : sumF K p = 1) (hn : (n : Rat) ≠ 0) :
    expN n p (fun s => DB K (fun j => countCls j s / n) q)
      = DB K p q + (1 / n) * (1 - sumF K (fun j => p j * p j)) := by
  have hfun : (fun s : Fin n → Fin K => DB K (fun j => countCls j s / n) q)
      = fun s : Fin n → Fin K =>
          sumF K (fun j => (countCls j s / n - q j) * (countCls j s / n - q j)) := by
    funext s
    unfold DB
    rfl
  rw [hfun, exp_sumF p hp]
  have hpt : ∀ j, expN n p (fun s => (countCls j s / n - q j) * (countCls j s / n - q j))
      = (p j - q j) * (p j - q j) + p j * (1 - p j) / n :=
    fun j => one_coord_plugin p j (q j) hp hn
  rw [sumF_congr K _ _ hpt, sumF_add]
  have hdiv : ∀ j, p j * (1 - p j) / n = (1 / n) * (p j * (1 - p j)) := by intro j; grind
  rw [sumF_congr K _ _ hdiv, sumF_mul_left, sum_bernoulli_var K p hp]
  unfold DB
  grind

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

/-! One threshold of the ranked probability score is a Bernoulli Brier score.
    Summing over thresholds gives excess RPS = ∑_t (F_q(t) - F_p(t))². -/
theorem bernoulli_square_excess (p q : Rat) :
    p * ((q - 1) * (q - 1)) + (1 - p) * (q * q)
      = (q - p) * (q - p) + p * (1 - p) := by
  grind

/-- On {0,1,2,3,4,5}: a point mass at 2 and a 50/50 law on {0,4}
    have the same mean 2, Brier gap 3/2, and RPS gap 1 (normalized gap 1/5). -/
theorem ordinal_same_mean_different_law :
    ((2 : Rat) = 2) ∧
    ((0 : Rat) * (1 / 2) + 4 * (1 / 2) = 2) ∧
    ((1 / 2) * (1 / 2) + (1 : Rat) + (1 / 2) * (1 / 2) = 3 / 2) ∧
    ((4 : Rat) * ((1 / 2) * (1 / 2)) = 1) ∧
    ((1 : Rat) / 5 = 1 / 5) := by
  grind

theorem fair_coin_couplings :
    ((1 / 2 : Rat) * 0 + (1 / 2) * 0 = 0) ∧
    ((1 / 4 : Rat) * 0 + (1 / 4) * 1 + (1 / 4) * 1 + (1 / 4) * 0 = 1 / 2) ∧
    ((1 / 2 : Rat) * 1 + (1 / 2) * 1 = 1) := by
  grind

theorem binary_realized_prob_drops :
    ((1 : Rat) - (1 / 5) = 4 / 5) ∧
    ((1 : Rat) - (47 / 100) = 53 / 100) ∧
    ((53 : Rat) / 100 < 4 / 5) := by
  grind

theorem binary_brier_strictly_worse :
    (2 * ((1 / 5) * (1 / 5)) + 2 * ((1 / 5) * (1 / 5))) / 4 = (1 : Rat) / 25 ∧
      (2 * ((47 / 100) * (47 / 100)) + 2 * ((47 / 100) * (47 / 100))) / 4 = (2209 : Rat) / 10000 ∧
      (1 : Rat) / 25 < (2209 : Rat) / 10000 := by
  grind

theorem binary_witness_corr_tie :
    let covStar : Rat := (4 * ((3 / 10) * (1 / 2))) / 4
    let varStar : Rat := (4 * ((3 / 10) * (3 / 10))) / 4
    let varY : Rat := (4 * ((1 / 2) * (1 / 2))) / 4
    let covBad : Rat := (1 / 10) * covStar
    let varBad : Rat := ((1 / 10) * (1 / 10)) * varStar
    covStar * covStar / (varStar * varY) = covBad * covBad / (varBad * varY) := by
  grind

/-- A strictly decreasing local score of the realized probability gets worse when that probability falls. -/
theorem local_score_worsens (ce : Rat → Rat) (good bad : Rat)
    (hdec : ∀ a b, 0 < a → a < b → ce b < ce a)
    (hpos : 0 < bad) (hlt : bad < good) :
    ce good < ce bad :=
  hdec bad good hpos hlt

theorem collapsed_sqcorr_scales (a v y : Rat) (ha : a ≠ 0) (hv : v ≠ 0) (hy : y ≠ 0) :
    ((a * v) * (a * v)) / (((a * a) * v) * y) = v / y := by
  grind

/-- If `Var(μ)>0` and `E[σ²]>0`, then `0 < ρ < 1` and `ρ² < ρ`,
    where `ρ = Var(μ) / (Var(μ) + E[σ²])` is both `Corr(X_{p*}, Y)` and the
    squared correlation of the collapsed judge `X_δ = a μ + b`. -/
theorem anti_proper_from_parts (varMu eVar : Rat) (hv : 0 < varMu) (he : 0 < eVar) :
    let varY := varMu + eVar
    let corr := varMu / varY
    0 < corr ∧ corr < 1 ∧ corr * corr < corr := by
  intro varY corr
  have hy : 0 < varY := by grind
  have hlt : varMu < varY := by grind
  have hpos : 0 < corr := by
    simpa [corr, varY] using div_pos_of_pos varMu varY hv hy
  have hlt1 : corr < 1 := by
    simpa [corr, varY] using div_lt_one_of_lt varMu varY hy hlt
  have hsq : corr * corr < corr := by
    have h := rat_mul_lt_left corr corr 1 hpos hlt1
    grind
  exact ⟨hpos, hlt1, hsq⟩

theorem anti_proper_half_example :
    let eVar : Rat := (1 / 2) * 1 + (1 / 2) * 1
    let muBar : Rat := (1 / 2) * 1 + (1 / 2) * 3
    let varMu : Rat := (1 / 2) * ((1 - muBar) * (1 - muBar)) + (1 / 2) * ((3 - muBar) * (3 - muBar))
    let varY : Rat := eVar + varMu
    varMu / varY = 1 / 2 ∧ (varMu / varY) * (varMu / varY) < varMu / varY := by
  grind

theorem independent_sq_drop (mux muy varx vary : Rat) :
    (mux - muy) * (mux - muy) + varx + vary
      = (mux * mux + varx) + (muy * muy + vary) - 2 * (mux * muy) := by
  grind

#print axioms brier_decomposition
#print axioms brier_strictly_proper
#print axioms binary_brier_eq_two_l2
#print axioms standardized_mse_corr
#print axioms mse_decomposition
#print axioms pearson_affine_invariant
#print axioms empirical_l2_rate
#print axioms plugin_DB_bias
#print axioms DB_zero_iff
#print axioms bernoulli_square_excess
#print axioms ordinal_same_mean_different_law
#print axioms fair_coin_couplings
#print axioms binary_realized_prob_drops
#print axioms binary_brier_strictly_worse
#print axioms binary_witness_corr_tie
#print axioms local_score_worsens
#print axioms collapsed_sqcorr_scales
#print axioms anti_proper_from_parts
#print axioms anti_proper_half_example
#print axioms independent_sq_drop

-- audit: print axioms for every theorem not already printed
#print axioms sumF_add
#print axioms sumF_mul_left
#print axioms sumF_const
#print axioms sumF_congr
#print axioms sumF_delta
#print axioms sq_nonneg'
#print axioms sq_eq_zero'
#print axioms sumF_nonneg
#print axioms sumF_eq_zero_iff
#print axioms brier_expand
#print axioms expBrier_closed
#print axioms brier_proper
#print axioms l2_decomposition
#print axioms l2_mean_minimizer
#print axioms cov_raw
#print axioms pos_sqrt_unique
#print axioms finCons_zero
#print axioms finCons_succ
#print axioms expN_const
#print axioms expN_add
#print axioms expN_mul
#print axioms indEq_sq
#print axioms count_cons
#print axioms exp_ind_pick
#print axioms exp_count
#print axioms exp_count_sq
#print axioms exp_sumF
#print axioms centered_count_sq
#print axioms sum_bernoulli_var
#print axioms brier_excess_eq_DB
#print axioms DB_nonneg
#print axioms one_coord_plugin
#print axioms rat_mul_lt_left
#print axioms div_neg
#print axioms div_lt_one_of_lt
#print axioms div_pos_of_pos
