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

-- Print the axioms each main theorem depends on (should be only Lean's standard
-- `propext`, `Classical.choice`, `Quot.sound` — no `sorryAx`, no custom axioms).
#print axioms brier_decomposition
#print axioms brier_proper
#print axioms brier_strictly_proper
#print axioms l2_decomposition
#print axioms l2_mean_minimizer
#print axioms binary_brier_eq_two_l2
#print axioms standardized_mse_corr
#print axioms mse_decomposition
#print axioms pearson_affine_invariant
