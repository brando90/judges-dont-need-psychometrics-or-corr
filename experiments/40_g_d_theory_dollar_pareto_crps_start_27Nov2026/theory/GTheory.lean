set_option linter.unusedSimpArgs false
set_option maxHeartbeats 800000
/-
  G-theory blindness, the distribution-free proper-risk decomposition, and
  finite-grid ranked probability score / CRPS.

  Core Lean 4 only (no Mathlib), over `Rat`. Zero `sorry`. Zero custom axioms.
  No normality, additivity, or homoscedasticity hypothesis is used for the
  scoring-rule identities. G-theory coefficients below are functions of the
  cell means and the cell variances only.
-/

/-! ## Finite sums -/

def sumF : (n : Nat) → (Fin n → Rat) → Rat
  | 0, _ => 0
  | n + 1, f => f 0 + sumF n (fun i => f i.succ)

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

theorem sumF_swap (n m : Nat) (f : Fin n → Fin m → Rat) :
    sumF n (fun i => sumF m (fun j => f i j)) = sumF m (fun j => sumF n (fun i => f i j)) := by
  induction n with
  | zero =>
      simp only [sumF]
      have : sumF m (fun _ => (0 : Rat)) = 0 := by rw [sumF_const]; grind
      exact this.symm
  | succ n ih =>
      simp only [sumF]
      rw [ih (fun i j => f i.succ j)]
      symm
      exact sumF_add m (fun j => f 0 j) (fun j => sumF n (fun i => f i.succ j))

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

/-! ## Prefix sums (CDF masses) -/

/-- Sum of the first `t` coordinates, `q 0 + … + q (t-1)`. -/
def mass (n : Nat) : Nat → (Fin n → Rat) → Rat
  | 0, _ => 0
  | t + 1, q => mass n t q + if h : t < n then q ⟨t, h⟩ else 0

theorem mass_succ_lt (n t : Nat) (q : Fin n → Rat) (h : t < n) :
    mass n (t + 1) q = mass n t q + q ⟨t, h⟩ := by
  simp [mass, h]

theorem sumF_prefix_step (n t : Nat) (q : Fin n → Rat) (ht : t < n) :
    sumF n (fun j => if j.val < t + 1 then q j else 0)
      = sumF n (fun j => if j.val < t then q j else 0) + q ⟨t, ht⟩ := by
  induction n generalizing t with
  | zero => exact (Nat.not_lt_zero t ht).elim
  | succ n ih =>
      cases t with
      | zero =>
          simp only [sumF]
          have htail : sumF n (fun j => if j.succ.val < 1 then q j.succ else 0)
              = sumF n (fun _ => (0 : Rat)) := by
            apply sumF_congr
            intro j
            have hv : j.succ.val = j.val + 1 := Fin.val_succ j
            have hnot : ¬ j.succ.val < 1 := by rw [hv]; omega
            simp [hnot]
          have htail0 : sumF n (fun j => if j.succ.val < 0 then q j.succ else 0)
              = sumF n (fun _ => (0 : Rat)) := by
            apply sumF_congr
            intro j
            simp
          rw [htail, htail0, sumF_const]
          have h0 : ((0 : Fin (n + 1))).val = 0 := by simp
          simp [h0]
          grind
      | succ t =>
          have ht' : t < n := by omega
          simp only [sumF]
          have hhead : ((0 : Fin (n + 1))).val < t.succ + 1 := by
            simp
          have hhead' : ((0 : Fin (n + 1))).val < t.succ := by
            simp
          simp only [hhead, hhead', if_true]
          have ih' := ih t (fun j => q j.succ) ht'
          have hL : sumF n (fun j => if j.succ.val < t.succ + 1 then q j.succ else 0)
              = sumF n (fun j => if j.val < t + 1 then q j.succ else 0) := by
            apply sumF_congr
            intro j
            have hv : j.succ.val = j.val + 1 := Fin.val_succ j
            by_cases hj : j.val < t + 1
            · simp [hj, hv]
            · have : ¬ j.succ.val < t.succ + 1 := by
                rw [hv]
                omega
              simp [hj, this]
          have hR : sumF n (fun j => if j.succ.val < t.succ then q j.succ else 0)
              = sumF n (fun j => if j.val < t then q j.succ else 0) := by
            apply sumF_congr
            intro j
            have hv : j.succ.val = j.val + 1 := Fin.val_succ j
            by_cases hj : j.val < t
            · simp [hj, hv]
            · have : ¬ j.succ.val < t.succ := by rw [hv]; omega
              simp [hj, this]
          have hidx : Fin.succ (Fin.mk t ht') = Fin.mk (t + 1) ht := by
            apply Fin.ext
            simp [Fin.val_succ, Fin.mk]
          have hq : q (Fin.mk (t + 1) ht) = q (Fin.succ (Fin.mk t ht')) := by
            rw [hidx]
          rw [hL, hR, ih', hq]
          grind

theorem mass_eq_prefix (n t : Nat) (q : Fin n → Rat) (ht : t ≤ n) :
    mass n t q = sumF n (fun j => if j.val < t then q j else 0) := by
  induction t with
  | zero =>
      simp only [mass]
      have h0 : sumF n (fun j => if j.val < 0 then q j else 0) = sumF n (fun _ => (0 : Rat)) := by
        apply sumF_congr; intro j; simp
      rw [h0, sumF_const]; grind
  | succ t ih =>
      have hlt : t < n := Nat.lt_of_succ_le ht
      rw [mass_succ_lt n t q hlt, ih (Nat.le_of_lt hlt)]
      exact (sumF_prefix_step n t q hlt).symm

theorem mass_total (n : Nat) (q : Fin n → Rat) (hq : sumF n q = 1) : mass n n q = 1 := by
  rw [mass_eq_prefix n n q (Nat.le_refl n)]
  have hpt : ∀ j : Fin n, (if j.val < n then q j else 0) = q j := by
    intro j
    have : j.val < n := j.isLt
    simp [this]
  rw [sumF_congr n _ _ hpt, hq]

theorem mass_determines (n : Nat) (p q : Fin n → Rat)
    (hlast : mass n n p = mass n n q)
    (hpref : ∀ t, t < n → mass n t p = mass n t q) :
    ∀ i, p i = q i := by
  intro i
  have hp := mass_succ_lt n i.val p i.isLt
  have hq := mass_succ_lt n i.val q i.isLt
  have hprev : mass n i.val p = mass n i.val q := hpref i.val i.isLt
  have hnext : mass n (i.val + 1) p = mass n (i.val + 1) q := by
    by_cases hlasti : i.val + 1 = n
    · rw [hlasti]; exact hlast
    · have hlt : i.val + 1 < n := by omega
      exact hpref (i.val + 1) hlt
  grind

/-! ## Brier decomposition, distribution-free -/

def brier (K : Nat) (q : Fin K → Rat) (i : Fin K) : Rat :=
  sumF K (fun j => (q j - (if i = j then (1 : Rat) else 0)) * (q j - (if i = j then (1 : Rat) else 0)))

def expBrier (K : Nat) (p q : Fin K → Rat) : Rat :=
  sumF K (fun i => p i * brier K q i)

theorem sumF_delta (n : Nat) (i : Fin n) (f : Fin n → Rat) :
    sumF n (fun j => (if i = j then (1 : Rat) else 0) * f j) = f i := by
  induction n with
  | zero => exact i.elim0
  | succ n ih =>
      simp only [sumF]
      cases i using Fin.cases with
      | zero =>
          have h0 : sumF n (fun j => (if (0 : Fin (n + 1)) = j.succ then (1 : Rat) else 0) * f j.succ)
              = sumF n (fun _ => 0) := by
            apply sumF_congr
            intro j
            have : (0 : Fin (n + 1)) ≠ j.succ := fun h => Fin.succ_ne_zero j h.symm
            simp [this]
          rw [h0, sumF_const]
          grind
      | succ i' =>
          have h1 : sumF n (fun j => (if i'.succ = j.succ then (1 : Rat) else 0) * f j.succ)
              = sumF n (fun j => (if i' = j then (1 : Rat) else 0) * f j.succ) := by
            apply sumF_congr
            intro j
            simp [Fin.succ_inj]
          rw [h1, ih i' (fun j => f j.succ)]
          have : i'.succ ≠ (0 : Fin (n + 1)) := Fin.succ_ne_zero i'
          simp [this]
          grind

theorem brier_expand (K : Nat) (q : Fin K → Rat) (i : Fin K) :
    brier K q i = sumF K (fun j => q j * q j) + (1 - 2 * q i) := by
  unfold brier
  have hpt : ∀ j, (q j - (if i = j then (1 : Rat) else 0)) * (q j - (if i = j then (1 : Rat) else 0))
      = q j * q j + (if i = j then (1 : Rat) else 0) * (1 - 2 * q j) := by
    intro j
    by_cases hij : i = j <;> simp [hij] <;> grind
  rw [sumF_congr K _ _ hpt, sumF_add, sumF_delta K i (fun j => 1 - 2 * q j)]

theorem expBrier_closed (K : Nat) (p q : Fin K → Rat) (hp : sumF K p = 1) :
    expBrier K p q = sumF K (fun j => q j * q j) + 1 - 2 * sumF K (fun i => p i * q i) := by
  unfold expBrier
  have hpt : ∀ i, p i * brier K q i
      = (sumF K (fun j => q j * q j) + 1) * p i + (-2) * (p i * q i) := by
    intro i
    rw [brier_expand]
    grind
  rw [sumF_congr K _ _ hpt, sumF_add, sumF_mul_left, sumF_mul_left, hp]
  grind

/-- Distribution-free Brier decomposition.
`E[BS(q, Y)] = ‖q - p‖² + E[BS(p, Y)]`. Only `∑ p = 1` is used. -/
theorem brier_risk_decomposition (K : Nat) (p q : Fin K → Rat) (hp : sumF K p = 1) :
    expBrier K p q = sumF K (fun j => (q j - p j) * (q j - p j)) + expBrier K p p := by
  have hclosedq := expBrier_closed K p q hp
  have hclosedp := expBrier_closed K p p hp
  have hpt : ∀ j, (q j - p j) * (q j - p j)
      = q j * q j + ((-2) * (p j * q j) + p j * p j) := by
    intro j
    grind
  have hsum : sumF K (fun j => (q j - p j) * (q j - p j))
      = sumF K (fun j => q j * q j) - 2 * sumF K (fun j => p j * q j) + sumF K (fun j => p j * p j) := by
    rw [sumF_congr K _ _ hpt, sumF_add, sumF_add, sumF_mul_left]
    grind
  grind

theorem brier_strictly_proper (K : Nat) (p q : Fin K → Rat) (hp : sumF K p = 1) :
    expBrier K p q = expBrier K p p ↔ ∀ j, q j = p j := by
  have h := brier_risk_decomposition K p q hp
  have hz := sumF_eq_zero_iff K (fun j => (q j - p j) * (q j - p j)) (fun j => sq_nonneg' _)
  constructor
  · intro he j
    have : sumF K (fun j => (q j - p j) * (q j - p j)) = 0 := by grind
    have := sq_eq_zero' _ (hz.mp this j)
    grind
  · intro hq
    have : sumF K (fun j => (q j - p j) * (q j - p j)) = 0 :=
      hz.mpr (fun j => by rw [hq j]; grind)
    grind

/-! ## Finite-grid RPS / discretized CRPS -/

/-- Squared CDF error at threshold `t` (0-based, `t.val ≤ K-2`). Unit spacing, so this
term is the finite-grid CRPS contribution of that cut. -/
def rpsTerm (K : Nat) (q : Fin K → Rat) (y : Fin K) (t : Fin (K - 1)) : Rat :=
  let ft := mass K (t.val + 1) q
  let ind := if y.val ≤ t.val then (1 : Rat) else 0
  (ft - ind) * (ft - ind)

def rps (K : Nat) (q : Fin K → Rat) (y : Fin K) : Rat :=
  sumF (K - 1) (fun t => rpsTerm K q y t)

def expRps (K : Nat) (p q : Fin K → Rat) : Rat :=
  sumF K (fun y => p y * rps K q y)

def rpsDivergence (K : Nat) (p q : Fin K → Rat) : Rat :=
  sumF (K - 1) (fun t =>
    let fp := mass K (t.val + 1) p
    let fq := mass K (t.val + 1) q
    (fq - fp) * (fq - fp))

theorem indicator_is_bit (t y : Nat) :
    let ind := if y ≤ t then (1 : Rat) else 0
    ind * ind = ind := by
  by_cases h : y ≤ t <;> simp [h]

theorem bernoulli_square (n : Nat) (p ind : Fin n → Rat) (c mu : Rat)
    (hp : sumF n p = 1) (hbit : ∀ i, ind i * ind i = ind i)
    (hmu : sumF n (fun i => p i * ind i) = mu) :
    sumF n (fun i => p i * ((ind i - c) * (ind i - c)))
      = (mu - c) * (mu - c) + mu * (1 - mu) := by
  have hpt : ∀ i, p i * ((ind i - c) * (ind i - c))
      = p i * ind i + ((-2) * c) * (p i * ind i) + (c * c) * p i := by
    intro i
    have hbiti := hbit i
    grind
  rw [sumF_congr n _ _ hpt, sumF_add, sumF_add, sumF_mul_left, sumF_mul_left, hmu, hp]
  grind

theorem exp_rpsTerm (K : Nat) (p q : Fin K → Rat) (t : Fin (K - 1)) (hp : sumF K p = 1) :
    sumF K (fun y => p y * rpsTerm K q y t)
      = (mass K (t.val + 1) q - mass K (t.val + 1) p) * (mass K (t.val + 1) q - mass K (t.val + 1) p)
        + (mass K (t.val + 1) p) * (1 - mass K (t.val + 1) p) := by
  let ind : Fin K → Rat := fun y => if y.val ≤ t.val then 1 else 0
  have hbit : ∀ y, ind y * ind y = ind y := by
    intro y
    simpa [ind] using indicator_is_bit t.val y.val
  have hmu : sumF K (fun y => p y * ind y) = mass K (t.val + 1) p := by
    have hpt : ∀ y : Fin K, p y * ind y = if y.val < t.val + 1 then p y else 0 := by
      intro y
      by_cases h : y.val ≤ t.val
      · have hlt : y.val < t.val + 1 := by omega
        simp [ind, h, hlt]
      · have hnot : ¬ y.val < t.val + 1 := by omega
        simp [ind, h, hnot]
    rw [sumF_congr K _ _ hpt]
    have ht : t.val + 1 ≤ K := by
      have := t.isLt
      omega
    exact (mass_eq_prefix K (t.val + 1) p ht).symm
  have hterm : ∀ y, rpsTerm K q y t = (mass K (t.val + 1) q - ind y) * (mass K (t.val + 1) q - ind y) := by
    intro y
    simp [rpsTerm, ind]
  have hsum : sumF K (fun y => p y * rpsTerm K q y t)
      = sumF K (fun y => p y * ((ind y - mass K (t.val + 1) q) * (ind y - mass K (t.val + 1) q))) := by
    apply sumF_congr
    intro y
    have := hterm y
    grind
  rw [hsum]
  have hber := bernoulli_square K p ind (mass K (t.val + 1) q) (mass K (t.val + 1) p) hp hbit hmu
  grind

/-- Distribution-free RPS decomposition.
`E[RPS(q, Y)] = Σ_t (F_q(t) - F_p(t))² + E[RPS(p, Y)]`. -/
theorem rps_risk_decomposition (K : Nat) (p q : Fin K → Rat) (hp : sumF K p = 1) :
    expRps K p q = rpsDivergence K p q + expRps K p p := by
  unfold expRps rps rpsDivergence
  have hswap := sumF_swap K (K - 1) (fun y t => p y * rpsTerm K q y t)
  have hswapp := sumF_swap K (K - 1) (fun y t => p y * rpsTerm K p y t)
  have hpt : ∀ y t, p y * rpsTerm K q y t = p y * rpsTerm K q y t := by intro y t; rfl
  -- rewrite the double sum into a sum of threshold expectations
  have hq : sumF K (fun y => p y * sumF (K - 1) (fun t => rpsTerm K q y t))
      = sumF (K - 1) (fun t => sumF K (fun y => p y * rpsTerm K q y t)) := by
    have hmul : ∀ y, p y * sumF (K - 1) (fun t => rpsTerm K q y t)
        = sumF (K - 1) (fun t => p y * rpsTerm K q y t) := by
      intro y
      exact (sumF_mul_left (K - 1) (p y) (fun t => rpsTerm K q y t)).symm
    rw [sumF_congr K _ _ hmul]
    exact hswap
  have hp' : sumF K (fun y => p y * sumF (K - 1) (fun t => rpsTerm K p y t))
      = sumF (K - 1) (fun t => sumF K (fun y => p y * rpsTerm K p y t)) := by
    have hmul : ∀ y, p y * sumF (K - 1) (fun t => rpsTerm K p y t)
        = sumF (K - 1) (fun t => p y * rpsTerm K p y t) := by
      intro y
      exact (sumF_mul_left (K - 1) (p y) (fun t => rpsTerm K p y t)).symm
    rw [sumF_congr K _ _ hmul]
    exact hswapp
  rw [hq, hp']
  have hterm : ∀ t, sumF K (fun y => p y * rpsTerm K q y t)
      = (mass K (t.val + 1) q - mass K (t.val + 1) p) * (mass K (t.val + 1) q - mass K (t.val + 1) p)
        + sumF K (fun y => p y * rpsTerm K p y t) := by
    intro t
    have hq := exp_rpsTerm K p q t hp
    have hpterm := exp_rpsTerm K p p t hp
    grind
  rw [sumF_congr (K - 1) _ _ hterm, sumF_add]

theorem mass_of_funext (n t : Nat) (p q : Fin n → Rat) (h : ∀ j, p j = q j) (ht : t ≤ n) :
    mass n t p = mass n t q := by
  induction t with
  | zero => simp [mass]
  | succ t ih =>
      have hlt : t < n := Nat.lt_of_succ_le ht
      rw [mass_succ_lt n t p hlt, mass_succ_lt n t q hlt]
      have ih' := ih (Nat.le_of_lt hlt)
      have hq := h ⟨t, hlt⟩
      grind

theorem rps_strictly_proper (K : Nat) (p q : Fin K → Rat) (hp : sumF K p = 1) (hq : sumF K q = 1) :
    expRps K p q = expRps K p p ↔ ∀ j, q j = p j := by
  have hdec := rps_risk_decomposition K p q hp
  have hz := sumF_eq_zero_iff (K - 1)
      (fun t => (mass K (t.val + 1) q - mass K (t.val + 1) p) * (mass K (t.val + 1) q - mass K (t.val + 1) p))
      (fun t => sq_nonneg' _)
  constructor
  · intro he
    have hdiv : sumF (K - 1) (fun t =>
        (mass K (t.val + 1) q - mass K (t.val + 1) p) * (mass K (t.val + 1) q - mass K (t.val + 1) p)) = 0 := by
      unfold rpsDivergence at hdec
      grind
    have hmass : ∀ t : Fin (K - 1), mass K (t.val + 1) q = mass K (t.val + 1) p := by
      intro t
      have hsq := hz.mp hdiv t
      have hzero := sq_eq_zero' _ hsq
      grind
    have hpref : ∀ t, t < K → mass K t p = mass K t q := by
      intro t ht
      cases t with
      | zero => simp [mass]
      | succ t =>
          have ht' : t < K - 1 := by omega
          have hm := hmass ⟨t, ht'⟩
          exact hm.symm
    have hlast : mass K K p = mass K K q := by
      rw [mass_total K p hp, mass_total K q hq]
    intro j
    exact (mass_determines K p q hlast hpref j).symm
  · intro hqq
    have hmass : ∀ t, t ≤ K → mass K t q = mass K t p := fun t ht => mass_of_funext K t q p hqq ht
    have hdiv : rpsDivergence K p q = 0 := by
      unfold rpsDivergence
      apply (sumF_eq_zero_iff (K - 1) _ (fun t => sq_nonneg' _)).mpr
      intro t
      have ht : t.val + 1 ≤ K := by have := t.isLt; omega
      have hm := hmass (t.val + 1) ht
      grind
    grind

/-- Adding a threshold cannot decrease RPS: refinement of the cut set is monotone. -/
theorem rps_term_nonneg (K : Nat) (q : Fin K → Rat) (y : Fin K) (t : Fin (K - 1)) :
    0 ≤ rpsTerm K q y t := by
  unfold rpsTerm
  exact sq_nonneg' _

theorem refinement_adds_a_nonnegative_term (a extra : Rat) (h : 0 ≤ extra) : a ≤ a + extra := by
  grind

/-! ## Concrete 4-class witness -/

def pStar (j : Fin 4) : Rat := 1 / 4

def qBad (j : Fin 4) : Rat :=
  if j.val = 0 then 1 / 5
  else if j.val = 1 then 2 / 5
  else if j.val = 2 then 1 / 10
  else 3 / 10

def mean4 (q : Fin 4 → Rat) : Rat := q 0 * 0 + q 1 * 1 + q 2 * 2 + q 3 * 3

def second4 (q : Fin 4 → Rat) : Rat := q 0 * 0 + q 1 * 1 + q 2 * 4 + q 3 * 9

theorem pStar_sum : sumF 4 pStar = 1 := by
  simp [sumF, pStar]
  grind

theorem qBad_sum : sumF 4 qBad = 1 := by
  simp [sumF, qBad]
  grind

theorem moments_agree :
    mean4 pStar = mean4 qBad ∧ second4 pStar = second4 qBad ∧
      mean4 pStar = 3 / 2 ∧ second4 pStar = 7 / 2 := by
  simp [mean4, second4, pStar, qBad]
  grind

/-- Cell variance `E[Y²] - (E[Y])²` agrees, so a second-moment G-study cannot see the distortion. -/
theorem cell_variance_agrees :
    second4 pStar - mean4 pStar * mean4 pStar = 5 / 4 ∧
      second4 qBad - mean4 qBad * mean4 qBad = 5 / 4 := by
  have h := moments_agree
  grind

/-- Shared 2×2 cell-mean design. Both judges are assigned this table. -/
def cellMean (i j : Fin 2) : Rat :=
  if i.val = 0 then
    if j.val = 0 then 3 / 2 else 1
  else
    if j.val = 0 then 3 / 4 else 1 / 4

def itemMean (i : Fin 2) : Rat := (cellMean i 0 + cellMean i 1) / 2

def raterMean (j : Fin 2) : Rat := (cellMean 0 j + cellMean 1 j) / 2

def grandMean : Rat := (itemMean 0 + itemMean 1) / 2

def sigmaC2 : Rat := ((itemMean 0 - grandMean) * (itemMean 0 - grandMean)
    + (itemMean 1 - grandMean) * (itemMean 1 - grandMean)) / 2

def sigmaR2 : Rat := ((raterMean 0 - grandMean) * (raterMean 0 - grandMean)
    + (raterMean 1 - grandMean) * (raterMean 1 - grandMean)) / 2

/-- Within-cell variance is the shared cell variance `5/4` in every cell. -/
def sigmaRes2 : Rat := 5 / 4

def eRho2 (sc se : Rat) : Rat := sc / (sc + se)

def phiCoeff (sc sr se : Rat) : Rat := sc / (sc + sr + se)

/-- D-study absolute coefficient at `k` ratings. It depends on `k` and the three
variance components, and not on the number of items `N`. -/
def phiAt (k sc sr se : Rat) : Rat := sc / (sc + (sr + se) / k)

theorem cellMean_00 : cellMean 0 0 = 3 / 2 := by simp [cellMean]
theorem cellMean_01 : cellMean 0 1 = 1 := by simp [cellMean]
theorem cellMean_10 : cellMean 1 0 = 3 / 4 := by simp [cellMean]
theorem cellMean_11 : cellMean 1 1 = 1 / 4 := by simp [cellMean]

theorem itemMean_0 : itemMean 0 = 5 / 4 := by simp [itemMean, cellMean_00, cellMean_01]; grind
theorem itemMean_1 : itemMean 1 = 1 / 2 := by simp [itemMean, cellMean_10, cellMean_11]; grind
theorem raterMean_0 : raterMean 0 = 9 / 8 := by simp [raterMean, cellMean_00, cellMean_10]; grind
theorem raterMean_1 : raterMean 1 = 5 / 8 := by simp [raterMean, cellMean_01, cellMean_11]; grind
theorem grandMean_value : grandMean = 7 / 8 := by simp [grandMean, itemMean_0, itemMean_1]; grind

theorem sigmaC2_value : sigmaC2 = 9 / 64 := by
  simp [sigmaC2, itemMean_0, itemMean_1, grandMean_value]; grind
theorem sigmaR2_value : sigmaR2 = 1 / 16 := by
  simp [sigmaR2, raterMean_0, raterMean_1, grandMean_value]; grind

theorem g_components_values :
    sigmaC2 = 9 / 64 ∧ sigmaR2 = 1 / 16 ∧ sigmaRes2 = 5 / 4 ∧
      eRho2 sigmaC2 sigmaRes2 = 9 / 89 ∧ phiCoeff sigmaC2 sigmaR2 sigmaRes2 = 3 / 31 := by
  have hc := sigmaC2_value
  have hr := sigmaR2_value
  have he : sigmaRes2 = 5 / 4 := rfl
  refine ⟨hc, hr, he, ?_, ?_⟩
  · simp [eRho2, hc, he]; grind
  · simp [phiCoeff, hc, hr, he]; grind

theorem lt_of_sub_eq (a b c : Rat) (hc : 0 < c) (h : b - a = c) : a < b := by
  have h1 : a < a + c := by grind
  have h2 : a + c = b := by grind
  grind

theorem dstudy_depends_only_on_k :
    phiAt 1 sigmaC2 sigmaR2 sigmaRes2 = 3 / 31 ∧
      phiAt 4 sigmaC2 sigmaR2 sigmaRes2 = 3 / 10 ∧
      phiAt 1 sigmaC2 sigmaR2 sigmaRes2 < phiAt 4 sigmaC2 sigmaR2 sigmaRes2 := by
  have hc := sigmaC2_value
  have hr := sigmaR2_value
  have he : sigmaRes2 = 5 / 4 := rfl
  refine ⟨?_, ?_, ?_⟩
  · simp [phiAt, hc, hr, he]; grind
  · simp [phiAt, hc, hr, he]; grind
  · have h1 : phiAt 1 sigmaC2 sigmaR2 sigmaRes2 = 3 / 31 := by simp [phiAt, hc, hr, he]; grind
    have h4 : phiAt 4 sigmaC2 sigmaR2 sigmaRes2 = 3 / 10 := by simp [phiAt, hc, hr, he]; grind
    grind

theorem brier_gap_witness :
    expBrier 4 pStar qBad - expBrier 4 pStar pStar = 1 / 20 := by
  have h := brier_risk_decomposition 4 pStar qBad pStar_sum
  have hs : sumF 4 (fun j => (qBad j - pStar j) * (qBad j - pStar j)) = 1 / 20 := by
    simp [sumF, qBad, pStar]
    grind
  grind

theorem rps_gap_witness :
    expRps 4 pStar qBad - expRps 4 pStar pStar = 3 / 200 := by
  have h := rps_risk_decomposition 4 pStar qBad pStar_sum
  -- three CDF gaps: 1/20, -1/10, 1/20, squares sum to 3/200
  have hs : rpsDivergence 4 pStar qBad = 3 / 200 := by
    simp [rpsDivergence, sumF, mass, pStar, qBad]
    grind
  grind

/-- Rational certificate that uniform cross-entropy is strictly smaller.
For `p = (1/4,…,1/4)`, `KL(p‖q) = (1/4) ln(((1/4)^4) / ∏ q_i)`.
Core Lean has no real logarithm. `∏ q < (1/4)^4` is the sign of that gap because
`ln` is strictly increasing and `ln 1 = 0` (calculus; the nats are in pytest). -/
theorem ce_gap_rational_certificate :
    ((1 : Rat) / 5) * (2 / 5) * (1 / 10) * (3 / 10) < (1 / 4) * (1 / 4) * (1 / 4) * (1 / 4) := by
  have h1 : ((1 : Rat) / 5) * (2 / 5) = 2 / 25 := by grind
  have h2 : ((2 : Rat) / 25) * (1 / 10) = 1 / 125 := by grind
  have h3 : ((1 : Rat) / 125) * (3 / 10) = 3 / 1250 := by grind
  have h4 : ((1 : Rat) / 4) * (1 / 4) * (1 / 4) * (1 / 4) = 1 / 256 := by grind
  grind

theorem qBad_not_pStar : ¬ ∀ j, qBad j = pStar j := by
  intro h
  have := h 0
  simp [qBad, pStar] at this
  grind

/-- The G-theory map of the shared moment table is the same number for both judges,
while Brier, RPS, and the rational cross-entropy certificate all separate them. -/
theorem g_theory_blindness_witness :
    sigmaC2 = 9 / 64 ∧ sigmaR2 = 1 / 16 ∧ sigmaRes2 = 5 / 4 ∧
      eRho2 sigmaC2 sigmaRes2 = 9 / 89 ∧ phiCoeff sigmaC2 sigmaR2 sigmaRes2 = 3 / 31 ∧
      (second4 pStar - mean4 pStar * mean4 pStar = second4 qBad - mean4 qBad * mean4 qBad) ∧
      expBrier 4 pStar pStar < expBrier 4 pStar qBad ∧
      expRps 4 pStar pStar < expRps 4 pStar qBad ∧
      ((1 : Rat) / 5) * (2 / 5) * (1 / 10) * (3 / 10) < (1 / 4) * (1 / 4) * (1 / 4) * (1 / 4) := by
  have gc := g_components_values
  have cv := cell_variance_agrees
  refine ⟨gc.1, gc.2.1, gc.2.2.1, gc.2.2.2.1, gc.2.2.2.2,
    cv.1.trans cv.2.symm, ?_, ?_, ce_gap_rational_certificate⟩
  · exact lt_of_sub_eq _ _ (1 / 20) (by grind) brier_gap_witness
  · exact lt_of_sub_eq _ _ (3 / 200) (by grind) rps_gap_witness

/-- Hypotheses of the witness can hold: both vectors are probability distributions. -/
theorem witness_probabilities_hold : sumF 4 pStar = 1 ∧ sumF 4 qBad = 1 :=
  ⟨pStar_sum, qBad_sum⟩

/-! ## Refinement on the witness grid: the middle cut is one RPS term -/

def coarseMiddle (q : Fin 4 → Rat) (y : Fin 4) : Rat :=
  let ft := q 0 + q 1
  let ind := if y.val ≤ 1 then (1 : Rat) else 0
  (ft - ind) * (ft - ind)

/-- The three cut terms of 4-class RPS, written without the `Fin` index wrapper. -/
def rps4Terms (q : Fin 4 → Rat) (y : Fin 4) : Rat :=
  (q 0 - if y.val ≤ 0 then (1 : Rat) else 0) * (q 0 - if y.val ≤ 0 then (1 : Rat) else 0)
    + (q 0 + q 1 - if y.val ≤ 1 then (1 : Rat) else 0) * (q 0 + q 1 - if y.val ≤ 1 then (1 : Rat) else 0)
    + (q 0 + q 1 + q 2 - if y.val ≤ 2 then (1 : Rat) else 0)
      * (q 0 + q 1 + q 2 - if y.val ≤ 2 then (1 : Rat) else 0)

theorem rps_refines_middle_cut (q : Fin 4 → Rat) (y : Fin 4) :
    coarseMiddle q y ≤ rps4Terms q y := by
  simp only [coarseMiddle, rps4Terms]
  have h0 : 0 ≤ (q 0 - if y.val ≤ 0 then (1 : Rat) else 0) * (q 0 - if y.val ≤ 0 then (1 : Rat) else 0) :=
    sq_nonneg' _
  have h2 : 0 ≤ (q 0 + q 1 + q 2 - if y.val ≤ 2 then (1 : Rat) else 0)
      * (q 0 + q 1 + q 2 - if y.val ≤ 2 then (1 : Rat) else 0) := sq_nonneg' _
  grind

#print axioms brier_risk_decomposition
#print axioms brier_strictly_proper
#print axioms rps_risk_decomposition
#print axioms rps_strictly_proper
#print axioms g_theory_blindness_witness
#print axioms ce_gap_rational_certificate
#print axioms rps_refines_middle_cut
#print axioms witness_probabilities_hold
#print axioms dstudy_depends_only_on_k
