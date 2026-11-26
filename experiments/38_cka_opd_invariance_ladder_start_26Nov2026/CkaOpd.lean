def sumF : (n : Nat) → (Fin n → Rat) → Rat
  | 0, _ => 0
  | n+1, f => f 0 + sumF n (fun i => f i.succ)

theorem sumF_zero (n : Nat) : sumF n (fun _ => (0 : Rat)) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [sumF, ih]; grind

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

theorem sumF_congr (n : Nat) (f g : Fin n → Rat) (h : ∀ i, f i = g i) :
    sumF n f = sumF n g := by
  have : f = g := funext h
  rw [this]

theorem sumF_swap (n m : Nat) (f : Fin n → Fin m → Rat) :
    sumF n (fun i => sumF m (fun j => f i j)) =
      sumF m (fun j => sumF n (fun i => f i j)) := by
  induction n with
  | zero =>
      simp only [sumF]
      exact (sumF_zero m).symm
  | succ n ih =>
      simp only [sumF]
      rw [ih (fun i j => f i.succ j)]
      have :
          sumF m (fun j => f 0 j) + sumF m (fun j => sumF n (fun i => f i.succ j))
            = sumF m (fun j => f 0 j + sumF n (fun i => f i.succ j)) := by
        simpa using (sumF_add m (fun j => f 0 j) (fun j => sumF n (fun i => f i.succ j))).symm
      exact this

theorem sumF_mul_sum (n m : Nat) (f : Fin n → Rat) (g : Fin m → Rat) :
    sumF n f * sumF m g = sumF n (fun i => sumF m (fun j => f i * g j)) := by
  have h1 : sumF m g * sumF n f = sumF n (fun i => sumF m g * f i) := by
    simpa using (sumF_mul_left n (sumF m g) f).symm
  have h2 : ∀ i, sumF m g * f i = sumF m (fun j => f i * g j) := by
    intro i
    have : f i * sumF m g = sumF m (fun j => f i * g j) := by
      simpa using (sumF_mul_left m (f i) g).symm
    grind
  have h3 : sumF n (fun i => sumF m g * f i) = sumF n (fun i => sumF m (fun j => f i * g j)) :=
    sumF_congr n _ _ h2
  grind


theorem sumF_swap3_jkl (p q : Nat) (f : Fin q → Fin p → Fin p → Rat) :
    sumF q (fun j => sumF p (fun k => sumF p (fun l => f j k l))) =
      sumF p (fun k => sumF p (fun l => sumF q (fun j => f j k l))) := by
  have h1 : sumF q (fun j => sumF p (fun k => sumF p (fun l => f j k l))) =
      sumF p (fun k => sumF q (fun j => sumF p (fun l => f j k l))) :=
    sumF_swap q p (fun j k => sumF p (fun l => f j k l))
  rw [h1]
  apply sumF_congr
  intro k
  exact sumF_swap q p (fun j l => f j k l)

theorem reorder_ijkl (p q : Nat) (f : Fin p → Fin q → Fin p → Fin p → Rat) :
    sumF p (fun i => sumF q (fun j => sumF p (fun k => sumF p (fun l => f i j k l)))) =
      sumF p (fun k => sumF p (fun l => sumF p (fun i => sumF q (fun j => f i j k l)))) := by
  have hstep : sumF p (fun i => sumF q (fun j => sumF p (fun k => sumF p (fun l => f i j k l)))) =
      sumF p (fun i => sumF p (fun k => sumF p (fun l => sumF q (fun j => f i j k l)))) := by
    apply sumF_congr
    intro i
    exact sumF_swap3_jkl p q (fun j k l => f i j k l)
  rw [hstep]
  -- sum_i sum_k sum_l g = sum_k sum_i sum_l g = sum_k sum_l sum_i g
  have h2 := sumF_swap p p (fun i k => sumF p (fun l => sumF q (fun j => f i j k l)))
  rw [h2]
  apply sumF_congr
  intro k
  exact sumF_swap p p (fun i l => sumF q (fun j => f i j k l))


abbrev Mat (n m : Nat) := Fin n → Fin m → Rat
def frobSq (n m : Nat) (A : Mat n m) : Rat :=
  sumF n (fun i => sumF m (fun j => A i j * A i j))
def mul (n m p : Nat) (A : Mat n m) (B : Mat m p) : Mat n p :=
  fun i j => sumF m (fun k => A i k * B k j)
def colDot (p : Nat) (Q : Mat p p) (k l : Fin p) : Rat :=
  sumF p (fun i => Q i k * Q i l)
def kron (p : Nat) (k l : Fin p) : Rat := if k = l then 1 else 0

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
      rw [h0, sumF_zero]; grind
    | succ i' =>
      have h1 : sumF n (fun j => (if i'.succ = j.succ then (1 : Rat) else 0) * f j.succ)
          = sumF n (fun j => (if i' = j then (1 : Rat) else 0) * f j.succ) := by
        apply sumF_congr; intro j; simp [Fin.succ_inj]
      rw [h1, ih i' (fun j => f j.succ)]
      have : i'.succ ≠ (0 : Fin (n+1)) := Fin.succ_ne_zero i'
      simp [this]; grind

theorem sumF_kron_contract (n : Nat) (g : Fin n → Fin n → Rat) :
    sumF n (fun k => sumF n (fun l => kron n k l * g k l)) = sumF n (fun k => g k k) := by
  apply sumF_congr
  intro k
  have : sumF n (fun l => kron n k l * g k l) = g k k := by
    have h := sumF_delta n k (fun l => g k l)
    have heq : ∀ l, kron n k l * g k l = (if k = l then (1 : Rat) else 0) * g k l := by
      intro l; unfold kron; grind
    rw [sumF_congr n _ _ heq]
    exact h
  exact this

theorem frobSq_left (p q : Nat) (Q : Mat p p) (M : Mat p q)
    (hQ : ∀ k l, colDot p Q k l = kron p k l) :
    frobSq p q (mul p p q Q M) = frobSq p q M := by
  unfold frobSq mul colDot at *
  have hexp : sumF p (fun i => sumF q (fun j =>
      (sumF p (fun k => Q i k * M k j)) * (sumF p (fun l => Q i l * M l j))))
      = sumF p (fun i => sumF q (fun j =>
          sumF p (fun k => sumF p (fun l => (Q i k * M k j) * (Q i l * M l j))))) := by
    apply sumF_congr; intro i; apply sumF_congr; intro j
    simpa using sumF_mul_sum p p (fun k => Q i k * M k j) (fun l => Q i l * M l j)
  rw [hexp]
  have hord := reorder_ijkl p q (fun i j k l => (Q i k * M k j) * (Q i l * M l j))
  rw [hord]
  have hfac : ∀ k l, sumF p (fun i => sumF q (fun j => (Q i k * M k j) * (Q i l * M l j)))
      = (sumF p (fun i => Q i k * Q i l)) * (sumF q (fun j => M k j * M l j)) := by
    intro k l
    have hcomm : ∀ i j, (Q i k * M k j) * (Q i l * M l j) = (Q i k * Q i l) * (M k j * M l j) := by
      intro i j; grind
    have h1 : sumF p (fun i => sumF q (fun j => (Q i k * M k j) * (Q i l * M l j)))
        = sumF p (fun i => sumF q (fun j => (Q i k * Q i l) * (M k j * M l j))) := by
      apply sumF_congr; intro i; apply sumF_congr; intro j; exact hcomm i j
    rw [h1]
    have h2 : sumF p (fun i => sumF q (fun j => (Q i k * Q i l) * (M k j * M l j)))
        = sumF p (fun i => (Q i k * Q i l) * sumF q (fun j => M k j * M l j)) := by
      apply sumF_congr; intro i
      exact sumF_mul_left q (Q i k * Q i l) (fun j => M k j * M l j)
    rw [h2]
    have hcomm2 : sumF p (fun i => (Q i k * Q i l) * sumF q (fun j => M k j * M l j))
        = sumF p (fun i => sumF q (fun j => M k j * M l j) * (Q i k * Q i l)) := by
      apply sumF_congr; intro i; grind
    rw [hcomm2]
    have h3 := sumF_mul_left p (sumF q (fun j => M k j * M l j)) (fun i => Q i k * Q i l)
    grind
  have hfac' : sumF p (fun k => sumF p (fun l =>
        sumF p (fun i => sumF q (fun j => (Q i k * M k j) * (Q i l * M l j)))))
      = sumF p (fun k => sumF p (fun l =>
          (sumF p (fun i => Q i k * Q i l)) * (sumF q (fun j => M k j * M l j)))) := by
    apply sumF_congr; intro k; apply sumF_congr; intro l; exact hfac k l
  rw [hfac']
  have hQ' : sumF p (fun k => sumF p (fun l =>
        (sumF p (fun i => Q i k * Q i l)) * (sumF q (fun j => M k j * M l j))))
      = sumF p (fun k => sumF p (fun l => kron p k l * (sumF q (fun j => M k j * M l j)))) := by
    apply sumF_congr; intro k; apply sumF_congr; intro l
    have := hQ k l
    grind
  rw [hQ']
  exact sumF_kron_contract p (fun k l => sumF q (fun j => M k j * M l j))


def transpose (n m : Nat) (A : Mat n m) : Mat m n := fun i j => A j i

theorem frobSq_transpose (n m : Nat) (A : Mat n m) :
    frobSq m n (transpose n m A) = frobSq n m A := by
  unfold frobSq transpose
  exact (sumF_swap n m (fun i j => A i j * A i j)).symm

theorem transpose_mul (n m p : Nat) (A : Mat n m) (B : Mat m p) :
    transpose n p (mul n m p A B) = mul p m n (transpose m p B) (transpose n m A) := by
  unfold transpose mul
  funext i j
  -- sum_k B k i * A j k, while (A B)^T i j = sum_k A j k * B k i
  have : sumF m (fun k => A j k * B k i) = sumF m (fun k => B k i * A j k) := by
    apply sumF_congr; intro k; grind
  simpa using this

def rowDot (p : Nat) (Q : Mat p p) (k l : Fin p) : Rat :=
  sumF p (fun j => Q k j * Q l j)

theorem frobSq_right (p q : Nat) (M : Mat q p) (Q : Mat p p)
    (hrow : ∀ k l, rowDot p Q k l = kron p k l) :
    frobSq q p (mul q p p M Q) = frobSq q p M := by
  rw [← frobSq_transpose q p (mul q p p M Q)]
  rw [transpose_mul]
  have hcol : ∀ k l, colDot p (transpose p p Q) k l = kron p k l := by
    intro k l
    unfold colDot transpose rowDot at *
    simpa using hrow k l
  have hleft := frobSq_left p q (transpose p p Q) (transpose q p M) hcol
  have hMT : frobSq p q (transpose q p M) = frobSq q p M := frobSq_transpose q p M
  -- frob (Q^T M^T) = frob (M^T) = frob M
  grind


def smulMat (n m : Nat) (c : Rat) (A : Mat n m) : Mat n m := fun i j => c * A i j

theorem frobSq_smul (n m : Nat) (c : Rat) (A : Mat n m) :
    frobSq n m (smulMat n m c A) = (c * c) * frobSq n m A := by
  unfold frobSq smulMat
  have h : ∀ i, sumF m (fun j => (c * A i j) * (c * A i j))
      = (c * c) * sumF m (fun j => A i j * A i j) := by
    intro i
    have h1 : sumF m (fun j => (c * A i j) * (c * A i j))
        = sumF m (fun j => (c * c) * (A i j * A i j)) := by
      apply sumF_congr; intro j; grind
    rw [h1, sumF_mul_left]
  have h2 : sumF n (fun i => sumF m (fun j => (c * A i j) * (c * A i j)))
      = sumF n (fun i => (c * c) * sumF m (fun j => A i j * A i j)) := sumF_congr n _ _ h
  rw [h2, sumF_mul_left]

def gramFrob (n p q : Nat) (X : Mat n p) (Y : Mat n q) : Rat :=
  frobSq p q (mul p n q (transpose n p X) Y)

theorem sumF_const (n : Nat) (c : Rat) : sumF n (fun _ => c) = (n : Rat) * c := by
  induction n with
  | zero => simp only [sumF]; grind
  | succ n ih => simp only [sumF]; rw [ih]; grind

theorem gramFrob_smul_left (n p q : Nat) (a : Rat) (X : Mat n p) (Y : Mat n q) :
    gramFrob n p q (smulMat n p a X) Y = (a * a) * gramFrob n p q X Y := by
  unfold gramFrob
  have hmat : mul p n q (transpose n p (smulMat n p a X)) Y
      = smulMat p q a (mul p n q (transpose n p X) Y) := by
    funext u v
    unfold mul smulMat transpose
    have : sumF n (fun k => (a * X k u) * Y k v) = sumF n (fun k => a * (X k u * Y k v)) := by
      apply sumF_congr; intro k; grind
    rw [this]
    exact sumF_mul_left n a (fun k => X k u * Y k v)
  rw [hmat]
  exact frobSq_smul p q a _

def colMean (n m : Nat) (A : Mat n m) (j : Fin m) : Rat :=
  sumF n (fun i => A i j) / (n : Rat)

def centerCols (n m : Nat) (A : Mat n m) : Mat n m :=
  fun i j => A i j - colMean n m A j

def addShift (n m : Nat) (A : Mat n m) (b : Fin m → Rat) : Mat n m :=
  fun i j => A i j + b j

theorem natCast_ne_zero {n : Nat} (h : n ≠ 0) : (n : Rat) ≠ 0 := by
  cases n with
  | zero => exact (h rfl).elim
  | succ n =>
    intro hz
    have hpos : 0 ≤ (n : Rat) := by
      induction n with
      | zero => simp
      | succ n ih =>
        simp
        grind
    have : (n : Rat) + 1 = 0 := by simpa using hz
    grind

theorem center_kills_shift (n m : Nat) (A : Mat n m) (b : Fin m → Rat) (hn : n ≠ 0) :
    centerCols n m (addShift n m A b) = centerCols n m A := by
  funext i j
  unfold centerCols addShift colMean
  have hsum : sumF n (fun i => A i j + b j) = sumF n (fun i => A i j) + (n : Rat) * b j := by
    have h1 : sumF n (fun i => A i j + b j) = sumF n (fun i => A i j) + sumF n (fun _ => b j) :=
      sumF_add n (fun i => A i j) (fun _ => b j)
    rw [h1, sumF_const]
  have hnrat : (n : Rat) ≠ 0 := natCast_ne_zero hn
  -- (s + n*b)/n = s/n + b, so the added shift cancels.
  have hdiv : (sumF n (fun i => A i j) + (n : Rat) * b j) / (n : Rat)
      = sumF n (fun i => A i j) / (n : Rat) + b j := by
    grind
  grind

theorem center_smul (n m : Nat) (c : Rat) (A : Mat n m) :
    centerCols n m (smulMat n m c A) = smulMat n m c (centerCols n m A) := by
  funext i j
  unfold centerCols smulMat colMean
  rw [sumF_mul_left]
  grind

theorem center_right (n p : Nat) (A : Mat n p) (Q : Mat p p) :
    centerCols n p (mul n p p A Q) = mul n p p (centerCols n p A) Q := by
  funext i j
  unfold centerCols mul colMean
  have hsum : sumF n (fun r => sumF p (fun k => A r k * Q k j))
      = sumF p (fun k => sumF n (fun r => A r k) * Q k j) := by
    have hswap := sumF_swap n p (fun r k => A r k * Q k j)
    rw [hswap]
    apply sumF_congr
    intro k
    have : sumF n (fun r => A r k * Q k j) = sumF n (fun r => A r k) * Q k j := by
      have hcomm : sumF n (fun r => A r k * Q k j) = sumF n (fun r => Q k j * A r k) := by
        apply sumF_congr; intro r; grind
      rw [hcomm, sumF_mul_left]
      grind
    exact this
  -- Goal, after unfolding: (X Q) centered equals (center X) Q.
  have hsub :
      (sumF n (fun r => sumF p (fun k => A r k * Q k j))) / (n : Rat)
        = (sumF p (fun k => (sumF n (fun r => A r k)) * Q k j)) / (n : Rat) := by
    exact congrArg (fun z => z / (n : Rat)) hsum
  have h1 : ∀ k, (A i k - sumF n (fun r => A r k) / (n : Rat)) * Q k j
      = A i k * Q k j + (-1) * ((sumF n (fun r => A r k) * Q k j) / (n : Rat)) := by
    intro k; grind
  have hsummand :
      sumF p (fun k => (A i k - sumF n (fun r => A r k) / (n : Rat)) * Q k j)
        = sumF p (fun k => A i k * Q k j)
          + sumF p (fun k => (-1) * ((sumF n (fun r => A r k) * Q k j) / (n : Rat))) := by
    rw [sumF_congr p _ _ h1, sumF_add]
  rw [hsummand]
  have hpull :
      sumF p (fun k => (-1) * ((sumF n (fun r => A r k) * Q k j) / (n : Rat)))
        = (-1) * sumF p (fun k => (sumF n (fun r => A r k) * Q k j) / (n : Rat)) :=
    sumF_mul_left p (-1) (fun k => (sumF n (fun r => A r k) * Q k j) / (n : Rat))
  rw [hpull]
  have hscale : sumF p (fun k => (sumF n (fun r => A r k) * Q k j) / (n : Rat))
      = (sumF p (fun k => (sumF n (fun r => A r k)) * Q k j)) / (n : Rat) := by
    have hk : ∀ k, (sumF n (fun r => A r k) * Q k j) / (n : Rat)
        = (1 / (n : Rat)) * ((sumF n (fun r => A r k)) * Q k j) := by
      intro k; grind
    rw [sumF_congr p _ _ hk, sumF_mul_left]
    grind
  grind


theorem mul_assoc (n m p r : Nat) (A : Mat n m) (B : Mat m p) (C : Mat p r) :
    mul n p r (mul n m p A B) C = mul n m r A (mul m p r B C) := by
  funext i j
  unfold mul
  have hL : sumF p (fun t => (sumF m (fun k => A i k * B k t)) * C t j)
      = sumF p (fun t => sumF m (fun k => C t j * (A i k * B k t))) := by
    apply sumF_congr; intro t
    have hcomm : (sumF m (fun k => A i k * B k t)) * C t j
        = C t j * sumF m (fun k => A i k * B k t) := by grind
    have hmul := (sumF_mul_left m (C t j) (fun k => A i k * B k t)).symm
    exact hcomm.trans hmul
  have hL2 : sumF p (fun t => sumF m (fun k => C t j * (A i k * B k t)))
      = sumF p (fun t => sumF m (fun k => (A i k * B k t) * C t j)) := by
    apply sumF_congr; intro t; apply sumF_congr; intro k; grind
  rw [hL, hL2]
  have hs := sumF_swap p m (fun t k => (A i k * B k t) * C t j)
  rw [hs]
  apply sumF_congr; intro k
  have hcomm : sumF p (fun t => (A i k * B k t) * C t j)
      = sumF p (fun t => A i k * (B k t * C t j)) := by
    apply sumF_congr; intro t; grind
  rw [hcomm]
  exact sumF_mul_left p (A i k) (fun t => B k t * C t j)

def negI (p : Nat) : Mat p p := fun i j => if i = j then -1 else 0

theorem sumF_unique (n : Nat) (k : Fin n) (f : Fin n → Rat)
    (h : ∀ i, i ≠ k → f i = 0) : sumF n f = f k := by
  have hrep : ∀ i, f i = (if k = i then (1 : Rat) else 0) * f i := by
    intro i
    by_cases hk : k = i
    · simp [hk]
    · have : f i = 0 := h i (Ne.symm hk)
      simp [hk, this]
  rw [sumF_congr n _ _ hrep, sumF_delta]

theorem negI_cols (p : Nat) (k l : Fin p) : colDot p (negI p) k l = kron p k l := by
  unfold colDot negI kron
  have h : ∀ i, i ≠ k → (if i = k then (-1 : Rat) else 0) * (if i = l then -1 else 0) = 0 := by
    intro i hik; simp [hik]
  rw [sumF_unique p k _ h]
  grind

theorem negI_rows (p : Nat) (k l : Fin p) : rowDot p (negI p) k l = kron p k l := by
  unfold rowDot negI kron
  have h : ∀ j, j ≠ k → (if k = j then (-1 : Rat) else 0) * (if l = j then -1 else 0) = 0 := by
    intro j hjk
    have hk : ¬ k = j := hjk.symm
    simp [hk]
  rw [sumF_unique p k _ h]
  grind

structure IsOrth (p : Nat) (Q : Mat p p) : Prop where
  cols : ∀ k l, colDot p Q k l = kron p k l
  rows : ∀ k l, rowDot p Q k l = kron p k l

theorem negI_orth (p : Nat) : IsOrth p (negI p) :=
  ⟨fun k l => negI_cols p k l, fun k l => negI_rows p k l⟩

#print axioms mul_assoc
#print axioms negI_orth

def ckaSq (n p q : Nat) (X : Mat n p) (Y : Mat n q) : Rat :=
  let N := gramFrob n p q X Y
  let Sx := gramFrob n p p X X
  let Sy := gramFrob n q q Y Y
  (N * N) / (Sx * Sy)

theorem gram_after_right (n p q : Nat) (X : Mat n p) (Y : Mat n q) (Q : Mat p p)
    (hQ : IsOrth p Q) :
    gramFrob n p q (mul n p p X Q) Y = gramFrob n p q X Y := by
  unfold gramFrob
  have hT : transpose n p (mul n p p X Q) = mul p p n (transpose p p Q) (transpose n p X) :=
    transpose_mul n p p X Q
  have hassoc : mul p n q (transpose n p (mul n p p X Q)) Y
      = mul p p q (transpose p p Q) (mul p n q (transpose n p X) Y) := by
    rw [hT]
    exact mul_assoc p p n q (transpose p p Q) (transpose n p X) Y
  rw [hassoc]
  have hcol : ∀ k l, colDot p (transpose p p Q) k l = kron p k l := by
    intro k l
    simpa [colDot, rowDot, transpose] using hQ.rows k l
  exact frobSq_left p q (transpose p p Q) (mul p n q (transpose n p X) Y) hcol

theorem gram_after_right_on_Y (n p q : Nat) (X : Mat n p) (Y : Mat n q) (Q : Mat q q)
    (hQ : IsOrth q Q) :
    gramFrob n p q X (mul n q q Y Q) = gramFrob n p q X Y := by
  unfold gramFrob
  have hmul : mul p n q (transpose n p X) (mul n q q Y Q)
      = mul p q q (mul p n q (transpose n p X) Y) Q := by
    exact (mul_assoc p n q q (transpose n p X) Y Q).symm
  rw [hmul]
  exact frobSq_right q p (mul p n q (transpose n p X) Y) Q hQ.rows

theorem gram_self_orth (n p : Nat) (X : Mat n p) (Q : Mat p p) (hQ : IsOrth p Q) :
    gramFrob n p p (mul n p p X Q) (mul n p p X Q) = gramFrob n p p X X := by
  have h1 := gram_after_right n p p X (mul n p p X Q) Q hQ
  have h2 := gram_after_right_on_Y n p p X X Q hQ
  grind

theorem gram_both_scaled (n p : Nat) (Z : Mat n p) (a : Rat) :
    gramFrob n p p (smulMat n p a Z) (smulMat n p a Z) = ((a * a) * (a * a)) * gramFrob n p p Z Z := by
  have hL := gramFrob_smul_left n p p a Z (smulMat n p a Z)
  have hR : gramFrob n p p Z (smulMat n p a Z) = (a * a) * gramFrob n p p Z Z := by
    unfold gramFrob
    have hmat : mul p n p (transpose n p Z) (smulMat n p a Z)
        = smulMat p p a (mul p n p (transpose n p Z) Z) := by
      funext u v
      unfold mul smulMat
      have : sumF n (fun k => transpose n p Z u k * (a * Z k v))
          = sumF n (fun k => a * (transpose n p Z u k * Z k v)) := by
        apply sumF_congr; intro k; grind
      rw [this]
      exact sumF_mul_left n a (fun k => transpose n p Z u k * Z k v)
    rw [hmat, frobSq_smul]
  grind

/-- Squared linear CKA is unchanged by `X ↦ a • X Q` when `Q` is orthogonal and `a ≠ 0`. -/
theorem ckaSq_right_scale_orth (n p q : Nat) (X : Mat n p) (Y : Mat n q) (Q : Mat p p) (a : Rat)
    (ha : a ≠ 0) (hQ : IsOrth p Q)
    (hSx : gramFrob n p p X X ≠ 0) (hSy : gramFrob n q q Y Y ≠ 0) :
    ckaSq n p q (smulMat n p a (mul n p p X Q)) Y = ckaSq n p q X Y := by
  unfold ckaSq
  have hN : gramFrob n p q (smulMat n p a (mul n p p X Q)) Y
      = (a * a) * gramFrob n p q X Y := by
    have h1 := gramFrob_smul_left n p q a (mul n p p X Q) Y
    have h2 := gram_after_right n p q X Y Q hQ
    grind
  have hS : gramFrob n p p (smulMat n p a (mul n p p X Q)) (smulMat n p a (mul n p p X Q))
      = ((a * a) * (a * a)) * gramFrob n p p X X := by
    have h1 := gram_both_scaled n p (mul n p p X Q) a
    have h2 := gram_self_orth n p X Q hQ
    grind
  have ha4 : (a * a) * (a * a) ≠ 0 := by grind
  grind

#print axioms ckaSq_right_scale_orth
#print axioms negI_orth
#print axioms gram_self_orth

theorem center_full (n p : Nat) (X : Mat n p) (Q : Mat p p) (b : Fin p → Rat) (a : Rat) (hn : n ≠ 0) :
    centerCols n p (addShift n p (smulMat n p a (mul n p p X Q)) b)
      = smulMat n p a (mul n p p (centerCols n p X) Q) := by
  rw [center_kills_shift n p (smulMat n p a (mul n p p X Q)) b hn,
      center_smul n p a (mul n p p X Q),
      center_right n p X Q]

theorem ckaSq_shift_scale_orth (n p q : Nat) (X : Mat n p) (Y : Mat n q)
    (Q : Mat p p) (b : Fin p → Rat) (a : Rat)
    (hn : n ≠ 0) (ha : a ≠ 0) (hQ : IsOrth p Q)
    (hSx : gramFrob n p p (centerCols n p X) (centerCols n p X) ≠ 0)
    (hSy : gramFrob n q q (centerCols n q Y) (centerCols n q Y) ≠ 0) :
    ckaSq n p q (centerCols n p (addShift n p (smulMat n p a (mul n p p X Q)) b)) (centerCols n q Y)
      = ckaSq n p q (centerCols n p X) (centerCols n q Y) := by
  have hcent := center_full n p X Q b a hn
  rw [hcent]
  exact ckaSq_right_scale_orth n p q (centerCols n p X) (centerCols n q Y) Q a ha hQ hSx hSy

/-! ## One-dimensional linear CKA is squared Pearson correlation -/

def dot (n : Nat) (x y : Fin n → Rat) : Rat := sumF n (fun i => x i * y i)
def mean1 (n : Nat) (x : Fin n → Rat) : Rat := sumF n x / (n : Rat)
def center1 (n : Nat) (x : Fin n → Rat) : Fin n → Rat := fun i => x i - mean1 n x

def cov1 (n : Nat) (x y : Fin n → Rat) : Rat := dot n (center1 n x) (center1 n y) / (n : Rat)
def sqPearson (n : Nat) (x y : Fin n → Rat) : Rat :=
  let c := cov1 n x y
  (c * c) / (cov1 n x x * cov1 n y y)

def linearCKA1d (n : Nat) (x y : Fin n → Rat) : Rat :=
  let xc := center1 n x
  let yc := center1 n y
  (dot n xc yc * dot n xc yc) / (dot n xc xc * dot n yc yc)

theorem linearCKA1d_eq_sqPearson (n : Nat) (x y : Fin n → Rat) (hn : n ≠ 0)
    (hx : dot n (center1 n x) (center1 n x) ≠ 0)
    (hy : dot n (center1 n y) (center1 n y) ≠ 0) :
    linearCKA1d n x y = sqPearson n x y := by
  unfold linearCKA1d sqPearson cov1
  have hn0 : (n : Rat) ≠ 0 := natCast_ne_zero hn
  grind

theorem center1_affine (n : Nat) (a b : Rat) (x : Fin n → Rat) (hn : n ≠ 0) :
    center1 n (fun i => a * x i + b) = fun i => a * center1 n x i := by
  funext i
  unfold center1 mean1
  have hsum : sumF n (fun i => a * x i + b) = a * sumF n x + (n : Rat) * b := by
    rw [sumF_add, sumF_mul_left, sumF_const]
  rw [hsum]
  have hn0 : (n : Rat) ≠ 0 := natCast_ne_zero hn
  grind

theorem sqPearson_scale_shift (n : Nat) (a b : Rat) (x y : Fin n → Rat)
    (hn : n ≠ 0) (ha : a ≠ 0)
    (hx : dot n (center1 n x) (center1 n x) ≠ 0)
    (hy : dot n (center1 n y) (center1 n y) ≠ 0) :
    sqPearson n (fun i => a * x i + b) y = sqPearson n x y := by
  have hc := center1_affine n a b x hn
  unfold sqPearson cov1 at *
  have hn0 : (n : Rat) ≠ 0 := natCast_ne_zero hn
  -- Dots against the centered image scale by a and a*a.
  have hdoty : dot n (center1 n (fun i => a * x i + b)) (center1 n y)
      = a * dot n (center1 n x) (center1 n y) := by
    rw [hc]
    unfold dot
    have hmul : sumF n (fun i => (a * center1 n x i) * center1 n y i)
        = sumF n (fun i => a * (center1 n x i * center1 n y i)) := by
      apply sumF_congr; intro i; grind
    rw [hmul]
    exact sumF_mul_left n a (fun i => center1 n x i * center1 n y i)
  have hdotx : dot n (center1 n (fun i => a * x i + b)) (center1 n (fun i => a * x i + b))
      = (a * a) * dot n (center1 n x) (center1 n x) := by
    rw [hc]
    unfold dot
    have h1 : sumF n (fun i => (a * center1 n x i) * (a * center1 n x i))
        = sumF n (fun i => (a * a) * (center1 n x i * center1 n x i)) := by
      apply sumF_congr; intro i; grind
    rw [h1, sumF_mul_left]
  grind

/-! ## One-dimensional normalized OPD decreases exactly when squared correlation increases. -/

def opdSqFromCorr (r : Rat) : Rat := 2 - 2 * r

theorem opdSq_anti (r1 r2 : Rat) (h1 : 0 ≤ r1) (hlt : r1 < r2) :
    opdSqFromCorr r2 < opdSqFromCorr r1 := by
  unfold opdSqFromCorr
  grind

theorem sq_mono_nonneg (a b : Rat) (_ha : 0 ≤ a) (_hb : 0 ≤ b) (h : a ≤ b) : a * a ≤ b * b := by
  have hdiff : b * b - a * a = (b - a) * (b + a) := by grind
  have h1 : 0 ≤ b - a := by grind
  have h2 : 0 ≤ b + a := by grind
  have hnn : 0 ≤ (b - a) * (b + a) := Rat.mul_nonneg h1 h2
  grind

theorem corr_lt_of_sq_lt (r1 r2 : Rat) (h1 : 0 ≤ r1) (h2 : 0 ≤ r2) (hs : r1 * r1 < r2 * r2) :
    r1 < r2 := by
  have hnot : ¬ r2 ≤ r1 := by
    intro hle
    have hsq := sq_mono_nonneg r2 r1 h2 h1 hle
    grind
  grind

/-- If `r` is the nonnegative square root of linear CKA, normalized 1D OPD squared is `2 - 2 r`. -/
theorem opdSq_of_cka (cka r : Rat) (hr : 0 ≤ r) (hsq : r * r = cka) :
    opdSqFromCorr r = 2 - 2 * r := by
  unfold opdSqFromCorr
  grind

/-! ## Permutation matrices are orthogonal. -/

structure FinPerm (p : Nat) where
  fwd : Fin p → Fin p
  rev : Fin p → Fin p
  fwd_rev : ∀ i, rev (fwd i) = i
  rev_fwd : ∀ i, fwd (rev i) = i

def permMat {p : Nat} (e : FinPerm p) : Mat p p :=
  fun i j => if e.fwd i = j then (1 : Rat) else 0

theorem perm_rows {p : Nat} (e : FinPerm p) (k l : Fin p) :
    rowDot p (permMat e) k l = kron p k l := by
  unfold rowDot permMat kron
  let j0 := e.fwd k
  have hzero : ∀ j, j ≠ j0 → (if e.fwd k = j then (1 : Rat) else 0) * (if e.fwd l = j then 1 else 0) = 0 := by
    intro j hj
    have : e.fwd k ≠ j := hj.symm
    simp [this]
  rw [sumF_unique p j0 _ hzero]
  by_cases hkl : k = l
  · subst hkl
    simp [j0]
  · have hne : e.fwd k ≠ e.fwd l := by
      intro h
      have := congrArg e.rev h
      rw [e.fwd_rev, e.fwd_rev] at this
      exact hkl this
    have : ¬ e.fwd l = e.fwd k := hne.symm
    simp [j0, this, hkl]

theorem perm_cols {p : Nat} (e : FinPerm p) (k l : Fin p) :
    colDot p (permMat e) k l = kron p k l := by
  unfold colDot permMat kron
  let i0 := e.rev k
  have hzero : ∀ i, i ≠ i0 → (if e.fwd i = k then (1 : Rat) else 0) * (if e.fwd i = l then 1 else 0) = 0 := by
    intro i hi
    have hfk : e.fwd i ≠ k := by
      intro h
      have : i = e.rev k := by
        have := congrArg e.rev h
        rw [e.fwd_rev] at this
        exact this
      exact hi (by simpa [i0] using this)
    simp [hfk]
  rw [sumF_unique p i0 _ hzero]
  by_cases hkl : k = l
  · subst hkl
    simp [i0, e.rev_fwd]
  · have hne : e.fwd (e.rev k) ≠ l := by simpa [e.rev_fwd] using hkl
    simp [i0, hne, hkl]

theorem perm_orth {p : Nat} (e : FinPerm p) : IsOrth p (permMat e) :=
  ⟨fun k l => perm_cols e k l, fun k l => perm_rows e k l⟩

/-! ## Proper-score witnesses: reversal and a transposition strictly worsen Brier and RPS. -/

def brier3 (q0 q1 q2 : Rat) (y : Nat) : Rat :=
  let d0 : Rat := if y = 0 then 1 else 0
  let d1 : Rat := if y = 1 then 1 else 0
  let d2 : Rat := if y = 2 then 1 else 0
  (q0 - d0) * (q0 - d0) + (q1 - d1) * (q1 - d1) + (q2 - d2) * (q2 - d2)

def rps3 (q0 q1 q2 : Rat) (y : Nat) : Rat :=
  let f0 := q0
  let f1 := q0 + q1
  let g0 : Rat := if y = 0 then 1 else 0
  let g1 : Rat := if y ≤ 1 then 1 else 0
  (f0 - g0) * (f0 - g0) + (f1 - g1) * (f1 - g1)

theorem reversal_brier_strict :
    brier3 (1/2) (1/3) (1/6) 0 < brier3 (1/6) (1/3) (1/2) 0 := by
  unfold brier3; grind

theorem reversal_rps_strict :
    rps3 (1/2) (1/3) (1/6) 0 < rps3 (1/6) (1/3) (1/2) 0 := by
  unfold rps3; grind

theorem transposition_brier_strict :
    brier3 (1/2) (1/3) (1/6) 0 < brier3 (1/3) (1/2) (1/6) 0 := by
  unfold brier3; grind

theorem transposition_rps_strict :
    rps3 (1/2) (1/3) (1/6) 0 < rps3 (1/3) (1/2) (1/6) 0 := by
  unfold rps3; grind

/-- Any strictly decreasing local score of the true-class probability, including cross-entropy,
    strictly worsens when a reversal or a swap moves mass off the realised class. -/
theorem local_score_worsens (score : Rat → Rat) (good bad : Rat)
    (hdec : ∀ a b, 0 < a → a < b → score b < score a)
    (hpos : 0 < bad) (hlt : bad < good) :
    score good < score bad :=
  hdec bad good hpos hlt

theorem reversal_true_class_prob : (1 / 6 : Rat) < 1 / 2 := by grind
theorem transposition_true_class_prob : (1 / 3 : Rat) < 1 / 2 := by grind

/-! ## Conditional-variance collapse: exact finite sample. -/

def wX : Fin 6 → Rat
  | 0 => 1 | 1 => -1 | 2 => 0 | 3 => 4 | 4 => 2 | 5 => 3
def wY : Fin 6 → Rat
  | 0 => 0 | 1 => 0 | 2 => 1 | 3 => 3 | 4 => 3 | 5 => 4
def wXc : Fin 6 → Rat
  | 0 => 0 | 1 => 0 | 2 => 0 | 3 => 3 | 4 => 3 | 5 => 3

theorem collapse_cka_strict :
    linearCKA1d 6 wX wY < linearCKA1d 6 wXc wY := by
  simp only [linearCKA1d, center1, mean1, dot, sumF, wX, wY, wXc]
  grind

theorem collapse_cka_values :
    linearCKA1d 6 wX wY = 2187 / 3115 ∧ linearCKA1d 6 wXc wY = 81 / 89 := by
  simp only [linearCKA1d, center1, mean1, dot, sumF, wX, wY, wXc]
  grind

/-- Point mass at the conditional mean class versus the calibrated 3-class law on outcomes 0,1,1,2. -/
theorem collapse_pointmass_brier :
    (brier3 (1/4) (1/2) (1/4) 0 + brier3 (1/4) (1/2) (1/4) 1
      + brier3 (1/4) (1/2) (1/4) 1 + brier3 (1/4) (1/2) (1/4) 2) / 4
    < (brier3 0 1 0 0 + brier3 0 1 0 1 + brier3 0 1 0 1 + brier3 0 1 0 2) / 4 := by
  unfold brier3; grind

theorem collapse_pointmass_rps :
    (rps3 (1/4) (1/2) (1/4) 0 + rps3 (1/4) (1/2) (1/4) 1
      + rps3 (1/4) (1/2) (1/4) 1 + rps3 (1/4) (1/2) (1/4) 2) / 4
    < (rps3 0 1 0 0 + rps3 0 1 0 1 + rps3 0 1 0 1 + rps3 0 1 0 2) / 4 := by
  unfold rps3; grind

theorem collapse_pointmass_ce_support :
    (0 : Rat) < 1 / 4 ∧ (0 : Rat) < 1 / 2 ∧ (1 / 4 : Rat) < 1 := by
  grind

/-- Hypotheses of the invariance theorem can hold: sign flip on a perfectly correlated pair. -/
def instX : Fin 2 → Rat | 0 => 0 | 1 => 2
def instY : Fin 2 → Rat | 0 => 1 | 1 => 5

theorem instance_perfect_correlation :
    sqPearson 2 instX instY = 1 := by
  simp only [sqPearson, cov1, center1, mean1, dot, sumF, instX, instY]
  grind

theorem instance_sign_shift_preserves :
    sqPearson 2 (fun i => (-2) * instX i + 5) instY = sqPearson 2 instX instY := by
  have hx : dot 2 (center1 2 instX) (center1 2 instX) ≠ 0 := by
    simp only [dot, center1, mean1, sumF, instX]; grind
  have hy : dot 2 (center1 2 instY) (center1 2 instY) ≠ 0 := by
    simp only [dot, center1, mean1, sumF, instY]; grind
  exact sqPearson_scale_shift 2 (-2) 5 instX instY (by decide) (by decide) hx hy

#print axioms linearCKA1d_eq_sqPearson
#print axioms sqPearson_scale_shift
#print axioms ckaSq_shift_scale_orth
#print axioms negI_orth
#print axioms perm_orth
#print axioms reversal_brier_strict
#print axioms reversal_rps_strict
#print axioms transposition_brier_strict
#print axioms transposition_rps_strict
#print axioms local_score_worsens
#print axioms collapse_cka_strict
#print axioms collapse_pointmass_brier
#print axioms collapse_pointmass_rps
#print axioms opdSq_anti
#print axioms corr_lt_of_sq_lt
#print axioms instance_perfect_correlation
