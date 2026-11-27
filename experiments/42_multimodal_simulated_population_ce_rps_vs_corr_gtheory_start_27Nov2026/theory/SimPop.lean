set_option linter.unusedSimpArgs false
set_option maxHeartbeats 800000
/-
  Simulated-population identities.

  A generalizability coefficient is a function of the item-mean vector and the
  item-variance vector only. Adding a constant to every coordinate does not
  change centered deviations, so it does not change Pearson correlation.
  Two explicit laws on {0,1,2,3,4,5} share the mean 5/2 and the second moment
  41/4, hence the variance 4, while the total-variation distance is 9/16 and
  the normalized ranked-probability-score gap is 9/512.

  Core Lean 4 only. No Mathlib. No `sorry`.
-/

def sumF : (n : Nat) → (Fin n → Rat) → Rat
  | 0, _ => 0
  | n + 1, f => f 0 + sumF n (fun i => f i.succ)

theorem sumF_add (n : Nat) (f g : Fin n → Rat) :
    sumF n (fun i => f i + g i) = sumF n f + sumF n g := by
  induction n with
  | zero =>
      simp only [sumF]
      grind
  | succ n ih =>
      simp only [sumF]
      rw [ih (fun i => f i.succ) (fun i => g i.succ)]
      grind

theorem sumF_const (n : Nat) (c : Rat) : sumF n (fun _ => c) = (n : Rat) * c := by
  induction n with
  | zero =>
      simp only [sumF]
      grind
  | succ n ih =>
      simp only [sumF]
      rw [ih]
      grind

theorem sumF_congr (n : Nat) (f g : Fin n → Rat) (h : ∀ i, f i = g i) :
    sumF n f = sumF n g := by
  have : f = g := funext h
  rw [this]

def meanOf (n : Nat) (x : Fin n → Rat) : Rat := sumF n x / (n : Rat)

def centered (n : Nat) (x : Fin n → Rat) (i : Fin n) : Rat := x i - meanOf n x

theorem add_div_right (a b d : Rat) : (a + b) / d = a / d + b / d := by
  show (a + b) * d⁻¹ = a * d⁻¹ + b * d⁻¹
  exact Rat.add_mul a b d⁻¹

theorem mul_div_cancel_of_ne_zero (n c : Rat) (hn : n ≠ 0) : (n * c) / n = c := by
  show (n * c) * n⁻¹ = c
  rw [Rat.mul_comm n c, Rat.mul_assoc, Rat.mul_inv_cancel n hn]
  exact Rat.mul_one c

/-- Adding a constant to every coordinate does not change a centered deviation. -/
theorem centered_add_const (n : Nat) (hn : n ≠ 0) (c : Rat) (x : Fin n → Rat) (i : Fin n) :
    centered n (fun j => x j + c) i = centered n x i := by
  unfold centered meanOf
  have hsum : sumF n (fun j => x j + c) = sumF n x + (n : Rat) * c := by
    have h1 : sumF n (fun j => x j + c) = sumF n x + sumF n (fun _ => c) := by
      simpa using sumF_add n x (fun _ => c)
    rw [h1, sumF_const]
  rw [hsum]
  have hn' : (n : Rat) ≠ 0 := by
    exact_mod_cast hn
  rw [add_div_right, mul_div_cancel_of_ne_zero (n : Rat) c hn']
  grind

def vu (n : Nat) (m : Fin n → Rat) : Rat :=
  let mu := meanOf n m
  sumF n (fun i => (m i - mu) * (m i - mu)) / (n : Rat)

def ve (n : Nat) (v : Fin n → Rat) : Rat := sumF n v / (n : Rat)

def erho (n : Nat) (m v : Fin n → Rat) : Rat := vu n m / (vu n m + ve n v)

theorem erho_depends_only_on_moments
    (n : Nat) (m1 m2 v1 v2 : Fin n → Rat)
    (hm : ∀ i, m1 i = m2 i) (hv : ∀ i, v1 i = v2 i) :
    erho n m1 v1 = erho n m2 v2 := by
  rw [show m1 = m2 from funext hm, show v1 = v2 from funext hv]

theorem val3 : ((3 : Fin 6).val) = 3 := by rfl
theorem val4 : ((4 : Fin 6).val) = 4 := by rfl
theorem val5 : ((5 : Fin 6).val) = 5 := by rfl

def score (i : Fin 6) : Rat := i.val

def m1 (i : Fin 6) : Rat :=
  match i with
  | 0 => 5 / 16
  | 1 => 0
  | 2 => 3 / 16
  | 3 => 3 / 16
  | 4 => 0
  | 5 => 5 / 16

def m2 (i : Fin 6) : Rat :=
  match i with
  | 0 => 7 / 32
  | 1 => 9 / 32
  | 2 => 0
  | 3 => 0
  | 4 => 9 / 32
  | 5 => 7 / 32

theorem m1_sums_to_one : sumF 6 m1 = 1 := by
  simp [sumF, m1]
  grind

theorem m2_sums_to_one : sumF 6 m2 = 1 := by
  simp [sumF, m2]
  grind

theorem m1_mean : sumF 6 (fun i => score i * m1 i) = 5 / 2 := by
  simp [sumF, m1, score, val3, val4, val5]
  grind

theorem m2_mean : sumF 6 (fun i => score i * m2 i) = 5 / 2 := by
  simp [sumF, m2, score, val3, val4, val5]
  grind

theorem m1_second : sumF 6 (fun i => score i * score i * m1 i) = 41 / 4 := by
  simp [sumF, m1, score, val3, val4, val5]
  grind

theorem m2_second : sumF 6 (fun i => score i * score i * m2 i) = 41 / 4 := by
  simp [sumF, m2, score, val3, val4, val5]
  grind

theorem same_variance :
    (sumF 6 (fun i => score i * score i * m1 i) - (5 / 2) * (5 / 2))
      = (sumF 6 (fun i => score i * score i * m2 i) - (5 / 2) * (5 / 2)) := by
  rw [m1_second, m2_second]

theorem variance_is_four :
    sumF 6 (fun i => score i * score i * m1 i) - (5 / 2) * (5 / 2) = 4 := by
  rw [m1_second]
  grind

def absRat (a : Rat) : Rat := if a < 0 then -a else a

theorem tvd_m1_m2 :
    sumF 6 (fun i => absRat (m1 i - m2 i)) / 2 = 9 / 16 := by
  simp [sumF, m1, m2, absRat]
  grind

def cdfAt (p : Fin 6 → Rat) (t : Nat) : Rat :=
  sumF 6 (fun i => if i.val ≤ t then p i else 0)

def rpsGapTerm (q p : Fin 6 → Rat) : Rat :=
  ((cdfAt q 0 - cdfAt p 0) ^ 2 + (cdfAt q 1 - cdfAt p 1) ^ 2
    + (cdfAt q 2 - cdfAt p 2) ^ 2 + (cdfAt q 3 - cdfAt p 3) ^ 2
    + (cdfAt q 4 - cdfAt p 4) ^ 2) / 5

theorem rps_gap_m2_on_m1 : rpsGapTerm m2 m1 - rpsGapTerm m1 m1 = 9 / 512 := by
  simp [rpsGapTerm, cdfAt, sumF, m1, m2, val3, val4, val5]
  grind

theorem rps_gap_positive : rpsGapTerm m2 m1 - rpsGapTerm m1 m1 > 0 := by
  rw [rps_gap_m2_on_m1]
  grind

def panelMean : Fin 3 → Rat
  | 0 => 3 / 2
  | 1 => 5 / 2
  | 2 => 7 / 2

def panelVar : Fin 3 → Rat
  | 0 => 2
  | 1 => 4
  | 2 => 2

theorem panel_vu : vu 3 panelMean = 2 / 3 := by
  simp [vu, meanOf, sumF, panelMean]
  grind

theorem panel_ve : ve 3 panelVar = 8 / 3 := by
  simp [ve, sumF, panelVar]
  grind

theorem panel_erho : erho 3 panelMean panelVar = 1 / 5 := by
  simp [erho, panel_vu, panel_ve]
  grind

def bin3 : Fin 3 → Rat
  | 0 => 8 / 5
  | 1 => 2
  | 2 => 12 / 5

theorem strict_shift_is_invisible_to_centered (i : Fin 3) :
    centered 3 (fun j => bin3 j - 4 / 5) i = centered 3 bin3 i := by
  have hfun : (fun j => bin3 j - 4 / 5) = fun j => bin3 j + (-(4 / 5)) := by
    funext j
    rw [Rat.sub_eq_add_neg]
  rw [hfun]
  exact centered_add_const 3 (by decide) (-(4 / 5)) bin3 i

def lawA : Fin 6 → Rat
  | 0 => 11 / 24
  | 1 => 0
  | 2 => 1 / 8
  | 3 => 5 / 12
  | 4 => 0
  | 5 => 0

def lawB : Fin 6 → Rat
  | 0 => 5 / 12
  | 1 => 1 / 8
  | 2 => 0
  | 3 => 11 / 24
  | 4 => 0
  | 5 => 0

theorem lawA_mean : sumF 6 (fun i => score i * lawA i) = 3 / 2 := by
  simp [sumF, lawA, score, val3, val4, val5]
  grind

theorem lawB_mean : sumF 6 (fun i => score i * lawB i) = 3 / 2 := by
  simp [sumF, lawB, score, val3, val4, val5]
  grind

theorem lawA_second : sumF 6 (fun i => score i * score i * lawA i) = 17 / 4 := by
  simp [sumF, lawA, score, val3, val4, val5]
  grind

theorem lawB_second : sumF 6 (fun i => score i * score i * lawB i) = 17 / 4 := by
  simp [sumF, lawB, score, val3, val4, val5]
  grind

theorem hypotheses_can_hold :
    sumF 6 m1 = 1 ∧ sumF 6 m2 = 1 ∧
      sumF 6 (fun i => score i * m1 i) = sumF 6 (fun i => score i * m2 i) ∧
      sumF 6 (fun i => score i * score i * m1 i) = sumF 6 (fun i => score i * score i * m2 i) ∧
      sumF 6 (fun i => absRat (m1 i - m2 i)) / 2 = 9 / 16 ∧
      rpsGapTerm m2 m1 - rpsGapTerm m1 m1 = 9 / 512 ∧
      erho 3 panelMean panelVar = 1 / 5 := by
  refine ⟨m1_sums_to_one, m2_sums_to_one, ?_, ?_, tvd_m1_m2, rps_gap_m2_on_m1, panel_erho⟩
  · rw [m1_mean, m2_mean]
  · rw [m1_second, m2_second]

#print axioms sumF_add
#print axioms sumF_const
#print axioms centered_add_const
#print axioms erho_depends_only_on_moments
#print axioms m1_sums_to_one
#print axioms m1_mean
#print axioms m2_mean
#print axioms m1_second
#print axioms m2_second
#print axioms variance_is_four
#print axioms tvd_m1_m2
#print axioms rps_gap_m2_on_m1
#print axioms panel_erho
#print axioms strict_shift_is_invisible_to_centered
#print axioms lawA_mean
#print axioms lawB_mean
#print axioms hypotheses_can_hold
