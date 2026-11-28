import Mathlib

/-!
# Point 2 note: encoded squared-error scores (Prop. `prop:encoding`), Brier (Prop. `prop:onehot`),
RPS (Prop. `prop:rps`), and the scalar encoding.

Categories are a finite type `κ` (for RPS, `Fin (n+1)`, i.e. `K = n + 1` ordered ratings).
A probability vector is `p : κ → ℝ`, nonnegative with sum 1. An encoding is a matrix
`A : ρ → κ → ℝ` (rows `ρ`, finite). The encoded score is `S_A(q, y) = ‖A (q - e_y)‖²`.
-/

open Finset BigOperators

namespace Point2

set_option linter.unusedSectionVars false

variable {κ ρ : Type*} [Fintype κ] [DecidableEq κ] [Fintype ρ]

def IsProb (p : κ → ℝ) : Prop := (∀ k, 0 ≤ p k) ∧ ∑ k, p k = 1

/-- `(A v)_j`. -/
def mulVec' (A : ρ → κ → ℝ) (v : κ → ℝ) (j : ρ) : ℝ := ∑ k, A j k * v k

/-- one-hot vector `e_y`. -/
def onehot (y : κ) : κ → ℝ := fun k => if k = y then 1 else 0

/-- Encoded squared-error score `S_A(q, y) = ‖A (q - e_y)‖²`. -/
def encScore (A : ρ → κ → ℝ) (q : κ → ℝ) (y : κ) : ℝ :=
  ∑ j, (mulVec' A (fun k => q k - onehot y k) j) ^ 2

/-- Expected score under the truth `p`. -/
def expScore (A : ρ → κ → ℝ) (q p : κ → ℝ) : ℝ := ∑ y, p y * encScore A q y

/-- `tr(A Σ_p Aᵀ)` with `Σ_p = diag p - p pᵀ`, written row by row. -/
def traceTerm (A : ρ → κ → ℝ) (p : κ → ℝ) : ℝ :=
  ∑ j, ((∑ k, A j k ^ 2 * p k) - (mulVec' A p j) ^ 2)

lemma mulVec'_sub_onehot (A : ρ → κ → ℝ) (q : κ → ℝ) (y : κ) (j : ρ) :
    mulVec' A (fun k => q k - onehot y k) j = mulVec' A q j - A j y := by
  unfold mulVec' onehot
  simp [mul_sub, sum_sub_distrib]

/-- **Prop. `prop:encoding`, identity.** `E_{Y∼p} S_A(q,Y) = ‖A(q-p)‖² + tr(A Σ_p Aᵀ)`. -/
theorem expScore_eq (A : ρ → κ → ℝ) (q p : κ → ℝ) (hp : ∑ k, p k = 1) :
    expScore A q p = (∑ j, (mulVec' A (fun k => q k - p k) j) ^ 2) + traceTerm A p := by
  unfold expScore encScore traceTerm
  simp_rw [mulVec'_sub_onehot, mul_sum]
  rw [sum_comm, ← sum_add_distrib]
  refine sum_congr rfl (fun j _ => ?_)
  have hsub : mulVec' A (fun k => q k - p k) j = mulVec' A q j - mulVec' A p j := by
    unfold mulVec'; simp [mul_sub, sum_sub_distrib]
  rw [hsub]
  have h : ∀ y, p y * (mulVec' A q j - A j y) ^ 2 =
      mulVec' A q j ^ 2 * p y + (-2 * mulVec' A q j) * (A j y * p y) + A j y ^ 2 * p y := by
    intro y; ring
  simp_rw [h]
  rw [sum_add_distrib, sum_add_distrib, ← mul_sum, ← mul_sum, hp]
  have hAp : ∑ y, A j y * p y = mulVec' A p j := rfl
  rw [hAp]; ring

/-- The expected score depends on `q` only through `A q`. -/
theorem expScore_congr (A : ρ → κ → ℝ) (q q' p : κ → ℝ) (hp : ∑ k, p k = 1)
    (h : ∀ j, mulVec' A q j = mulVec' A q' j) : expScore A q p = expScore A q' p := by
  rw [expScore_eq A q p hp, expScore_eq A q' p hp]
  congr 1
  refine sum_congr rfl (fun j _ => ?_)
  have e : ∀ r : κ → ℝ, mulVec' A (fun k => r k - p k) j = mulVec' A r j - mulVec' A p j := by
    intro r; unfold mulVec'; simp [mul_sub, sum_sub_distrib]
  rw [e, e, h]

/-- `S_A` is proper. -/
theorem enc_proper (A : ρ → κ → ℝ) (q p : κ → ℝ) (hp : ∑ k, p k = 1) :
    expScore A p p ≤ expScore A q p := by
  rw [expScore_eq A q p hp, expScore_eq A p p hp]
  simp only [sub_self]
  have : ∀ j, mulVec' A (fun _ => (0:ℝ)) j = 0 := fun j => by simp [mulVec']
  simp only [this]
  simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, sum_const_zero, zero_add]
  have := sum_nonneg (fun j (_ : j ∈ univ) => sq_nonneg (mulVec' A (fun k => q k - p k) j))
  linarith

/-- Injective on `D = {v : ∑ v = 0}`. -/
def InjOnD (A : ρ → κ → ℝ) : Prop :=
  ∀ v : κ → ℝ, ∑ k, v k = 0 → (∀ j, mulVec' A v j = 0) → v = 0

def StrictlyProper (A : ρ → κ → ℝ) : Prop :=
  ∀ p q : κ → ℝ, IsProb p → IsProb q → q ≠ p → expScore A p p < expScore A q p

/-- **Prop. `prop:encoding`, characterization.** `S_A` is strictly proper iff `A` is injective on
`D = {v : ∑ₖ vₖ = 0}`. -/
theorem strictlyProper_iff (A : ρ → κ → ℝ) : StrictlyProper A ↔ InjOnD A := by
  constructor
  · intro hSP v hv hAv
    by_contra hne
    have hK : Nonempty κ := by
      by_contra h; apply hne; funext k; exact (h ⟨k⟩).elim
    set N : ℝ := (Fintype.card κ : ℝ) with hN
    have hNpos : 0 < N := by rw [hN]; exact_mod_cast Fintype.card_pos
    set M : ℝ := ∑ k, |v k| + 1 with hM
    have hMpos : 0 < M := by rw [hM]; positivity
    set t : ℝ := 1 / (N * M) with ht
    have htpos : 0 < t := by rw [ht]; positivity
    let u : κ → ℝ := fun _ => 1 / N
    let q : κ → ℝ := fun k => u k + t * v k
    have hu : IsProb u := by
      refine ⟨fun _ => by positivity, ?_⟩
      simp only [u, sum_const, card_univ, nsmul_eq_mul]; rw [← hN]; field_simp
    have hq : IsProb q := by
      refine ⟨fun k => ?_, ?_⟩
      · have hvk : |v k| ≤ ∑ k, |v k| := single_le_sum (fun i _ => abs_nonneg (v i)) (mem_univ k)
        have h1 : t * |v k| < 1 / N := by
          rw [ht]
          rw [div_mul_eq_mul_div, one_mul, div_lt_div_iff₀ (by positivity) hNpos]
          nlinarith
        have h2 : -(t * |v k|) ≤ t * v k := by
          have := neg_abs_le (v k); nlinarith
        simp only [q, u]; linarith
      · simp only [q, sum_add_distrib, ← mul_sum, hv, mul_zero, add_zero]; exact hu.2
    have hqne : q ≠ u := by
      intro h; apply hne; funext k
      have := congrFun h k; simp only [q, add_eq_left, mul_eq_zero] at this
      rcases this with h0 | h0
      · exact absurd h0 htpos.ne'
      · simpa using h0
    have hlt := hSP u q hu hq hqne
    have heq : expScore A q u = expScore A u u := by
      apply expScore_congr A q u u hu.2
      intro j
      simp only [q, mulVec', mul_add, sum_add_distrib]
      have : ∑ k, A j k * (t * v k) = t * mulVec' A v j := by
        unfold mulVec'; rw [mul_sum]; exact sum_congr rfl (fun k _ => by ring)
      rw [this, hAv j, mul_zero, add_zero]
    linarith
  · intro hinj p q hp hq hne
    rw [expScore_eq A q p hp.2, expScore_eq A p p hp.2]
    simp only [sub_self]
    have hz : ∀ j, mulVec' A (fun _ => (0:ℝ)) j = 0 := fun j => by simp [mulVec']
    simp only [hz, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, sum_const_zero,
      zero_add]
    have hD : ∑ k, (q k - p k) = 0 := by rw [sum_sub_distrib, hq.2, hp.2, sub_self]
    have hnz : ¬ ∀ j, mulVec' A (fun k => q k - p k) j = 0 := by
      intro h; apply hne; funext k
      have := congrFun (hinj _ hD h) k; simp only [Pi.zero_apply] at this; linarith
    push Not at hnz
    obtain ⟨j, hj⟩ := hnz
    have hsq : 0 < (mulVec' A (fun k => q k - p k) j) ^ 2 :=
      lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 hj))
    have hpos := lt_of_lt_of_le hsq (single_le_sum (f := fun j =>
      (mulVec' A (fun k => q k - p k) j) ^ 2) (fun i _ => sq_nonneg _) (mem_univ j))
    linarith

/-! ### One-hot: the multiclass Brier score, any finite `K` (Prop. `prop:onehot`). -/

def idEnc : κ → κ → ℝ := fun j k => if j = k then 1 else 0

def brier (q : κ → ℝ) (y : κ) : ℝ := ∑ k, (q k - onehot y k) ^ 2

lemma brier_eq_encScore (q : κ → ℝ) (y : κ) : brier q y = encScore idEnc q y := by
  unfold brier encScore mulVec' idEnc; simp

theorem brier_expected (q p : κ → ℝ) (hp : ∑ k, p k = 1) :
    ∑ y, p y * brier q y = ∑ k, (q k - p k) ^ 2 + (1 - ∑ k, p k ^ 2) := by
  simp_rw [brier_eq_encScore]
  have := expScore_eq idEnc q p hp
  unfold expScore at this; rw [this]
  unfold traceTerm mulVec' idEnc
  simp only [ite_mul, one_mul, zero_mul, sum_ite_eq, mem_univ, ite_true, ite_pow, one_pow,
    ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow]
  rw [sum_sub_distrib, hp]

theorem brier_strictly_proper : StrictlyProper (idEnc : κ → κ → ℝ) := by
  rw [strictlyProper_iff]
  intro v _ h; funext k
  have := h k; unfold mulVec' idEnc at this; simpa using this

/-! ### Cumulative encoding and RPS on `K = n + 1` ordered ratings (Prop. `prop:rps`). -/

/-- CDF at a natural-number threshold. -/
def cdfN {n : ℕ} (q : Fin (n + 1) → ℝ) (t : ℕ) : ℝ := ∑ k, if k.val ≤ t then q k else 0

/-- Cumulative encoding `L`, rows `t = 0, …, K-2` (that is `t : Fin n`). -/
def cumEnc (n : ℕ) : Fin n → Fin (n + 1) → ℝ := fun t k => if k.val ≤ t.val then 1 else 0

/-- RPS (un-normalized) `∑_{t=0}^{K-2} (F_q(t) - 1[y ≤ t])²`. -/
def rps {n : ℕ} (q : Fin (n + 1) → ℝ) (y : Fin (n + 1)) : ℝ :=
  ∑ t : Fin n, (cdfN q t.val - if y.val ≤ t.val then 1 else 0) ^ 2

lemma mulVec'_cumEnc {n : ℕ} (v : Fin (n + 1) → ℝ) (t : Fin n) :
    mulVec' (cumEnc n) v t = cdfN v t.val := by
  unfold mulVec' cumEnc cdfN; simp [ite_mul]

lemma cdfN_sub {n : ℕ} (q p : Fin (n + 1) → ℝ) (t : ℕ) :
    cdfN (fun k => q k - p k) t = cdfN q t - cdfN p t := by
  unfold cdfN; rw [← sum_sub_distrib]
  refine sum_congr rfl (fun k _ => ?_); split_ifs <;> simp

lemma cdfN_onehot {n : ℕ} (y : Fin (n + 1)) (t : ℕ) :
    cdfN (onehot y) t = if y.val ≤ t then 1 else 0 := by
  unfold cdfN onehot
  rw [show (∑ k : Fin (n+1), if k.val ≤ t then (if k = y then (1:ℝ) else 0) else 0)
      = ∑ k : Fin (n+1), if k = y then (if y.val ≤ t then (1:ℝ) else 0) else 0 from
      sum_congr rfl (fun k _ => by by_cases h : k = y <;> simp [h])]
  rw [sum_ite_eq']; simp

lemma rps_eq_encScore {n : ℕ} (q : Fin (n + 1) → ℝ) (y : Fin (n + 1)) :
    rps q y = encScore (cumEnc n) q y := by
  unfold rps encScore
  refine sum_congr rfl (fun t _ => ?_)
  rw [mulVec'_cumEnc, cdfN_sub, cdfN_onehot]

lemma cdfN_succ {n : ℕ} (q : Fin (n + 1) → ℝ) (t : ℕ) (h : t + 1 < n + 1) :
    cdfN q (t + 1) = cdfN q t + q ⟨t + 1, h⟩ := by
  unfold cdfN
  rw [← sub_eq_iff_eq_add', ← sum_sub_distrib]
  rw [show (∑ k : Fin (n+1), ((if k.val ≤ t + 1 then q k else 0) - if k.val ≤ t then q k else 0))
      = ∑ k : Fin (n+1), if k = ⟨t + 1, h⟩ then q k else 0 from ?_]
  · rw [sum_ite_eq']; simp
  refine sum_congr rfl (fun k _ => ?_)
  by_cases h1 : k.val ≤ t
  · have : k ≠ ⟨t+1, h⟩ := by intro e; rw [e] at h1; simp at h1
    simp [h1, show k.val ≤ t + 1 by omega, this]
  · by_cases h2 : k.val = t + 1
    · have : k = ⟨t+1, h⟩ := Fin.ext h2
      simp [h1, h2, this]
    · have : k ≠ ⟨t+1, h⟩ := by intro e; apply h2; rw [e]
      simp [h1, show ¬ k.val ≤ t + 1 by omega, this]

lemma cdfN_zero {n : ℕ} (q : Fin (n + 1) → ℝ) : cdfN q 0 = q 0 := by
  unfold cdfN
  rw [show (∑ k : Fin (n+1), if k.val ≤ 0 then q k else 0)
      = ∑ k : Fin (n+1), if k = 0 then q k else 0 from
      sum_congr rfl (fun k _ => by
        have : (k.val ≤ 0) ↔ k = 0 := by rw [Fin.ext_iff, Nat.le_zero]; rfl
        simp only [this])]
  rw [sum_ite_eq']; simp

/-- `L` is injective on `D`: CDF values at thresholds `0..K-2` plus total mass 0 determine `v`. -/
theorem cumEnc_injOnD (n : ℕ) : InjOnD (cumEnc n) := by
  intro v hv hL
  have hc : ∀ t, t < n → cdfN v t = 0 := fun t ht => by
    have := hL ⟨t, ht⟩; rwa [mulVec'_cumEnc] at this
  have hcn : cdfN v n = 0 := by
    unfold cdfN; rw [← hv]; refine sum_congr rfl (fun k _ => ?_); simp [Nat.lt_succ_iff.mp k.isLt]
  have hall : ∀ t, t ≤ n → cdfN v t = 0 := fun t ht => by
    rcases Nat.lt_or_ge t n with h | h
    · exact hc t h
    · rw [show t = n by omega]; exact hcn
  funext k
  obtain ⟨m, hm⟩ := k
  cases m with
  | zero => have := hall 0 (Nat.zero_le _); rw [cdfN_zero] at this; simpa using this
  | succ m =>
    have h1 := hall (m + 1) (by omega)
    have h0 := hall m (by omega)
    rw [cdfN_succ v m hm, h0, zero_add] at h1
    simpa using h1

/-- **Prop. `prop:rps`, expectation.** `E_{Y∼p} RPS(q,Y) = ∑_t (F_q(t) - F_p(t))² + ∑_t F_p(t)(1 - F_p(t))`. -/
theorem rps_expected {n : ℕ} (q p : Fin (n + 1) → ℝ) (hp : ∑ k, p k = 1) :
    ∑ y, p y * rps q y
      = ∑ t : Fin n, (cdfN q t.val - cdfN p t.val) ^ 2
        + ∑ t : Fin n, cdfN p t.val * (1 - cdfN p t.val) := by
  simp_rw [rps_eq_encScore]
  have := expScore_eq (cumEnc n) q p hp
  unfold expScore at this; rw [this]
  congr 1
  · refine sum_congr rfl (fun t _ => ?_)
    rw [mulVec'_cumEnc, cdfN_sub]
  · unfold traceTerm
    refine sum_congr rfl (fun t _ => ?_)
    rw [mulVec'_cumEnc]
    have : ∑ k, cumEnc n t k ^ 2 * p k = cdfN p t.val := by
      unfold cumEnc cdfN; refine sum_congr rfl (fun k _ => ?_); split_ifs <;> simp
    rw [this]; ring

/-- **Prop. `prop:rps`, strict propriety** on `K = n + 1` ordered categories. -/
theorem rps_strictly_proper {n : ℕ} (p q : Fin (n + 1) → ℝ) (hp : IsProb p) (hq : IsProb q)
    (hne : q ≠ p) : ∑ y, p y * rps p y < ∑ y, p y * rps q y := by
  have h := (strictlyProper_iff (cumEnc n)).mpr (cumEnc_injOnD n) p q hp hq hne
  unfold expScore at h
  simp_rw [rps_eq_encScore]; exact h

/-! ### The scalar encoding `A = (0, 1, …, K-1)` (rank one). -/

def scalarEnc (K : ℕ) : Unit → Fin K → ℝ := fun _ k => (k.val : ℝ)

/-- Scalar squared error `(mean(q) - y)²` is the scalar-encoded score. -/
lemma scalar_encScore (K : ℕ) (q : Fin K → ℝ) (y : Fin K) :
    encScore (scalarEnc K) q y = ((∑ k, (k.val : ℝ) * q k) - y.val) ^ 2 := by
  unfold encScore mulVec' scalarEnc onehot
  simp [mul_sub, sum_sub_distrib]

/-- Same mean ⇒ same expected scalar score under every truth. -/
theorem scalar_same_mean_tie (K : ℕ) (q q' p : Fin K → ℝ) (hp : ∑ k, p k = 1)
    (h : ∑ k, (k.val : ℝ) * q k = ∑ k, (k.val : ℝ) * q' k) :
    expScore (scalarEnc K) q p = expScore (scalarEnc K) q' p :=
  expScore_congr _ q q' p hp (fun _ => by simpa [mulVec', scalarEnc] using h)

/-- For `K ≥ 3` the scalar encoding is not strictly proper. -/
theorem scalar_not_strictly_proper (m : ℕ) : ¬ StrictlyProper (scalarEnc (m + 3)) := by
  rw [strictlyProper_iff]
  intro hinj
  let v : Fin (m + 3) → ℝ := fun k => if k.val = 0 then 1 else if k.val = 1 then -2 else
    if k.val = 2 then 1 else 0
  have hv0 : v ≠ 0 := by
    intro h; have := congrFun h 0; simp [v] at this
  apply hv0
  apply hinj
  · simp only [v, Fin.sum_univ_succ]
    simp [Fin.succ]
    norm_num
  · intro j
    simp only [mulVec', scalarEnc, v, Fin.sum_univ_succ]
    simp [Fin.succ]

/-- For `K = 2` the scalar encoding is injective on `D`, hence strictly proper. -/
theorem scalar_two_strictly_proper : StrictlyProper (scalarEnc 2) := by
  rw [strictlyProper_iff]
  intro v hv h
  have h1 := h (); simp [mulVec', scalarEnc, Fin.sum_univ_two] at h1
  simp [Fin.sum_univ_two] at hv
  funext k; fin_cases k <;> simp <;> linarith

end Point2
