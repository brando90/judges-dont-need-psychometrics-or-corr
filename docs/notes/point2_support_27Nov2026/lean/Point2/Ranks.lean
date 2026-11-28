import Point2.Moments

/-!
# Point 2 note: Spearman and Kendall (Remark "Rank correlations", Cor. `cor:tie`)

Population versions on a finite law. Spearman is Pearson correlation of the mid-rank transforms
`R_S(i) = P(S < S i) + ½ P(S = S i)` (the usual tie-corrected ranks, scaled to `[0,1]`); Kendall is
`τ = E[sign((S - S')(Y - Y'))]` over two independent draws `i, j` from the law.
-/

open Finset BigOperators

namespace Point2
namespace FinLaw

set_option linter.unusedSectionVars false

variable {ι : Type*} [Fintype ι] (P : FinLaw ι)

noncomputable def midrank (S : ι → ℝ) (i : ι) : ℝ :=
  ∑ j, P.w j * ((if S j < S i then 1 else 0) + (if S j = S i then 1 / 2 else 0))

noncomputable def spearman (S Y : ι → ℝ) : ℝ := P.corr (P.midrank S) (P.midrank Y)

noncomputable def kendall (S Y : ι → ℝ) : ℝ :=
  ∑ i, ∑ j, P.w i * P.w j * ((SignType.sign ((S i - S j) * (Y i - Y j)) : SignType) : ℝ)

/-- Mid-ranks are invariant under any strictly increasing map. -/
theorem midrank_strictMono (S : ι → ℝ) (f : ℝ → ℝ) (hf : StrictMono f) :
    P.midrank (fun i => f (S i)) = P.midrank S := by
  funext i; unfold midrank
  simp only [hf.lt_iff_lt, hf.injective.eq_iff]

lemma sign_sub_strictMono (f : ℝ → ℝ) (hf : StrictMono f) (a b : ℝ) :
    SignType.sign (f a - f b) = SignType.sign (a - b) := by
  rcases lt_trichotomy a b with h | h | h
  · rw [sign_neg (sub_neg.mpr (hf h)), sign_neg (sub_neg.mpr h)]
  · rw [h, sub_self, sub_self]
  · rw [sign_pos (sub_pos.mpr (hf h)), sign_pos (sub_pos.mpr h)]

theorem kendall_strictMono (S Y : ι → ℝ) (f : ℝ → ℝ) (hf : StrictMono f) :
    P.kendall (fun i => f (S i)) Y = P.kendall S Y := by
  unfold kendall
  simp only [sign_mul, sign_sub_strictMono f hf]

theorem spearman_strictMono (S Y : ι → ℝ) (f : ℝ → ℝ) (hf : StrictMono f) :
    P.spearman (fun i => f (S i)) Y = P.spearman S Y := by
  unfold spearman; rw [P.midrank_strictMono S f hf]

/-- Pearson is invariant under strictly increasing *affine* maps only; the rank statistics under
all strictly increasing maps. Example in the note: `exp` keeps Spearman and Kendall. -/
example (S Y : ι → ℝ) : P.spearman (fun i => Real.exp (S i)) Y = P.spearman S Y :=
  P.spearman_strictMono S Y Real.exp Real.exp_strictMono

/-! ### Strictly decreasing maps flip the sign. -/

lemma sign_sub_strictAnti (f : ℝ → ℝ) (hf : StrictAnti f) (a b : ℝ) :
    SignType.sign (f a - f b) = - SignType.sign (a - b) := by
  rcases lt_trichotomy a b with h | h | h
  · rw [sign_pos (sub_pos.mpr (hf h)), sign_neg (sub_neg.mpr h)]; rfl
  · subst h; simp
  · rw [sign_neg (sub_neg.mpr (hf h)), sign_pos (sub_pos.mpr h)]

theorem kendall_strictAnti (S Y : ι → ℝ) (f : ℝ → ℝ) (hf : StrictAnti f) :
    P.kendall (fun i => f (S i)) Y = - P.kendall S Y := by
  unfold kendall
  simp only [sign_mul, sign_sub_strictAnti f hf, neg_mul, SignType.coe_neg, mul_neg,
    sum_neg_distrib]

theorem midrank_strictAnti (S : ι → ℝ) (f : ℝ → ℝ) (hf : StrictAnti f) :
    P.midrank (fun i => f (S i)) = fun i => -1 * P.midrank S i + 1 := by
  funext i; unfold midrank
  simp only [hf.lt_iff_gt, hf.injective.eq_iff]
  have h1 : ∑ j, P.w j * ((if S i < S j then (1:ℝ) else 0) + (if S j = S i then 1 / 2 else 0))
      = ∑ j, (P.w j - P.w j * ((if S j < S i then 1 else 0) + (if S j = S i then 1 / 2 else 0))) := by
    refine sum_congr rfl (fun j _ => ?_)
    rcases lt_trichotomy (S j) (S i) with h | h | h
    · simp [h, h.ne, not_lt.mpr h.le]
    · simp [h]; ring
    · simp [h, h.ne', not_lt.mpr h.le]
  rw [h1, sum_sub_distrib, P.sum_one]; ring

theorem spearman_strictAnti (S Y : ι → ℝ) (f : ℝ → ℝ) (hf : StrictAnti f) :
    P.spearman (fun i => f (S i)) Y = - P.spearman S Y := by
  unfold spearman; rw [P.midrank_strictAnti S f hf]
  exact P.pearson_affine_neg _ _ (-1) 1 (by norm_num)

/-! ### Cor. `cor:tie`: same conditional mean ⇒ the expected-score readouts coincide, so every
statistic of (readout, Y) ties. The joint law is over `C × Fin K`: item `c`, human rating `y`. -/

/-- Expected-score readout `μ_q(c) = ∑ₖ k q(k | c)`, as a function on the joint space. -/
def meanReadout {C : Type*} {K : ℕ} (q : C → Fin K → ℝ) : C × Fin K → ℝ :=
  fun x => ∑ k, (k.val : ℝ) * q x.1 k

def rating {C : Type*} {K : ℕ} : C × Fin K → ℝ := fun x => (x.2.val : ℝ)

/-- **Cor. `cor:tie`.** If `μ_q(c) = μ_{q'}(c)` for every item, then on any joint law of
(item, human rating) the two forecasts tie on Pearson, Spearman, Kendall and the mean squared
error of the readout, whatever their spread. -/
theorem same_mean_tie {C : Type*} [Fintype C] {K : ℕ} (P : FinLaw (C × Fin K))
    (q q' : C → Fin K → ℝ) (h : ∀ c, ∑ k, (k.val : ℝ) * q c k = ∑ k, (k.val : ℝ) * q' c k) :
    P.corr (meanReadout q) rating = P.corr (meanReadout q') rating ∧
    P.spearman (meanReadout q) rating = P.spearman (meanReadout q') rating ∧
    P.kendall (meanReadout q) rating = P.kendall (meanReadout q') rating ∧
    P.E (fun x => (meanReadout q x - rating x) ^ 2)
      = P.E (fun x => (meanReadout q' x - rating x) ^ 2) := by
  have e : meanReadout q = meanReadout q' := by funext x; exact h x.1
  rw [e]; exact ⟨rfl, rfl, rfl, rfl⟩

end FinLaw
end Point2
