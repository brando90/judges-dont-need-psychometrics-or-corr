/-!
# Point 1 checks (core Lean 4, no Mathlib)

Toolchain: `leanprover/lean4:v4.36.0` (see `lean-toolchain`).  Build: `lean Point1Checks.lean`.

Exact scope, as `AGENTS.md` ("Every proof in Lean 4") requires:

* `agrees_iff_mode` and `agreement_depends_only_on_mode` are GENERAL: any number of categories `K`,
  integer-scaled forecasts, agreement with a label defined as "the label is the forecast's strict mode".
  They prove that agreement with a label depends on a forecast only through its mode.
* Every `*_instance` theorem is a CLOSED FINITE INSTANCE ("Lean-checked special case"):
  three categories, probabilities scaled by 10, squared distances (the Brier excess up to a constant
  factor) scaled by 100.  General strict propriety of the Brier score is in
  `docs/notes/point2_support_27Nov2026/lean` (`Point2.brier_strictly_proper`), not re-proved here.
-/

namespace Point1

/-- `m` is the strict mode of the integer-scaled forecast `q`. -/
def IsStrictMode {K : Nat} (q : Fin K → Int) (m : Fin K) : Prop :=
  ∀ k, k ≠ m → q k < q m

/-- A forecast agrees with the label `y` when `y` is its strict mode. -/
def Agrees {K : Nat} (q : Fin K → Int) (y : Fin K) : Prop := IsStrictMode q y

/-- If `m` is the strict mode of `q`, then `q` agrees with `y` exactly when `y = m`. -/
theorem agrees_iff_mode {K : Nat} {q : Fin K → Int} {m : Fin K}
    (h : IsStrictMode q m) (y : Fin K) : Agrees q y ↔ y = m := by
  constructor
  · intro hy
    by_cases hym : y = m
    · exact hym
    · have h1 : q y < q m := h y hym
      have h2 : q m < q y := hy m (fun hmy => hym hmy.symm)
      omega
  · intro hym
    subst hym
    exact h

/-- Two forecasts with the same strict mode agree with exactly the same labels. -/
theorem agreement_depends_only_on_mode {K : Nat} {q q' : Fin K → Int} {m : Fin K}
    (h : IsStrictMode q m) (h' : IsStrictMode q' m) (y : Fin K) :
    Agrees q y ↔ Agrees q' y := by
  rw [agrees_iff_mode h y, agrees_iff_mode h' y]

/-- Squared Euclidean distance on three categories (the Brier excess, up to scale). -/
def sq3 (q p : Fin 3 → Int) : Int :=
  (q 0 - p 0) * (q 0 - p 0) + (q 1 - p 1) * (q 1 - p 1) + (q 2 - p 2) * (q 2 - p 2)

/-- Build a three-category forecast from its three scaled entries. -/
def f3 (a b c : Int) : Fin 3 → Int := fun i => if i = 0 then a else if i = 1 then b else c

/-- Human distribution of pool A: (0.5, 0.3, 0.2), scaled by 10. -/
def pA : Fin 3 → Int := f3 5 3 2
/-- Human distribution of pool B: (0.2, 0.3, 0.5), scaled by 10. -/
def pB : Fin 3 → Int := f3 2 3 5
/-- A forecast with the same strict mode (category 0) as `pA`, but sharper. -/
def qSame : Fin 3 → Int := f3 8 1 1

/-- INSTANCE (Lean-checked special case): same strict mode, so the same agreement with every label,
yet the Brier excess against the human distribution is 0 for the truthful forecast and 14 (units 1/100)
for the sharper one. -/
theorem same_mode_ties_agreement_but_brier_separates_instance :
    (∀ y, Agrees pA y ↔ Agrees qSame y) ∧ sq3 pA pA = 0 ∧ sq3 qSame pA = 14 ∧ sq3 pA pA < sq3 qSame pA := by
  have hA : IsStrictMode pA 0 := by unfold IsStrictMode pA f3; decide
  have hS : IsStrictMode qSame 0 := by unfold IsStrictMode qSame f3; decide
  exact ⟨fun y => agreement_depends_only_on_mode hA hS y, by decide⟩

/-- INSTANCE: the pool is the construct.  A judge that is exactly right for pool A has Brier excess 18
(units 1/100) against pool B, and the judge exactly right for pool B has excess 0 there. -/
theorem pool_changes_the_ruler_instance :
    sq3 pA pA = 0 ∧ sq3 pB pB = 0 ∧ sq3 pA pB = 18 ∧ sq3 pB pA = 18 := by
  unfold sq3 pA pB f3; decide

/-- Consensus item: the human distribution is a point mass on category 0. -/
def pC : Fin 3 → Int := f3 10 0 0
/-- A soft forecast and a sharpened forecast, both with strict mode 0. -/
def qSoft : Fin 3 → Int := f3 8 1 1
def qSharp : Fin 3 → Int := f3 9 1 0

/-- INSTANCE: on a consensus item (point-mass human distribution) the sharper forecast has the lower
Brier excess (2 versus 6, units 1/100) and the point mass itself has excess 0. -/
theorem consensus_item_rewards_peaked_forecast_instance :
    sq3 pC pC = 0 ∧ sq3 qSharp pC = 2 ∧ sq3 qSoft pC = 6 ∧ sq3 qSharp pC < sq3 qSoft pC := by
  unfold sq3 pC qSoft qSharp f3; decide

/-- Contested item: the human distribution is (0.4, 0.3, 0.3), scaled by 10. -/
def pK : Fin 3 → Int := f3 4 3 3
/-- A sharpened forecast with the same strict mode. -/
def qSharpK : Fin 3 → Int := f3 6 2 2

/-- INSTANCE: on a contested item the truthful forecast beats the sharpened one (0 versus 6, units 1/100),
so peakedness is rewarded only where the human distribution is peaked. -/
theorem contested_item_penalizes_sharpening_instance :
    sq3 pK pK = 0 ∧ sq3 qSharpK pK = 6 ∧ sq3 pK pK < sq3 qSharpK pK := by
  unfold sq3 pK qSharpK f3; decide

end Point1

#print axioms Point1.agrees_iff_mode
#print axioms Point1.agreement_depends_only_on_mode
#print axioms Point1.same_mode_ties_agreement_but_brier_separates_instance
#print axioms Point1.pool_changes_the_ruler_instance
#print axioms Point1.consensus_item_rewards_peaked_forecast_instance
#print axioms Point1.contested_item_penalizes_sharpening_instance
