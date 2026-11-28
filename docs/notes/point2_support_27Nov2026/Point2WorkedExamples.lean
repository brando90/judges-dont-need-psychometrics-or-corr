/-
Exact rational checks of the worked-example numbers in the Point 2 note (27 Nov 2026).
No Mathlib; Lean 4.36.0.  The score functions are defined here (one-hot Brier and RPS on ratings 0..5,
expected under a law p), and each worked-example value is proved by evaluation.  This file does NOT
prove general theorems; the general statements of the note are proved with Mathlib in the Lake
project `lean/` next to this file (`lake build`; see lean/README.md).
-/

abbrev L6 := Fin 6 → Rat

def sum6 (f : Fin 6 → Rat) : Rat := f 0 + f 1 + f 2 + f 3 + f 4 + f 5

/-- cumulative distribution function at threshold t (t = 0..4) -/
def cdf (q : L6) (t : Nat) : Rat :=
  (List.range 6).foldl (fun acc k => if k ≤ t then acc + q ⟨k % 6, by omega⟩ else acc) 0

/-- one-hot Brier of forecast q at realized rating y -/
def brier1 (q : L6) (y : Fin 6) : Rat := sum6 (fun k => (q k - (if k = y then 1 else 0)) ^ 2)

/-- RPS (un-normalized) of forecast q at realized rating y -/
def rps1 (q : L6) (y : Fin 6) : Rat :=
  (List.range 5).foldl (fun acc t => acc + (cdf q t - (if y.val ≤ t then 1 else 0)) ^ 2) 0

/-- expected score under the truth p -/
def expB (q p : L6) : Rat := sum6 (fun y => p y * brier1 q y)
def expR (q p : L6) : Rat := sum6 (fun y => p y * rps1 q y)

def A : L6 := fun i => match i.val with | 2 => 1 | _ => 0
def B : L6 := fun i => match i.val with | 0 => 1/2 | 4 => 1/2 | _ => 0

def mean (p : L6) : Rat := sum6 (fun k => (k.val : Rat) * p k)

theorem same_mean : mean A = 2 ∧ mean B = 2 := by decide +kernel

theorem brier_table :
    expB A A = 0 ∧ expB A B = 2 ∧ expB B A = 3 / 2 ∧ expB B B = 1 / 2 := by decide +kernel

theorem rps_table :
    expR A A = 0 ∧ expR A B = 2 ∧ expR B A = 1 ∧ expR B B = 1 := by decide +kernel

/-- distance sensitivity: realized rating 0; point masses at 1 and at 5 -/
def P1 : L6 := fun i => match i.val with | 1 => 1 | _ => 0
def P5 : L6 := fun i => match i.val with | 5 => 1 | _ => 0
theorem distance_sensitivity :
    brier1 P1 0 = 2 ∧ brier1 P5 0 = 2 ∧ rps1 P1 0 = 1 ∧ rps1 P5 0 = 5 := by decide +kernel

/-! Population example (note, Section "A population version"): three equally likely items with
truths `½δ₀+½δ₂`, `½δ₀+½δ₄`, `½δ₂+½δ₄`; truthful judge T, mean-collapsed judge D (`δ₁, δ₂, δ₃`),
pooled base rate `(1/3, 0, 1/3, 0, 1/3, 0)`. Added 27 Nov 2026. -/

def T1 : L6 := fun i => match i.val with | 0 => 1/2 | 2 => 1/2 | _ => 0
def T2 : L6 := fun i => match i.val with | 0 => 1/2 | 4 => 1/2 | _ => 0
def T3 : L6 := fun i => match i.val with | 2 => 1/2 | 4 => 1/2 | _ => 0
def D1 : L6 := fun i => match i.val with | 1 => 1 | _ => 0
def D2 : L6 := fun i => match i.val with | 2 => 1 | _ => 0
def D3 : L6 := fun i => match i.val with | 3 => 1 | _ => 0
def Base : L6 := fun i => match i.val with | 0 => 1/3 | 2 => 1/3 | 4 => 1/3 | _ => 0

/-- population-average expected score of forecasts `(q1,q2,q3)` against the three truths -/
def popB (q1 q2 q3 : L6) : Rat := (expB q1 T1 + expB q2 T2 + expB q3 T3) / 3
def popR (q1 q2 q3 : L6) : Rat := (expR q1 T1 + expR q2 T2 + expR q3 T3) / 3

theorem population_table :
    popB T1 T2 T3 = 1/2 ∧ popB D1 D2 D3 = 2 ∧ popR T1 T2 T3 = 2/3 ∧ popR D1 D2 D3 = 4/3 ∧
    popB Base Base Base = 2/3 ∧ popR Base Base Base = 8/9 ∧
    1 - popB T1 T2 T3 / popB Base Base Base = 1/4 ∧ 1 - popB D1 D2 D3 / popB Base Base Base = -2 ∧
    1 - popR T1 T2 T3 / popR Base Base Base = 1/4 ∧ 1 - popR D1 D2 D3 / popR Base Base Base = -1/2 ∧
    mean T1 = 1 ∧ mean T2 = 2 ∧ mean T3 = 3 := by decide +kernel

/-- Shift-and-stretch example: `E[Y²] = 17/2` on the eight-item list `0,1,2,3,4,5,2,3`. -/
theorem shift_stretch_mse :
    ([0, 1, 2, 3, 4, 5, 2, 3].map (fun y : Rat => y ^ 2)).sum / 8 = 17 / 2 := by decide +kernel

#print axioms population_table
#print axioms shift_stretch_mse

#print axioms same_mean
#print axioms brier_table
#print axioms rps_table
#print axioms distance_sensitivity
