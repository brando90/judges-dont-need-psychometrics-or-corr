import Lake
open Lake DSL

package «gtheory39» where

-- Core Lean only. Do not add a Mathlib dependency: this package must not build Mathlib.
@[default_target]
lean_lib GTheory where
  srcDir := "theory"
