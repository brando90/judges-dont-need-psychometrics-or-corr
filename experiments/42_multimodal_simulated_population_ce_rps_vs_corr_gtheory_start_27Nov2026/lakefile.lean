import Lake
open Lake DSL

package «simpop41» where

-- Core Lean only. Do not add a Mathlib dependency.
@[default_target]
lean_lib SimPop where
  srcDir := "theory"
