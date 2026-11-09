# Cross-entropy versus Pearson: execution status

**Status, 11-09-2026:** Handoff accepted and its SHA-256 verified. Historical source snapshots and a training-only diagnostic runner are prepared. No new empirical fits have run.

The cluster reconnect guard remains in its shared cooldown. A local CPU fallback question is pending because the shared execution rule requires SNAP. Synthetic deterministic implementation checks passed; they are not empirical comparison results.

The proposed new condition uses the same simplex of cached probability forecasts for both objectives. This is distinct from the original affine-softmax condition. The final protocol will be frozen only after reproducing and reviewing a historical failing fit. Neither objective is presumed to win.

See [ckpt.md](ckpt.md) for the complete source handoff, [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) for exact copied-source hashes, and [SESSION_ACCESS.md](SESSION_ACCESS.md) for verified cross-profile continuation access and remaining native-history limits.
