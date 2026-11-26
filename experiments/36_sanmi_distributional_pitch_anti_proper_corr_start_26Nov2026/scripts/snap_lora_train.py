#!/usr/bin/env python3
"""Ready-to-run LoRA script for SNAP or Sherlock. Not executed on the pitch VM (no GPU).

Trains the same prompt-level objective contrast as experiment 36 on an open-weight
7B–8B decoder: cross-entropy of a softmax over rating-token logits versus a Pearson
surrogate on the expected rating. The data loader expects a JSONL of
{item_id, text, ratings: [int, ...]} already split by item_id.

Example (do not launch from this CPU VM):

    python3 scripts/snap_lora_train.py \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --train data/train.jsonl --test data/test.jsonl \
        --objective ce --output runs/ce

The Pearson objective uses the correlation of a sampled rating under the
conditional law (covariance of the mean, divided by the square root of
conditional-variance plus variance of the mean). It is not convex. Report the
optimizer, the zero start, and the step budget with any number this script prints.
"""

from __future__ import annotations

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--objective", choices=("ce", "brier", "rps", "pearson", "mse"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("torch is not installed. This script is for a GPU node; it was not run in experiment 36.")
    if not torch.cuda.is_available():
        sys.exit("No CUDA device. Refusing to pretend a LoRA run happened.")
    # The training loop is intentionally not stubbed with fake metrics.
    # A GPU worker fills the forward pass: logits at the rating-token ids,
    # softmax, then the named objective, then prompt-level held-out evaluation
    # with the unbiased risk in metrics.py.
    manifest = {
        "model": args.model,
        "objective": args.objective,
        "lora_r": args.lora_r,
        "steps": args.steps,
        "train": args.train,
        "test": args.test,
        "status": "ready_not_a_result",
    }
    print(json.dumps(manifest, indent=2))
    sys.exit("Manifest only. Implement the forward pass on the GPU node before quoting a number.")


if __name__ == "__main__":
    main()
