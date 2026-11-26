#!/usr/bin/env python3
"""One forward pass over rating-token logits, versus M sampled generations.

Not run in experiment 36 (no GPU, no local open-weight checkpoint).

The single pass with no rationale is: tokenize the prompt, read the logits at
the vocabulary ids of the rating strings (for example "0".."5" or "1".."5"),
and softmax those K logits. That vector is q(y | c).

If the prompt asks for a chain-of-thought rationale r first, the rating-token
logits at the end of that text are q(y | c, r), one sampled trajectory. They
are not the marginal sum_r q(r | c) q(y | c, r). Estimate that marginal either
by the no-rationale forward pass or by averaging M chain-of-thought samples.

The Monte Carlo arm draws M completions, parses a rating token, and averages
the one-hots. M in {1, 5, 20} matches the linear-head stand-in already reported
in figures/single_pass_vs_rollout.png.
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True, help="JSONL with fields item_id and text")
    parser.add_argument("--rating-tokens", default="0,1,2,3,4,5")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("torch is not installed. This script is for a GPU node.")
    if not torch.cuda.is_available():
        sys.exit("No CUDA device. The linear-head rollout in experiment 36 is the stand-in, not this script.")
    print("Rating tokens:", args.rating_tokens)
    print("Samples M:", args.samples)
    sys.exit("Hook the model's next-token logits at these ids before quoting a readout.")


if __name__ == "__main__":
    main()
