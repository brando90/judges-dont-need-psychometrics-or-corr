# Transcription of the handwriting on a printed 6.437 notes page (11-26-2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/assets/TRANSCRIPTION_6437_PAGE_11-26-2026.md>

Date: 11-26-2026. One photograph (PNG, 3,024 by 4,032 pixels) of a printed page of lecture notes that the project owner calls the 6.437 notes (the printed page number is 2), covered in handwritten margin notes in blue, teal and black ink. It was supplied to the session at about 12:25 PST, with the request to show that the Brier score is proper in the same style of argument as this page. The answer is in [`../extended_checks_11-26-2026/brier_proper_6437_style_11-26-2026/`](../extended_checks_11-26-2026/brier_proper_6437_style_11-26-2026/README.md).

| Item | Bytes | SHA-256 |
| --- | --- | --- |
| Supplied photograph (PNG) | 16,189,478 | `55147ee8e898dab14c938cea5d2b0ea5357ec69e1a9e58879abac1e09db8673f` |
| Encrypted copy in this directory, [`private_6437_annotated_page_11-26-2026.png.age`](private_6437_annotated_page_11-26-2026.png.age) | 16,193,630 | `4a075de2f31607c04f1bc35d570ea39e8a339ed0db9c41e5c085d9345dd60d00` |

Most of the photographed page is someone else's printed text, so the supplied bytes are kept here only as an age-encrypted file. A full decrypt-and-compare round trip on 11-26-2026 reproduced the supplied bytes exactly (`cmp` reported no difference). The recipient, the identity file and the decrypt command are in [`ARCHIVE_11-26-2026.md`](ARCHIVE_11-26-2026.md); the same identity opens this file. A downscaled viewing copy (1,500 by 2,000 pixels, JPEG) is in the project Google Drive folder. It is not the original bytes; the full-size photograph has not reached Drive because the Drive connector refuses uploads above about 3 MB per call.

## Conventions

- `[?]` marks a reading I am not sure of. `[struck: ...]` marks writing with a line through it. `[scribbled out]` marks ink that blots out the writing underneath, so it cannot be read. `[illegible]` marks writing I could not read at all.
- The printed text is not reproduced here. The printed items are named by their labels: Definition 1 with equation (4) (a cost function is "proper"), Claim 1 and its proof with equations (5) and (6) (log-loss is proper), Definition 2 (a cost function is "local"), Claim 2 (log-loss is local) and Theorem 1 (for an alphabet of at least three values, log-loss is the only smooth local proper cost function).
- Pen colors: black for the left-margin block, blue for most of the rest, teal for the question in the right-hand middle and the note under Theorem 1.
- The desk, the pen at the top and the two other pages that show at the left and right edges of the photograph are not page text. The edge pages carry scraps of the derivations transcribed in [`TRANSCRIPTION_DERIVATION_PAGES_11-26-2026.md`](TRANSCRIPTION_DERIVATION_PAGES_11-26-2026.md) (for example `X, Y`, `(X~ -`, `(a - b`, `E[(X~`, `Z~ = X~ =`).

## Marks on the printed text (not writing)

1. In the paragraph before Definition 1, the printed word "peaky" (in quotation marks) is boxed in blue. A blue arrow runs from the box to the note `Q: meaning?`.
2. In Definition 1, the printed word "proper" is boxed. Under equation (4), a small arrow points up at the expectation and carries `x ~ p_{x|y}(.|y)`.
3. In the sentence after Definition 1, "true" is circled, "true belief p_{x|y}" is underlined, and the printed word "honest" starts a second underline. A blue arrow leaves that sentence and points to the teal question below.
4. In the paragraph after Definition 2, the phrase "assesses the quality of the estimated belief q only in terms of the probability assigned to the actual outcome x" is underlined in blue.
5. In Theorem 1, the printed word "only" is boxed, and "smooth", "local" and "proper" are each underlined.
6. A long curved blue line leaves the start of the top-left question, runs down the left side of the printed block and ends at the left end of equation (4). A second long vertical blue line leaves the top-right question and runs down to the right end of equation (4).

## Left margin, black ink, top to bottom

```text
TODO!
Show Brier is
local in this
style of
argument.
Also, it's obvious
log loss is proper;
it's [circled mark: "y" or "4"?] literally say[ing]
find optimal
distribution.
if your optim[?]
log of distribution,
which, logs
basically just
p!! log is a 1-1
function with
its input!
```

The word after "optim" runs into the printed text and is cut off. The circled mark is above the line, between "it's" and "literally".

## Top left, blue ink

```text
besides having
now trillions of tokens.
```

A long thin line runs from the word "tokens" down to the start of the question below.

```text
Q: [arrow up to "tokens"] is fundamentally the reason why ML/DL use
estimation (for deciding theta|x for x) because
having uncertainty over theta|x is hard due to the
partition function ? (unknown all the data).
```

The first word after "use" could also be read "we". The symbol I give as `theta|x` is a Greek letter written like a theta, followed by a slash or bar and `x`.

## Next to the paragraph about "peaky" distributions

```text
Q: meaning?
```

## Top right, blue ink

```text
A^(.) = argmin  E_{H | y ~ p_{H|y}(.|y)} [ C( A^(y), H~ ) ]
        A^(.) in H = {H_0, H_1}
        {A^ : y -> H}
        [illegible, about three words]        H_0, H_1 -> hyp. tests [?]
```

The `H` in `A^(.) in H` and `{A^ : y -> H}` is a script capital. A tilde sits under the second argument of `C`. A small `x` is written under `y -> H` (it may belong to the set-builder notation). Below this block, to the right of the long vertical line:

```text
Q we are using the thing | want to estimate
   -> to estimate it? with estimation in which we
   [if?] it was point estimates & have was a
   way to plug them in to the [estimator?] even if it
                      depended on
                      P_{x|y} (the posterior)
                      that is hard to
                      compute usually.
Q: [illegible, about four words, perhaps ending "needs?"]
```

The vertical line cuts through the first row between "thing" and "want". The last question is small and crowded against the page edge, and I could not read it.

## Right middle, teal ink

```text
Q! why is this important? [scribbled out word] why is it worth
defining? Is it because we can rarely compute
the true pdf due to e.g. the partition function ?
So we can't in approx[?]
Inference.                       ("Inference." is underlined)
   -> defined as the
      P_{x|y}(x|y) for so
      unknown[?] x [?]
```

## Under Theorem 1, teal ink

```text
[small upward tick] Cite [or "Like"?] if it's good...
```
