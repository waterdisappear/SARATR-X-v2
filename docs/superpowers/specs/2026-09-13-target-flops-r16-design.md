# Design: Hardware-Independent FLOPs Accounting for Multi-Scale Structural Target (R1.6)

**Date:** 2026-09-13  
**Status:** Approved  
**Goal:** Provide reviewer R1.6 with a hardware-independent efficiency comparison of target construction (`pixel` / `single_s1` / `single_s6` / `multi`), add an appendix table, and draft the response letter entry in the style of `SARATR_X_v2_GRSM_R1/R1.Respone.tex`.

## Problem

Reviewer 1 Comment 6 asks for basic efficiency information (throughput, target-generation overhead, or GPU memory) to judge the cost of six structural operators plus fusion versus pixel or single-scale targets.

Absolute runtime and memory are GPU-dependent. The agreed approach is to report **approximate FLOPs of target construction** (analytic, hardware-independent), not measured img/s or peak VRAM.

## Decisions (confirmed)

| Decision | Choice |
|---|---|
| Experiment style | Local analytic FLOPs script (no required GPU timing) |
| Absolute GPU metrics | **Not reported** |
| Primary metric | Approximate target-construction FLOPs + relative ratios |
| Target modes in table | `pixel`, `single_s1`, `single_s6`, `multi` |
| Paper edits | Appendix paragraph + table; fill R1.6 in `R1.Respone.tex` |
| Style reference | Existing completed responses in `R1.Respone.tex` (e.g. R1.7) |

## Scope

### In scope

1. Script under `pre-training/` that computes approximate FLOPs for the four target modes from the same operator definitions as `models/masked_autoencoder.py`.
2. Appendix addition in `SARATR_X_v2_GRSM_R1/Data/Appendix.tex` (near Implementation Details for Pre-training / Method).
3. Replace `\PLACEHOLDER{response to R1.6}` and its changelist placeholder in `R1.Respone.tex`.

### Out of scope

- Measured throughput / step time / GPU memory tables.
- Re-running full 1200-epoch pre-training.
- Changing the structural operators or training code behavior (benchmark/accounting only).

## FLOPs accounting rules (hardware-independent)

Canonical input: single-channel SAR image of size \(H = W = 224\) (paper pre-training resolution). Report **per image** (batch = 1). Relative ratios are invariant to batch size under linear scaling.

### What is counted

Count **multiply–accumulate (MAC) operations of the depthwise-style `conv2d` kernels as implemented**, plus the cheap fused linear combination for `multi`.

Do **not** count as primary FLOPs: `log`, `sigmoid`, padding, elementwise subtract/norm (order-of-magnitude smaller than the convolutions; mention qualitatively in prose that they are omitted).

### Per-operator formulas

Let spatial output size after valid convolution on reflect-padded input be \(N = H \times W = 224^2\).

**S1 (`SAR_Lay`, \(r=1\), \(3\times3\) kernel):** one convolution with 9 taps  
\[
\mathrm{FLOPs}_{S1} = N \cdot 9.
\]

**S\(k\) for \(k\in\{2,\ldots,6\}\) (`SAR_Layer`, radius \(r\in\{3,5,9,13,17\}\), kernel \(M=2r+1\)):** four convolutions of size \(M\times M\) (as in code: left/right/up/down half-region filters)  
\[
\mathrm{FLOPs}_{S_k} = 4 \cdot N \cdot M^2.
\]
(Counting dense kernel MACs matches the `F.conv2d` implementation; zero taps are not skipped at the CUDA kernel level for these buffers.)

**Fusion (`multi` only):** six scalar–map products and five adds  
\[
\mathrm{FLOPs}_{\mathrm{fuse}} = 6N + 5N = 11N
\]
(approximate; negligible vs convolutions).

### Per target mode

| Mode | Formula |
|---|---|
| `pixel` | \(0\) (no structural extractor; reconstruct raw pixels) |
| `single_s1` | \(\mathrm{FLOPs}_{S1}\) |
| `single_s6` | \(\mathrm{FLOPs}_{S6}\) with \(r=17\), \(M=35\) |
| `multi` | \(\sum_{s=1}^{6}\mathrm{FLOPs}_{S_s} + \mathrm{FLOPs}_{\mathrm{fuse}}\) |

Relative columns: \(\mathrm{FLOPs}/\mathrm{FLOPs}_{S1}\) or vs `multi`, and a short note that learnable parameters of the target head are only the **6 fusion scalars** (softmax), while the six operators run under `torch.no_grad()`.

Optional secondary row (if easy from existing `util/flop_count` or published iTPN-B forward FLOPs): ratio of target FLOPs to one backbone forward — only if a cited/reproducible backbone FLOPs number is available without inventing claims. If not available cleanly, omit from the table and keep a qualitative sentence (“orders of magnitude below the hierarchical encoder–decoder”).

## Script design

**Path:** `pre-training/bench_target_flops.py`

**Behavior:**

- Pure Python/`numpy` (no GPU required).
- Encode the formulas above with \(H=W=224\) and radii `{1,3,5,9,13,17}`.
- Print a markdown/LaTeX-friendly table and write `pre-training/bench_target_flops.json` (or under `results/` if preferred for paper artifacts — default: next to the script as JSON for reproducibility).
- Assert that `multi` FLOPs equals sum of six branches + fusion.

No dependency on loading `itpn` weights.

## Appendix manuscript change

Insert under **Implementation Details for Method** or immediately after **Implementation Details for Pre-training** a short subsection, e.g. **Computational Cost of Structural Target Construction**, containing:

1. One paragraph: operators are fixed; extraction under `no_grad`; only six fusion weights are trainable; FLOPs are analytic MACs of the implemented convolutions at \(224\times224\); independent of GPU.
2. Table `\label{tab:target_flops}` with columns roughly: Target / Approx. FLOPs (MACs) / Relative to `single_s1` / Relative to `multi`.
3. Blue text for response-letter changelist quoting, consistent with other R1 revisions.

Numbers in the table must come from running the script (no hand-waved digits).

## Response letter (R1.6)

Follow the completed-response pattern (e.g. R1.7):

1. Thank the reviewer and agree that efficiency should be clearer.
2. Clarify: six extractors are **fixed** structural operators under `no_grad`; fusion adds **six learnable scalars**; no second trainable encoder for the target.
3. Point to the new appendix table of **approximate target-construction FLOPs** (hardware-independent) for `pixel` / `single_s1` / `single_s6` / `multi`.
4. Interpret: `multi` costs the sum of six branches (dominated by large-support `S6`), still a cheap preprocessing relative to the masked hierarchical backbone; `pixel` has zero target FLOPs but weaker supervision as shown elsewhere.
5. `\star Changes in manuscripts` + `changelist` citing the new appendix paragraph/table (with `\textcolor{blue}{...}` as elsewhere).

Do not claim measured throughput/memory unless later added; R1.6 explicitly listed those as examples (“such as”), so FLOPs addressing overhead is in scope.

## Acceptance criteria

- Script runs without GPU and prints FLOPs for all four modes.
- Appendix table numbers match script output.
- `R1.Respone.tex` R1.6 has no `\PLACEHOLDER` remaining for that comment.
- No absolute GPU timing/memory presented as primary evidence.
- No new technical claims beyond FLOPs accounting and already-established method facts (`no_grad`, six scalars).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Dense vs nonzero MAC counting disputed | State explicitly: count as implemented dense `conv2d` MACs |
| Reviewer still wants wall-clock | Response notes FLOPs chosen for hardware independence; optional future timing not required for this revision |
| Backbone FLOPs citation weak | Omit backbone ratio column if no solid number |
