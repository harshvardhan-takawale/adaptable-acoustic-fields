# P4-7 status report for the manager — 2026-10-17, 10:50 EDT

**State.**
- **Training:** one job. M3-fam (the modal hybrid) is queued for a 48 GB L40S on scavenger. It resumes from 26K and needs about 6 h once it starts.
- **Finished, final reads recorded:** m1, m2 (A-fam, A-500), m4's arm GX, and G3 at 120K.
- **M3-fam:** read at 20K as an interim. Its 30K and 60K standard reports launch automatically from detached watchers.
- **Decisions:** D169 (pre-registration), D170 (m1 + m3 pre-registration), D171 (m2 / GX / G3 finals) and D172 (m3 interim), in [DECISIONS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md).
- **Write-up:** [CHUNK_P4_7_RESULTS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/CHUNK_P4_7_RESULTS.md).

**Protocol for every number below.**
- Receivers: off-lattice (Rule 11).
- r: per-bin column Pearson of dB, band mean over 25–100 / 100–200 / 200–300 Hz. LSD: mean |dB error|.
- CIs: paired bootstrap, 10k resamples, seed 20261013. Over the 15 test shapes, or over the 6 out-of-family rooms (n = 6 is weak and is flagged where used).
- "Out of family" (OOF): D111's dense T, DN and U rooms, two per family.

---

## 1. The headline: physics carries the field; the network does not add to it

| model | 15 test shapes: r | OOF mean (T, DN, U): r | held-out U: r |
|---|---|---|---|
| **H_modal alone** (m1, physics, no training) | **0.988 / 0.976 / 0.960** | **0.978 / 0.974 / 0.962** | **0.973 / 0.972 / 0.960** |
| **M3-fam @20K** (H_modal + PB3 residual, family corpus, U held out) | 0.989 / 0.976 / 0.961 | — (U is its out-of-family family) | **0.977** / 0.972 / 0.960 |
| G1 @60K (best network, 500 shapes) | 0.931 / 0.721 / 0.581 | 0.628 / 0.372 / 0.358 | 0.718 / 0.434 / 0.402 |
| GX @60K (G1's recipe + 500 diverse rooms) | 0.907 / 0.646 / 0.504 | 0.738 / 0.484 / 0.419 | 0.790 / 0.495 / 0.463 |
| G2 @60K (family corpus, U held out) | 0.774 / 0.475 / 0.309 | 0.714 / 0.425 / 0.359 | 0.554 / 0.257 / 0.294 |

- **m1 (D170).** Each room's modes come from the solver's own discrete Laplacian (to 600 Hz, first-order Robin damping β = tanh(m/4), one global constant C).
  - With the zero mode's exact 2 s record, it reads 0.97–0.99 everywhere, in and out of family.
  - The pre-registered primary formula (continuous zero mode) reads 0.92 at 25–100 Hz out of family, against the 0.85 threshold. **m3 launched.**
  - **Checks:**
    - C is within 0.1% of the analytic dA/(c dt)²; the +ikη sign is confirmed.
    - Damping ×0.5 or ×2 loses 0.02–0.09 r, so the analytic damping is optimal.
    - Modes to 800 Hz change nothing (truncation converged).
    - Mode counts match Weyl's law within ±3.
  - **D101 refined:** the DC level is 1/(βP) exactly (corr 1.000); 1/area was a proxy.
- **m3 interim (D172), M3-fam @20K against H_modal alone on the same receivers:**
  - **(1) Out-of-family repair: MET.** U is 0.977 at 25–100 Hz, against G2's 0.554.
  - **(2) The network adds over physics: NOT MET at 20K.** The gain is +0.001 r [+0.001, +0.002] and −0.01 to −0.08 dB LSD; the threshold was ≥ +0.01 r or ≥ 0.2 dB.
  - **(3) Harm: none.**
  - The training-to-test gap is 0.004.
- **Caveat** (labelled; [OPEN_QUESTIONS Q24](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/OPEN_QUESTIONS.md)): the baseline IS the solver's operator, so its accuracy on FDTD truth is near-tautological. It needs exact geometry, source and β, and a solve costs about as much as the FDTD (median 102 s vs 58 s per dense room).

## 2. Finished — configuration diversity (m2) and the diverse corpus (m4 / GX), at 60K

| arm | Gate-2 off-lattice [CI] | test r | LSD < 100 Hz | training-room fit r (gap 200–300) | OOF mean r | U r |
|---|---|---|---|---|---|---|
| A-fam (G2 + D4 flips + reciprocity) | 0.833 [0.815, 0.851] | 0.806 / 0.521 / 0.405 | 2.22 | 0.937 / 0.709 / 0.571 (0.166) | 0.745 / 0.479 / 0.411 | **0.599** / 0.353 / 0.364 |
| G2 (parent) | 0.832 [0.810, 0.848] | 0.774 / 0.475 / 0.309 | 2.35 | 0.965 / 0.864 / 0.793 (0.484) | 0.714 / 0.425 / 0.359 | 0.554 / 0.257 / 0.294 |
| A-500 (G1 + flips + reciprocity) | 0.945 [0.937, 0.955] | 0.918 / 0.648 / 0.485 | 1.34 | 0.937 / 0.680 / 0.492 (0.008) | **0.603** / 0.375 / 0.389 | 0.726 / 0.384 / 0.392 |
| GX (C500 + CX500, per-room source) | 0.935 [0.921, 0.948] | 0.907 / 0.646 / 0.504 | 1.46 | 0.923 / 0.681 / 0.529 (0.025) | **0.738** / 0.484 / 0.419 | 0.790 / 0.495 / 0.463 |
| G1 (parent) | 0.955 [0.950, 0.960] | 0.931 / 0.721 / 0.581 | 1.17 | 0.960 / 0.806 / 0.675 (0.094) | 0.628 / 0.372 / 0.358 | 0.718 / 0.434 / 0.402 |

- **m2 pre-registration FAILS on both halves (D171(a)).**
  - A-fam's U rises +0.045, where ≥ +0.10 was needed.
  - A-500's OOF mean moves −0.025 [−0.080, +0.038].
  - **The in-family cost is real:** A-500 − G1 on the test shapes is −0.013 / −0.073 / −0.097, every CI excluding 0. Training-room fit falls just as far, so the augmented arm under-fits at matched iterations.
  - A-fam gains on the 15 L test shapes: +0.03 / +0.05 / +0.10. The family corpus has only NW L rooms, and the flips add the other corners.
  - **Leakage pre-flight** ([json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_7/m2_leak_preflight.json)): 0 U-topology images under all 8 D4 elements. The transposes leak the d̂ slab (43 / 240 and 88 / 500), so both arms used the flips.
- **GX (D171(b)), no pre-registered threshold:**
  - OOF +0.110 [+0.049, +0.178] / +0.112 / +0.061 over G1 (n = 6); U +0.07.
  - In family it costs −0.024 / −0.076 / −0.078, every CI excluding 0. GX has 1000 rooms at 60K iterations, so each room is seen half as often.
  - Labelled reading: genuinely new geometries and sources move the out-of-family field, where flips of the same rooms do not. Even so, 0.738 is far below physics' 0.978.
- **CX500 corpus** (m4): 500 rooms with 1–2 notches at any corner or side midpoint, U topology and both slabs excluded. 499 distinct sources, ≥ 0.5 m from every edge. 800 receivers per room, B6-style. 1.7 GB, built in ~1 h of CPU array time. Trainer gained `per_room_source` (default off; mixed-source corpora are now refused when it is off).

## 3. Finished — G3 at 120K (D171(c))

| | test r | training-room fit r | gap 200–300 |
|---|---|---|---|
| G3 @120K | 0.893 / 0.625 / **0.469** | 0.981 / 0.922 / 0.880 | **0.411** |
| G3 @60K | 0.897 / 0.642 / 0.491 | 0.968 / 0.871 / 0.798 | 0.307 |

- **Pre-registered test 200–300 Hz r ≥ 0.52 by 120K: FAILS** (0.469).
- 120K − 60K: −0.004 / −0.017 / −0.023, every CI excluding 0.
- 4× receivers plus a long schedule memorise the training rooms. Density does not buy transfer (with D168).

## 4. Pre-registration scorecard (P4-7)

| item | pre-registered | result |
|---|---|---|
| m1 | OOF 25–100 Hz r ≥ 0.85 → m3 launches | **MET**, 0.923 (0.978 with the exact zero mode) |
| m2 | A-fam U +0.10 over G2 AND A-500 OOF +0.10 over G1 | **FAILS** (+0.045; −0.025) |
| m2 | in-family within the parents' CI | **FAILS** for A-500 (−0.07 to −0.10 at 100–300 Hz); mixed for A-fam |
| GX | report against G1 | OOF +0.110; in-family −0.02 to −0.08 |
| G3 | test 200–300 r ≥ 0.52 by 120K | **FAILS** (0.469) |
| m3 (1) | held-out U 25–100 r ≥ 0.85 | **MET at 20K** (0.977); final at 60K |
| m3 (2) | the network adds ≥ 0.01 r @200–300 or ≥ 0.2 dB | **NOT MET at 20K** (+0.001; ≤ 0.08 dB) |
| m3 (3) | no band > 0.02 below H_modal alone | **MET at 20K** (no harm) |

## 5. Incidents and corrections this round

- **M3-fam ran out of GPU memory at 26.6K on an A5000** (24 GB, fragmentation).
  - A resume with the allocator option died in its first backward.
  - It was moved to a 48 GB L40S on scavenger and is queued. It carries ~1 GB more than G2 for the prior.
- **The first precompute array silently received one data set**, because `sbatch --export` splits on commas. Sets are now `+`-separated and the missing sets were recomputed.
- **Six precompute tasks died on legacy43**; that node is now excluded.
- **CX500 has 499 distinct sources, not 500** (two rooms share one); corrected.
- **D101 refined:** the DC level scales as 1/(βP), not 1/area (D170(d)).

## 6. Awaiting the manager's decision (nothing launched)

1. **Q24, now decisive.** On simulated data the INR's measured contribution over a per-room modal solve is about 0: m3 at 20K adds +0.001 r. What is the INR for? The options are in Q24:
   - (a) residual over physics, judged where the prior is WRONG;
   - (b) physics as a feature or regulariser for an INR that must work without exact geometry;
   - (c) a different target (real-time, inverse problems).
2. **A probe that would make m3 informative under (a)** (not launched): train M3-fam against a deliberately wrong prior (β ×2, or geometry snapped to 10 cm) and measure whether the residual network recovers the truth. m1 says the unaided prior loses 0.02–0.09 r and 1.0–2.4 dB LSD under β ×2.
3. **M3-500** (m3 on G1's corpus) is pre-registered. Its 500 training priors are computed. The config is a one-line change of P4_7_M3FAM.yaml (G1's corpus). It needs a 48 GB card, as M3-fam does.
   - Recommend deciding after M3-fam's 60K read, given (1).
4. **GX at matched exposure** (120K, each room seen as often as in G1) would separate "diverse data helps OOF" from "dilution costs in-family". Not launched.
5. **Scale augmentation** is unblocked in principle: D170(c) gives the zero mode's exact record, which is the subtraction p2 needed. Parked per the manager.

**Code added this round (all default off, behavioural tests per Rule 8):**
- `aaf/data/cx_rooms.py` (8 tests);
- trainer `per_room_source` (8 tests);
- `aaf/eval/modal_prior.py` plus the trainer / model / renderer / `load_model` modal-prior path (8 tests, including the prior against the actual FDTD on a small room and evaluator-path == trainer-path);
- 113 existing trainer tests pass.

**Scripts:**
- `scripts/p4_7_m1_modal.py`, `p4_7_m3_precompute.py`, `p4_7_m2_leak_preflight.py`, `build_p4_7_cx500.py`;
- `p4_7_report_tables.py` ([report_tables.json](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_7/report_tables.json));
- `p4_7_watch_report.sh` (detached per-checkpoint standard report).

**Links:**
- [m1_modal.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_7/m1_modal.md)
- [CHUNK_P4_7_RESULTS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/tasks/CHUNK_P4_7_RESULTS.md)
- [CONTEXT_FOR_MANAGER.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/CONTEXT_FOR_MANAGER.md)
- [DECISIONS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md) (D169–D172)
- [OPEN_QUESTIONS.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/OPEN_QUESTIONS.md) (Q24)
