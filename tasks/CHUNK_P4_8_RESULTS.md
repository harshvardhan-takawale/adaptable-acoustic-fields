# Chunk P4-8 — Multi-band models and spatial losses (network-only): RESULTS (live)

**Status (2026-10-18):**
- Step 0 is done (D175): the topology audit, the frozen unseen-family set, and the references re-scored on it. The S1 gate fails, so S1 is not launched.
- All eight wave-1 arms are training.
- Wave 2 is decided after the 60K reads.
- The S1 gate diagnostic and S2 (CX1000) are under way.

Pre-registration is [D174](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/DECISIONS.md), recorded before any P4-8 number. M3-fam's final read is D173, recorded for the next meeting; per the manager it is kept out of this round's tables and figures.

**Primary metrics** (off-lattice, paired CIs):
- **HF:** in-family test r at 200–300 Hz on the 15 frozen shapes.
- **UF:** unseen-family r at 25–100 Hz (also 100–200 and 200–300) on the frozen set below.

---

## Step 0 — evaluation prep

**0a. Topology audit** ([topology_audit.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_8/topology_audit.md)). Read from each room's polygon:

| corpus | rect | L | T-like | DN | U | LT | TT | C3 |
|---|---|---|---|---|---|---|---|---|
| C250 / C500 train | 8 | 242 / 492 | 0 | 0 | 0 | 0 | 0 | 0 |
| CX500 | 0 | 143 | **174** | **19** | 0 | 110 | 54 | 0 |
| CX1000 (S2) | 0 | 268 | 342 | 41 | 0 | 230 | 119 | 0 |
| family (FAM) train | 60 | 60 | 60 | 60 | 60 | 0 | 0 | 0 |

- **CX500 contains T-like and DN rooms, so T and DN are in-distribution for every GX-corpus arm.**
- GX's P4-7 "+0.110 out of family" splits into T +0.051, DN +0.206 (both in-distribution) and U +0.072 (truly unseen).
- No corpus that any P4-8 arm or reference trained on contains U or any 3-notch room.

**0b. The unseen-family set, FROZEN** ([manifest](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/configs/sweeps_2d_mat/p4_8_unseen_manifest.json), rows_sha256 `ba34b17c…`):
- **8 U rooms:** D111's 2, plus FAM test-split indices 3, 6, 2, 7, 1, 8, i.e. "next nearest to the median d_hat, alternating".
- **8 C3 rooms:** NW + NE + SE, the three-corner, 10-vertex, Z-corridor-like topology. The first 8 valid draws of seed 20261018.
- **Simulation:** dense as D111 (64×64 lattice clipped to the polygon, 3158–3850 receivers per room), source (0.5, 1.6).

**0c. References on the unseen set** (D175(c)). Off-lattice r at 25–100 / 100–200 / 200–300 Hz:

| model | HF | UF (16) | U (8) | C3 (8) |
|---|---|---|---|---|
| G1 @60K | 0.581 | 0.384 / 0.300 / 0.298 | 0.444 / 0.347 / 0.349 | 0.323 / 0.253 / 0.246 |
| **GX @60K** (reference) | 0.504 | **0.460 / 0.389 / 0.375** | 0.480 / 0.417 / 0.418 | 0.439 / 0.361 / 0.333 |
| GX @30K (kill reference) | 0.413 | 0.466 / 0.390 / 0.365 | 0.493 / 0.411 / 0.388 | 0.440 / 0.370 / 0.342 |
| PB3 @60K | 0.496 | 0.357 / 0.288 / 0.273 | 0.423 / 0.326 / 0.304 | 0.291 / 0.249 / 0.242 |
| B5 @60K | 0.467 | 0.405 / 0.311 / 0.284 | 0.473 / 0.360 / 0.355 | 0.337 / 0.261 / 0.213 |

- **GX vs G1:** GX − G1 is UF +0.076 [+0.044, +0.114] and HF −0.078. GX's UF is flat from 30K to 60K, while its HF still climbs.
- **The expanded set is much harder than D111's two U rooms:** G1 scores 0.444 on the 8 U rooms vs 0.718 on the 2.

## Wave 1 (GX corpus; each arm differs from GX in one thing)

| arm | change | hardware, speed | pre-registered (D174(e)) | status |
|---|---|---|---|---|
| R0 | 120K cosine (matched exposure) | tron A5000, 0.76 s/it | HF ≥ 0.56 AND UF not below GX@60K within CI | training; reads at 30 / 60 / 90 / 120K |
| MB1 | per-band models, wavelength-matched grids (6 levels, 1 m → 12.4 / 8.75 / 7.15 cm; 98K / 178K / 250K grid params per band, all dense) | L40S, 0.53 s/it | UF ≥ +0.05 & HF ≥ −0.02, or HF ≥ +0.03 & UF ≥ −0.02 | training |
| MB2 | per-band models, default grid per band (33.8M grid params per band) | L40S, 0.57 s/it | as MB1 | training |
| LS1A / LS1B | neighbour-difference loss, λ 0.3 / 1.0 (ratio to the spec terms at init; S_init 4.92) | tron A5000, 0.76 s/it | HF ≥ +0.03 or UF ≥ +0.05, no primary or band more than 0.02 worse | training |
| LS2A / LS2B | per-room correlation loss, λ 0.1 / 0.3; 1 room × 32 receivers, exact two-pass step, 30K steps | scavenger A5000, 1.93 s/it | as LS1, plus band LSD ≤ +0.2 dB | training |
| LS3 | floored log-amplitude (B5F keys) | tron A5000, 0.76 s/it | as LS1 | training |

**LS1 sampler:**
- 698,941 anchors have a neighbour within 0.30 m and 1,059 (0.15%) have none; anchors with none are never drawn.
- Median 7 neighbours per anchor; mean pair distance 0.198 m.
- Each step uses 8 pairs, with each pair rendered in one renderer call.

**Kill rule at 30K** (LS2: 15K): an arm is stopped only if BOTH primaries are below GX@30K, with paired CIs excluding 0.

**Wave 2** (after the 60K reads): the best MB combined with the best LS, at 120K, compared with R0@120K.

## Stretch

- **S1 gate** ([s1_gate.md](https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p4_8/s1_gate.md), D175(d)): **FAILS, so S1 is not launched.**
  - (i) holds: the FDTD residual is 0.057.
  - (ii) fails: G1 / GX fields violate the Helmholtz equation massively (relative residual 4.2 / 5.4), but equally in family and out (ratio 1.05 / 1.02, needed ≥ 2).
- **S2:** CX1000 (1000 rooms, 3.4 GB) is built. PB3 at 180K on C500 + CX500 + CX1000 runs only if slots free up after wave 1.

## Engineering (eced517; all default-off; 19 new Rule-8 tests, 129 existing pass)

- **Model:** separate per-band models built after every default module. The grid spec is a buffer checked on load.
- **Trainer:**
  - `ls_mode` selects the step function, so the default `_step` is byte-identical.
  - LS1: interleaved pairs in one renderer call. LS2: an exact two-pass step that requires pass 2 to reproduce pass 1 bit-for-bit.
  - P4-8 keys are in the resume guard (compared with their defaults when absent).
  - Unknown YAML keys are now FATAL.
- **Renderer:** a `jitter=` override.
- **Reports:**
  - `scripts/p4_8_watch_report.sh`: the standard report plus UF, launched per checkpoint.
  - `scripts/p4_8_primary_table.py`: HF / UF paired against GX, the verdicts and the kill rule.
  - `scripts/p4_8_meeting_figs.py`: the four meeting figures.
  - `scripts/p4_8_finalize.sh`: builds the table and figures automatically once the finals exist.
