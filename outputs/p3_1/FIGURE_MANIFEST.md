# P3-1 — figure manifest (data report)

Index of the figures generated for P3-1. Numbers, axes, and sources only; interpretation left to the
reader. All figures are the edit-sweep set (Part C of the plan). The broader 8-panel `meeting_assets`
pack described in the plan was **not generated**; the figures that exist are the six listed below.

Commit: `dcf9a3c` (branch `main`). Raw-URL base:
`https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/`

## What the figures plot

All six come from one driver (`scripts/p3_1_edit_sweep.py`), run per arm over an 18-point unseen
L-sweep with W=4.0 m and H=3.25 m fixed (center receiver, in-band 0–300 Hz). G+ figures use the
~6,000-iter checkpoint.

- **`waterfall.png`** — `pcolormesh`: x = frequency 0–300 Hz, y = L (the edited dimension, 18 sweep
  points), color = predicted |H| (dB) at the center receiver. Overlays: analytic axial-L modes
  f = c·n/2L for n=1..4 (cyan); (0,1,0)=c/2W (green dashed, invariant); (0,0,1)=c/2H (yellow dotted,
  invariant).
- **`tracked_peaks.png`** — x = L, y = mode frequency (Hz, 0–300); analytic c·n/2L lines for n=1,2,3
  with the matched predicted peaks scattered on top; legend reports per-mode tracking MAE (Hz) and
  recall (matched sweep points / 18).

## Figures + source numbers

Per-arm tracking MAE (Hz) / recall over the first three axial-L modes, from
`outputs/p3_1/edits/{arm}/edit_sweep_summary.json` (n_sweep = 18):

| Arm (checkpoint) | mode 1 MAE / recall | mode 2 MAE / recall | mode 3 MAE / recall |
|---|---|---|---|
| L (best) | 1.46 / 0.389 | 1.61 / 0.722 | 2.69 / 0.778 |
| G (best) | 1.40 / 0.722 | 1.43 / 0.833 | 3.05 / 0.778 |
| G+ (~6K) | 1.98 / 0.778 | 2.15 / 0.500 | 2.81 / 0.778 |

| # | Figure | Raw URL |
|---|---|---|
| 1 | Arm L — waterfall | `https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p3_1/edits/arm_L/waterfall.png` |
| 2 | Arm L — tracked peaks | `https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p3_1/edits/arm_L/tracked_peaks.png` |
| 3 | Arm G — waterfall | `https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p3_1/edits/arm_G/waterfall.png` |
| 4 | Arm G — tracked peaks | `https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p3_1/edits/arm_G/tracked_peaks.png` |
| 5 | Arm G+ — waterfall | `https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p3_1/edits/arm_Gplus/waterfall.png` |
| 6 | Arm G+ — tracked peaks | `https://raw.githubusercontent.com/harshvardhan-takawale/adaptable-acoustic-fields/main/outputs/p3_1/edits/arm_Gplus/tracked_peaks.png` |

## Reproduction

```bash
export PYTHONPATH="$PWD"
python scripts/p3_1_edit_sweep.py --arm-dir outputs/p3_1/arm_L     --out outputs/p3_1/edits/arm_L
python scripts/p3_1_edit_sweep.py --arm-dir outputs/p3_1/arm_G     --out outputs/p3_1/edits/arm_G
python scripts/p3_1_edit_sweep.py --arm-dir outputs/p3_1/arm_Gplus --out outputs/p3_1/edits/arm_Gplus
```

Sources: `scripts/p3_1_edit_sweep.py`, `outputs/p3_1/edits/{arm_L,arm_G,arm_Gplus}/edit_sweep_summary.json`,
`configs/sweeps_3d/p3_1_edit_sweep.yaml`. Headline zero-shot numbers: `outputs/p3_1/HEADTOHEAD.md`.

## Not generated
- The 8-panel `outputs/p3_1/meeting_assets/` pack from the plan (three-arm headline table image,
  modal-placement comparison, disentanglement 3-panel, median-room spectrum overlay, mode-shape
  spatial slices, `make_p3_1_meeting_pack.py`).
- Disentanglement figures (single-axis L/W/H sweeps): the eval was not run.
