# `demo_edit_2d.py` — live material-edit demo (P3-2)

Renders one trained model twice for a rectangle it has never seen: once with all four walls
at the M0 baseline, once with **one** wall swapped to a new material. Nothing is looked up
and nothing is optimised at demo time — the only thing that differs between the two renders
is the 64-d conditioning vector `fourier_features_2d(L, W, alphas)`, computed from the
numbers on the command line. Produces one figure + a falsifiable number on stdout in ~20 s.

## Run it

```bash
# needs a GPU (tinycudann); a login node cannot import the model
sbatch scripts/slurm/demo_edit_2d.sh \
    --L 4.51 --W 4.00 --wall west --material curtain --gt data/track_c_2d
```

`--wall` = `west | east | south | north` (west/east are x = 0 / x = L; south/north are
y = 0 / y = W). `--material` accepts any alias in `aaf.walls`: `baseline|brick`,
`concrete|hard`, `curtain`, `absorber|panel`, or `M0`..`M3`.

| flag | default | note |
|---|---|---|
| `--receiver` | `corner` | `corner`, `center`, or an integer 0–63 |
| `--gt` | *(none)* | ISM ground truth: the data **directory** or the edited config's `.h5` |
| `--checkpoint` | newest in `outputs/p3_2/p3_2_main` | any `ckpt_iter*.pt` |
| `--out` | `outputs/p3_2/demo` | writes `demo_L<L>_W<W>_<wall>_<material>.{png,json}` |
| `--rx-chunk` | 8 | receivers per forward pass; lower it if the GPU is tight |

**Why the far corner is the default receiver.** On the 8×8 grid the near-centre receiver
sits close to a node of the odd axial modes (mode (1,0) is ~14 dB down there) and receiver 0
is dominated by direct sound, so both make the single-receiver panel harder to read than the
physics deserves. `--receiver center` still works and is honestly labelled. The *numbers*
never depend on this choice: they come from projecting all 64 receivers onto the analytic
mode shapes (`aaf.eval.modal_projection`), so every measurement is attributable to a single
(n_x, n_y).

## What the figure shows

1. **Spectrum, 0–300 Hz** at the chosen receiver — edited over baseline (plus ISM ground
   truth when `--gt` is given), with the analytic eigenfrequencies of the family the edited
   wall drives marked. The inset zooms the most-affected mode's **modal-projected** spectrum
   and draws its −3 dB width, which is where the headline number actually comes from.
2. **Spatial |field|** on the 8×8 grid at that mode, baseline vs edited, shared dB scale;
   the edited wall is drawn in red where it physically is.
3. **Band-limited (0–300 Hz) impulse response** at the same receiver, thin, with the energy
   decay curve over it thick — the edited room decays visibly faster.

## The claim, and its scope

> Editing a wall broadens **its own** mode family's −3 dB bandwidth and leaves the
> orthogonal family essentially untouched.

stdout prints ΔBW and Δlevel for the driven family, the other family and the tangential
family, the selectivity ratio, and both competing damping laws' predictions for the same
quantity. Bandwidth — not level — is the observable: measured level selectivity is only
~4:1 while bandwidth selectivity is ~29:1 (`outputs/p3_2/SIM_VALIDATION.md`).

**Scoping — say this out loud in the meeting.** That ~29:1 is a property of the ISM
simulator, which uses an angle-independent reflection coefficient and therefore has no
grazing-incidence absorption: a purely axial mode is damped only by the wall pair it bounces
between, exactly. Real locally-reacting walls follow Kuttruff and would give only ~2:1, with
no invariant family. The claim this demo supports is **"the model learns the simulator's
per-wall law"** — both predictions are printed side by side so the audience can see which
one the model reproduces.

Absolute widths run ~1.66× the γ/π prediction (the physics gate's T5 calibration slope); it
is the **ratio**, not the width, that is under test.

## Suggested demo sequence

```bash
--L 4.51 --W 4.00 --wall west  --material curtain    # held-out combo, unseen geometry
--L 4.51 --W 4.00 --wall north --material absorber   # the other held-out combo
--L 4.51 --W 4.00 --wall east  --material curtain    # the twin: same T60, other wall
--L 4.51 --W 4.00 --wall west  --material concrete   # concrete SHARPENS (ΔBW < 0)
```

The third one is the point of the design: `(west, M2)` and `(east, M2)` have identical mean
absorption and identical T60 and differ only in *where* the absorber sits, so a model that
had learned a scalar effective absorption could not tell them apart. The fourth shows the
material axis is signed, not just "more absorption = worse".

`(west, curtain)` and `(north, absorber)` are the held-out combos — never trained in any
geometry. The script prints whether the requested combo was held out and how far the
requested geometry is from the nearest trained one, so the zero-shot claim is checkable at
demo time rather than asserted.

## Outputs

- `outputs/p3_2/demo/demo_L4.51_W4.00_west_M2.png` — the figure
- `outputs/p3_2/demo/demo_L4.51_W4.00_west_M2.json` — every printed number, plus the
  checkpoint path/iteration, the receiver and mode choices and the ground-truth comparison
