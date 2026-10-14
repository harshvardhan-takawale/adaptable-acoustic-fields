# P4-5 meeting figures (2026-10-14)

All panels: existing 60K checkpoints (C250 = `outputs/p4_4/p4_4_C250`, B5 = `outputs/p4_5/arms/B5`), existing dense FDTD test simulations, OFF the training lattice, 15-shape protocol unless stated. 2560x1440, white background. Generator: `scripts/p4_5_meeting_figs.py`.

| file | what it shows | selection rule |
|---|---|---|
| S1_a_shape1000_fields.png | S1 (a): dB magnitude field maps, FDTD vs C250 @60K vs B5 @60K, rows 50/100/200/280 Hz (single 0.5 Hz bins), shared per-row limits = GT 2-98th pct; corner r/LSD off-lattice, frozen metric functions | lower-median shape by B5 @60K off-lattice 200-300 Hz band-mean r (rank 8 of 15, r = 0.471) (L_L5.54_W5.28_d1.42_w1.52) |
| S1_b_shape1001_fields.png | S1 (b): dB magnitude field maps, FDTD vs C250 @60K vs B5 @60K, rows 50/100/200/280 Hz (single 0.5 Hz bins), shared per-row limits = GT 2-98th pct; corner r/LSD off-lattice, frozen metric functions | most wall segments (6), tie-break lowest shape id (the first pick coincided with (a); next-lowest id taken) (L_L6.06_W4.52_d1.08_w2.44) |
| S1b_U_family_heldout_fields.png | S1b: same layout, held-out shape FAMILY (U): a U-room, a family neither model trained on | D111's median-d_hat U test room (rule fixed in D111) |
| S3_paired_scatter.png | S3: per-shape off-lattice band-mean dB spatial r, C250 (x) vs B5 (y), 100-200 and 200-300 Hz | all 15 frozen test shapes (no selection) |
| S2_a_shape1000_abs_error.png | S2 (a): absolute dB error maps, C250 vs B5, shared 0-15 dB, rows 50/100/200/280 Hz | same shape as S1 (a) |
| S2_b_shape1001_abs_error.png | S2 (b): absolute dB error maps, C250 vs B5, shared 0-15 dB, rows 50/100/200/280 Hz | same shape as S1 (b) |
| S4_kspace_radial.png | S4: radial wavenumber spectrum (D108's field_stats, complex field, 2x zero-pad), FDTD vs C250 vs B5, 100-300 Hz (every 4th bin), k0 and 1.5 k0 marked | all 15 frozen test shapes |
| S5_lattice_distance.png | S5: band-mean dB r by distance class to the nearest 0.12+0.06k lattice point (0 / 2 / 2.8 cm), C250 vs B5, 25-100 and 200-300 Hz | all 15 frozen test shapes |
