# P4-8 meeting figures (D174(h))

Network models only. Best wave-1 arm by rule: **LS3_60K** (no arm passed its pre-registered test; largest dHF + dUF).

| file | what | data |
|---|---|---|
| F1_freq_curve_in_family.png | per-bin off-lattice r, mean over 15 shapes, 11-bin smoothing | gate2_dense caches |
| F2_freq_curve_unseen.png | same, frozen unseen set, U and C3 separately | outputs/p4_8/cache/unseen_* |
| F3_fields_in_family.png | S1-layout field maps, rows 50/100/200/280 Hz, columns FDTD / G1 / R0 / best; per-row limits = FDTD 2-98th pct; corner r off-lattice | lower-median in-family shape by G1 @60K 200-300 Hz r (0.592) |
| F3_fields_U.png | S1-layout field maps, rows 50/100/200/280 Hz, columns FDTD / G1 / R0 / best; per-row limits = FDTD 2-98th pct; corner r off-lattice | lower-median U room by G1 @60K 25-100 Hz r (0.367) |
| F3_fields_C3.png | S1-layout field maps, rows 50/100/200/280 Hz, columns FDTD / G1 / R0 / best; per-row limits = FDTD 2-98th pct; corner r off-lattice | lower-median C3 room by G1 @60K 25-100 Hz r (0.272) |
| F4_scatter_unseen.png | per-room off-lattice r on the unseen set, best arm vs G1 @60K, per band | outputs/p4_8/unseen/*.json |
