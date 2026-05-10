# Visual sanity notes

After eyeballing all 16 PDFs (15 per-room + 1 cross-room) generated against the rebuilt 8192-sample dataset.

## What looks right

- **Modal-tracking scatter (cross-room page 1)**: every picked ISM peak across all 15 rooms lies on the y=x diagonal versus the nearest analytical eigenfrequency. Visually no points fall off the diagonal at any frequency. Confirms the sub-Hertz MAE finding from the noise-floor report.
- **Mode-frequency-vs-L curves (page 2)**: the (1,0)/(2,0)/(3,0) hyperbolae and the (0,1)/(0,2) horizontals are followed cleanly by the picked peaks — the (1,0) blue × tracks the c/(2L) curve perfectly for L=2.5–4.25 m and is correctly absent at L≥4.5 m, where the (1,0) frequency drops below the (0,1) line at 42.875 Hz and the picker can't separate them. This is expected behaviour, not a bug.
- **T60 vs L (page 3)**: pra's `rt60_theory(sabine)` exactly equals the textbook 3D Sabine formula `T60 = 0.161·V/(α·S)` applied with V=A and S=P (the dashed line and blue dots overlap perfectly). The EDC-measured T60 from the actual RIR is consistently 30–40% lower than Sabine prediction (515 ms vs 826 ms at L=2.5; 826 ms vs 1330 ms at L=6.5). This Sabine-overestimates-EDC pattern is well-known at moderate absorption (α=0.15) when the diffuse-field assumption is weak — pyroomacoustics is mathematically self-consistent; the disagreement is a model-vs-measurement gap, not a bug.
- **Per-room IR + EDC (each L\*.pdf page 1)**: the linear RIR shows the expected bright initial impulse + reverberant tail. Schroeder EDC is monotonic and clean across all 15 rooms. The 2.0-second IR window (n_time_samples=8192 at fs=4096) covers ~1.5–2.5× the EDC T60 in every room, so the -5..-35 dB regression segment is comfortably within the captured signal.
- **Modal-regime zoom (page 1 bottom-right)**: at L=4.00 m the spectrum shows clear peaks aligned with the orange analytical ticks. The (1,0)/(0,1) degenerate pair at 42.875 Hz appears as a thicker tick (multiplicity-aware width) and the spectrum has a single broad peak there — confirms degeneracy is real and the dedup machinery handles it.
- **8×8 sparkline grids (page 2)**: every receiver shows a non-trivial IR with similar overall envelope. No dead spots. Each panel is auto-scaled so relative amplitudes aren't comparable visually, but the temporal shapes are uniformly reasonable.

## What surprised me

- **Modal-regime spectra are surprisingly smooth.** At α=0.15 in 2D I expected sharper Q peaks, but the per-room 0–200 Hz views show broad humps rather than narrow resonances. Likely consequence of the higher-than-expected modal density even below 200 Hz (in the L=2.5 m room there are ~16 analytical modes ≤ 200 Hz). Each individual mode has a half-width on the order of 4–6 Hz from Sabine damping, and they overlap.
- **Source-receiver geometry asymmetry visible in sparklines**: at L=4 the sparklines in the right-half columns (x ≥ 2.7 m) show longer-looking decay tails than the left-half (x ≤ 1.3 m). Because the source is at (0.5, 0.5), receivers far from the source have weaker direct path and proportionally stronger reverberant tail — the auto-scaled sparkline visualisation amplifies this. Not an anomaly, just an artifact of the per-panel y-axis normalisation.
- **EDC T60 is consistently ~60% of pra's Sabine prediction**, ratio is stable across L. This suggests the multiplicative gap is dominated by a shared factor (probably the diffuse-field assumption breakdown at modest α) rather than something L-dependent. For Chunk 2's neural model, **train/eval against EDC values, not Sabine**, since EDC is what the data actually exhibits.

## What needs follow-up (not blocking)

- **Picker missed (1,0) at L ≥ 4.5 m**: in the cross-room page-2 plot, the blue × markers for the (1,0) family disappear once `f_(1,0) = c/(2L)` drops below ~38 Hz. This is the (1,0) being subsumed by the broader (0,1) peak at 42.875 Hz. Not a problem for the dataset (the IR encodes it correctly); the eigenfrequency probe just can't isolate it. Affects how Chunk 2 reports per-mode MAE for L > 4.5 m.
- **The Sabine vs EDC gap is large enough to matter for any energy-decay loss in Chunk 3+.** If we ever add an L1 loss on EDC slope, we should compare model EDC to dataset EDC, not to Sabine.
