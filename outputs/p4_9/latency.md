# P4-9 0a: latency by edit type (D180; D181 A3, A4, E5)

Room: L5.54_W5.28_d1.42_w1.52.h5 (3753 grid points). Network = R0 @120K on NVIDIA RTX A5000, fp32 (the shipped dtype, A4); physics and FDTD on one CPU node (legacy42.umiacs.umd.edu, Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz). Medians of 20 runs (5 above 5 s, D182(c)).

| edit | network, 1 listener (601-bin) | network, 64x64 grid (601-bin / current 4097-bin path) | physics, cached modes (1 listener / grid) | physics re-solve to 600 Hz | FDTD re-run | note |
|---|---|---|---|---|---|---|
| listener move | 9.9 ms | 32.4 s / 57.1 s | 2 ms / 49 ms | 55.6 s | 49.4 s | modes unchanged: lookup + sum |
| source move | 9.9 ms | 32.4 s / 57.1 s | 2 ms / 49 ms | 55.6 s | 49.4 s | modes unchanged: lookup + sum |
| material change | 12 ms | 32.4 s / 57.1 s | 2 ms / 49 ms | 55.6 s | 49.4 s | eta rescales (beta linear); network accuracy UNTESTED (never trained on materials) |
| geometry change | 12 ms | 32.1 s / 56.5 s | 55.6 s / 55.6 s | 55.6 s | 49.4 s | modes change: re-solve (no cache) |

**Single-listener update rate:** network 101 Hz (601-bin path; 61 Hz on the current path); physics with cached modes 496 Hz.
**Render-time projection** (64x64 lattice, 551 bins): 23.2 s; token build 1.8 ms.
**Largest dense room** (L6.70_W5.38_d0.36_w0.72.h5, 4056 points): network grid 35 s (601-bin) / 61.7 s (4097-bin); FDTD 59.6 s; modal re-solve 111 s.
**A 2D FDTD on a GPU:** ~0.1-0.3 s (ESTIMATE, not measured: ~7e4 air nodes x 61440 steps = 4.5e9 node updates at 15-50 G updates/s on one GPU).

Reading (labelled, D181 E5): with cached modes, physics answers listener, source and material edits faster than the network; the network's latency case is the GEOMETRY edit, where physics must re-solve.
