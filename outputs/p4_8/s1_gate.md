# P4-8 S1 gate -- finite-difference Helmholtz residual (D174(g))

On each dense room's 64x64 lattice (the off-lattice evaluation receivers), per bin f in 25-300 Hz:
    R = lap_h F + k^2 (F - <F>),     rel(f) = ||R|| / ||k^2 (F - <F>)||   over INTERIOR stencils
where <F> is the complex spatial mean (the room-uniform record component, D170(c), removed as the manager specifies;
a constant has zero Laplacian, so only the k^2 term carries it), and lap_h is the 5-point stencil with UNEQUAL
spacings taken from the stored (solver-snapped) receiver coordinates: d2F/dx2 = 2[(F+ - F0)/h+ - (F0 - F-)/h-] /
(h+ + h-), likewise in y. A stencil is interior when its 4 neighbours are on the room's lattice; stencils within
0.3 m of the source are dropped (the field is singular there).
Gate (i): the FDTD ground truth's band-mean rel residual <= 0.10 (D174(g)); (ii) on G1 / GX predictions, the
residual on unseen-topology rooms is >= 2x the in-family one. Reported either way. Writes outputs/p4_8/s1_gate.{json,md}.

| quantity | value |
|---|---|
| (i) FDTD relative residual, band-mean 25-300 Hz, mean over 31 rooms | 0.057 (threshold 0.10) |
| (ii) G1_60K: in-family (15 test shapes) / unseen (16) / U / C3 | 4.199 / 4.398 / 4.251 / 4.545; ratio 1.05 (threshold 2) |
| (ii) GX_60K: in-family (15 test shapes) / unseen (16) / U / C3 | 5.415 / 5.501 / 5.427 / 5.576; ratio 1.02 (threshold 2) |

**Gate:** {"i_fdtd_rel_mean": 0.05737805385444093, "i_threshold": 0.1, "i_holds": true, "ii_ratios": {"G1_60K": 1.0473308694563013, "GX_60K": 1.016008229479832}, "ii_threshold": 2.0, "ii_holds": false, "launch_S1": false}
