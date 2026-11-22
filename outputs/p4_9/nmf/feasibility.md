# P4-9 NMF feasibility (physics only; plan I9)

P4-9 2c feasibility (D180 S-b; D181 A10; plan I9), PHYSICS ONLY, before any NMF code trains.

**S-b (pre-registered, D180): N = 24 modes reach 0.984 in family at 25-100 Hz -> HOLDS.**

| r at 25-100 Hz (off-lattice, vs FDTD) | in-family 15 | unseen 16 |
|---|---|---|
| modes: all | 0.988 | 0.978 |
| modes: 12 | 0.958 | 0.956 |
| modes: 16 | 0.976 | 0.967 |
| modes: 24 | 0.984 | 0.974 |
| modes: 32 | 0.985 | 0.975 |
| modes: 48 | 0.986 | 0.976 |
| modes: 24+static | 0.988 | 0.978 |
| 24 modes, eigenfrequency error 0.25% | 0.982 | 0.974 |
| 24 modes, eigenfrequency error 0.50% | 0.978 | 0.970 |
| 24 modes, eigenfrequency error 1.00% | 0.965 | 0.956 |
| 24 modes, eigenfrequency error 2.00% | 0.920 | 0.923 |
| 24 modes, eigenfrequency error 5.00% | 0.768 | 0.786 |
| 24 modes, mode-shape mixing delta 0.05 | 0.969 | 0.962 |
| 24 modes, mode-shape mixing delta 0.1 | 0.934 | 0.933 |
| 24 modes, mode-shape mixing delta 0.2 | 0.836 | 0.841 |
| 24 modes, mode-shape mixing delta 0.4 | 0.617 | 0.641 |

The 24th nonzero mode lies at 164 Hz (in-family median) / 166 Hz (unseen median).
N1's bars: unseen >= 0.60, in-family >= 0.90 (NMF or NMF-P). The error rows say how accurate a learned modal field must be to clear them.
