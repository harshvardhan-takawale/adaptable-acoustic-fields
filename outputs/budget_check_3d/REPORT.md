# 3D dataset budget check

**Status**: PASS

- Per-room wall-clock limit: 600 s
- Per-room file-size limit: 500 MB
- Worst-case wall observed: 23.6 s
- Worst-case size observed: 48.8 MB

## Per-room measurements
| Label | L | W | H | wall (s) | t_ISM (s) | t_analytical (s) | size (MB) | max_order | T60 (s) | n_modes |
|---|---:|---:|---:|---------:|---------:|----------------:|----------:|----------:|--------:|--------:|
| smallest | 3.00 | 3.00 | 2.50 | 5.7 | 1.4 | 4.4 | 45.9 | 12 (capped) | 0.50 | 19978 |
| largest | 6.00 | 5.00 | 4.00 | 23.6 | 1.2 | 22.4 | 48.8 | 12 (capped) | 0.87 | 103611 |
