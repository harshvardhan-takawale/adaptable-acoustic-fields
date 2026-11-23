# P4-9 S-4: sm_86/89 tinycudann build -- parity and throughput (D181 E8, c5)

R0 @120K, 15 in-family dense test rooms, off-lattice r (frozen scorer); both envs rendered on one node (NVIDIA RTX A5000).

| band | aaf (cc 52 wheel) | aaf_sm89 | delta | parity (abs <= 0.003) |
|---|---|---|---|---|
| 25-100 Hz | 0.9185 | 0.9185 | +0.0000 | yes |
| 100-200 Hz | 0.7009 | 0.7009 | +0.0000 | yes |
| 200-300 Hz | 0.5818 | 0.5818 | +0.0000 | yes |

Max relative field difference (bins 0-600, any receiver): 3.89e-06.
Training throughput (R0's recipe, 300 steps, same node): aaf 0.761 s/it, aaf_sm89 0.573 s/it (1.33x).

**Parity HOLDS.** Adoption is a P4-10 decision (E8); no P4-9 arm switches environments.
