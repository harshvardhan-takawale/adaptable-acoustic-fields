# P4-10 V7: the late floor under ISO 3382-1 style compensation (D216; descriptive)

Per room: receiver median of |model - truth| / truth; table entries are means over rooms (95% CI over rooms when there are at least 3). Same receivers before and after; receivers that cannot be compensated are dropped from both.

| rooms | model | octave | receivers (valid in truth / not compensable) | EDT before -> after | T20 before -> after | T30 before -> after | receivers used for EDT / T20 / T30 |
|---|---|---|---|---|---|---|---|
| in-family 15 (15) | R1 | 63 Hz | 11421 / 118 | 14% -> 14% | 32% -> 33% | 31% -> 35% | 11302 / 11297 / 11282 |
| in-family 15 (15) | R1 | 125 Hz | 11421 / 117 | 24% -> 24% | 57% -> 61% | 514% -> 55% | 11301 / 11288 / 11280 |
| in-family 15 (15) | R1 | 250 Hz | 11157 / 141 | 30% -> 29% | 52% -> 60% | 546% -> 55% | 11015 / 11002 / 10993 |
| in-family 15 (15) | R1-M | 63 Hz | 11421 / 117 | 18% -> 18% | 35% -> 36% | 35% -> 38% | 11304 / 11299 / 11288 |
| in-family 15 (15) | R1-M | 125 Hz | 11421 / 81 | 25% -> 25% | 60% -> 62% | 489% -> 59% | 11338 / 11332 / 11321 |
| in-family 15 (15) | R1-M | 250 Hz | 11157 / 136 | 34% -> 34% | 59% -> 65% | 518% -> 61% | 11016 / 11000 / 10989 |
| S-1 rooms (2) | R1 | 63 Hz | 1568 / 8 | 13% -> 13% | 35% -> 36% | 34% -> 40% | 1560 / 1558 / 1558 |
| S-1 rooms (2) | R1 | 125 Hz | 1568 / 7 | 26% -> 26% | 52% -> 55% | 390% -> 50% | 1561 / 1559 / 1557 |
| S-1 rooms (2) | R1 | 250 Hz | 1505 / 15 | 30% -> 30% | 43% -> 49% | 474% -> 43% | 1489 / 1489 / 1487 |
| S-1 rooms (2) | R1-M | 63 Hz | 1568 / 6 | 17% -> 17% | 38% -> 38% | 37% -> 41% | 1562 / 1561 / 1561 |
| S-1 rooms (2) | R1-M | 125 Hz | 1568 / 11 | 28% -> 28% | 56% -> 58% | 357% -> 55% | 1556 / 1555 / 1555 |
| S-1 rooms (2) | R1-M | 250 Hz | 1505 / 12 | 36% -> 36% | 50% -> 55% | 445% -> 50% | 1492 / 1491 / 1490 |
| S-1 rooms (2) | S-1 | 63 Hz | 1568 / 2 | 0% -> 0% | 1% -> 1% | 6% -> 2% | 1566 / 1566 / 1566 |
| S-1 rooms (2) | S-1 | 125 Hz | 1568 / 0 | 1% -> 1% | 2% -> 1% | 18% -> 2% | 1568 / 1568 / 1568 |
| S-1 rooms (2) | S-1 | 250 Hz | 1505 / 1 | 2% -> 2% | 3% -> 3% | 35% -> 8% | 1504 / 1504 / 1504 |

ISO 3382-1 range (model, peak to noise): receivers below 35 dB (T20) and below 45 dB (T30), summed over rooms and octaves, per model: R1: 130 / 15821; R1-M: 93 / 12936; S-1: 14 / 1522

The truth is not compensated. At 250 Hz the truth's own record is limited (floor about -40 dB), so T30 there is the weakest reading.
