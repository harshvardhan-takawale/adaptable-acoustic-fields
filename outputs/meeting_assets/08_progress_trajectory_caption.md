**Modal-regime zero-shot LSD across project iterations.** Pure modal (0-250 Hz)
LSD on the same 6 unseen L (3.25, 3.75, 4.25, 4.75, 5.25, 5.75 m) throughout.
Bars: Chunk 3 → 3.70 dB, Chunk 3.5 best → 3.66, Chunk 3.6 best → 3.51, Chunk
3.7 D1+B1 → **2.55 dB** (the I1 denser-training-sweep result). The 1.15 dB drop
from Chunk 3 to Chunk 3.7 is driven entirely by training-data density —
Chunks 3.6's full set of inner-loop-strategy and architecture experiments
moved modal LSD by ~0.15 dB total, while Chunk 3.7's I1 (15 rooms at 0.2 m
spacing instead of 7 at 0.5 m) gave us 1 dB in a single retrain.

L=5.25 and L=5.75 individually land at modal LSD 2.33 and 2.28 dB
respectively — within 0.3 dB of the 2 dB Phase-1 target.

The Chunk-3 bar (*) is a retrospective estimate: modal-band LSD wasn't
measured at the time, but Chunk 3.6 Track A's band-limited recompute on R0
(the closest architectural relative of the original Chunk-3 run) gives modal
3.69 dB (rounded to 3.70 for the bar). The Chunk-3.5/3.6/3.7 bars are direct
measurements from the respective `metrics.json` band_metrics_held fields.
