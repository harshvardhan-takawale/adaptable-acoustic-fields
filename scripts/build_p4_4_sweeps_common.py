"""Receiver-sampling constants for the demo-v2 sweeps, held to figN's corpus exactly.

These are `build_p4_2_sweep.py`'s values, not new ones. They live in their own module so that a
v2 strip and `figN_morph` are sampled identically -- a different step or clearance would change
the spatial Pearson without changing the model, and the two figures are meant to be read side by
side.
"""
RX_STEP = 0.08             # dense enough for a field-map strip; the trainer's 800 is not
RX_WALL_CLEAR = 0.12       # keeps receivers off the boundary condition
RX_CORNER_CLEAR = 0.15     # clears the r^(2/3) reflex-corner singularity, at EVERY corner
