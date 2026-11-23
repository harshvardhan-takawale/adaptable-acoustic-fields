# P4-9 0c: first-order directional output, zero-shot

Unit test (E3): passes = True; the code's per-ray error equals the closed form |H1/(jH0) - 1| = True; its largest deviation from 1/(2kr) over kr 1-30 is 0.086; omni-FD reproduces the analytic gradient to 9.30e-04.

Directional r = Re of the normalised complex inner product with FDTD grad p / (jk), per bin over off-lattice receivers >= 0.5 m from the source and both components, band means (|rho| in the JSON).

### In-family (15)

| readout | Re rho 25-100 / 100-200 / 200-300 | abs rho 25-100 |
|---|---|---|
| R0_120K per-ray | 0.835 / 0.795 / 0.780 | 0.901 |
| R0_120K omni-FD | 0.848 / 0.739 / 0.747 | 0.857 |
| R0_120K proj-FD | 0.943 / 0.844 / 0.803 | 0.953 |
| S2_120K per-ray | 0.822 / 0.763 / 0.719 | 0.894 |
| S2_120K omni-FD | 0.751 / 0.624 / 0.623 | 0.760 |
| S2_120K proj-FD | 0.935 / 0.808 / 0.742 | 0.946 |

### Unseen (16)

| readout | Re rho 25-100 / 100-200 / 200-300 | abs rho 25-100 |
|---|---|---|
| R0_120K per-ray | 0.553 / 0.567 / 0.596 | 0.673 |
| R0_120K omni-FD | 0.580 / 0.549 / 0.580 | 0.628 |
| R0_120K proj-FD | 0.651 / 0.625 / 0.624 | 0.704 |
| S2_120K per-ray | 0.563 / 0.601 / 0.636 | 0.681 |
| S2_120K omni-FD | 0.534 / 0.502 / 0.551 | 0.572 |
| S2_120K proj-FD | 0.670 / 0.652 / 0.659 | 0.719 |

### U (8)

| readout | Re rho 25-100 / 100-200 / 200-300 | abs rho 25-100 |
|---|---|---|
| R0_120K per-ray | 0.569 / 0.589 / 0.636 | 0.674 |
| R0_120K omni-FD | 0.606 / 0.571 / 0.618 | 0.643 |
| R0_120K proj-FD | 0.676 / 0.646 / 0.665 | 0.717 |
| S2_120K per-ray | 0.571 / 0.602 / 0.661 | 0.687 |
| S2_120K omni-FD | 0.553 / 0.508 / 0.574 | 0.587 |
| S2_120K proj-FD | 0.685 / 0.652 / 0.682 | 0.728 |

### C3 (8)

| readout | Re rho 25-100 / 100-200 / 200-300 | abs rho 25-100 |
|---|---|---|
| R0_120K per-ray | 0.537 / 0.544 / 0.556 | 0.672 |
| R0_120K omni-FD | 0.554 / 0.528 / 0.542 | 0.612 |
| R0_120K proj-FD | 0.625 / 0.604 / 0.584 | 0.690 |
| S2_120K per-ray | 0.556 / 0.600 / 0.611 | 0.674 |
| S2_120K omni-FD | 0.515 / 0.496 / 0.528 | 0.557 |
| S2_120K proj-FD | 0.655 / 0.652 / 0.636 | 0.710 |

