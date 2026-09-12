# Robustness Report: `20260913-013747_convnext_tiny_e4_canonical_convnext_s42`

- **Overall OOF BA**: 0.7094 @ t*=0.565
- **Shortcut-Wrong BA**: 0.3368 (N=616)
- **Worst Slice**: Azimuth [135°, 180°) (BA=0.4910)

### Per-Slice Breakdown

| Slice | Samples | Balanced Accuracy |
|---|---|---|
| Azimuth [  0°,  45°) | 0 | N/A |
| Azimuth [ 45°,  90°) | 0 | N/A |
| Azimuth [ 90°, 135°) | 0 | N/A |
| Azimuth [135°, 180°) | 458 | 0.4910 |
| Azimuth [180°, 225°) | 328 | 0.5452 |
| Azimuth [225°, 270°) | 0 | N/A |
| Azimuth [270°, 315°) | 786 | 0.5086 |
| Azimuth [315°, 360°) | 0 | N/A |
| Brightness Q1 (Darkest) | 393 | 0.6941 |
| Brightness Q2 (Medium-Dark) | 393 | 0.7681 |
| Brightness Q3 (Medium-Bright) | 393 | 0.7272 |
| Brightness Q4 (Brightest) | 393 | 0.6543 |
| Contrast Q1 (Lowest) | 393 | 0.6077 |
| Contrast Q2 (Low-Med) | 393 | 0.7156 |
| Contrast Q3 (Med-High) | 393 | 0.7668 |
| Contrast Q4 (Highest) | 393 | 0.7473 |
