# Robustness Report: `20260913-010428_resnet18_e3_raw_labelflip_resnet18_s42`

- **Overall OOF BA**: 0.5664 @ t*=0.560
- **Shortcut-Wrong BA**: 0.4230 (N=616)
- **Worst Slice**: Azimuth [135°, 180°) (BA=0.4479)

### Per-Slice Breakdown

| Slice | Samples | Balanced Accuracy |
|---|---|---|
| Azimuth [  0°,  45°) | 0 | N/A |
| Azimuth [ 45°,  90°) | 0 | N/A |
| Azimuth [ 90°, 135°) | 0 | N/A |
| Azimuth [135°, 180°) | 458 | 0.4479 |
| Azimuth [180°, 225°) | 328 | 0.4607 |
| Azimuth [225°, 270°) | 0 | N/A |
| Azimuth [270°, 315°) | 786 | 0.5296 |
| Azimuth [315°, 360°) | 0 | N/A |
| Brightness Q1 (Darkest) | 393 | 0.6002 |
| Brightness Q2 (Medium-Dark) | 393 | 0.5698 |
| Brightness Q3 (Medium-Bright) | 393 | 0.5477 |
| Brightness Q4 (Brightest) | 393 | 0.5381 |
| Contrast Q1 (Lowest) | 393 | 0.5655 |
| Contrast Q2 (Low-Med) | 393 | 0.5980 |
| Contrast Q3 (Med-High) | 393 | 0.5186 |
| Contrast Q4 (Highest) | 393 | 0.5860 |
