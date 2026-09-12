# Robustness Report: `20260913-014540_convnext_tiny_e5_raw_balanced_counterfactual_convnext_s42`

- **Overall OOF BA**: 0.7921 @ t*=0.585
- **Shortcut-Wrong BA**: 0.2624 (N=616)
- **Worst Slice**: Azimuth [135°, 180°) (BA=0.5000)

### Per-Slice Breakdown

| Slice | Samples | Balanced Accuracy |
|---|---|---|
| Azimuth [  0°,  45°) | 0 | N/A |
| Azimuth [ 45°,  90°) | 0 | N/A |
| Azimuth [ 90°, 135°) | 0 | N/A |
| Azimuth [135°, 180°) | 458 | 0.5000 |
| Azimuth [180°, 225°) | 328 | 0.5000 |
| Azimuth [225°, 270°) | 0 | N/A |
| Azimuth [270°, 315°) | 786 | 0.5000 |
| Azimuth [315°, 360°) | 0 | N/A |
| Brightness Q1 (Darkest) | 393 | 0.8058 |
| Brightness Q2 (Medium-Dark) | 393 | 0.8133 |
| Brightness Q3 (Medium-Bright) | 393 | 0.7872 |
| Brightness Q4 (Brightest) | 393 | 0.7610 |
| Contrast Q1 (Lowest) | 393 | 0.7697 |
| Contrast Q2 (Low-Med) | 393 | 0.8165 |
| Contrast Q3 (Med-High) | 393 | 0.7955 |
| Contrast Q4 (Highest) | 393 | 0.7837 |
