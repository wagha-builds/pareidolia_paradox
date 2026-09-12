# Robustness Report: `20260913-015033_convnext_tiny_e4_canonical_convnext_s42`

- **Overall OOF BA**: 0.6633 @ t*=0.512
- **Shortcut-Wrong BA**: 0.3593 (N=2852)
- **Worst Slice**: Azimuth [270°, 315°) (BA=0.4964)

### Per-Slice Breakdown

| Slice | Samples | Balanced Accuracy |
|---|---|---|
| Azimuth [  0°,  45°) | 693 | 0.5296 |
| Azimuth [ 45°,  90°) | 693 | 0.5124 |
| Azimuth [ 90°, 135°) | 722 | 0.5086 |
| Azimuth [135°, 180°) | 706 | 0.5184 |
| Azimuth [180°, 225°) | 712 | 0.5340 |
| Azimuth [225°, 270°) | 685 | 0.5319 |
| Azimuth [270°, 315°) | 1818 | 0.4964 |
| Azimuth [315°, 360°) | 1825 | 0.5103 |
| Brightness Q1 (Darkest) | 1965 | 0.6322 |
| Brightness Q2 (Medium-Dark) | 1962 | 0.6954 |
| Brightness Q3 (Medium-Bright) | 1964 | 0.6908 |
| Brightness Q4 (Brightest) | 1963 | 0.6338 |
| Contrast Q1 (Lowest) | 1964 | 0.5970 |
| Contrast Q2 (Low-Med) | 1963 | 0.6669 |
| Contrast Q3 (Med-High) | 1963 | 0.6878 |
| Contrast Q4 (Highest) | 1964 | 0.7015 |
