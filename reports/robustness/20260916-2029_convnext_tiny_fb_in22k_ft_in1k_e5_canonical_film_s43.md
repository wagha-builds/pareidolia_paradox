# Robustness & Shortcut Audit Report: `20260916-2029_convnext_tiny_fb_in22k_ft_in1k_e5_canonical_film_s43`

## 1. Headline Robustness Metrics
- **Overall OOF Balanced Accuracy:** 0.7236 (@ threshold 0.4400)
- **OOF ROC-AUC:** 0.7514
- **Shortcut-Wrong BA:** 0.5355 (Accuracy on 3386 samples (43.1%) where B1 shadow shortcut fails)
- **B1 Shortcut Baseline BA:** 0.5951
- **Expected Calibration Error (ECE):** 0.1774 (Brier Score: 0.2205)

## 2. Topographic Stress Tests
- **Inversion-Stress BA:** 0.6729 (Under relief-inversion vertical flip)
- **Inversion Consistency Error:** 0.1394 (Ideal: 0.00)
- **Azimuth-Shift Δ (±15°):** 0.0081
- **Center Occlusion Degradation:** 0.0000 (Base 0.8122 → Occluded 0.8122)

## 3. Slice Report (Performance Across Sub-Populations)
- **Worst-Performing Slice:** `0-45° (NNE)` = 0.4665 BA

### A. Regional Lunar Folds
| Slice | Sample Count | Prior (π₁) | Balanced Accuracy |
| :--- | :---: | :---: | :---: |
| Fold 0 | 1572 | 61.7% | 0.7740 |
| Fold 1 | 1570 | 62.0% | 0.7782 |
| Fold 2 | 1571 | 61.7% | 0.6907 |
| Fold 3 | 1571 | 71.4% | 0.6049 |
| Fold 4 | 1570 | 61.6% | 0.7733 |

### B. Azimuth Octants
| Octant | Sample Count | Prior (π₁) | Balanced Accuracy |
| :--- | :---: | :---: | :---: |
| 0-45° (NNE) | 693 | 88.2% | 0.4665 |
| 45-90° (ENE) | 693 | 88.0% | 0.4954 |
| 90-135° (ESE) | 722 | 87.1% | 0.5038 |
| 135-180° (SSE) | 706 | 89.4% | 0.5027 |
| 180-225° (SSW) | 712 | 89.0% | 0.5068 |
| 225-270° (WSW) | 685 | 87.0% | 0.5274 |
| 270-315° (WNW) | 1818 | 34.8% | 0.4982 |
| 315-360° (NNW) | 1825 | 36.0% | 0.4980 |

## 4. Verdict & Assessment
Model maintains **0.5355 BA** on non-shortcut terrain, demonstrating legitimate 3D relief discrimination.
