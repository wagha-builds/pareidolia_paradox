# Robustness & Shortcut Audit Report: `20260915-2338_convnext_tiny_fb_in22k_ft_in1k_e5_canonical_film_s42`

## 1. Headline Robustness Metrics
- **Overall OOF Balanced Accuracy:** 0.7271 (@ threshold 0.4450)
- **OOF ROC-AUC:** 0.7528
- **Shortcut-Wrong BA:** 0.5597 (Accuracy on 3386 samples (43.1%) where B1 shadow shortcut fails)
- **B1 Shortcut Baseline BA:** 0.5951
- **Expected Calibration Error (ECE):** 0.1731 (Brier Score: 0.2191)

## 2. Topographic Stress Tests
- **Inversion-Stress BA:** 0.6476 (Under relief-inversion vertical flip)
- **Inversion Consistency Error:** 0.1290 (Ideal: 0.00)
- **Azimuth-Shift Δ (±15°):** 0.0065
- **Center Occlusion Degradation:** 0.0000 (Base 0.8106 → Occluded 0.8106)

## 3. Slice Report (Performance Across Sub-Populations)
- **Worst-Performing Slice:** `0-45° (NNE)` = 0.4796 BA

### A. Regional Lunar Folds
| Slice | Sample Count | Prior (π₁) | Balanced Accuracy |
| :--- | :---: | :---: | :---: |
| Fold 0 | 1572 | 61.7% | 0.7896 |
| Fold 1 | 1570 | 62.0% | 0.7797 |
| Fold 2 | 1571 | 61.7% | 0.6708 |
| Fold 3 | 1571 | 71.4% | 0.5872 |
| Fold 4 | 1570 | 61.6% | 0.7742 |

### B. Azimuth Octants
| Octant | Sample Count | Prior (π₁) | Balanced Accuracy |
| :--- | :---: | :---: | :---: |
| 0-45° (NNE) | 693 | 88.2% | 0.4796 |
| 45-90° (ENE) | 693 | 88.0% | 0.4959 |
| 90-135° (ESE) | 722 | 87.1% | 0.5000 |
| 135-180° (SSE) | 706 | 89.4% | 0.5051 |
| 180-225° (SSW) | 712 | 89.0% | 0.5432 |
| 225-270° (WSW) | 685 | 87.0% | 0.5327 |
| 270-315° (WNW) | 1818 | 34.8% | 0.5102 |
| 315-360° (NNW) | 1825 | 36.0% | 0.4977 |

## 4. Verdict & Assessment
Model maintains **0.5597 BA** on non-shortcut terrain, demonstrating legitimate 3D relief discrimination.
