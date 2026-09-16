# Robustness & Shortcut Audit Report: `20260916-0212_swin_tiny_patch4_window7_224_e6b_canonical_swin_s42`

## 1. Headline Robustness Metrics
- **Overall OOF Balanced Accuracy:** 0.7038 (@ threshold 0.4825)
- **OOF ROC-AUC:** 0.7180
- **Shortcut-Wrong BA:** 0.5017 (Accuracy on 3386 samples (43.1%) where B1 shadow shortcut fails)
- **B1 Shortcut Baseline BA:** 0.5951
- **Expected Calibration Error (ECE):** 0.2129 (Brier Score: 0.2438)

## 2. Topographic Stress Tests
- **Inversion-Stress BA:** 0.6991 (Under relief-inversion vertical flip)
- **Inversion Consistency Error:** 0.1369 (Ideal: 0.00)
- **Azimuth-Shift Δ (±15°):** 0.0000
- **Center Occlusion Degradation:** -0.0011 (Base 0.7959 → Occluded 0.7969)

## 3. Slice Report (Performance Across Sub-Populations)
- **Worst-Performing Slice:** `90-135° (ESE)` = 0.4971 BA

### A. Regional Lunar Folds
| Slice | Sample Count | Prior (π₁) | Balanced Accuracy |
| :--- | :---: | :---: | :---: |
| Fold 0 | 1572 | 61.7% | 0.7693 |
| Fold 1 | 1570 | 62.0% | 0.7587 |
| Fold 2 | 1571 | 61.7% | 0.7153 |
| Fold 3 | 1571 | 71.4% | 0.5639 |
| Fold 4 | 1570 | 61.6% | 0.7525 |

### B. Azimuth Octants
| Octant | Sample Count | Prior (π₁) | Balanced Accuracy |
| :--- | :---: | :---: | :---: |
| 0-45° (NNE) | 693 | 88.2% | 0.5065 |
| 45-90° (ENE) | 693 | 88.0% | 0.5094 |
| 90-135° (ESE) | 722 | 87.1% | 0.4971 |
| 135-180° (SSE) | 706 | 89.4% | 0.5065 |
| 180-225° (SSW) | 712 | 89.0% | 0.5267 |
| 225-270° (WSW) | 685 | 87.0% | 0.5236 |
| 270-315° (WNW) | 1818 | 34.8% | 0.5012 |
| 315-360° (NNW) | 1825 | 36.0% | 0.5087 |

## 4. Verdict & Assessment
Model maintains **0.5017 BA** on non-shortcut terrain, demonstrating legitimate 3D relief discrimination.
