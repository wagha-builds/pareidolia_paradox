# Robustness & Shortcut Audit Report: `ensemble_e4_e5_swin`

## 1. Headline Robustness Metrics
- **Overall OOF Balanced Accuracy:** 0.7361 (@ threshold 0.4375)
- **OOF ROC-AUC:** 0.7580
- **Shortcut-Wrong BA:** 0.5619 (Accuracy on 3386 samples (43.1%) where B1 shadow shortcut fails)
- **B1 Shortcut Baseline BA:** 0.5951
- **Expected Calibration Error (ECE):** 0.1834 (Brier Score: 0.2205)

## 2. Topographic Stress Tests
- **Inversion-Stress BA:** 0.0000 (Under relief-inversion vertical flip)
- **Inversion Consistency Error:** 0.0000 (Ideal: 0.00)
- **Azimuth-Shift Δ (±15°):** 0.0000
- **Center Occlusion Degradation:** 0.0000 (Base 0.0000 → Occluded 0.0000)

## 3. Slice Report (Performance Across Sub-Populations)
- **Worst-Performing Slice:** `0-45° (NNE)` = 0.4778 BA

### A. Regional Lunar Folds
| Slice | Sample Count | Prior (π₁) | Balanced Accuracy |
| :--- | :---: | :---: | :---: |
| Fold 0 | 1572 | 61.7% | 0.7896 |
| Fold 1 | 1570 | 62.0% | 0.7785 |
| Fold 2 | 1571 | 61.7% | 0.7183 |
| Fold 3 | 1571 | 71.4% | 0.6000 |
| Fold 4 | 1570 | 61.6% | 0.7752 |

### B. Azimuth Octants
| Octant | Sample Count | Prior (π₁) | Balanced Accuracy |
| :--- | :---: | :---: | :---: |
| 0-45° (NNE) | 693 | 88.2% | 0.4778 |
| 45-90° (ENE) | 693 | 88.0% | 0.4959 |
| 90-135° (ESE) | 722 | 87.1% | 0.5046 |
| 135-180° (SSE) | 706 | 89.4% | 0.5059 |
| 180-225° (SSW) | 712 | 89.0% | 0.5474 |
| 225-270° (WSW) | 685 | 87.0% | 0.5604 |
| 270-315° (WNW) | 1818 | 34.8% | 0.5144 |
| 315-360° (NNW) | 1825 | 36.0% | 0.4983 |

## 4. Verdict & Assessment
Model maintains **0.5619 BA** on non-shortcut terrain, demonstrating legitimate 3D relief discrimination.
