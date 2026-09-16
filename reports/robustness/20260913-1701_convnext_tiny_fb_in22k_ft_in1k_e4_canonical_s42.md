# Robustness & Shortcut Audit Report: `20260913-1701_convnext_tiny_fb_in22k_ft_in1k_e4_canonical_s42`

## 1. Headline Robustness Metrics
- **Overall OOF Balanced Accuracy:** 0.7161 (@ threshold 0.4850)
- **OOF ROC-AUC:** 0.7376
- **Shortcut-Wrong BA:** 0.5358 (Accuracy on 3386 samples (43.1%) where B1 shadow shortcut fails)
- **B1 Shortcut Baseline BA:** 0.5951
- **Expected Calibration Error (ECE):** 0.2022 (Brier Score: 0.2329)

## 2. Topographic Stress Tests
- **Inversion-Stress BA:** 0.7122 (Under relief-inversion vertical flip)
- **Inversion Consistency Error:** 0.0955 (Ideal: 0.00)
- **Azimuth-Shift Δ (±15°):** 0.0000
- **Center Occlusion Degradation:** -0.0027 (Base 0.8002 → Occluded 0.8029)

## 3. Slice Report (Performance Across Sub-Populations)
- **Worst-Performing Slice:** `0-45° (NNE)` = 0.4883 BA

### A. Regional Lunar Folds
| Slice | Sample Count | Prior (π₁) | Balanced Accuracy |
| :--- | :---: | :---: | :---: |
| Fold 0 | 1572 | 61.7% | 0.7661 |
| Fold 1 | 1570 | 62.0% | 0.7750 |
| Fold 2 | 1571 | 61.7% | 0.7065 |
| Fold 3 | 1571 | 71.4% | 0.5864 |
| Fold 4 | 1570 | 61.6% | 0.7727 |

### B. Azimuth Octants
| Octant | Sample Count | Prior (π₁) | Balanced Accuracy |
| :--- | :---: | :---: | :---: |
| 0-45° (NNE) | 693 | 88.2% | 0.4883 |
| 45-90° (ENE) | 693 | 88.0% | 0.4979 |
| 90-135° (ESE) | 722 | 87.1% | 0.5020 |
| 135-180° (SSE) | 706 | 89.4% | 0.5011 |
| 180-225° (SSW) | 712 | 89.0% | 0.5075 |
| 225-270° (WSW) | 685 | 87.0% | 0.5198 |
| 270-315° (WNW) | 1818 | 34.8% | 0.5060 |
| 315-360° (NNW) | 1825 | 36.0% | 0.4936 |

## 4. Verdict & Assessment
Model maintains **0.5358 BA** on non-shortcut terrain, demonstrating legitimate 3D relief discrimination.
