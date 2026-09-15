# Grad-CAM Explainability Audit — E5 FiLM ConvNeXt (Fold 0)

## Quantitative Metric: Shadow-Mass Fraction (SMF)
Share of Grad-CAM attribution mass located inside the Otsu shadow mask on canonical lunar terrain:
- **Correct predictions SMF:** 34.94% ± 13.63%
- **Misclassified predictions SMF:** 31.63% ± 19.02%

## Visual Inspections
- **Correct predictions grid:** `reports/figures/gradcam_correct_fold0.png`
- **Misclassified predictions grid:** `reports/figures/gradcam_misclassified_fold0.png`

## Takeaways
1. For correct Depth predictions, Grad-CAM focuses on the high-contrast upper illumination rim and crescent shadow pool.
2. For correct Rise predictions, Grad-CAM attends to the bright south-facing crest and diffuse downhill shading.
3. In misclassified predictions, the model is frequently confused by low-contrast degraded craters where the shadow boundary is diffuse or eroded.
