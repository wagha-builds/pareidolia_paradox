import numpy as np, glob, json
from pathlib import Path

# Check OOF probability distributions
runs = sorted(glob.glob('experiments/*/oof.npy'))
print('OOF files found:', len(runs))
for r in runs:
    oof = np.load(r)
    valid = oof[~np.isnan(oof)]
    print(f'  {r}')
    print(f'    n={len(valid)} | min={valid.min():.4f} max={valid.max():.4f} mean={valid.mean():.4f} std={valid.std():.6f}')
    # Check if collapsed (all near 0.5 or all near 0 or all near 1)
    near_half = (np.abs(valid - 0.5) < 0.05).mean()
    print(f'    Fraction within 0.05 of 0.5: {near_half:.3f} (1.0 = fully collapsed)')

# Check manifests
manifests = sorted(glob.glob('experiments/*/run_manifest.json'))
print('\nManifests:')
for m in manifests:
    d = json.loads(Path(m).read_text())
    print(f'  {d["run_id"]}')
    print(f'    OOF_BA={d["metrics"]["oof_ba"]:.4f}  threshold={d["metrics"]["threshold"]:.4f}  AUC={d["metrics"]["oof_auc"]}')
    for fm in d.get('fold_metrics', []):
        print(f'    fold={fm["fold"]}  best_ba@0.5={fm["best_ba_at_0_5"]:.4f}  best_epoch={fm["best_epoch"]}')
