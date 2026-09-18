import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate_submission(
    path: str | Path, test_meta_path: str | Path = "data/raw/test_metadata.csv"
):
    path = Path(path)
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise AssertionError("UTF-8 BOM present")
    if b"\r" in raw:
        raise AssertionError("CR characters found; use LF line endings")
    text = raw.decode("utf-8")
    if not text.endswith("\n") or text.endswith("\n\n"):
        raise AssertionError("must end with exactly one newline")
    lines = text[:-1].split("\n")
    if lines[0] != "image_id,label":
        raise AssertionError(f"bad header {lines[0]!r}")
    expected = pd.read_csv(test_meta_path, dtype={"image_id": str})["image_id"].tolist()
    if len(lines) != len(expected) + 1:
        raise AssertionError(f"expected {len(expected) + 1} lines, got {len(lines)}")
    rows = [ln.split(",") for ln in lines[1:]]
    if not all(len(r) == 2 for r in rows):
        raise AssertionError("every row needs exactly 2 fields")
    ids = [r[0] for r in rows]
    labels = [r[1] for r in rows]
    if not all(v in ("0", "1") for v in labels):
        raise AssertionError("labels must be literal 0 or 1")
    if not all(i.endswith(".png") and i == i.strip() for i in ids):
        raise AssertionError("image_id must keep .png and have no surrounding spaces")
    if len(set(ids)) != len(ids):
        raise AssertionError("duplicate image_id")
    if set(ids) != set(expected):
        raise AssertionError("image_id set differs from test_metadata.csv")
    if ids != expected:
        raise AssertionError("row order differs from test_metadata.csv")
    return path


def run_inversion_check(
    run_dir: str | Path, threshold: float, n_check_per_class: int = 10
) -> dict:
    """Pass known training images through the exact production inference path.

    Asserts that predictions match known labels (AGENTS.md §5, PRD §Acceptance).
    Catches catastrophic 0 <-> 1 class inversions prior to CSV generation.
    """
    from .infer import predict_run
    from .metrics import apply_threshold

    run_dir = Path(run_dir)
    train_meta_path = Path("data/raw/train_metadata.csv")
    if not train_meta_path.exists():
        raise FileNotFoundError(f"Missing {train_meta_path} for inversion check")
    train_meta = pd.read_csv(train_meta_path, dtype={"image_id": str})

    depth_idx = train_meta[train_meta["label"] == 0].index[:n_check_per_class].tolist()
    rise_idx = train_meta[train_meta["label"] == 1].index[:n_check_per_class].tolist()
    test_indices = depth_idx + rise_idx
    y_true = train_meta.iloc[test_indices]["label"].to_numpy()

    # Pass through the exact same predict_run path with TTA
    preds = predict_run(run_dir, split="train", tta=True, indices=test_indices)
    y_pred = apply_threshold(preds["p_rise"].to_numpy(), threshold)

    acc = float(np.mean(y_pred == y_true))
    if acc < 0.65:
        raise AssertionError(
            f"Label-inversion check FAILED! Accuracy on {len(test_indices)} known training images is {acc:.2f} (< 0.65). "
            f"Class orientation may be inverted!"
        )
    print(
        f"[PASS] Label-inversion check passed: {int(acc * len(test_indices))}/{len(test_indices)} "
        f"known samples correct ({acc * 100:.1f}%)"
    )
    return {"accuracy": acc, "n_checked": len(test_indices)}


def generate_sanity_report(
    sub_path: Path, p_rise: np.ndarray, threshold: float
) -> Path:
    """Generate and save sanity report covering class balance, probability stats, and comparison with previous submissions."""
    df = pd.read_csv(sub_path)
    n_total = len(df)
    n_rise = int((df["label"] == 1).sum())
    n_depth = int((df["label"] == 0).sum())
    frac_rise = n_rise / n_total

    p_min = float(np.min(p_rise))
    p_25 = float(np.percentile(p_rise, 25))
    p_median = float(np.median(p_rise))
    p_mean = float(np.mean(p_rise))
    p_75 = float(np.percentile(p_rise, 75))
    p_max = float(np.max(p_rise))

    report_lines = [
        "=" * 65,
        f"SUBMISSION SANITY REPORT: {sub_path.name}",
        "=" * 65,
        f"File:                  {sub_path}",
        f"Rows:                  {n_total}",
        f"Applied Threshold:     {threshold:.4f}",
        "",
        "Class Balance:",
        f"  Depth (0):           {n_depth:4d} ({n_depth / n_total * 100:.1f}%)",
        f"  Rise  (1):           {n_rise:4d} ({n_rise / n_total * 100:.1f}%)",
        "",
        "P(Rise) Distribution:",
        f"  Min:                 {p_min:.4f}",
        f"  25th percentile:     {p_25:.4f}",
        f"  Median:              {p_median:.4f}",
        f"  Mean:                {p_mean:.4f}",
        f"  75th percentile:     {p_75:.4f}",
        f"  Max:                 {p_max:.4f}",
    ]

    # Check class balance plausibility
    if frac_rise < 0.35 or frac_rise > 0.85:
        report_lines.append(
            f"  [WARNING] Implausible class balance: Rise fraction {frac_rise:.3f} (AGENTS.md §11)"
        )
    else:
        report_lines.append(
            f"  [OK] Class balance matches expected lunar relief distribution ({frac_rise * 100:.1f}% Rise)."
        )

    # Check against previous submissions
    submissions_dir = sub_path.parent
    prev_subs = sorted(
        [
            f
            for f in submissions_dir.glob("sub_*.csv")
            if f.name != sub_path.name and not f.name.startswith("sanity")
        ]
    )
    if prev_subs:
        prev = prev_subs[-1]
        prev_df = pd.read_csv(prev)
        agreement = float((df["label"] == prev_df["label"]).mean() * 100)
        disagreement = 100.0 - agreement
        report_lines.extend(
            [
                "",
                f"Comparison with Previous ({prev.name}):",
                f"  Agreement:           {agreement:.1f}%",
                f"  Disagreement:        {disagreement:.1f}%",
            ]
        )
        if disagreement > 15.0:
            report_lines.append(
                f"  [WARNING] Disagreement > 15% ({disagreement:.1f}%)! Review carefully before uploading (AGENTS.md §11)."
            )
        else:
            report_lines.append(
                "  [OK] Agreement within normal envelope (disagreement <= 15%)."
            )

    report_lines.append("=" * 65)
    report_text = "\n".join(report_lines)
    print("\n" + report_text + "\n")

    report_path = submissions_dir / f"sanity_{sub_path.stem}.txt"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def generate_spot_check_grid(
    sub_path: Path, p_rise: np.ndarray, output_path: Path | None = None
) -> Path:
    """Generate a 40-image spot-check grid (20 predicted Depth + 20 predicted Rise)."""
    import matplotlib.pyplot as plt
    from PIL import Image

    df = pd.read_csv(sub_path)
    df["p_rise"] = p_rise

    # Select top-20 most confident depth (lowest p) and top-20 most confident rise (highest p)
    depth_samples = df[df["label"] == 0].sort_values("p_rise").head(20)
    rise_samples = df[df["label"] == 1].sort_values("p_rise", ascending=False).head(20)

    test_img_dir = Path("data/raw/eval_images")
    if not test_img_dir.exists():
        print(f"eval_images dir {test_img_dir} not found; skipping spot-check grid.")
        return Path()

    fig, axes = plt.subplots(8, 5, figsize=(15, 24))
    axes = axes.ravel()

    # First 20: Depth
    for i, (_, row) in enumerate(depth_samples.iterrows()):
        ax = axes[i]
        img_path = test_img_dir / row["image_id"]
        if img_path.exists():
            img = Image.open(img_path).convert("L")
            ax.imshow(img, cmap="gray")
        ax.set_title(
            f"Depth: {row['image_id'][:10]}\np={row['p_rise']:.3f}",
            fontsize=8,
            color="blue",
        )
        ax.axis("off")

    # Next 20: Rise
    for i, (_, row) in enumerate(rise_samples.iterrows()):
        ax = axes[20 + i]
        img_path = test_img_dir / row["image_id"]
        if img_path.exists():
            img = Image.open(img_path).convert("L")
            ax.imshow(img, cmap="gray")
        ax.set_title(
            f"Rise: {row['image_id'][:10]}\np={row['p_rise']:.3f}",
            fontsize=8,
            color="red",
        )
        ax.axis("off")

    plt.suptitle(
        "Spot-Check Grid: Top 20 Predicted Depth (Top 4 Rows) & Top 20 Predicted Rise (Bottom 4 Rows)",
        fontsize=14,
    )
    plt.tight_layout()

    if output_path is None:
        out_dir = Path("reports")
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / f"spot_check_{sub_path.stem}.png"
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Spot-check grid saved to {output_path}")
    return output_path


def build_submission(
    run_dir: str | Path,
    threshold: float | None = None,
    skip_inversion_check: bool = False,
) -> Path:
    from .infer import predict_run
    from .metrics import apply_threshold

    run_dir = Path(run_dir)
    if threshold is None:
        if (run_dir / "threshold.json").exists():
            thresh_meta = json.loads(
                (run_dir / "threshold.json").read_text(encoding="utf-8")
            )
            threshold = float(thresh_meta["threshold"])
        elif (run_dir / "run_manifest.json").exists():
            manifest = json.loads(
                (run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            threshold = float(manifest["metrics"]["threshold"])
        else:
            raise ValueError(f"Could not determine threshold from {run_dir}")

    # 1. Mandatory Label-Inversion Check
    if not skip_inversion_check:
        run_inversion_check(run_dir, threshold)

    # 2. Test-time Inference with TTA
    preds = predict_run(run_dir, split="test", tta=True)
    p_rise = preds["p_rise"].to_numpy()
    labels = apply_threshold(p_rise, threshold)

    test_meta = pd.read_csv("data/raw/test_metadata.csv", dtype={"image_id": str})
    out = test_meta[["image_id"]].merge(
        pd.DataFrame({"image_id": preds["image_id"], "label": labels.astype(int)}),
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    if out["label"].isna().any():
        raise RuntimeError("Missing predictions for some test images")

    out_dir = Path("submissions")
    out_dir.mkdir(exist_ok=True)
    path = (
        out_dir / f"sub_{pd.Timestamp.now().strftime('%Y%m%d-%H%M')}_{run_dir.name}.csv"
    )
    out.to_csv(path, index=False, lineterminator="\n")

    # 3. Format Validation
    validate_submission(path)

    # 4. Sanity Report
    generate_sanity_report(path, p_rise, threshold)

    # 5. Spot-Check Grid
    generate_spot_check_grid(path, p_rise)

    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=str)
    parser.add_argument("--validate", type=str)
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--skip-inversion-check", action="store_true", help="Skip inversion check"
    )
    args = parser.parse_args()
    if args.validate:
        validate_submission(args.validate)
        print(f"Validated {args.validate}")
    elif args.artifact:
        path = build_submission(
            args.artifact,
            threshold=args.threshold,
            skip_inversion_check=args.skip_inversion_check,
        )
        print(f"Wrote and validated {path}")
    else:
        parser.error("pass --artifact or --validate")


if __name__ == "__main__":
    main()
