import argparse
import json
from pathlib import Path

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


def build_submission(run_dir: str | Path, threshold: float | None = None) -> Path:
    from .infer import predict_run
    from .metrics import apply_threshold

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if threshold is None:
        threshold = float(manifest["metrics"]["threshold"])
    preds = predict_run(run_dir)
    labels = apply_threshold(preds["p_rise"].to_numpy(), threshold)
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
    validate_submission(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=str)
    parser.add_argument("--validate", type=str)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    if args.validate:
        validate_submission(args.validate)
        print(f"Validated {args.validate}")
    elif args.artifact:
        path = build_submission(args.artifact, threshold=args.threshold)
        print(f"Wrote and validated {path}")
    else:
        parser.error("pass --artifact or --validate")


if __name__ == "__main__":
    main()
