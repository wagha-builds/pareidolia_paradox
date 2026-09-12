"""
Protected submission builder and strict format validator for Pareidolia.
Enforces all platform and competition formatting rules per AGENTS.md §5 & §8.
"""

import os
import csv
import argparse
from typing import List, Tuple, Union
import numpy as np
import pandas as pd

from .metrics import apply_threshold


class ValidationError(ValueError):
    """Raised when a submission file violates formatting or integrity rules."""
    pass


def validate_submission(
    filepath: str,
    test_metadata_path: str = "data/raw/test_metadata.csv",
    expected_rows: int = 2000,
) -> Tuple[bool, str]:
    """
    Validates a submission CSV file against the competition specifications.
    Enforces the broken-file rejection suite from AGENTS.md §8:
      - Header must be exactly ['image_id', 'label'] (no index column)
      - Exactly expected_rows data rows
      - Integer labels '0' or '1' only (no floats, no booleans)
      - IDs must retain '.png'
      - No duplicate image_ids
      - Correct order matching test_metadata.csv
      - Rejects UTF-8 BOM
      - Rejects multiple trailing newlines
    """
    if not os.path.exists(filepath):
        raise ValidationError(f"File does not exist: {filepath}")

    if os.path.getsize(filepath) == 0:
        raise ValidationError("File is empty.")

    # 1. Raw byte inspection for BOM and trailing newlines
    with open(filepath, "rb") as f:
        raw_bytes = f.read()

    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        raise ValidationError("File contains UTF-8 BOM encoding, which is prohibited.")

    raw_str = raw_bytes.decode("utf-8")
    if raw_str.endswith("\n\n") or raw_str.endswith("\r\n\r\n"):
        raise ValidationError("File contains extra trailing newlines at the end.")

    # 2. Parse lines with standard csv reader
    lines = raw_str.splitlines()
    if len(lines) != expected_rows + 1:
        raise ValidationError(
            f"Expected exactly {expected_rows + 1} lines (1 header + {expected_rows} rows), got {len(lines)}."
        )

    reader = csv.reader(lines)
    rows = list(reader)

    # 3. Header inspection
    header = rows[0]
    if header != ["image_id", "label"]:
        raise ValidationError(
            f"Header must be strictly ['image_id', 'label']. Found: {header}"
        )

    # 4. Load expected test metadata if available
    expected_ids = None
    if os.path.exists(test_metadata_path):
        test_df = pd.read_csv(test_metadata_path)
        expected_ids = [str(x) for x in test_df["image_id"]]

    data_rows = rows[1:]
    seen_ids = set()
    parsed_ids = []

    for idx, row in enumerate(data_rows, start=1):
        if len(row) != 2:
            raise ValidationError(f"Row {idx} does not have exactly 2 columns: {row}")

        img_id, label_str = row[0].strip(), row[1].strip()

        # Image ID checks
        if not img_id.endswith(".png"):
            raise ValidationError(f"Row {idx}: image_id '{img_id}' does not end with '.png'.")

        if img_id in seen_ids:
            raise ValidationError(f"Duplicate image_id found at row {idx}: '{img_id}'.")
        seen_ids.add(img_id)
        parsed_ids.append(img_id)

        # Label checks
        if label_str not in ("0", "1"):
            raise ValidationError(
                f"Row {idx}: label must strictly be '0' or '1'. Found '{label_str}'."
            )

    # 5. Order and completeness matching test metadata
    if expected_ids is not None:
        if len(seen_ids) != len(expected_ids):
            raise ValidationError(
                f"ID count mismatch: expected {len(expected_ids)} unique IDs, found {len(seen_ids)}."
            )
        if parsed_ids != expected_ids:
            raise ValidationError(
                "Submission image_id order does not match test_metadata.csv exactly."
            )

    return True, "Submission file passed all validation checks."


def check_label_inversion(labels: Union[List[int], np.ndarray], prior: float = 0.6366) -> None:
    """
    Sanity checks predicted class balance to catch accidental label inversion (0 <-> 1).
    Throws ValidationError if the proportion of Rise (1) is outside the plausible range [0.20, 0.90].
    """
    labels_arr = np.asarray(labels, dtype=int)
    pos_rate = float(labels_arr.mean())

    if pos_rate < 0.20 or pos_rate > 0.90:
        raise ValidationError(
            f"Label inversion or severe distribution shift detected! "
            f"Share of class 1 is {pos_rate:.4f} (training prior is {prior:.4f}). "
            f"Expected positive rate in range [0.20, 0.90]."
        )


def build_submission(
    image_ids: List[str],
    probs: Union[List[float], np.ndarray],
    threshold: float,
    output_path: str,
    test_metadata_path: str = "data/raw/test_metadata.csv",
) -> str:
    """
    Builds, validates, and writes a submission CSV file.

    Args:
        image_ids: List of test image_ids (preserving .png).
        probs: Model probability predictions for P(Rise).
        threshold: Decision threshold t* (predict 1 iff p >= t*).
        output_path: Target CSV path.
        test_metadata_path: Path to reference test_metadata.csv.
    Returns:
        Verified output file path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    probs_arr = np.asarray(probs, dtype=float)
    binary_labels = apply_threshold(probs_arr, threshold)

    check_label_inversion(binary_labels)

    sub_df = pd.DataFrame({
        "image_id": [str(x) for x in image_ids],
        "label": binary_labels,
    })

    expected_rows = 2000
    # If reference test metadata exists, align strictly to its order
    if os.path.exists(test_metadata_path):
        ref_df = pd.read_csv(test_metadata_path)
        expected_rows = len(ref_df)
        ref_ids = pd.DataFrame({"image_id": [str(x) for x in ref_df["image_id"]]})
        sub_df = ref_ids.merge(sub_df, on="image_id", how="left")

        if sub_df["label"].isna().any():
            missing = sub_df[sub_df["label"].isna()]["image_id"].tolist()
            raise ValidationError(f"Missing predictions for test images: {missing[:5]} (total {len(missing)})")

        sub_df["label"] = sub_df["label"].astype(int)

    # Write CSV with LF line endings and single trailing newline
    sub_df.to_csv(output_path, index=False, lineterminator="\n")

    # Run internal validator immediately
    is_valid, msg = validate_submission(
        output_path, test_metadata_path=test_metadata_path, expected_rows=expected_rows
    )
    if not is_valid:
        raise ValidationError(f"Generated submission failed validation: {msg}")

    print(f"Validated submission written successfully to {output_path} ({len(sub_df)} rows).")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submission validator CLI")
    parser.add_argument("--validate", type=str, help="Path to submission CSV to validate")
    args = parser.parse_args()

    if args.validate:
        try:
            ok, message = validate_submission(args.validate)
            print(f"[SUCCESS] {message}")
        except ValidationError as e:
            print(f"[REJECTED] {e}")
            exit(1)
