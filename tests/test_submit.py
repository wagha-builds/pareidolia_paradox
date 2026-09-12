"""
Tests for submission validation and broken-file rejection suite in src/submit.py.
Enforces the mandatory requirements of AGENTS.md §8.
"""

import os
import pytest
import pandas as pd

from src.submit import (
    ValidationError,
    validate_submission,
    check_label_inversion,
    build_submission,
)


@pytest.fixture
def test_metadata_file(tmp_path):
    """Creates a mock test_metadata.csv with 10 rows for fast testing."""
    file_path = tmp_path / "test_metadata.csv"
    df = pd.DataFrame({
        "image_id": [f"eval_{i:04d}.png" for i in range(10)],
        "sun_azimuth_angle": [float(i * 36) for i in range(10)],
    })
    df.to_csv(file_path, index=False)
    return str(file_path)


def test_valid_submission_passes(tmp_path, test_metadata_file):
    """A perfectly formatted submission file passes validation."""
    sub_path = tmp_path / "valid_sub.csv"
    df = pd.DataFrame({
        "image_id": [f"eval_{i:04d}.png" for i in range(10)],
        "label": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    })
    df.to_csv(sub_path, index=False, lineterminator="\n")

    ok, msg = validate_submission(str(sub_path), test_metadata_path=test_metadata_file, expected_rows=10)
    assert ok is True


def test_reject_index_column(tmp_path, test_metadata_file):
    """Files with an extra index column (e.g. from df.to_csv without index=False) are rejected."""
    sub_path = tmp_path / "sub_index.csv"
    with open(sub_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(",image_id,label\n")
        for i in range(10):
            f.write(f"{i},eval_{i:04d}.png,1\n")

    with pytest.raises(ValidationError, match="Header must be strictly"):
        validate_submission(str(sub_path), test_metadata_path=test_metadata_file, expected_rows=10)


def test_reject_float_labels(tmp_path, test_metadata_file):
    """Files with float labels like '1.0' or '0.0' are rejected."""
    sub_path = tmp_path / "sub_floats.csv"
    with open(sub_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("image_id,label\n")
        for i in range(10):
            f.write(f"eval_{i:04d}.png,1.0\n")

    with pytest.raises(ValidationError, match="label must strictly be '0' or '1'"):
        validate_submission(str(sub_path), test_metadata_path=test_metadata_file, expected_rows=10)


def test_reject_boolean_labels(tmp_path, test_metadata_file):
    """Files with boolean labels ('True' / 'False') are rejected."""
    sub_path = tmp_path / "sub_bool.csv"
    with open(sub_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("image_id,label\n")
        for i in range(10):
            f.write(f"eval_{i:04d}.png,True\n")

    with pytest.raises(ValidationError, match="label must strictly be '0' or '1'"):
        validate_submission(str(sub_path), test_metadata_path=test_metadata_file, expected_rows=10)


def test_reject_stripped_png(tmp_path, test_metadata_file):
    """Files where the '.png' extension was stripped from image_id are rejected."""
    sub_path = tmp_path / "sub_nopng.csv"
    with open(sub_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("image_id,label\n")
        for i in range(10):
            f.write(f"eval_{i:04d},1\n")

    with pytest.raises(ValidationError, match="does not end with '.png'"):
        validate_submission(str(sub_path), test_metadata_path=test_metadata_file, expected_rows=10)


def test_reject_missing_row(tmp_path, test_metadata_file):
    """Files with missing rows are rejected."""
    sub_path = tmp_path / "sub_missing.csv"
    with open(sub_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("image_id,label\n")
        for i in range(9):  # 9 instead of 10
            f.write(f"eval_{i:04d}.png,1\n")

    with pytest.raises(ValidationError, match="Expected exactly 11 lines"):
        validate_submission(str(sub_path), test_metadata_path=test_metadata_file, expected_rows=10)


def test_reject_duplicate_id(tmp_path, test_metadata_file):
    """Files with duplicate IDs are rejected."""
    sub_path = tmp_path / "sub_dup.csv"
    with open(sub_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("image_id,label\n")
        f.write("eval_0000.png,1\n")
        f.write("eval_0000.png,0\n")  # Duplicate
        for i in range(2, 10):
            f.write(f"eval_{i:04d}.png,1\n")

    with pytest.raises(ValidationError, match="Duplicate image_id found"):
        validate_submission(str(sub_path), test_metadata_path=test_metadata_file, expected_rows=10)


def test_reject_bom_encoding(tmp_path, test_metadata_file):
    """Files containing UTF-8 BOM encoding are rejected."""
    sub_path = tmp_path / "sub_bom.csv"
    with open(sub_path, "wb") as f:
        f.write(b"\xef\xbb\xbfimage_id,label\n")
        for i in range(10):
            f.write(f"eval_{i:04d}.png,1\n".encode("utf-8"))

    with pytest.raises(ValidationError, match="contains UTF-8 BOM"):
        validate_submission(str(sub_path), test_metadata_path=test_metadata_file, expected_rows=10)


def test_reject_shuffled_order(tmp_path, test_metadata_file):
    """Files where row ordering does not match test_metadata.csv are rejected."""
    sub_path = tmp_path / "sub_shuffled.csv"
    df = pd.DataFrame({
        "image_id": [f"eval_{i:04d}.png" for i in reversed(range(10))],
        "label": [1] * 10,
    })
    df.to_csv(sub_path, index=False, lineterminator="\n")

    with pytest.raises(ValidationError, match="order does not match"):
        validate_submission(str(sub_path), test_metadata_path=test_metadata_file, expected_rows=10)


def test_reject_extra_trailing_newline(tmp_path, test_metadata_file):
    """Files with extra blank trailing newlines are rejected."""
    sub_path = tmp_path / "sub_trailing.csv"
    with open(sub_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("image_id,label\n")
        for i in range(10):
            f.write(f"eval_{i:04d}.png,1\n")
        f.write("\n")  # Extra newline

    with pytest.raises(ValidationError, match="extra trailing newlines"):
        validate_submission(str(sub_path), test_metadata_path=test_metadata_file, expected_rows=10)


def test_label_inversion_check():
    """Detects inverted labels (e.g. 95% depth or 95% rise)."""
    # 5% rise -> extreme skew / inversion
    with pytest.raises(ValidationError, match="Label inversion"):
        check_label_inversion([0] * 95 + [1] * 5)

    # 60% rise -> within plausible range
    check_label_inversion([0] * 40 + [1] * 60)


def test_build_submission_end_to_end(tmp_path, test_metadata_file):
    """Verifies build_submission constructs, validates, and writes CSV cleanly."""
    out_file = str(tmp_path / "output_sub.csv")
    ids = [f"eval_{i:04d}.png" for i in range(10)]
    probs = [0.8, 0.2, 0.9, 0.7, 0.1, 0.85, 0.3, 0.9, 0.75, 0.4]

    result_path = build_submission(
        image_ids=ids,
        probs=probs,
        threshold=0.5,
        output_path=out_file,
        test_metadata_path=test_metadata_file,
    )

    assert os.path.exists(result_path)
    ok, _ = validate_submission(result_path, test_metadata_path=test_metadata_file, expected_rows=10)
    assert ok is True
