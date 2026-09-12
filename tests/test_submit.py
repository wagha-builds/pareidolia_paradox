from pathlib import Path

import pandas as pd
import pytest

from src.submit import validate_submission


def test_validate_submission_accepts_correct_file(tmp_path: Path):
    meta = tmp_path / "test_metadata.csv"
    meta.write_text("image_id,sun_azimuth_angle\neval_00001.png,0\neval_00002.png,90\n")
    sub = tmp_path / "sub.csv"
    sub.write_bytes(b"image_id,label\neval_00001.png,0\neval_00002.png,1\n")
    assert validate_submission(sub, meta) == sub


@pytest.mark.parametrize(
    "content",
    [
        "image_id,label\neval_00001,0\neval_00002.png,1\n",
        "image_id,label\neval_00001.png,0.0\neval_00002.png,1\n",
        "image_id,label\neval_00002.png,1\neval_00001.png,0\n",
        "index,image_id,label\n0,eval_00001.png,0\n1,eval_00002.png,1\n",
        "image_id,label\neval_00001.png,0\neval_00001.png,1\n",
        "image_id,label\neval_00001.png,0\neval_00002.png,1\n\n",
    ],
)
def test_validate_submission_rejects_broken_files(tmp_path: Path, content: str):
    meta = tmp_path / "test_metadata.csv"
    meta.write_text("image_id,sun_azimuth_angle\neval_00001.png,0\neval_00002.png,90\n")
    sub = tmp_path / "sub.csv"
    sub.write_bytes(content.encode("utf-8"))
    with pytest.raises(AssertionError):
        validate_submission(sub, meta)


def test_submission_labels_are_plain_ints(tmp_path: Path):
    meta = tmp_path / "test_metadata.csv"
    meta.write_text("image_id,sun_azimuth_angle\neval_00001.png,0\n")
    sub = tmp_path / "sub.csv"
    pd.DataFrame({"image_id": ["eval_00001.png"], "label": [1]}).to_csv(
        sub, index=False, lineterminator="\n"
    )
    assert validate_submission(sub, meta) == sub
