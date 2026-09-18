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


def test_generate_sanity_report(tmp_path: Path):
    from src.submit import generate_sanity_report
    import numpy as np

    sub = tmp_path / "sub_test.csv"
    sub.write_bytes(b"image_id,label\neval_00001.png,0\neval_00002.png,1\n")
    p_rise = np.array([0.2, 0.8], dtype=np.float32)
    report_path = generate_sanity_report(sub, p_rise, threshold=0.5)
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "SUBMISSION SANITY REPORT" in content
    assert "Rise  (1):              1 (50.0%)" in content
    assert "Depth (0):              1 (50.0%)" in content


def test_inversion_check_catches_inverted_predictions(monkeypatch, tmp_path: Path):
    import numpy as np
    import src.infer
    from src.submit import run_inversion_check

    def mock_predict_run(run_dir, split="test", tta=True, indices=None):
        # 10 depth predicted as 1.0 (inverted), 10 rise predicted as 0.0 (inverted)
        p = np.array([1.0] * 10 + [0.0] * 10, dtype=np.float32)
        return pd.DataFrame({"image_id": [f"img_{i}" for i in range(20)], "p_rise": p})

    monkeypatch.setattr(src.infer, "predict_run", mock_predict_run)

    with pytest.raises(AssertionError, match="Label-inversion check FAILED"):
        run_inversion_check(tmp_path, threshold=0.5, n_check_per_class=10)
