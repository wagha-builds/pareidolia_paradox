from src.reproduce import reproduce_artifact


def test_reproduce_production_artifact():
    res = reproduce_artifact("artifacts/20260916_ensemble_e4_e5_swin_ba0.7361")
    assert res["diff"] < 0.002
    assert res["reproduced_ba"] >= 0.73
