import numpy as np
from src.canonical import az_to_img, img_to_az, canonical_rotation_deg
from tests.fixtures.synthetic import render

def test_az_to_img():
    # If s=1, delta=45: az=0 -> theta=45
    assert az_to_img(0, 45, 1) == 45
    assert az_to_img(360, 45, 1) == 45

def test_img_to_az():
    assert img_to_az(45, 45, 1) == 0

def test_canonical_rotation():
    # If sun is at theta=45, rotation to make it 90 is +45
    assert canonical_rotation_deg(0, 45, 1) == 45
    
def test_synthetic_render():
    img = render("dome", 90)
    # Dome lit from top (90) should be bright at top (low rows) and dark at bottom (high rows)
    # top-minus-bottom should be > 0
    top_mean = img[:128, :].mean()
    bottom_mean = img[128:, :].mean()
    assert top_mean > bottom_mean

    img_pit = render("pit", 90)
    top_mean_pit = img_pit[:128, :].mean()
    bottom_mean_pit = img_pit[128:, :].mean()
    assert top_mean_pit < bottom_mean_pit
