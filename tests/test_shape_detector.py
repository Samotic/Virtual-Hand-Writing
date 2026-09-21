"""Detector behaviour tests.

The thresholds in shape_detector.py are hand-tuned, so these tests pin down both
sides of the trade-off: supported shapes must still be recognized, and ordinary
scribbling must not be mistaken for one.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import shape_detector as sd
import strokes


# --- supported shapes are still recognized -------------------------------


@pytest.mark.parametrize("jitter", [0.0, 3.0, 6.0])
def test_clean_heart_is_recognized(jitter):
    match = sd.recognize_shape(strokes.heart(seed=1, jitter=jitter), strokes.CANVAS_SHAPE)
    assert match is not None and match.name == "heart"


@pytest.mark.parametrize("scale", [5.0, 9.0, 15.0])
def test_heart_is_recognized_across_sizes(scale):
    match = sd.recognize_shape(strokes.heart(scale=scale), strokes.CANVAS_SHAPE)
    assert match is not None and match.name == "heart"


@pytest.mark.parametrize("jitter", [0.0, 3.0])
def test_star_is_recognized(jitter):
    match = sd.recognize_shape(strokes.star(seed=2, jitter=jitter), strokes.CANVAS_SHAPE)
    assert match is not None and match.name == "star"


@pytest.mark.parametrize("radius", [45.0, 100.0, 190.0])
def test_circle_maps_to_smiley(radius):
    match = sd.recognize_shape(strokes.circle(radius=radius), strokes.CANVAS_SHAPE)
    assert match is not None and match.name == "smiley"


def test_recognition_rate_stays_high():
    hits = sum(
        1
        for seed in range(40)
        for jitter in (0.0, 3.0, 6.0)
        if (m := sd.recognize_shape(strokes.heart(seed=seed * 7, jitter=jitter), strokes.CANVAS_SHAPE))
        and m.name == "heart"
    )
    assert hits >= 110, f"heart recall regressed: {hits}/120"


# --- ordinary drawing is left alone --------------------------------------


def test_random_scribbles_are_not_shapes():
    """Regression: a disabled similarity gate once turned 28% of these into hearts."""
    false_positives = [
        seed
        for seed in range(200)
        if sd.recognize_shape(strokes.scribble(seed), strokes.CANVAS_SHAPE) is not None
    ]
    assert not false_positives, f"{len(false_positives)} scribbles matched a shape"


def test_doodles_rarely_match():
    hits = sum(
        1
        for seed in range(200)
        if sd.recognize_shape(strokes.doodle(seed), strokes.CANVAS_SHAPE) is not None
    )
    assert hits <= 15, f"doodle false-positive rate regressed: {hits}/200"


@pytest.mark.parametrize("stroke_name", ["triangle", "square", "line", "open_arc", "wide_ellipse"])
def test_unsupported_strokes_are_ignored(stroke_name):
    stroke = {
        "triangle": strokes.polygon(3),
        "square": strokes.polygon(4),
        "line": strokes.straight_line(),
        "open_arc": strokes.circle(span=1.2 * np.pi),
        "wide_ellipse": [
            (int(strokes.CENTER_X + 160 * np.cos(t)), int(strokes.CENTER_Y + 55 * np.sin(t)))
            for t in np.linspace(0, 2 * np.pi, 200)
        ],
    }[stroke_name]
    assert sd.recognize_shape(stroke, strokes.CANVAS_SHAPE) is None


def test_heart_requires_template_similarity():
    """The similarity gate must apply unconditionally, not only when no notch is found."""
    stroke = strokes.heart(seed=5)
    mask = sd.stroke_points_to_mask(stroke, strokes.CANVAS_SHAPE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    features = sd.extract_stroke_features(stroke)

    assert sd.detect_heart(contour, features) is not None

    original = sd.HEART_MATCH_SCORE_GATE
    sd.HEART_MATCH_SCORE_GATE = 0.0  # nothing can be this similar
    try:
        assert sd.detect_heart(contour, features) is None
    finally:
        sd.HEART_MATCH_SCORE_GATE = original


# --- edge cases ----------------------------------------------------------


def test_empty_and_tiny_strokes_are_safe():
    assert sd.recognize_shape([], strokes.CANVAS_SHAPE) is None
    assert sd.recognize_shape([(10, 10)], strokes.CANVAS_SHAPE) is None
    assert sd.recognize_shape([(10, 10), (12, 12), (14, 14)], strokes.CANVAS_SHAPE) is None


def test_stroke_points_require_canvas_shape():
    with pytest.raises(ValueError):
        sd.recognize_shape(strokes.heart())


def test_mask_input_is_accepted():
    mask = sd.stroke_points_to_mask(strokes.circle(), strokes.CANVAS_SHAPE)
    match = sd.recognize_shape(mask)
    assert match is not None and match.name == "smiley"


def test_extract_stroke_features_needs_enough_points():
    assert sd.extract_stroke_features([(1, 1), (2, 2)]) is None
    assert sd.extract_stroke_features(strokes.heart()) is not None
