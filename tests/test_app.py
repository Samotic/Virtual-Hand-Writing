"""Tests for the canvas/overlay helpers that surround the detector."""

from __future__ import annotations

import numpy as np
import pytest

import main as app
import strokes
from overlay_utils import overlay_png


def blank_canvas():
    return np.full(strokes.CANVAS_SHAPE, 255, dtype=np.uint8)


def solid_sticker(size=64, color=(0, 0, 255), alpha=255):
    sticker = np.zeros((size, size, 4), dtype=np.uint8)
    sticker[..., :3] = color
    sticker[..., 3] = alpha
    return sticker


# --- overlay_png ---------------------------------------------------------


def test_overlay_blends_onto_canvas():
    canvas = blank_canvas()
    assert overlay_png(canvas, solid_sticker(), 100, 100, 64, 64) is True
    assert tuple(canvas[130, 130]) == (0, 0, 255)
    assert tuple(canvas[10, 10]) == (255, 255, 255)


def test_overlay_respects_alpha():
    canvas = blank_canvas()
    overlay_png(canvas, solid_sticker(alpha=0), 100, 100, 64, 64)
    assert tuple(canvas[130, 130]) == (255, 255, 255)


def test_overlay_clips_at_canvas_edges():
    canvas = blank_canvas()
    assert overlay_png(canvas, solid_sticker(), -20, -20, 64, 64) is True
    assert overlay_png(canvas, solid_sticker(), 700, 500, 64, 64) is False


@pytest.mark.parametrize("box", [(0, 0, 0, 10), (0, 0, 10, 0), (0, 0, -5, 10)])
def test_overlay_rejects_empty_boxes(box):
    assert overlay_png(blank_canvas(), solid_sticker(), *box) is False


def test_overlay_requires_alpha_channel():
    with pytest.raises(ValueError):
        overlay_png(blank_canvas(), np.zeros((10, 10, 3), dtype=np.uint8), 0, 0, 10, 10)


# --- finalize_stroke -----------------------------------------------------


def test_recognized_shape_replaces_the_drawing():
    canvas = blank_canvas()
    status = app.StatusMessage()
    app.finalize_stroke(canvas, strokes.heart(), {"heart": solid_sticker()}, status, 100.0)
    assert status.text.startswith("Detected: heart")
    assert status.color == app.STATUS_SUCCESS_COLOR


def test_unrecognized_stroke_is_drawn_as_is():
    canvas = blank_canvas()
    status = app.StatusMessage()
    app.finalize_stroke(canvas, strokes.scribble(3), {"heart": solid_sticker()}, status, 100.0)
    assert status.text == "Detected: none"
    assert (canvas != 255).any(), "the stroke should still be committed to the canvas"


def test_overlay_failure_message_survives(monkeypatch):
    """Regression: the warning used to be overwritten by 'Detected: none'."""
    monkeypatch.setattr(app, "overlay_png", lambda *a, **k: False)
    status = app.StatusMessage()
    app.finalize_stroke(blank_canvas(), strokes.heart(), {"heart": solid_sticker()}, status, 100.0)
    assert "overlay failed" in status.text
    assert status.color == app.STATUS_WARNING_COLOR


def test_empty_stroke_sets_no_status():
    status = app.StatusMessage()
    app.finalize_stroke(blank_canvas(), [], {}, status, 100.0)
    assert status.text == ""


def test_short_stroke_skips_recognition():
    status = app.StatusMessage()
    canvas = blank_canvas()
    app.finalize_stroke(canvas, [(10, 10), (12, 11), (14, 12)], {}, status, 100.0)
    assert status.text == ""
    assert (canvas != 255).any()


# --- geometry and input helpers ------------------------------------------


def test_pen_state_has_hysteresis():
    assert app.update_pen_state(False, 0.25) is True
    assert app.update_pen_state(False, 0.35) is False   # needs a firm pinch to start
    assert app.update_pen_state(True, 0.35) is True     # but tolerates drift while drawing
    assert app.update_pen_state(True, 0.50) is False


def test_scaled_box_stays_inside_canvas():
    x, y, w, h = app.scale_bounding_box((600, 450, 80, 60), 1.15, strokes.CANVAS_SHAPE)
    assert x >= 0 and y >= 0
    assert x + w <= strokes.CANVAS_SHAPE[1]
    assert y + h <= strokes.CANVAS_SHAPE[0]


def test_scaled_box_grows_around_its_center():
    x, y, w, h = app.scale_bounding_box((200, 200, 100, 100), 1.5, strokes.CANVAS_SHAPE)
    assert (w, h) == (150, 150)
    assert (x + w // 2, y + h // 2) == (250, 250)


def test_cursor_gain_scales_movement():
    canvas_shape = strokes.CANVAS_SHAPE
    cursor = np.array([100.0, 100.0], dtype=np.float32)
    last_raw = np.array([200.0, 200.0], dtype=np.float32)
    moved, raw = app.update_cursor_position(cursor, last_raw, (220, 200), canvas_shape, 0.5)
    assert tuple(moved) == (110.0, 100.0)   # 20px of hand travel -> 10px of cursor
    assert tuple(raw) == (220.0, 200.0)


def test_cursor_ignores_tracking_jumps():
    cursor = np.array([100.0, 100.0], dtype=np.float32)
    last_raw = np.array([100.0, 100.0], dtype=np.float32)
    moved, _ = app.update_cursor_position(cursor, last_raw, (500, 400), strokes.CANVAS_SHAPE, 1.0)
    assert tuple(moved) == (100.0, 100.0)


def test_cursor_seeds_from_first_sighting():
    moved, raw = app.update_cursor_position(None, None, (321, 123), strokes.CANVAS_SHAPE, 0.5)
    assert tuple(moved) == (321.0, 123.0) == tuple(raw)


def test_stroke_accumulates_points_above_the_minimum_distance():
    state = app.StrokeState()
    app.start_stroke(state, (100, 100), 0.0)
    app.add_stroke_point(state, (101, 100), 0.1)   # below MIN_DRAW_DISTANCE
    assert len(state.points) == 1
    app.add_stroke_point(state, (120, 100), 0.2)
    assert len(state.points) == 2
    finished = app.end_stroke(state)
    assert finished == [(100, 100), (120, 100)]
    assert state.active is False and state.points == []


def test_shape_recognition_needs_a_real_stroke():
    assert app.should_try_shape_recognition([(1, 1)]) is False
    assert app.should_try_shape_recognition(strokes.straight_line()) is True
    tight = [(100 + i % 3, 100) for i in range(30)]
    assert app.should_try_shape_recognition(tight) is False


def test_supported_drawings_text_lists_loaded_stickers():
    assert app.describe_supported_drawings({"heart": None}) == "heart"
    assert app.describe_supported_drawings({"heart": None, "star": None}) == "heart or star"
    assert (
        app.describe_supported_drawings({"heart": None, "star": None, "smiley": None})
        == "heart, star, or circle"
    )
    assert app.describe_supported_drawings({}) == "shapes"
