"""Helpers for blending transparent sticker PNGs onto the canvas."""

from __future__ import annotations

import cv2
import numpy as np


def overlay_png(
    canvas: np.ndarray,
    png: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> bool:
    """Resize a transparent PNG and alpha-blend it onto the canvas."""
    if w <= 0 or h <= 0:
        return False
    if png.ndim != 3 or png.shape[2] != 4:
        raise ValueError("overlay_png expects an RGBA image.")

    resized = cv2.resize(
        png,
        (w, h),
        interpolation=cv2.INTER_AREA if w < png.shape[1] else cv2.INTER_LINEAR,
    )

    canvas_height, canvas_width = canvas.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(canvas_width, x + w)
    y2 = min(canvas_height, y + h)
    if x1 >= x2 or y1 >= y2:
        return False

    sticker = resized[y1 - y : y2 - y, x1 - x : x2 - x]
    alpha = sticker[:, :, 3:4].astype(np.float32) / 255.0
    sticker_rgb = sticker[:, :, :3].astype(np.float32)
    canvas_roi = canvas[y1:y2, x1:x2].astype(np.float32)

    blended = (alpha * sticker_rgb) + ((1.0 - alpha) * canvas_roi)
    canvas[y1:y2, x1:x2] = blended.astype(np.uint8)
    return True
