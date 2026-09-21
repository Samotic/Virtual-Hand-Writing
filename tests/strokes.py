"""Synthetic stroke generators that stand in for hand-drawn input in tests."""

from __future__ import annotations

import numpy as np


CANVAS_SHAPE = (480, 640, 3)
CENTER_X, CENTER_Y = 320, 240


def heart(scale: float = 9.0, points: int = 200, seed: int = 0, jitter: float = 0.0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi, points)
    x = 16.0 * np.sin(t) ** 3
    y = 13.0 * np.cos(t) - 5.0 * np.cos(2 * t) - 2.0 * np.cos(3 * t) - np.cos(4 * t)
    return [
        (
            int(CENTER_X + scale * a + rng.normal(0.0, jitter)),
            int(CENTER_Y - scale * b + rng.normal(0.0, jitter)),
        )
        for a, b in zip(x, y)
    ]


def star(outer: float = 110.0, inner: float = 46.0, step: int = 20, seed: int = 0, jitter: float = 0.0):
    rng = np.random.default_rng(seed)
    corners = []
    for index in range(11):
        angle = (-np.pi / 2.0) + (index * np.pi / 5.0)
        radius = outer if index % 2 == 0 else inner
        corners.append((CENTER_X + radius * np.cos(angle), CENTER_Y + radius * np.sin(angle)))

    stroke = []
    for start, end in zip(corners, corners[1:]):
        for k in range(step):
            f = k / step
            stroke.append(
                (
                    int(start[0] + (end[0] - start[0]) * f + rng.normal(0.0, jitter)),
                    int(start[1] + (end[1] - start[1]) * f + rng.normal(0.0, jitter)),
                )
            )
    return stroke


def circle(radius: float = 100.0, points: int = 200, seed: int = 0, jitter: float = 0.0, span: float = 2 * np.pi):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, span, points)
    return [
        (
            int(CENTER_X + radius * np.cos(a) + rng.normal(0.0, jitter)),
            int(CENTER_Y + radius * np.sin(a) + rng.normal(0.0, jitter)),
        )
        for a in t
    ]


def polygon(vertices: int, radius: float = 100.0, step: int = 25):
    corners = [
        (
            CENTER_X + radius * np.cos(2 * np.pi * i / vertices - np.pi / 2),
            CENTER_Y + radius * np.sin(2 * np.pi * i / vertices - np.pi / 2),
        )
        for i in range(vertices + 1)
    ]
    stroke = []
    for start, end in zip(corners, corners[1:]):
        for k in range(step):
            f = k / step
            stroke.append(
                (int(start[0] + (end[0] - start[0]) * f), int(start[1] + (end[1] - start[1]) * f))
            )
    return stroke


def scribble(seed: int, points: int = 120):
    """Fast random jabs - the worst case for false positives."""
    rng = np.random.default_rng(seed)
    return [
        (int(CENTER_X + rng.integers(-120, 120)), int(CENTER_Y + rng.integers(-120, 120)))
        for _ in range(points)
    ]


def doodle(seed: int, points: int = 150):
    """A smoother random walk, closer to an idle hand wandering."""
    rng = np.random.default_rng(1000 + seed)
    x, y = float(CENTER_X), float(CENTER_Y)
    stroke = []
    for _ in range(points):
        x += rng.normal(0.0, 14.0)
        y += rng.normal(0.0, 14.0)
        stroke.append((int(np.clip(x, 10, 630)), int(np.clip(y, 10, 470))))
    return stroke


def straight_line(points: int = 60):
    return [(150 + i * 5, 240) for i in range(points)]
