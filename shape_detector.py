"""Contour-based shape detection helpers for the virtual painter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


STROKE_MASK_THICKNESS = 18
MORPH_KERNEL_SIZE = 9
MIN_CONTOUR_AREA = 1200.0
MIN_CONTOUR_PERIMETER = 140.0
HEART_MATCH_SCORE_MAX = 0.28
# Hard reject gate: a stroke this dissimilar to the template is never a heart,
# no matter how its other features score. Real hearts land around 0.014-0.038.
HEART_MATCH_SCORE_GATE = 0.08
HEART_EXTENT_MAX = 0.72
HEART_CONFIDENCE_THRESHOLD = 0.62
STAR_MATCH_SCORE_MAX = 0.30
STAR_CONFIDENCE_THRESHOLD = 0.65
CIRCLE_MATCH_SCORE_MAX = 0.10
CIRCLE_CONFIDENCE_THRESHOLD = 0.70


@dataclass
class ShapeMatch:
    name: str
    confidence: float
    bounding_box: tuple[int, int, int, int]
    contour: np.ndarray


@dataclass
class ContourFeatures:
    area: float
    perimeter: float
    bounding_box: tuple[int, int, int, int]
    aspect_ratio: float
    convex_hull: np.ndarray
    solidity: float
    extent: float
    circularity: float
    significant_defects: int
    top_notch_found: bool


@dataclass
class StrokeFeatures:
    bounding_box: tuple[int, int, int, int]
    closure_ratio: float
    top_notch_depth_ratio: float
    bottom_center_offset_ratio: float
    lobe_balance_ratio: float


def _build_heart_template(size: int = 256) -> np.ndarray:
    samples = np.linspace(0.0, 2.0 * np.pi, 360, dtype=np.float32)
    x = 16.0 * np.sin(samples) ** 3
    y = (
        13.0 * np.cos(samples)
        - 5.0 * np.cos(2.0 * samples)
        - 2.0 * np.cos(3.0 * samples)
        - np.cos(4.0 * samples)
    )

    points = np.column_stack((x, -y))
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    normalized = (points - min_xy) / span
    scaled = normalized * (size - 40) + 20
    return scaled.astype(np.int32).reshape(-1, 1, 2)


HEART_TEMPLATE = _build_heart_template()
SUPPORTED_SHAPE_NAMES = ("heart", "star", "smiley")


def _build_star_template(size: int = 256) -> np.ndarray:
    center = size / 2.0
    outer_radius = size * 0.40
    inner_radius = outer_radius * 0.42
    points: list[tuple[int, int]] = []
    for index in range(10):
        angle = (-np.pi / 2.0) + (index * np.pi / 5.0)
        radius = outer_radius if index % 2 == 0 else inner_radius
        x = center + (radius * np.cos(angle))
        y = center + (radius * np.sin(angle))
        points.append((int(round(x)), int(round(y))))
    return np.array(points, dtype=np.int32).reshape(-1, 1, 2)


def _build_circle_template(size: int = 256) -> np.ndarray:
    samples = np.linspace(0.0, 2.0 * np.pi, 360, dtype=np.float32)
    radius = size * 0.38
    center = size / 2.0
    x = center + (radius * np.cos(samples))
    y = center + (radius * np.sin(samples))
    points = np.column_stack((x, y))
    return np.round(points).astype(np.int32).reshape(-1, 1, 2)


STAR_TEMPLATE = _build_star_template()
CIRCLE_TEMPLATE = _build_circle_template()


def stroke_points_to_mask(
    points: Iterable[tuple[int, int]],
    canvas_shape: tuple[int, int] | tuple[int, int, int],
) -> np.ndarray:
    """Convert stroke points into a binary mask that can be contoured."""
    mask = np.zeros(canvas_shape[:2], dtype=np.uint8)
    point_list = list(points)
    if not point_list:
        return mask

    if len(point_list) == 1:
        cv2.circle(
            mask,
            point_list[0],
            STROKE_MASK_THICKNESS,
            255,
            -1,
            lineType=cv2.LINE_AA,
        )
        return mask

    contour = np.array(point_list, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        mask,
        [contour],
        False,
        255,
        STROKE_MASK_THICKNESS,
        lineType=cv2.LINE_AA,
    )

    x, y, w, h = cv2.boundingRect(contour)
    diagonal = float(np.hypot(w, h))
    endpoint_gap = float(np.linalg.norm(np.subtract(point_list[0], point_list[-1])))
    if diagonal > 0 and endpoint_gap <= diagonal * 0.28:
        cv2.line(
            mask,
            point_list[0],
            point_list[-1],
            255,
            STROKE_MASK_THICKNESS,
            lineType=cv2.LINE_AA,
        )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def extract_stroke_features(points: Iterable[tuple[int, int]]) -> StrokeFeatures | None:
    point_list = list(points)
    if len(point_list) < 5:
        return None

    point_array = np.array(point_list, dtype=np.float32)
    min_xy = point_array.min(axis=0)
    max_xy = point_array.max(axis=0)
    width = max(float(max_xy[0] - min_xy[0]), 1.0)
    height = max(float(max_xy[1] - min_xy[1]), 1.0)
    diagonal = max(float(np.hypot(width, height)), 1.0)
    center_x = float(min_xy[0] + (width / 2.0))

    bins = np.linspace(min_xy[0], max_xy[0], 6)
    top_profile: list[float] = []
    for index in range(5):
        left = bins[index]
        right = bins[index + 1]
        if index == 4:
            mask = (point_array[:, 0] >= left) & (point_array[:, 0] <= right)
        else:
            mask = (point_array[:, 0] >= left) & (point_array[:, 0] < right)
        top_profile.append(float(point_array[mask, 1].min()) if np.any(mask) else np.nan)

    # A heart usually has a dip in the middle of the top edge, unlike a circle.
    if all(np.isfinite(value) for value in top_profile[1:4]):
        top_notch_depth = max(
            0.0,
            top_profile[2] - min(top_profile[1], top_profile[3]),
        )
        lobe_balance = abs(top_profile[1] - top_profile[3]) / height
    else:
        top_notch_depth = 0.0
        lobe_balance = 1.0

    endpoint_gap = float(np.linalg.norm(point_array[0] - point_array[-1]))
    bottom_point = point_array[point_array[:, 1].argmax()]
    bottom_center_offset = abs(float(bottom_point[0]) - center_x) / width

    return StrokeFeatures(
        bounding_box=(
            int(round(min_xy[0])),
            int(round(min_xy[1])),
            int(round(width)),
            int(round(height)),
        ),
        closure_ratio=endpoint_gap / diagonal,
        top_notch_depth_ratio=top_notch_depth / height,
        bottom_center_offset_ratio=bottom_center_offset,
        lobe_balance_ratio=lobe_balance,
    )


def extract_contour_features(contour: np.ndarray) -> ContourFeatures:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / max(h, 1)
    convex_hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(convex_hull))
    solidity = area / hull_area if hull_area > 0 else 0.0
    extent = area / max(float(w * h), 1.0)
    circularity = (4.0 * np.pi * area) / max(perimeter * perimeter, 1.0)
    significant_defects, top_notch_found = _count_convexity_defects(contour, (x, y, w, h))
    return ContourFeatures(
        area=area,
        perimeter=perimeter,
        bounding_box=(x, y, w, h),
        aspect_ratio=aspect_ratio,
        convex_hull=convex_hull,
        solidity=solidity,
        extent=extent,
        circularity=circularity,
        significant_defects=significant_defects,
        top_notch_found=top_notch_found,
    )


def _count_convexity_defects(
    contour: np.ndarray,
    bounding_box: tuple[int, int, int, int],
) -> tuple[int, bool]:
    x, y, w, h = bounding_box
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 4:
        return 0, False

    defects = cv2.convexityDefects(contour, hull_indices)
    if defects is None:
        return 0, False

    center_x = x + (w / 2.0)
    significant = 0
    top_notch_found = False
    minimum_depth = max(6.0, min(w, h) * 0.05)

    for start_idx, end_idx, far_idx, depth_raw in defects[:, 0]:
        depth = depth_raw / 256.0
        if depth < minimum_depth:
            continue

        significant += 1
        farthest = contour[far_idx][0]
        in_top_half = farthest[1] <= y + (0.45 * h)
        near_center = abs(farthest[0] - center_x) <= (0.22 * w)
        if in_top_half and near_center:
            top_notch_found = True

    return significant, top_notch_found


def detect_heart(
    contour: np.ndarray,
    stroke_features: StrokeFeatures | None = None,
) -> ShapeMatch | None:
    """Recognize a heart using contour shape matching plus simple heuristics."""
    features = extract_contour_features(contour)
    if features.area < MIN_CONTOUR_AREA or features.perimeter < MIN_CONTOUR_PERIMETER:
        return None

    x, y, w, h = features.bounding_box
    center_x = x + (w / 2.0)
    bottom_point = contour[contour[:, :, 1].argmax()][0]
    contour_bottom_centered = abs(bottom_point[0] - center_x) <= (0.25 * w)
    stroke_notch_found = (
        stroke_features is not None and stroke_features.top_notch_depth_ratio >= 0.04
    )
    top_notch_found = features.top_notch_found or stroke_notch_found
    bottom_centered = contour_bottom_centered or (
        stroke_features is not None and stroke_features.bottom_center_offset_ratio <= 0.22
    )
    lobe_balance_ok = (
        stroke_features is None or stroke_features.lobe_balance_ratio <= 0.20
    )

    shape_score = cv2.matchShapes(
        contour, HEART_TEMPLATE, cv2.CONTOURS_MATCH_I1, 0.0
    )
    match_component = float(np.clip(1.0 - (shape_score / HEART_MATCH_SCORE_MAX), 0.0, 1.0))
    aspect_component = 1.0 if 0.65 <= features.aspect_ratio <= 1.35 else 0.0
    solidity_component = 1.0 if 0.55 <= features.solidity <= 0.995 else 0.0
    extent_component = 1.0 if 0.45 <= features.extent <= 0.88 else 0.0
    notch_component = 1.0 if top_notch_found else 0.0
    tip_component = 1.0 if bottom_centered else 0.0
    lobe_component = 1.0 if lobe_balance_ok else 0.0
    stroke_notch_component = (
        1.0
        if stroke_features is not None and stroke_features.top_notch_depth_ratio >= 0.06
        else 0.0
    )

    confidence = (
        match_component * 0.25
        + aspect_component * 0.10
        + solidity_component * 0.10
        + extent_component * 0.10
        + notch_component * 0.20
        + tip_component * 0.10
        + lobe_component * 0.10
        + stroke_notch_component * 0.05
    )

    if features.aspect_ratio < 0.55 or features.aspect_ratio > 1.45:
        return None
    if features.solidity < 0.42 or features.solidity > 0.998:
        return None
    if features.extent < 0.35 or features.extent > HEART_EXTENT_MAX:
        return None
    if stroke_features is not None and stroke_features.top_notch_depth_ratio < 0.035:
        return None
    if stroke_features is not None and stroke_features.lobe_balance_ratio > 0.28:
        return None
    if not top_notch_found:
        return None
    if shape_score > HEART_MATCH_SCORE_GATE:
        return None
    if confidence < HEART_CONFIDENCE_THRESHOLD:
        return None

    return ShapeMatch(
        name="heart",
        confidence=round(float(confidence), 2),
        bounding_box=features.bounding_box,
        contour=contour,
    )


def detect_star(
    contour: np.ndarray,
    stroke_features: StrokeFeatures | None = None,
) -> ShapeMatch | None:
    """Recognize a five-point star from its contour shape."""
    features = extract_contour_features(contour)
    if features.area < MIN_CONTOUR_AREA or features.perimeter < MIN_CONTOUR_PERIMETER:
        return None

    approx = cv2.approxPolyDP(contour, 0.02 * features.perimeter, True)
    shape_score = cv2.matchShapes(contour, STAR_TEMPLATE, cv2.CONTOURS_MATCH_I1, 0.0)
    defects_ok = 4 <= features.significant_defects <= 6
    vertices_ok = 8 <= len(approx) <= 12
    aspect_ok = 0.65 <= features.aspect_ratio <= 1.35
    solidity_ok = 0.45 <= features.solidity <= 0.80
    extent_ok = 0.30 <= features.extent <= 0.65
    closure_ok = stroke_features is None or stroke_features.closure_ratio <= 0.35

    match_component = float(np.clip(1.0 - (shape_score / STAR_MATCH_SCORE_MAX), 0.0, 1.0))
    confidence = (
        match_component * 0.35
        + (1.0 if defects_ok else 0.0) * 0.20
        + (1.0 if vertices_ok else 0.0) * 0.15
        + (1.0 if aspect_ok else 0.0) * 0.10
        + (1.0 if solidity_ok else 0.0) * 0.10
        + (1.0 if extent_ok else 0.0) * 0.10
    )

    if not defects_ok or not vertices_ok:
        return None
    if not aspect_ok or not solidity_ok or not extent_ok or not closure_ok:
        return None
    if shape_score > STAR_MATCH_SCORE_MAX:
        return None
    if confidence < STAR_CONFIDENCE_THRESHOLD:
        return None

    return ShapeMatch(
        name="star",
        confidence=round(float(confidence), 2),
        bounding_box=features.bounding_box,
        contour=contour,
    )


def detect_smiley(
    contour: np.ndarray,
    stroke_features: StrokeFeatures | None = None,
) -> ShapeMatch | None:
    """Treat a simple drawn circle as a smiley placeholder sticker."""
    features = extract_contour_features(contour)
    if features.area < MIN_CONTOUR_AREA or features.perimeter < MIN_CONTOUR_PERIMETER:
        return None

    shape_score = cv2.matchShapes(contour, CIRCLE_TEMPLATE, cv2.CONTOURS_MATCH_I1, 0.0)
    aspect_ok = 0.80 <= features.aspect_ratio <= 1.20
    solidity_ok = features.solidity >= 0.90
    extent_ok = 0.68 <= features.extent <= 0.86
    circularity_ok = features.circularity >= 0.74
    simple_outline_ok = features.significant_defects <= 1
    no_heart_notch = (
        stroke_features is None or stroke_features.top_notch_depth_ratio < 0.03
    )

    match_component = float(
        np.clip(1.0 - (shape_score / CIRCLE_MATCH_SCORE_MAX), 0.0, 1.0)
    )
    confidence = (
        match_component * 0.30
        + (1.0 if aspect_ok else 0.0) * 0.15
        + (1.0 if solidity_ok else 0.0) * 0.15
        + (1.0 if extent_ok else 0.0) * 0.15
        + (1.0 if circularity_ok else 0.0) * 0.15
        + (1.0 if simple_outline_ok else 0.0) * 0.10
    )

    if not aspect_ok or not solidity_ok or not extent_ok:
        return None
    if not circularity_ok or not simple_outline_ok or not no_heart_notch:
        return None
    if shape_score > CIRCLE_MATCH_SCORE_MAX:
        return None
    if confidence < CIRCLE_CONFIDENCE_THRESHOLD:
        return None

    return ShapeMatch(
        name="smiley",
        confidence=round(float(confidence), 2),
        bounding_box=features.bounding_box,
        contour=contour,
    )


def recognize_shape(
    points_or_mask: Iterable[tuple[int, int]] | np.ndarray,
    canvas_shape: tuple[int, int] | tuple[int, int, int] | None = None,
) -> ShapeMatch | None:
    """Recognize a supported shape from stroke points or a prepared binary mask."""
    stroke_features = None
    if isinstance(points_or_mask, np.ndarray):
        mask = points_or_mask
    else:
        if canvas_shape is None:
            raise ValueError("canvas_shape is required when passing stroke points.")
        point_list = list(points_or_mask)
        stroke_features = extract_stroke_features(point_list)
        mask = stroke_points_to_mask(point_list, canvas_shape)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    best_match = None
    for detector in (detect_heart, detect_star, detect_smiley):
        match = detector(largest_contour, stroke_features)
        if match is None:
            continue
        if best_match is None or match.confidence > best_match.confidence:
            best_match = match
    return best_match
