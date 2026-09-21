"""Virtual painter controlled by index finger movement and shape stickers.

Install dependencies:
    pip install opencv-python mediapipe numpy

Run:
    python main.py

You can also keep using:
    python virtual_painter.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from overlay_utils import overlay_png
from shape_detector import SUPPORTED_SHAPE_NAMES, recognize_shape


WINDOW_TITLE = "Virtual Painter"
MODEL_FILENAME = "hand_landmarker.task"
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
ASSETS_DIRECTORY = "assets"
WRIST_INDEX = 0
THUMB_TIP_INDEX = 4
INDEX_FINGER_MCP_INDEX = 5
INDEX_FINGER_TIP_INDEX = 8
MIDDLE_FINGER_MCP_INDEX = 9
PINKY_MCP_INDEX = 17
CANVAS_COLOR = 255
LINE_COLOR = (30, 30, 30)
LINE_THICKNESS = 3
DRAW_MARKER_COLOR = (0, 200, 0)
HOVER_MARKER_COLOR = (0, 200, 255)
STATUS_SUCCESS_COLOR = (50, 140, 50)
STATUS_INFO_COLOR = (80, 80, 80)
STATUS_WARNING_COLOR = (0, 140, 255)
PINCH_ON_RATIO = 0.30
PINCH_OFF_RATIO = 0.42
MIN_DRAW_DISTANCE = 2.5
DEFAULT_CURSOR_GAIN = 0.55
MIN_CURSOR_GAIN = 0.20
MAX_CURSOR_GAIN = 1.20
CURSOR_GAIN_STEP = 0.05
RAW_JUMP_THRESHOLD = 90.0
CURSOR_RADIUS = 8
STROKE_IDLE_TIMEOUT_SECONDS = 0.70
MIN_SHAPE_POINTS = 12
MIN_SHAPE_SPAN = 30.0
STICKER_BOX_SCALE = 1.15
STATUS_MESSAGE_SECONDS = 1.80
DRAWING_LABELS = {
    "heart": "heart",
    "star": "star",
    "smiley": "circle",
}


@dataclass
class HandState:
    index_tip: tuple[int, int]
    pinch_ratio: float


@dataclass
class StrokeState:
    active: bool = False
    points: list[tuple[int, int]] = field(default_factory=list)
    started_at: float = 0.0
    last_point_at: float = 0.0


@dataclass
class StatusMessage:
    text: str = ""
    color: tuple[int, int, int] = STATUS_INFO_COLOR
    expires_at: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw on a virtual canvas using your index finger."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index to open (default: 0).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Path to hand_landmarker.task. Required when MediaPipe exposes the "
            "newer Tasks API instead of mp.solutions."
        ),
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=None,
        help="Optional folder that contains sticker PNGs such as heart.png and star.png.",
    )
    return parser.parse_args()


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def resolve_model_path(model_arg: Path | None) -> Path:
    candidates = []
    if model_arg is not None:
        candidates.append(model_arg.expanduser().resolve())

    script_dir = Path(__file__).resolve().parent
    candidates.append(script_dir / MODEL_FILENAME)
    candidates.append(Path.cwd() / MODEL_FILENAME)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    expected_path = candidates[0] if model_arg is not None else script_dir / MODEL_FILENAME
    raise FileNotFoundError(
        f"MediaPipe Tasks mode requires a model file.\n"
        f"Expected: {expected_path}\n"
        f"Download: {MODEL_DOWNLOAD_URL}"
    )


def resolve_assets_directory(assets_dir_arg: Path | None) -> Path:
    candidates = []
    if assets_dir_arg is not None:
        candidates.append(assets_dir_arg.expanduser().resolve())

    script_dir = Path(__file__).resolve().parent
    candidates.append(script_dir / ASSETS_DIRECTORY)
    candidates.append(Path.cwd() / ASSETS_DIRECTORY)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    expected_path = candidates[0] if assets_dir_arg is not None else script_dir / ASSETS_DIRECTORY
    raise FileNotFoundError(f"Could not find assets directory: {expected_path}")


def load_transparent_png(path: Path) -> np.ndarray:
    png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if png is None:
        raise FileNotFoundError(f"Could not load PNG: {path}")
    if png.ndim != 3 or png.shape[2] != 4:
        raise ValueError(f"PNG must include an alpha channel: {path}")
    return png


def load_sticker_assets(assets_dir: Path) -> dict[str, np.ndarray]:
    sticker_assets: dict[str, np.ndarray] = {}
    for shape_name in SUPPORTED_SHAPE_NAMES:
        asset_path = assets_dir / f"{shape_name}.png"
        if not asset_path.is_file():
            continue
        sticker_assets[shape_name] = load_transparent_png(asset_path)
    if not sticker_assets:
        raise FileNotFoundError(f"No sticker PNGs were found in: {assets_dir}")
    return sticker_assets


def describe_supported_drawings(sticker_assets: dict[str, np.ndarray]) -> str:
    drawing_names = [
        DRAWING_LABELS.get(shape_name, shape_name)
        for shape_name in SUPPORTED_SHAPE_NAMES
        if shape_name in sticker_assets
    ]
    if not drawing_names:
        return "shapes"
    if len(drawing_names) == 1:
        return drawing_names[0]
    if len(drawing_names) == 2:
        return " or ".join(drawing_names)
    return ", ".join(drawing_names[:-1]) + f", or {drawing_names[-1]}"


def create_blank_canvas(frame: np.ndarray) -> np.ndarray:
    return np.full_like(frame, CANVAS_COLOR)


def landmark_xy(landmark) -> np.ndarray:
    return np.array([landmark.x, landmark.y], dtype=np.float32)


def landmark_to_pixel(landmark, frame_shape: tuple[int, int, int]) -> tuple[int, int]:
    frame_height, frame_width = frame_shape[:2]
    x = int(landmark.x * frame_width)
    y = int(landmark.y * frame_height)
    x = min(max(x, 0), frame_width - 1)
    y = min(max(y, 0), frame_height - 1)
    return (x, y)


def measure_pinch_ratio(landmarks) -> float:
    thumb_tip = landmark_xy(landmarks[THUMB_TIP_INDEX])
    index_tip = landmark_xy(landmarks[INDEX_FINGER_TIP_INDEX])
    hand_width = np.linalg.norm(
        landmark_xy(landmarks[INDEX_FINGER_MCP_INDEX])
        - landmark_xy(landmarks[PINKY_MCP_INDEX])
    )
    hand_height = np.linalg.norm(
        landmark_xy(landmarks[WRIST_INDEX])
        - landmark_xy(landmarks[MIDDLE_FINGER_MCP_INDEX])
    )
    hand_scale = max(hand_width, hand_height, 1e-6)
    pinch_distance = np.linalg.norm(index_tip - thumb_tip)
    return pinch_distance / hand_scale


def update_pen_state(pen_is_down: bool, pinch_ratio: float) -> bool:
    if pen_is_down:
        return pinch_ratio <= PINCH_OFF_RATIO
    return pinch_ratio <= PINCH_ON_RATIO


def draw_fingertip_marker(
    frame: np.ndarray, fingertip: tuple[int, int], should_draw: bool
) -> None:
    marker_color = DRAW_MARKER_COLOR if should_draw else HOVER_MARKER_COLOR
    cv2.circle(frame, fingertip, 10, marker_color, -1)


def clamp_float_point(point: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    frame_height, frame_width = shape[:2]
    return np.array(
        [
            np.clip(point[0], 0, frame_width - 1),
            np.clip(point[1], 0, frame_height - 1),
        ],
        dtype=np.float32,
    )


def update_cursor_position(
    cursor_point: np.ndarray | None,
    last_raw_tip: np.ndarray | None,
    raw_tip: tuple[int, int],
    canvas_shape: tuple[int, int, int],
    cursor_gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    raw_tip_point = np.array(raw_tip, dtype=np.float32)

    if cursor_point is None:
        return raw_tip_point, raw_tip_point

    if last_raw_tip is None:
        return cursor_point, raw_tip_point

    raw_delta = raw_tip_point - last_raw_tip
    if np.linalg.norm(raw_delta) > RAW_JUMP_THRESHOLD:
        return cursor_point, raw_tip_point

    next_cursor = clamp_float_point(cursor_point + raw_delta * cursor_gain, canvas_shape)
    return next_cursor, raw_tip_point


def point_to_int_tuple(point: np.ndarray) -> tuple[int, int]:
    return (int(round(point[0])), int(round(point[1])))


def detect_hand_state_with_solutions(
    frame: np.ndarray,
    hands_detector,
    drawing_utils,
    hands_module,
) -> HandState | None:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_frame)

    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = hand_landmarks.landmark
    fingertip = landmark_to_pixel(
        landmarks[hands_module.HandLandmark.INDEX_FINGER_TIP], frame.shape
    )
    pinch_ratio = measure_pinch_ratio(landmarks)

    drawing_utils.draw_landmarks(frame, hand_landmarks, hands_module.HAND_CONNECTIONS)
    return HandState(index_tip=fingertip, pinch_ratio=pinch_ratio)


def create_tasks_landmarker(model_path: Path):
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def detect_hand_state_with_tasks(
    frame: np.ndarray,
    landmarker,
    drawing_utils,
    connections,
    timestamp_ms: int,
) -> HandState | None:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.hand_landmarks:
        return None

    hand_landmarks = result.hand_landmarks[0]
    drawing_utils.draw_landmarks(frame, hand_landmarks, connections)
    fingertip = landmark_to_pixel(hand_landmarks[INDEX_FINGER_TIP_INDEX], frame.shape)
    pinch_ratio = measure_pinch_ratio(hand_landmarks)
    return HandState(index_tip=fingertip, pinch_ratio=pinch_ratio)


def start_stroke(
    stroke_state: StrokeState, point: tuple[int, int], timestamp: float
) -> None:
    stroke_state.active = True
    stroke_state.points = [point]
    stroke_state.started_at = timestamp
    stroke_state.last_point_at = timestamp


def add_stroke_point(
    stroke_state: StrokeState, point: tuple[int, int], timestamp: float
) -> None:
    if not stroke_state.active:
        start_stroke(stroke_state, point, timestamp)
        return

    if point == stroke_state.points[-1]:
        return

    distance = np.linalg.norm(np.subtract(point, stroke_state.points[-1]))
    if distance < MIN_DRAW_DISTANCE:
        return

    stroke_state.points.append(point)
    stroke_state.last_point_at = timestamp


def end_stroke(stroke_state: StrokeState) -> list[tuple[int, int]]:
    points = stroke_state.points.copy()
    stroke_state.active = False
    stroke_state.points.clear()
    stroke_state.started_at = 0.0
    stroke_state.last_point_at = 0.0
    return points


def draw_stroke(
    image: np.ndarray,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if not points:
        return

    if len(points) == 1:
        cv2.circle(image, points[0], thickness, color, -1, lineType=cv2.LINE_AA)
        return

    contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [contour], False, color, thickness, lineType=cv2.LINE_AA)


def stroke_span(points: list[tuple[int, int]]) -> float:
    if not points:
        return 0.0
    contour = np.array(points, dtype=np.float32)
    min_xy = contour.min(axis=0)
    max_xy = contour.max(axis=0)
    return float(np.linalg.norm(max_xy - min_xy))


def should_try_shape_recognition(points: list[tuple[int, int]]) -> bool:
    return len(points) >= MIN_SHAPE_POINTS and stroke_span(points) >= MIN_SHAPE_SPAN


def scale_bounding_box(
    box: tuple[int, int, int, int],
    scale: float,
    canvas_shape: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    canvas_height, canvas_width = canvas_shape[:2]
    center_x = x + (w / 2.0)
    center_y = y + (h / 2.0)
    scaled_w = max(1, int(round(w * scale)))
    scaled_h = max(1, int(round(h * scale)))
    scaled_x = int(round(center_x - (scaled_w / 2.0)))
    scaled_y = int(round(center_y - (scaled_h / 2.0)))

    scaled_x = max(0, min(scaled_x, canvas_width - 1))
    scaled_y = max(0, min(scaled_y, canvas_height - 1))
    scaled_w = min(scaled_w, canvas_width - scaled_x)
    scaled_h = min(scaled_h, canvas_height - scaled_y)
    return (scaled_x, scaled_y, scaled_w, scaled_h)


def set_status(
    status_message: StatusMessage,
    text: str,
    color: tuple[int, int, int],
    timestamp: float,
) -> None:
    status_message.text = text
    status_message.color = color
    status_message.expires_at = timestamp + STATUS_MESSAGE_SECONDS


def finalize_stroke(
    canvas: np.ndarray,
    stroke_points: list[tuple[int, int]],
    sticker_assets: dict[str, np.ndarray],
    status_message: StatusMessage,
    timestamp: float,
) -> None:
    if not stroke_points:
        return

    shape_match = None
    if should_try_shape_recognition(stroke_points):
        # The detector analyzes the finished stroke before we commit it to the canvas.
        shape_match = recognize_shape(stroke_points, canvas.shape)

    overlay_failed = False
    if shape_match is not None and shape_match.name in sticker_assets:
        sticker_box = scale_bounding_box(
            shape_match.bounding_box, STICKER_BOX_SCALE, canvas.shape
        )
        if overlay_png(canvas, sticker_assets[shape_match.name], *sticker_box):
            set_status(
                status_message,
                f"Detected: {shape_match.name} ({shape_match.confidence:.2f})",
                STATUS_SUCCESS_COLOR,
                timestamp,
            )
            return

        set_status(
            status_message,
            f"{shape_match.name} detected, but sticker overlay failed",
            STATUS_WARNING_COLOR,
            timestamp,
        )
        overlay_failed = True

    draw_stroke(canvas, stroke_points, LINE_COLOR, LINE_THICKNESS)
    # Keep the overlay warning visible instead of replacing it with "Detected: none".
    if not overlay_failed and should_try_shape_recognition(stroke_points):
        set_status(status_message, "Detected: none", STATUS_INFO_COLOR, timestamp)


def main() -> None:
    args = parse_args()
    assets_dir = resolve_assets_directory(args.assets_dir)
    sticker_assets = load_sticker_assets(assets_dir)
    supported_drawings_text = describe_supported_drawings(sticker_assets)

    hands_detector = None
    landmarker = None
    last_timestamp_ms = 0

    if hasattr(mp, "solutions"):
        hands_module = mp.solutions.hands
        drawing_utils = mp.solutions.drawing_utils
        hands_detector = hands_module.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        def detect_hand_state(frame: np.ndarray) -> HandState | None:
            return detect_hand_state_with_solutions(
                frame, hands_detector, drawing_utils, hands_module
            )

    else:
        model_path = resolve_model_path(args.model)
        landmarker = create_tasks_landmarker(model_path)
        drawing_utils = mp.tasks.vision.drawing_utils
        connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

        def detect_hand_state(frame: np.ndarray) -> HandState | None:
            nonlocal last_timestamp_ms
            timestamp_ms = time.monotonic_ns() // 1_000_000
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms
            return detect_hand_state_with_tasks(
                frame, landmarker, drawing_utils, connections, timestamp_ms
            )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    canvas = None
    pen_is_down = False
    cursor_point = None
    last_raw_tip = None
    cursor_gain = DEFAULT_CURSOR_GAIN
    stroke_state = StrokeState()
    waiting_for_pen_reset = False
    status_message = StatusMessage()

    try:
        while True:
            now = time.monotonic()
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)

            if canvas is None or canvas.shape != frame.shape:
                canvas = create_blank_canvas(frame)
                cursor_point = None
                last_raw_tip = None
                stroke_state = StrokeState()
                waiting_for_pen_reset = False
                status_message = StatusMessage()

            hand_state = detect_hand_state(frame)
            was_pen_down = pen_is_down
            pen_is_down = (
                update_pen_state(pen_is_down, hand_state.pinch_ratio)
                if hand_state is not None
                else False
            )

            if not pen_is_down:
                waiting_for_pen_reset = False

            if hand_state is not None:
                cursor_point, last_raw_tip = update_cursor_position(
                    cursor_point,
                    last_raw_tip,
                    hand_state.index_tip,
                    canvas.shape,
                    cursor_gain,
                )
                draw_fingertip_marker(frame, hand_state.index_tip, pen_is_down)
            else:
                last_raw_tip = None

            if cursor_point is not None and pen_is_down and not waiting_for_pen_reset:
                current_point = point_to_int_tuple(cursor_point)
                if not stroke_state.active:
                    start_stroke(stroke_state, current_point, now)
                else:
                    add_stroke_point(stroke_state, current_point, now)

            canvas_view = canvas.copy()
            if stroke_state.active:
                # Show the active stroke as a preview so we can replace it cleanly later.
                draw_stroke(canvas_view, stroke_state.points, LINE_COLOR, LINE_THICKNESS)
            if cursor_point is not None and hand_state is not None:
                cv2.circle(
                    canvas_view,
                    point_to_int_tuple(cursor_point),
                    CURSOR_RADIUS,
                    DRAW_MARKER_COLOR if pen_is_down else HOVER_MARKER_COLOR,
                    -1,
                )

            draw_label(frame, "Webcam", (10, 30))
            draw_label(frame, "Pinch thumb + index to write", (10, 65))
            draw_label(frame, "Release pinch or press S to finalize", (10, 100))

            if waiting_for_pen_reset and pen_is_down:
                mode_text = "Mode: Lift fingers to start the next stroke"
                mode_color = STATUS_WARNING_COLOR
            elif pen_is_down:
                mode_text = "Mode: Writing"
                mode_color = DRAW_MARKER_COLOR
            elif hand_state is not None:
                mode_text = "Mode: Move"
                mode_color = HOVER_MARKER_COLOR
            else:
                mode_text = "Mode: Idle"
                mode_color = (180, 180, 180)
            draw_label(frame, mode_text, (10, 135), mode_color)

            draw_label(canvas_view, "Canvas", (10, 30), (40, 40, 40))
            draw_label(
                canvas_view,
                f"Sensitivity: {cursor_gain:.2f}  (-/+= adjust)",
                (10, 65),
                (80, 80, 80),
            )
            draw_label(
                canvas_view,
                "C = clear   S = finalize stroke   Q = quit",
                (10, 100),
                (80, 80, 80),
            )
            draw_label(
                canvas_view,
                f"Draw {supported_drawings_text} to place a sticker",
                (10, 135),
                (80, 80, 80),
            )

            if status_message.text and now <= status_message.expires_at:
                draw_label(canvas_view, status_message.text, (10, 170), status_message.color)

            combined_view = np.hstack((frame, canvas_view))
            cv2.imshow(WINDOW_TITLE, combined_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                canvas = create_blank_canvas(frame)
                stroke_state = StrokeState()
                waiting_for_pen_reset = False
                status_message = StatusMessage()
                continue
            if key in (ord("-"), ord("_")):
                cursor_gain = max(MIN_CURSOR_GAIN, cursor_gain - CURSOR_GAIN_STEP)
            elif key in (ord("="), ord("+")):
                cursor_gain = min(MAX_CURSOR_GAIN, cursor_gain + CURSOR_GAIN_STEP)
            elif key == ord("q"):
                break

            finalize_reason = None
            wait_for_release = False
            if key == ord("s") and stroke_state.active:
                finalize_reason = "manual"
                wait_for_release = pen_is_down
            elif stroke_state.active and was_pen_down and not pen_is_down:
                finalize_reason = "release"
            elif (
                stroke_state.active
                and pen_is_down
                and now - stroke_state.last_point_at >= STROKE_IDLE_TIMEOUT_SECONDS
            ):
                finalize_reason = "idle"
                wait_for_release = True

            if finalize_reason is not None:
                finished_points = end_stroke(stroke_state)
                # Once the stroke ends, either place a sticker or commit the original drawing.
                finalize_stroke(
                    canvas,
                    finished_points,
                    sticker_assets,
                    status_message,
                    now,
                )
                waiting_for_pen_reset = wait_for_release
    finally:
        if hands_detector is not None:
            hands_detector.close()
        if landmarker is not None:
            landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
