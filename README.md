# Virtual Painter

Draw on a virtual canvas by moving your index finger in front of your webcam.
If you draw a supported shape, the rough sketch can be replaced by a clean
transparent sticker on the canvas.

## Install

```bash
pip install -r requirements.txt
```

## MediaPipe model

If your installed `mediapipe` package uses the newer Tasks API, download the
official hand landmarker model into this project folder as `hand_landmarker.task`:

```bash
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

PowerShell alternative:

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"
```

## Run

```bash
python main.py
```

Backward-compatible entry point:

```bash
python virtual_painter.py
```

If you saved the model somewhere else, pass its path explicitly:

```bash
python virtual_painter.py --model path/to/hand_landmarker.task
```

If your stickers are stored in a different folder, pass it explicitly:

```bash
python main.py --assets-dir path/to/assets
```

## Project layout

```text
main.py
shape_detector.py
overlay_utils.py
assets/heart.png
assets/star.png
assets/smiley.png
virtual_painter.py
tests/
```

## Controls

- Pinch your thumb and index finger together to write.
- Release the pinch to move your hand without drawing.
- Release the pinch or press `s` to finalize the current stroke for shape recognition.
- Press `-` or `+` to lower or raise pen sensitivity.
- Press `c` to clear the canvas.
- Press `q` to exit.

## Shape replacement

- Starter shapes included now:
  - Heart -> heart sticker
  - Star -> star sticker
  - Circle -> smiley sticker
- Stroke points are collected while you draw.
- When the stroke ends, the code converts the points into a binary mask and contour.
- The detector runs the stroke past each shape matcher in turn and keeps the
  highest-confidence result, so it is not hard-coded to only one form.
- Heart detection uses contour features plus a top-notch check from the stroke path.
- Star detection uses contour shape matching, convexity defects, and polygon corners.
- Circle detection is mapped to a smiley sticker.
- If a supported shape is recognized, the matching PNG in `assets/` is placed over
  the drawing area using alpha blending.
- If the shape is not recognized, the normal drawing stays on the canvas.

Each detector has to clear a hard similarity gate against its template before its
other features are even scored. That gate is what keeps ordinary scribbling from
being mistaken for a shape, so treat the thresholds at the top of
`shape_detector.py` as load-bearing and re-run the tests after changing them.

To add more forms later:

1. Add a new PNG to `assets/`.
2. Add a detector in `shape_detector.py` and list it in `recognize_shape`.
3. Add the shape name to `SUPPORTED_SHAPE_NAMES`, and a display label in
   `DRAWING_LABELS` in `main.py`.

## Tests

```bash
python -m pytest
```

The suite draws synthetic strokes and checks both directions: the supported
shapes still get recognized, and random scribbling does not trigger a sticker.
