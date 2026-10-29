# Driver Drowsiness Detection

A real-time drowsiness detection system built with Python, OpenCV, and Google MediaPipe. It tracks facial landmarks through your webcam, calculates the Eye Aspect Ratio (EAR) to measure eye closure duration, and sounds an audio alarm if it detects prolonged closure (microsleep or nodding off).

---

## Features

- **Live Landmark Tracking**: Uses MediaPipe's 468-point Face Landmarker (runs smoothly on CPU / Apple Silicon).
- **Temporal State Machine**: Distinguishes normal blinks (100–400 ms) from actual drowsiness:
  - **Normal**: Eyes open / natural blinking.
  - **Warning**: Eyes closed for > 0.9 seconds (yellow HUD alert).
  - **Drowsy**: Eyes closed for > 1.8 seconds (red pulsating alert + audio alarm).
- **Audio Alarm**: Non-blocking audio playback via Pygame with cooldown prevention and in-app mute toggle. Auto-generates a synthesized beep if the sound file is missing.
- **HUD Overlay**: Shows live EAR values, smoothed FPS, state badge, and eye landmark contours.
- **Demo Mode**: Built-in simulator for testing without needing camera access.

---

## Requirements

- Python 3.9+ (tested up to Python 3.12)
- Built-in webcam or external USB camera
- macOS, Linux, or Windows

---

## Installation

1. **Clone or open the repository**:
   ```bash
   cd drowsiness
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   *(On Windows: `.venv\Scripts\activate`)*

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

The required packages are:
- `opencv-python` (video capture and display)
- `mediapipe` (face mesh and landmark detection)
- `numpy` (vector math and landmark geometry)
- `pygame` (audio mixer for alarm sounds)

> **Note**: On first run, the app will automatically download the ~3.7 MB MediaPipe `face_landmarker.task` model into `assets/models/` if it isn't already present.

---

## How to Run

### 1. Default (Webcam)
Make sure your virtual environment is active, then run:
```bash
python main.py
```

### 2. Specify a Different Camera
If you have multiple webcams (e.g. an external USB camera or iPhone Continuity Camera):
```bash
python main.py --camera 1
```

### 3. Run on a Pre-recorded Video
To analyze a video file instead of live camera input:
```bash
python main.py --video sample.mp4
```

### 4. Demo Mode (No Camera Needed)
If you are testing on a machine without camera permissions or in an environment without a webcam:
```bash
python main.py --demo
```

### 5. Start Muted or Visual-Only
```bash
# Start with audio muted (you can unmute anytime with 'M')
python main.py --mute

# Completely disable audio subsystem
python main.py --no-audio
```

---

## In-App Controls

When the camera window is focused, you can use these keys:

| Key | Action |
|-----|--------|
| `Q` | Quit the application |
| `M` | Toggle mute / unmute alarm |
| `R` | Reset state machine and silence current alarm |

---

## CLI Options

You can fine-tune thresholds directly from the command line:

```bash
python main.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--camera` | `0` | Camera device index |
| `--video` | `None` | Path to video file |
| `--demo` | `False` | Run simulated demo mode |
| `--ear-threshold` | `0.21` | Eye Aspect Ratio below which eyes are considered closed |
| `--warning-time` | `0.9` | Seconds of closure before triggering WARNING state |
| `--drowsy-time` | `1.8` | Seconds of closure before triggering DROWSY alarm |
| `--mute` | `False` | Start with audio alarm muted |
| `--no-audio` | `False` | Disable audio entirely |

---

## How It Works

1. **Landmark Extraction**: MediaPipe identifies 468 3D facial landmarks from each video frame. Six specific landmark indices are extracted for each eye.
2. **Eye Aspect Ratio (EAR)**:
   $$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$
   Where $p_1, p_4$ are the eye corners and $p_2, p_3, p_5, p_6$ are top/bottom eyelid points.
3. **Temporal Smoothing**: EAR values are smoothed across frames using an exponential moving average (EMA) to prevent false triggers from rapid blinks or noise.
4. **State Machine**:
   - `EAR >= threshold`: Eyes open $\rightarrow$ state is **NORMAL**.
   - `EAR < threshold` for $> 0.9\text{s}$: state transitions to **WARNING**.
   - `EAR < threshold` for $> 1.8\text{s}$: state transitions to **DROWSY** and fires the alarm thread.
   - If face leaves the frame for $> 0.5\text{s}$: state reports **FACE_NOT_DETECTED**.

---

## Tuning for Your Eyes

If the detector is triggering too easily or not triggering when you close your eyes:

- **Wearing glasses or smaller eye opening**: Try slightly lowering the threshold:
  ```bash
  python main.py --ear-threshold 0.19
  ```
- **Larger eye shape / false negatives**: Try raising the threshold:
  ```bash
  python main.py --ear-threshold 0.23
  ```
- You can also permanently adjust defaults in `config.py`.

---

## Project Structure

```text
drowsiness/
├── assets/
│   ├── models/           # MediaPipe task models (face_landmarker.task)
│   └── sounds/           # Alert audio files (alarm.wav)
├── alarm.py              # Audio player, cooldown manager, and tone fallback
├── camera.py             # Video capture, AVFoundation integration, demo feed
├── config.py             # All thresholds, timings, colors, and landmark indices
├── drowsiness_detector.py # State machine tracking closure duration
├── eye_metrics.py        # EAR math, Euclidean distance, and smoothing
├── face_detector.py      # MediaPipe Face Landmarker wrapper & HUD drawing
├── main.py               # Main CLI app, event loop, and HUD renderer
└── requirements.txt      # Project dependencies
```

---

## Troubleshooting

- **macOS Camera Permission Error**:
  If the camera fails to open, go to:
  `System Settings` $\rightarrow$ `Privacy & Security` $\rightarrow$ `Camera`
  and make sure your terminal application (Terminal, iTerm, VS Code, Cursor, etc.) has permission enabled.
- **Audio Not Playing**:
  Ensure your system volume is up. If Pygame fails to initialize an audio device, run with `--no-audio` to inspect the visual HUD alerts without crashes.
- **Laggy / Low FPS**:
  The default resolution is set to `640x480` in `config.py` for high frame rates. If running on slower hardware, ensure no heavy background processes are using your GPU/camera.
