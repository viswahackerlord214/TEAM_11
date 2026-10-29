"""Configuration settings for Real-Time Driver Drowsiness Detection System.

All thresholds, timings, camera parameters, and landmark indices are
consolidated here for easy tuning and calibration.
"""

from pathlib import Path

# ==========================================
# File Paths and Assets
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = ASSETS_DIR / "models"
SOUNDS_DIR = ASSETS_DIR / "sounds"

# MediaPipe Face Landmarker bundle model path and official Google download URL
FACE_LANDMARKER_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# Audio alarm path
ALARM_SOUND_PATH = SOUNDS_DIR / "alarm.wav"

# ==========================================
# Camera Settings
# ==========================================
CAMERA_INDEX = 0             # Default webcam index (0 is typical for built-in Mac FaceTime HD camera)
FRAME_WIDTH = 640            # Capture width (balanced for high-speed M1 processing)
FRAME_HEIGHT = 480           # Capture height
TARGET_FPS = 30              # Target frame rate

# ==========================================
# Eye Aspect Ratio (EAR) Parameters
# ==========================================
# Eye closure threshold: EAR values below this indicate eyes are closed.
# Initial baseline is 0.21. Individual users and lighting conditions may require 0.19 - 0.24.
EAR_THRESHOLD = 0.21

# Method to combine left and right eye EAR:
# - "average": Mean of left and right EAR (standard, robust to asymmetric blinks)
# - "min": Minimum of left and right EAR (sensitive to one-eye closure/microsleeps)
# - "left_only": Only subject's left eye
# - "right_only": Only subject's right eye
EAR_COMBINATION_METHOD = "average"

# Exponential Moving Average factor (alpha) for smoothing EAR across frames (0.0 < alpha <= 1.0)
# alpha = 1.0 means no smoothing; 0.5 balances responsiveness with noise reduction.
EAR_SMOOTHING_FACTOR = 0.65

# ==========================================
# Temporal Drowsiness Thresholds (in Seconds)
# ==========================================
# Normal blink: typically 100ms - 400ms (0.1s - 0.4s).
# If eye closure duration exceeds WARNING_TIME_SECONDS, state moves to WARNING.
WARNING_TIME_SECONDS = 0.9

# If eye closure duration exceeds DROWSY_TIME_SECONDS, state moves to DROWSY and alarm triggers.
DROWSY_TIME_SECONDS = 1.8

# Alarm cooldown: minimum duration between alarm triggers to avoid overlapping/jarring sound repeats
ALARM_COOLDOWN_SECONDS = 2.0

# Face lost timeout: duration without face detection before marking state as FACE_NOT_DETECTED
FACE_LOST_TIMEOUT_SECONDS = 0.5

# ==========================================
# MediaPipe Facial Landmark Indices (468 Mesh)
# ==========================================
# Landmark points for calculating 6-point Eye Aspect Ratio:
# EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
#
# Subject's Right Eye (Screen Left in mirrored camera view):
# p1 = 33 (outer corner / lateral canthus)
# p4 = 133 (inner corner / medial canthus)
# p2 = 160 (top outer)
# p6 = 144 (bottom outer)
# p3 = 158 (top inner)
# p5 = 153 (bottom inner)
RIGHT_EYE_EAR_INDICES = {
    "corner_outer": 33,
    "corner_inner": 133,
    "top_outer": 160,
    "bottom_outer": 144,
    "top_inner": 158,
    "bottom_inner": 153,
}

# Subject's Left Eye (Screen Right in mirrored camera view):
# p1 = 362 (inner corner / medial canthus)
# p4 = 263 (outer corner / lateral canthus)
# p2 = 385 (top inner)
# p6 = 380 (bottom inner)
# p3 = 387 (top outer)
# p5 = 373 (bottom outer)
LEFT_EYE_EAR_INDICES = {
    "corner_inner": 362,
    "corner_outer": 263,
    "top_inner": 385,
    "bottom_inner": 380,
    "top_outer": 387,
    "bottom_outer": 373,
}

# Full eye contour landmark indices for rich visual overlay
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# ==========================================
# UI & HUD Styling (BGR Color Format for OpenCV)
# ==========================================
COLOR_NORMAL = (80, 200, 120)     # Emerald Green
COLOR_WARNING = (0, 190, 255)     # Amber / Gold
COLOR_DROWSY = (40, 40, 240)      # Bright Crimson Red
COLOR_NO_FACE = (160, 160, 160)   # Neutral Slate Gray
COLOR_CARD_BG = (20, 22, 28)      # Dark Slate for HUD panel
COLOR_CARD_BORDER = (50, 55, 65)  # Border for HUD panel
COLOR_TEXT_PRIMARY = (255, 255, 255)
COLOR_TEXT_SECONDARY = (190, 195, 205)
COLOR_ACCENT = (255, 180, 50)     # Cyan/Sky blue accent

# Keyboard controls
KEY_QUIT = ord('q')
KEY_QUIT_UPPER = ord('Q')
KEY_RESET = ord('r')
KEY_RESET_UPPER = ord('R')
KEY_MUTE = ord('m')
KEY_MUTE_UPPER = ord('M')
