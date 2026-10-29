"""MediaPipe Face Landmarker wrapper for facial landmark extraction and visualization.

Encapsulates model initialization, image conversion, landmark extraction,
coordinate projection to pixel space, and eye contour drawing.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import logging
import urllib.request
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import (
    FACE_LANDMARKER_MODEL_PATH,
    FACE_LANDMARKER_MODEL_URL,
    RIGHT_EYE_EAR_INDICES,
    LEFT_EYE_EAR_INDICES,
    RIGHT_EYE_CONTOUR,
    LEFT_EYE_CONTOUR,
    COLOR_NORMAL,
    COLOR_WARNING,
    COLOR_DROWSY,
    COLOR_NO_FACE,
)

logger = logging.getLogger(__name__)


def ensure_model_file(model_path: Path, url: str = FACE_LANDMARKER_MODEL_URL) -> Path:
    """Ensure the MediaPipe face landmarker task model file exists.

    If not found, attempts to download it from the official Google repository.

    Args:
        model_path: Target path for the .task model file.
        url: Official download URL.

    Returns:
        Path to the verified model file.

    Raises:
        FileNotFoundError: If the model is missing and download fails.
    """
    model_path = Path(model_path)
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Model file not found at %s. Downloading from %s...", model_path, url)
    try:
        urllib.request.urlretrieve(url, str(model_path))
        logger.info("Model downloaded successfully (%d bytes)", model_path.stat().st_size)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not load or download MediaPipe Face Landmarker model from {url}. "
            f"Please download it manually to '{model_path.resolve()}'. "
            f"Original error: {exc}"
        ) from exc

    return model_path


class FaceLandmarkerDetector:
    """Wrapper around MediaPipe FaceLandmarker Task for real-time landmark tracking."""

    def __init__(
        self,
        model_path: Path = FACE_LANDMARKER_MODEL_PATH,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        auto_download: bool = True,
    ) -> None:
        """Initialize MediaPipe Face Landmarker.

        Args:
            model_path: Path to the .task model file.
            min_detection_confidence: Detection confidence threshold.
            min_presence_confidence: Presence confidence threshold.
            min_tracking_confidence: Tracking confidence threshold.
            auto_download: Whether to automatically download the model if missing.
        """
        self.model_path = Path(model_path)
        if auto_download:
            self.model_path = ensure_model_file(self.model_path)
        elif not self.model_path.exists():
            raise FileNotFoundError(f"MediaPipe model file not found at {self.model_path}")

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self._detector: Optional[vision.FaceLandmarker] = (
            vision.FaceLandmarker.create_from_options(options)
        )
        logger.info("MediaPipe Face Landmarker initialized successfully.")

    def detect_landmarks(
        self, frame: np.ndarray
    ) -> Optional[List[Tuple[float, float, float]]]:
        """Detect 468+ facial landmarks from an OpenCV BGR frame.

        Args:
            frame: BGR numpy image frame from webcam.

        Returns:
            List of (pixel_x, pixel_y, pixel_z) tuples for the primary detected face,
            or None if no face is detected or frame is empty.
        """
        if self._detector is None:
            raise RuntimeError("FaceLandmarker is closed or not initialized.")

        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return None

        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            detection_result = self._detector.detect(mp_image)
        except Exception as exc:
            logger.warning("Error running FaceLandmarker detection: %s", exc)
            return None

        if not detection_result.face_landmarks:
            return None

        # Extract primary face landmarks and project normalized coords [0, 1] to pixel coords
        primary_face = detection_result.face_landmarks[0]
        pixel_landmarks: List[Tuple[float, float, float]] = []

        for lm in primary_face:
            px = float(lm.x * w)
            py = float(lm.y * h)
            pz = float(lm.z * w)
            pixel_landmarks.append((px, py, pz))

        return pixel_landmarks

    def draw_eye_landmarks(
        self,
        frame: np.ndarray,
        landmarks: Sequence[Tuple[float, float, float]],
        state: str = "NORMAL",
        draw_contours: bool = True,
        draw_ear_points: bool = True,
    ) -> None:
        """Draw eye contours and EAR measurement points directly onto the frame.

        Args:
            frame: Target BGR image frame (modified in place).
            landmarks: List of pixel landmark coordinates.
            state: Current detector state for color coding ('NORMAL', 'WARNING', 'DROWSY').
            draw_contours: Whether to draw full eye contour polylines.
            draw_ear_points: Whether to highlight the 6 EAR measurement points per eye.
        """
        if landmarks is None or len(landmarks) == 0:
            return

        # Choose color according to current state
        if state == "DROWSY":
            contour_color = COLOR_DROWSY
            pt_color = (0, 0, 255)
        elif state == "WARNING":
            contour_color = COLOR_WARNING
            pt_color = (0, 215, 255)
        elif state == "FACE_NOT_DETECTED":
            contour_color = COLOR_NO_FACE
            pt_color = (180, 180, 180)
        else:
            contour_color = COLOR_NORMAL
            pt_color = (100, 255, 150)

        n = len(landmarks)

        # Draw smooth eye contours
        if draw_contours:
            for contour_indices in (RIGHT_EYE_CONTOUR, LEFT_EYE_CONTOUR):
                pts = []
                for idx in contour_indices:
                    if idx < n:
                        x, y = int(round(landmarks[idx][0])), int(round(landmarks[idx][1]))
                        pts.append([x, y])
                if len(pts) > 2:
                    pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        frame,
                        [pts_arr],
                        isClosed=True,
                        color=contour_color,
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

        # Highlight EAR measurement points and vertical measurement lines
        if draw_ear_points:
            for eye_indices in (RIGHT_EYE_EAR_INDICES, LEFT_EYE_EAR_INDICES):
                try:
                    c1 = (int(round(landmarks[eye_indices["corner_outer" if "corner_outer" in eye_indices else "corner1"]][0])),
                          int(round(landmarks[eye_indices["corner_outer" if "corner_outer" in eye_indices else "corner1"]][1])))
                    c2 = (int(round(landmarks[eye_indices["corner_inner" if "corner_inner" in eye_indices else "corner2"]][0])),
                          int(round(landmarks[eye_indices["corner_inner" if "corner_inner" in eye_indices else "corner2"]][1])))
                    t1 = (int(round(landmarks[eye_indices["top_outer" if "top_outer" in eye_indices else "top1"]][0])),
                          int(round(landmarks[eye_indices["top_outer" if "top_outer" in eye_indices else "top1"]][1])))
                    b1 = (int(round(landmarks[eye_indices["bottom_outer" if "bottom_outer" in eye_indices else "bottom1"]][0])),
                          int(round(landmarks[eye_indices["bottom_outer" if "bottom_outer" in eye_indices else "bottom1"]][1])))
                    t2 = (int(round(landmarks[eye_indices["top_inner" if "top_inner" in eye_indices else "top2"]][0])),
                          int(round(landmarks[eye_indices["top_inner" if "top_inner" in eye_indices else "top2"]][1])))
                    b2 = (int(round(landmarks[eye_indices["bottom_inner" if "bottom_inner" in eye_indices else "bottom2"]][0])),
                          int(round(landmarks[eye_indices["bottom_inner" if "bottom_inner" in eye_indices else "bottom2"]][1])))

                    # Subtle vertical measurement line segments
                    cv2.line(frame, t1, b1, (200, 200, 200), 1, cv2.LINE_AA)
                    cv2.line(frame, t2, b2, (200, 200, 200), 1, cv2.LINE_AA)

                    # Highlight points
                    for pt in (c1, c2, t1, b1, t2, b2):
                        cv2.circle(frame, pt, 2, pt_color, -1, cv2.LINE_AA)
                except (KeyError, IndexError):
                    continue

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None

    def __enter__(self) -> "FaceLandmarkerDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
