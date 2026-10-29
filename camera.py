"""Webcam capture and hardware resource management using OpenCV.

Handles camera initialization with macOS AVFoundation backend, configuration of
resolution, frame acquisition, horizontal mirror flipping, and safe cleanup.
"""

from typing import Optional, Tuple, Union
import logging
import platform
import time
import cv2
import numpy as np

from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS

logger = logging.getLogger(__name__)


class CameraCapture:
    """Manages OpenCV VideoCapture lifecycle and frame acquisition."""

    def __init__(
        self,
        camera_source: Optional[Union[int, str]] = None,
        camera_index: Optional[int] = None,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        target_fps: int = TARGET_FPS,
        mirror_horizontal: bool = True,
        demo_mode: bool = False,
    ) -> None:
        """Initialize CameraCapture settings.

        Args:
            camera_source: Device index for webcam (int) or video file path (str).
            camera_index: Backward-compatible alias for camera_source index.
            width: Desired capture frame width.
            height: Desired capture frame height.
            target_fps: Desired frame rate.
            mirror_horizontal: Whether to mirror frame horizontally (natural webcam view).
            demo_mode: Whether to run synthetic demo camera simulator.
        """
        if camera_source is not None:
            self.camera_source = camera_source
        elif camera_index is not None:
            self.camera_source = camera_index
        else:
            self.camera_source = CAMERA_INDEX

        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.mirror_horizontal = mirror_horizontal and not isinstance(self.camera_source, str)
        self.demo_mode = demo_mode
        self._cap: Optional[cv2.VideoCapture] = None
        self._demo_start_time: float = 0.0

    def open(self) -> bool:
        """Open camera device or video source.

        On macOS Apple Silicon, utilizes AVFoundation backend for live cameras.

        Returns:
            True if camera or demo opened successfully, False otherwise.
        """
        if self.demo_mode:
            logger.info("Starting CameraCapture in DEMO simulation mode...")
            self._demo_start_time = time.perf_counter()
            return True

        if self._cap is not None and self._cap.isOpened():
            return True

        # If camera_source is a video file path
        if isinstance(self.camera_source, str):
            logger.info("Opening video source: %s", self.camera_source)
            cap = cv2.VideoCapture(self.camera_source)
            if not cap.isOpened():
                logger.error("Unable to open video file: %s", self.camera_source)
                return False
            self._cap = cap
            return True

        # Live webcam capture: Use AVFoundation on macOS for fast camera start
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY

        logger.info("Opening camera device %s with backend %s...", self.camera_source, backend)
        cap = cv2.VideoCapture(self.camera_source, backend)

        if not cap.isOpened():
            logger.warning("Camera %s failed with backend %s. Retrying default...", self.camera_source, backend)
            cap = cv2.VideoCapture(self.camera_source)

        if not cap.isOpened():
            logger.error(
                "Unable to open camera device %s. "
                "macOS Permission Required: "
                "Open System Settings -> Privacy & Security -> Camera, and enable access for your Terminal/IDE. "
                "Tip: You can also test immediately with simulated demo mode: python3 main.py --demo",
                self.camera_source,
            )
            return False

        # Set capture properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self._cap = cap
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info("Camera opened successfully: %dx%d @ %.1f FPS", actual_w, actual_h, actual_fps)
        return True

    @property
    def is_opened(self) -> bool:
        """Check if camera or demo is currently active."""
        if self.demo_mode:
            return True
        return self._cap is not None and self._cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture and return a single frame."""
        if self.demo_mode:
            return self._generate_demo_frame()

        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None or frame.size == 0:
            if isinstance(self.camera_source, str):
                # Loop video file seamlessly
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    return True, frame
            logger.warning("Failed to grab valid frame from camera device.")
            return False, None

        if self.mirror_horizontal:
            frame = cv2.flip(frame, 1)

        return True, frame

    def _generate_demo_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Synthesize a dynamic animated face frame demonstrating open, blink, and drowsy phases."""
        frame = np.full((self.height, self.width, 3), 35, dtype=np.uint8)

        # Elapsed time in demo
        t = time.perf_counter() - self._demo_start_time
        cycle = t % 10.0  # 10-second repeating scenario

        # Determine eye state:
        # 0.0s - 3.0s: Normal open eyes
        # 3.0s - 3.3s: Fast normal blink
        # 3.3s - 5.0s: Normal open eyes
        # 5.0s - 9.0s: Sustained closure (drowsy trigger!)
        # 9.0s - 10.0s: Reopen eyes (recovery)
        if 3.0 <= cycle <= 3.3:
            eye_h = 2
            phase_text = "DEMO: Normal Blink (<0.3s)"
        elif 5.0 <= cycle <= 9.0:
            eye_h = 2
            phase_text = f"DEMO: Sustained Closure ({cycle - 5.0:.1f}s / Warning & Drowsy Alarm!)"
        else:
            eye_h = 16
            phase_text = "DEMO: Eyes Open (Normal Driving)"

        # Draw a synthetic driver avatar
        cx, cy = self.width // 2, self.height // 2
        # Head contour
        cv2.ellipse(frame, (cx, cy), (120, 160), 0, 0, 360, (200, 180, 160), -1)
        cv2.ellipse(frame, (cx, cy), (120, 160), 0, 0, 360, (140, 120, 100), 2)

        # Eyes: Left and Right
        for eye_x in (cx - 45, cx + 45):
            cv2.ellipse(frame, (eye_x, cy - 20), (22, max(2, eye_h)), 0, 0, 360, (255, 255, 255), -1)
            cv2.ellipse(frame, (eye_x, cy - 20), (22, max(2, eye_h)), 0, 0, 360, (60, 40, 30), 1)
            if eye_h > 4:
                cv2.circle(frame, (eye_x, cy - 20), 7, (70, 50, 40), -1)
                cv2.circle(frame, (eye_x, cy - 20), 3, (15, 10, 5), -1)

        # Eyebrows
        cv2.line(frame, (cx - 65, cy - 42), (cx - 25, cy - 40), (80, 50, 30), 3)
        cv2.line(frame, (cx + 25, cy - 40), (cx + 65, cy - 42), (80, 50, 30), 3)

        # Nose & Mouth
        cv2.line(frame, (cx, cy - 10), (cx, cy + 25), (140, 100, 80), 2)
        cv2.ellipse(frame, (cx, cy + 60), (30, 8), 0, 0, 180, (120, 60, 60), 3)

        # Demo Banner
        cv2.putText(frame, phase_text, (20, self.height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        # Pace demo frame rate (~30 FPS)
        time.sleep(0.03)
        return True, frame

    def release(self) -> None:
        """Release camera hardware resources."""
        if self._cap is not None:
            logger.info("Releasing camera device %s...", self.camera_source)
            try:
                self._cap.release()
            except Exception as exc:
                logger.warning("Error releasing camera: %s", exc)
            self._cap = None

    def __enter__(self) -> "CameraCapture":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
