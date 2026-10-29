"""Main application entry point for Real-Time Driver Drowsiness Detection System.

Captures live webcam stream, processes facial landmarks via MediaPipe,
computes Eye Aspect Ratio (EAR), evaluates temporal vigilance state,
triggers audio alarms, and renders a heads-up display (HUD).
"""

from typing import Optional, Union
import argparse
import logging
import sys
import time
import cv2
import numpy as np

import config
from alarm import AlarmManager
from camera import CameraCapture
from drowsiness_detector import DrowsinessDetector, DetectionResult, DetectionState
from eye_metrics import Point
from face_detector import FaceLandmarkerDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("drowsiness_main")


class DrowsinessApp:
    """Orchestrates video capture, landmark detection, state machine, and HUD."""

    def __init__(
        self,
        camera_source: Optional[Union[int, str]] = None,
        camera_index: Optional[int] = None,
        ear_threshold: float = config.EAR_THRESHOLD,
        warning_time: float = config.WARNING_TIME_SECONDS,
        drowsy_time: float = config.DROWSY_TIME_SECONDS,
        muted: bool = False,
        no_audio: bool = False,
        demo_mode: bool = False,
    ) -> None:
        """Initialize all system modules outside the per-frame loop."""
        logger.info("Initializing Real-Time Drowsiness Detection System...")

        src = camera_source if camera_source is not None else (camera_index if camera_index is not None else config.CAMERA_INDEX)
        self.camera = CameraCapture(camera_source=src, demo_mode=demo_mode)
        self.face_detector = FaceLandmarkerDetector(auto_download=True)
        self.drowsiness_detector = DrowsinessDetector(
            ear_threshold=ear_threshold,
            warning_time_seconds=warning_time,
            drowsy_time_seconds=drowsy_time,
            alarm_cooldown_seconds=config.ALARM_COOLDOWN_SECONDS,
            ear_combination_method=config.EAR_COMBINATION_METHOD,
            ear_smoothing_factor=config.EAR_SMOOTHING_FACTOR,
        )
        self.alarm_manager = AlarmManager(
            initial_muted=muted,
            init_audio=not no_audio,
        )

        # Telemetry and FPS calculation
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._fps_start_time: float = time.perf_counter()
        self._window_name: str = "Driver Drowsiness Detection - MediaPipe & OpenCV"

    def _update_fps(self) -> None:
        """Calculate smoothed FPS over 10-frame intervals."""
        self._frame_count += 1
        if self._frame_count >= 10:
            now = time.perf_counter()
            elapsed = now - self._fps_start_time
            if elapsed > 0:
                self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_start_time = now

    def _render_hud(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        landmarks: Optional[list[Point]],
    ) -> None:
        """Render high-contrast, modern Heads-Up Display (HUD) overlay on frame."""
        h, w = frame.shape[:2]

        # 1. Draw eye landmarks & contours
        if landmarks is not None:
            self.face_detector.draw_eye_landmarks(
                frame,
                landmarks,
                state=result.state.value,
                draw_contours=True,
                draw_ear_points=True,
            )

        # 2. Pulsing warning border when DROWSY
        if result.state == DetectionState.DROWSY:
            pulse = int((time.perf_counter() * 5) % 2)
            border_thickness = 8 if pulse else 4
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), config.COLOR_DROWSY, border_thickness)

        # 3. Top Status Banner Card
        banner_h = 56
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (w - 12, 12 + banner_h), config.COLOR_CARD_BG, -1)
        cv2.rectangle(overlay, (12, 12), (w - 12, 12 + banner_h), config.COLOR_CARD_BORDER, 1)

        # Status Badge Color
        if result.state == DetectionState.NORMAL:
            badge_color = config.COLOR_NORMAL
            status_text = "STATE: NORMAL (VIGILANT)"
        elif result.state == DetectionState.WARNING:
            badge_color = config.COLOR_WARNING
            status_text = "STATE: WARNING (EYE CLOSURE)"
        elif result.state == DetectionState.DROWSY:
            badge_color = config.COLOR_DROWSY
            status_text = "STATE: DROWSY! SOUNDING ALARM!"
        else:
            badge_color = config.COLOR_NO_FACE
            status_text = "STATE: FACE NOT DETECTED"

        # Blend banner card background
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Draw status pill indicator
        cv2.circle(frame, (32, 12 + banner_h // 2), 10, badge_color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            status_text,
            (52, 12 + banner_h // 2 + 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            badge_color,
            1,
            cv2.LINE_AA,
        )

        # Right-aligned FPS counter
        fps_str = f"FPS: {self._fps:.1f}"
        cv2.putText(
            frame,
            fps_str,
            (w - 130, 12 + banner_h // 2 + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            config.COLOR_ACCENT,
            1,
            cv2.LINE_AA,
        )

        # 4. Left Telemetry Dashboard Card
        card_w, card_h = 260, 140
        card_x, card_y = 12, 80
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (card_x, card_y), (card_x + card_w, card_y + card_h), config.COLOR_CARD_BG, -1)
        cv2.rectangle(overlay2, (card_x, card_y), (card_x + card_w, card_y + card_h), config.COLOR_CARD_BORDER, 1)
        cv2.addWeighted(overlay2, 0.85, frame, 0.15, 0, frame)

        # Telemetry Text
        left_str = f"{result.left_ear:.3f}" if result.left_ear is not None else "--"
        right_str = f"{result.right_ear:.3f}" if result.right_ear is not None else "--"
        comb_str = f"{result.smoothed_ear:.3f}" if result.smoothed_ear is not None else "--"

        cv2.putText(
            frame,
            f"Left EAR: {left_str}  Right EAR: {right_str}",
            (card_x + 14, card_y + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            config.COLOR_TEXT_PRIMARY,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Avg EAR: {comb_str}  (Thr: {self.drowsiness_detector.ear_threshold:.2f})",
            (card_x + 14, card_y + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            config.COLOR_TEXT_PRIMARY,
            1,
            cv2.LINE_AA,
        )

        # Continuous closure timer and progress bar
        closure_str = f"Closure: {result.closure_duration:.2f}s / {self.drowsiness_detector.drowsy_time_seconds:.1f}s"
        cv2.putText(
            frame,
            closure_str,
            (card_x + 14, card_y + 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            config.COLOR_TEXT_PRIMARY,
            1,
            cv2.LINE_AA,
        )

        # Closure progress bar gauge
        bar_x = card_x + 14
        bar_y = card_y + 90
        bar_w = card_w - 28
        bar_h = 10
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_ratio = min(1.0, max(0.0, result.closure_duration / self.drowsiness_detector.drowsy_time_seconds))
        fill_w = int(bar_w * fill_ratio)
        bar_color = config.COLOR_NORMAL if fill_ratio < 0.5 else (config.COLOR_WARNING if fill_ratio < 1.0 else config.COLOR_DROWSY)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

        # Audio alarm status
        if self.alarm_manager.is_muted:
            alarm_status = "ALARM: MUTED [M to unmute]"
            alarm_col = (120, 120, 255)
        elif not self.alarm_manager.is_audio_available:
            alarm_status = "ALARM: AUDIO UNAVAILABLE"
            alarm_col = (100, 100, 100)
        elif result.alarm_in_cooldown:
            alarm_status = "ALARM: COOLDOWN ACTIVE"
            alarm_col = config.COLOR_WARNING
        else:
            alarm_status = "ALARM: ARMED & ACTIVE"
            alarm_col = config.COLOR_NORMAL

        cv2.putText(
            frame,
            alarm_status,
            (card_x + 14, card_y + 124),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            alarm_col,
            1,
            cv2.LINE_AA,
        )

        # 5. Bottom Keyboard Shortcut Help Bar
        footer_y = h - 14
        help_text = "[Q] Quit  |  [R] Reset State  |  [M] Mute/Unmute Alarm"
        cv2.putText(
            frame,
            help_text,
            (14, footer_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            config.COLOR_TEXT_SECONDARY,
            1,
            cv2.LINE_AA,
        )

    def run(self) -> None:
        """Start the webcam acquisition and detection loop."""
        if not self.camera.open():
            logger.error("Failed to start camera. Exiting.")
            return

        logger.info("Opening live detection window. Press 'Q' to exit, 'R' to reset, 'M' to mute.")

        try:
            while self.camera.is_opened:
                ret, frame = self.camera.read_frame()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                self._update_fps()

                # Step 1: Detect facial landmarks
                landmarks = self.face_detector.detect_landmarks(frame)

                # Step 2: Temporal state machine update
                result = self.drowsiness_detector.update(landmarks)

                # Step 3: Trigger alarm if drowsy condition met
                if result.should_trigger_alarm:
                    self.alarm_manager.trigger_alarm()

                # Step 4: Render UI HUD
                self._render_hud(frame, result, landmarks)

                # Step 5: Display window
                cv2.imshow(self._window_name, frame)

                # Step 6: Handle keyboard shortcuts
                key = cv2.waitKey(1) & 0xFF
                if key in (config.KEY_QUIT, config.KEY_QUIT_UPPER):
                    logger.info("Exit requested by user.")
                    break
                elif key in (config.KEY_RESET, config.KEY_RESET_UPPER):
                    logger.info("Detection state reset by user.")
                    self.drowsiness_detector.reset()
                    self.alarm_manager.stop()
                elif key in (config.KEY_MUTE, config.KEY_MUTE_UPPER):
                    is_muted = self.alarm_manager.toggle_mute()
                    logger.info("Alarm mute toggled. Currently muted: %s", is_muted)

        except KeyboardInterrupt:
            logger.info("Interrupted by keyboard signal.")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Release all hardware, ML model, and GUI resources cleanly."""
        logger.info("Cleaning up resources...")
        self.camera.release()
        self.face_detector.close()
        self.alarm_manager.close()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Driver Drowsiness Detection System (OpenCV & MediaPipe)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=config.CAMERA_INDEX,
        help="Webcam device index (0 is typical for built-in camera).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to an optional video file to analyze instead of a live webcam.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Run in interactive DEMO simulation mode without needing webcam permissions.",
    )
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=config.EAR_THRESHOLD,
        help="Eye Aspect Ratio (EAR) closure threshold.",
    )
    parser.add_argument(
        "--warning-time",
        type=float,
        default=config.WARNING_TIME_SECONDS,
        help="Duration in seconds before triggering WARNING state.",
    )
    parser.add_argument(
        "--drowsy-time",
        type=float,
        default=config.DROWSY_TIME_SECONDS,
        help="Duration in seconds before triggering DROWSY state and alarm.",
    )
    parser.add_argument(
        "--mute",
        action="store_true",
        default=False,
        help="Start with audio alarm muted.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        default=False,
        help="Disable audio subsystem completely (visual-only mode).",
    )
    return parser.parse_args()


def main() -> None:
    """Application entry point."""
    args = parse_args()
    camera_source = args.video if args.video else args.camera
    app = DrowsinessApp(
        camera_source=camera_source,
        ear_threshold=args.ear_threshold,
        warning_time=args.warning_time,
        drowsy_time=args.drowsy_time,
        muted=args.mute,
        no_audio=args.no_audio,
        demo_mode=args.demo,
    )
    app.run()


if __name__ == "__main__":
    main()
