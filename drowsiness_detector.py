"""Temporal state machine for drowsiness detection.

Evaluates Eye Aspect Ratio (EAR) over elapsed time, tracks eye closure durations,
distinguishes normal blinks from microsleeps and sustained drowsiness, handles
missing faces, and enforces audio alert cooldown periods.
"""

from dataclasses import dataclass
from enum import Enum
import time
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

from config import (
    EAR_THRESHOLD,
    WARNING_TIME_SECONDS,
    DROWSY_TIME_SECONDS,
    ALARM_COOLDOWN_SECONDS,
    FACE_LOST_TIMEOUT_SECONDS,
    EAR_COMBINATION_METHOD,
    EAR_SMOOTHING_FACTOR,
    RIGHT_EYE_EAR_INDICES,
    LEFT_EYE_EAR_INDICES,
)
from eye_metrics import (
    Point,
    calculate_eye_ear_from_landmarks,
    combine_ear,
    smooth_ear,
)


class DetectionState(str, Enum):
    """Enumeration of possible driver vigilance states."""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    DROWSY = "DROWSY"
    FACE_NOT_DETECTED = "FACE_NOT_DETECTED"


@dataclass
class DetectionResult:
    """Detection results and telemetry for a single video frame."""
    state: DetectionState
    left_ear: Optional[float]
    right_ear: Optional[float]
    combined_ear: Optional[float]
    smoothed_ear: Optional[float]
    closure_duration: float
    should_trigger_alarm: bool
    is_face_detected: bool
    consecutive_closed_frames: int
    alarm_in_cooldown: bool


class DrowsinessDetector:
    """Configurable temporal state machine for drowsiness detection."""

    def __init__(
        self,
        ear_threshold: float = EAR_THRESHOLD,
        warning_time_seconds: float = WARNING_TIME_SECONDS,
        drowsy_time_seconds: float = DROWSY_TIME_SECONDS,
        alarm_cooldown_seconds: float = ALARM_COOLDOWN_SECONDS,
        face_lost_timeout_seconds: float = FACE_LOST_TIMEOUT_SECONDS,
        ear_combination_method: str = EAR_COMBINATION_METHOD,
        ear_smoothing_factor: float = EAR_SMOOTHING_FACTOR,
        right_eye_indices: Optional[Dict[str, int]] = None,
        left_eye_indices: Optional[Dict[str, int]] = None,
        time_provider: Optional[Callable[[], float]] = None,
    ) -> None:
        """Initialize the drowsiness detector.

        Args:
            ear_threshold: EAR threshold below which eyes are considered closed.
            warning_time_seconds: Continuous closure time before WARNING state.
            drowsy_time_seconds: Continuous closure time before DROWSY state.
            alarm_cooldown_seconds: Minimum delay between alarm triggers.
            face_lost_timeout_seconds: Duration without a face before FACE_NOT_DETECTED.
            ear_combination_method: Strategy for combining both eyes ('average', 'min', etc.).
            ear_smoothing_factor: EMA alpha parameter for smoothing EAR.
            right_eye_indices: Semantic index map for right eye landmarks.
            left_eye_indices: Semantic index map for left eye landmarks.
            time_provider: Function providing current timestamp (default time.perf_counter).
        """
        self.ear_threshold = float(ear_threshold)
        self.warning_time_seconds = float(warning_time_seconds)
        self.drowsy_time_seconds = float(drowsy_time_seconds)
        self.alarm_cooldown_seconds = float(alarm_cooldown_seconds)
        self.face_lost_timeout_seconds = float(face_lost_timeout_seconds)
        self.ear_combination_method = ear_combination_method
        self.ear_smoothing_factor = float(ear_smoothing_factor)

        self.right_eye_indices = right_eye_indices or RIGHT_EYE_EAR_INDICES
        self.left_eye_indices = left_eye_indices or LEFT_EYE_EAR_INDICES
        self._get_time = time_provider or time.perf_counter

        # Internal state tracking
        self._state: DetectionState = DetectionState.NORMAL
        self._closure_start_time: Optional[float] = None
        self._last_face_seen_time: Optional[float] = None
        self._last_alarm_time: Optional[float] = None
        self._smoothed_ear: Optional[float] = None
        self._consecutive_closed_frames: int = 0
        self._consecutive_open_frames: int = 0

    @property
    def current_state(self) -> DetectionState:
        """Return the current detection state."""
        return self._state

    def reset(self) -> None:
        """Reset internal temporal counters, state, and smoothing history."""
        self._state = DetectionState.NORMAL
        self._closure_start_time = None
        self._last_face_seen_time = None
        self._last_alarm_time = None
        self._smoothed_ear = None
        self._consecutive_closed_frames = 0
        self._consecutive_open_frames = 0

    def update(
        self,
        landmarks: Optional[Sequence[Point]],
        current_timestamp: Optional[float] = None,
    ) -> DetectionResult:
        """Process a single frame's facial landmarks and advance state machine.

        Args:
            landmarks: Sequence of landmark coordinates, or None if no face detected.
            current_timestamp: Optional explicit timestamp (useful for offline/tests).

        Returns:
            DetectionResult instance with state, metrics, and alarm decision.
        """
        now = self._get_time() if current_timestamp is None else float(current_timestamp)

        # -------------------------------------------------------------
        # 1. Handle Missing Face
        # -------------------------------------------------------------
        if landmarks is None or len(landmarks) == 0:
            if self._last_face_seen_time is None:
                self._last_face_seen_time = now

            time_since_face = now - self._last_face_seen_time
            if time_since_face >= self.face_lost_timeout_seconds:
                self._state = DetectionState.FACE_NOT_DETECTED
                # Reset closure tracking when face is lost
                self._closure_start_time = None
                self._consecutive_closed_frames = 0

            return DetectionResult(
                state=self._state,
                left_ear=None,
                right_ear=None,
                combined_ear=None,
                smoothed_ear=self._smoothed_ear,
                closure_duration=0.0,
                should_trigger_alarm=False,
                is_face_detected=False,
                consecutive_closed_frames=0,
                alarm_in_cooldown=self._is_in_cooldown(now),
            )

        # Face detected
        self._last_face_seen_time = now

        # -------------------------------------------------------------
        # 2. Compute EAR for both eyes
        # -------------------------------------------------------------
        try:
            r_ear = calculate_eye_ear_from_landmarks(landmarks, self.right_eye_indices)
            l_ear = calculate_eye_ear_from_landmarks(landmarks, self.left_eye_indices)
            comb_ear = combine_ear(l_ear, r_ear, method=self.ear_combination_method)
        except (IndexError, KeyError, ValueError):
            # Degenerate or malformed landmark array
            return DetectionResult(
                state=self._state,
                left_ear=None,
                right_ear=None,
                combined_ear=None,
                smoothed_ear=self._smoothed_ear,
                closure_duration=0.0,
                should_trigger_alarm=False,
                is_face_detected=True,
                consecutive_closed_frames=0,
                alarm_in_cooldown=self._is_in_cooldown(now),
            )

        # Apply Exponential Moving Average (EMA) smoothing
        smoothed = smooth_ear(comb_ear, self._smoothed_ear, alpha=self.ear_smoothing_factor)
        self._smoothed_ear = smoothed

        # -------------------------------------------------------------
        # 3. Evaluate Eye Closure and Temporal State
        # -------------------------------------------------------------
        is_closed = smoothed < self.ear_threshold
        should_alarm = False
        closure_duration = 0.0

        if is_closed:
            self._consecutive_closed_frames += 1
            self._consecutive_open_frames = 0

            if self._closure_start_time is None:
                self._closure_start_time = now

            closure_duration = max(0.0, now - self._closure_start_time)

            if closure_duration >= self.drowsy_time_seconds:
                self._state = DetectionState.DROWSY
                # Check alarm cooldown
                if not self._is_in_cooldown(now):
                    should_alarm = True
                    self._last_alarm_time = now
            elif closure_duration >= self.warning_time_seconds:
                self._state = DetectionState.WARNING
            else:
                # Brief closure (normal blink window)
                self._state = DetectionState.NORMAL
        else:
            # Eyes are open: immediately reset closure timer
            self._consecutive_open_frames += 1
            self._consecutive_closed_frames = 0
            self._closure_start_time = None
            closure_duration = 0.0
            self._state = DetectionState.NORMAL

        return DetectionResult(
            state=self._state,
            left_ear=l_ear,
            right_ear=r_ear,
            combined_ear=comb_ear,
            smoothed_ear=smoothed,
            closure_duration=closure_duration,
            should_trigger_alarm=should_alarm,
            is_face_detected=True,
            consecutive_closed_frames=self._consecutive_closed_frames,
            alarm_in_cooldown=self._is_in_cooldown(now),
        )

    def _is_in_cooldown(self, current_time: float) -> bool:
        """Check whether alarm cooldown is currently active."""
        if self._last_alarm_time is None:
            return False
        return (current_time - self._last_alarm_time) < self.alarm_cooldown_seconds
