"""Eye Aspect Ratio (EAR) metric calculations and landmark geometry.

This module provides functions to calculate Euclidean distances, individual
eye aspect ratios based on facial landmark coordinates, combination of dual-eye
metrics, and exponential moving average smoothing.
"""

from typing import Dict, List, Sequence, Tuple, Union, Optional
import math
import numpy as np


Point = Union[Tuple[float, ...], List[float], np.ndarray]


def calculate_euclidean_distance(pt1: Point, pt2: Point) -> float:
    """Calculate Euclidean distance between two 2D or 3D points.

    Args:
        pt1: Coordinates of first point (x, y) or (x, y, z).
        pt2: Coordinates of second point (x, y) or (x, y, z).

    Returns:
        Euclidean distance as float.

    Raises:
        ValueError: If points are empty or dimension mismatch.
    """
    if pt1 is None or pt2 is None:
        raise ValueError("Points cannot be None")

    if len(pt1) == 0 or len(pt2) == 0:
        raise ValueError("Points cannot be empty")

    if len(pt1) != len(pt2):
        raise ValueError(f"Dimension mismatch: {len(pt1)} vs {len(pt2)}")

    # Compute Euclidean distance using 2D coordinates (x, y) by default
    # If 3D coordinates are passed, we consider 2D projection on the image plane
    dx = float(pt1[0]) - float(pt2[0])
    dy = float(pt1[1]) - float(pt2[1])

    if math.isnan(dx) or math.isnan(dy) or math.isinf(dx) or math.isinf(dy):
        raise ValueError("Point coordinates contain NaN or Inf values")

    return math.hypot(dx, dy)


def calculate_ear_from_points(
    corner1: Point,
    corner2: Point,
    top1: Point,
    bottom1: Point,
    top2: Point,
    bottom2: Point,
    epsilon: float = 1e-6,
) -> float:
    """Compute the Eye Aspect Ratio (EAR) using 6 landmark coordinates.

    Formula:
        EAR = (|top1 - bottom1| + |top2 - bottom2|) / (2 * |corner1 - corner2|)

    Args:
        corner1: First horizontal eye corner (e.g., outer canthus).
        corner2: Second horizontal eye corner (e.g., inner canthus).
        top1: First upper eyelid point.
        bottom1: First lower eyelid point corresponding to top1.
        top2: Second upper eyelid point.
        bottom2: Second lower eyelid point corresponding to top2.
        epsilon: Small tolerance to prevent division by zero.

    Returns:
        EAR value as a float. Returns 0.0 if horizontal distance is degenerate (<= epsilon).
    """
    v1 = calculate_euclidean_distance(top1, bottom1)
    v2 = calculate_euclidean_distance(top2, bottom2)
    h = calculate_euclidean_distance(corner1, corner2)

    if h <= epsilon:
        # Degenerate horizontal length (e.g. identical points or occluded eye)
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return float(ear)


def calculate_eye_ear_from_landmarks(
    landmarks: Sequence[Point],
    indices_map: Dict[str, int],
) -> float:
    """Calculate EAR for an eye given a sequence of facial landmarks and an index map.

    Expected keys in indices_map:
        - 'corner_outer' or 'corner1'
        - 'corner_inner' or 'corner2'
        - 'top_outer' or 'top1'
        - 'bottom_outer' or 'bottom1'
        - 'top_inner' or 'top2'
        - 'bottom_inner' or 'bottom2'

    Args:
        landmarks: Sequence of landmark coordinates (length >= max index).
        indices_map: Dictionary mapping semantic eye landmarks to landmark indices.

    Returns:
        Calculated EAR as float.

    Raises:
        IndexError: If an index is out of bounds of the landmarks array.
        KeyError: If required landmark semantic keys are missing.
    """
    # Normalize keys
    c1_idx = indices_map.get("corner_outer", indices_map.get("corner1"))
    c2_idx = indices_map.get("corner_inner", indices_map.get("corner2"))
    t1_idx = indices_map.get("top_outer", indices_map.get("top1"))
    b1_idx = indices_map.get("bottom_outer", indices_map.get("bottom1"))
    t2_idx = indices_map.get("top_inner", indices_map.get("top2"))
    b2_idx = indices_map.get("bottom_inner", indices_map.get("bottom2"))

    required = [c1_idx, c2_idx, t1_idx, b1_idx, t2_idx, b2_idx]
    if any(idx is None for idx in required):
        raise KeyError(
            f"Missing required landmark index in mapping. Got: {list(indices_map.keys())}"
        )

    num_landmarks = len(landmarks)
    for idx in required:
        if idx < 0 or idx >= num_landmarks:
            raise IndexError(
                f"Landmark index {idx} out of bounds for landmarks list of length {num_landmarks}"
            )

    return calculate_ear_from_points(
        corner1=landmarks[c1_idx],
        corner2=landmarks[c2_idx],
        top1=landmarks[t1_idx],
        bottom1=landmarks[b1_idx],
        top2=landmarks[t2_idx],
        bottom2=landmarks[b2_idx],
    )


def combine_ear(
    left_ear: float,
    right_ear: float,
    method: str = "average",
) -> float:
    """Combine left and right eye EAR measurements based on configured strategy.

    Args:
        left_ear: Left eye EAR.
        right_ear: Right eye EAR.
        method: Combination strategy:
            - 'average': Mean of left and right EAR.
            - 'min': Minimum EAR between both eyes (sensitive to single-eye closure).
            - 'left_only': Left eye EAR only.
            - 'right_only': Right eye EAR only.

    Returns:
        Combined EAR float value.

    Raises:
        ValueError: If method is unrecognized.
    """
    if method == "average":
        return float((left_ear + right_ear) / 2.0)
    elif method == "min":
        return float(min(left_ear, right_ear))
    elif method == "left_only":
        return float(left_ear)
    elif method == "right_only":
        return float(right_ear)
    else:
        raise ValueError(
            f"Unknown EAR combination method '{method}'. "
            "Supported methods are: 'average', 'min', 'left_only', 'right_only'."
        )


def smooth_ear(
    current_ear: float,
    previous_ear: Optional[float],
    alpha: float = 0.65,
) -> float:
    """Apply Exponential Moving Average (EMA) smoothing to EAR.

    Args:
        current_ear: Current frame raw EAR value.
        previous_ear: Smoothed EAR value from previous frame (or None if first frame).
        alpha: Smoothing factor between 0.0 and 1.0. Higher means more responsive;
               lower means smoother.

    Returns:
        Smoothed EAR value as float.
    """
    if previous_ear is None:
        return float(current_ear)

    alpha = max(0.0, min(1.0, float(alpha)))
    return float(alpha * current_ear + (1.0 - alpha) * previous_ear)
