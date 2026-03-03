"""
normalizer.py
-------------
Converts raw MediaPipe keypoints (33 x 3) into rotation/scale invariant
joint angle features. This is the core feature engineering step.

Why angles instead of raw coordinates?
- Raw x,y,z changes with camera angle and dancer position in frame
- Joint angles stay the same regardless of where in the frame the dancer stands
- e.g. an arabesque at 90 degrees hip extension looks the same from any horizontal angle
"""

import numpy as np
from typing import Optional


# Key joint triplets: (point_a, vertex, point_b)
# The angle is computed AT the vertex between the two arms a→vertex and b→vertex
JOINT_ANGLE_TRIPLETS = {
    "left_knee":       (23, 25, 27),   # hip → knee → ankle
    "right_knee":      (24, 26, 28),
    "left_hip":        (11, 23, 25),   # shoulder → hip → knee
    "right_hip":       (12, 24, 26),
    "left_elbow":      (11, 13, 15),   # shoulder → elbow → wrist
    "right_elbow":     (12, 14, 16),
    "left_shoulder":   (23, 11, 13),   # hip → shoulder → elbow
    "right_shoulder":  (24, 12, 14),
    "left_ankle":      (25, 27, 31),   # knee → ankle → foot index
    "right_ankle":     (26, 28, 32),
    "left_hip_abduct": (25, 23, 24),   # knee → left_hip → right_hip (turnout!)
    "right_hip_abduct":(26, 24, 23),
}

FEATURE_NAMES = list(JOINT_ANGLE_TRIPLETS.keys())

# Symmetric arm feature names — sorted pairs so left/right distinction is removed.
# elbow_high = whichever elbow is more extended; elbow_low = the more bent one.
# Used for positions where which arm is up doesn't matter (e.g. third, fourth position).
SYMMETRIC_ARM_FEATURE_NAMES = ["elbow_high", "elbow_low", "shoulder_high", "shoulder_low"]

# Symmetric leg feature names — sorted pairs so standing vs working leg distinction is removed.
# hip_high = whichever hip is more extended (working leg), hip_low = standing leg hip angle.
# Used for one-legged poses (arabesque, penche, tendu, attitude) so left/right don't matter.
SYMMETRIC_LEG_FEATURE_NAMES = [
    "hip_high", "hip_low",
    "knee_high", "knee_low",
    "ankle_high", "ankle_low",
    "hip_abduct_high", "hip_abduct_low",
]

ALL_FEATURE_NAMES = FEATURE_NAMES + SYMMETRIC_ARM_FEATURE_NAMES + SYMMETRIC_LEG_FEATURE_NAMES


def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute angle in degrees at vertex b, between vectors b→a and b→c.
    Returns value in [0, 180].
    """
    v1 = a - b
    v2 = c - b
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0

    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def center_and_scale(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalize keypoints to be translation and scale invariant.
    1. Translate so hip midpoint is at origin
    2. Scale so torso length = 1.0

    Args:
        keypoints: shape (33, 3)
    Returns:
        Normalized keypoints, shape (33, 3)
    """
    kp = keypoints.copy()

    # 1. Center on hip midpoint
    left_hip = kp[23]
    right_hip = kp[24]
    hip_center = (left_hip + right_hip) / 2.0
    kp -= hip_center

    # 2. Scale by torso length (hip center to shoulder center)
    left_shoulder = kp[11]
    right_shoulder = kp[12]
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    torso_length = np.linalg.norm(shoulder_center)  # distance from origin (hip center)

    if torso_length > 1e-6:
        kp /= torso_length

    return kp


def extract_joint_angles(keypoints: np.ndarray) -> np.ndarray:
    """
    Convert (33, 3) keypoints into a 1D feature vector of joint angles.

    Args:
        keypoints: shape (33, 3) — raw or normalized MediaPipe landmarks

    Returns:
        angles: shape (len(JOINT_ANGLE_TRIPLETS),) — one angle per joint
    """
    kp = center_and_scale(keypoints)

    angles = []
    for joint_name, (a_idx, b_idx, c_idx) in JOINT_ANGLE_TRIPLETS.items():
        angle = _angle_between(kp[a_idx], kp[b_idx], kp[c_idx])
        angles.append(angle)

    return np.array(angles)


def extract_all_features(keypoints: np.ndarray) -> np.ndarray:
    """
    Extract all features: 12 regular joint angles + 4 symmetric arm + 8 symmetric leg angles.

    Symmetric arm angles sort left/right pairs so the model cannot distinguish which arm
    is raised (used for third/fourth position).

    Symmetric leg angles sort left/right pairs so the model cannot distinguish which leg
    is the working leg (used for arabesque, penche, tendu, attitude, grand_battement, developpe).

    Returns:
        shape (24,) — 12 regular + 4 symmetric arm + 8 symmetric leg angles
    """
    regular = extract_joint_angles(keypoints)

    # Symmetric arms
    left_elbow     = regular[FEATURE_NAMES.index("left_elbow")]
    right_elbow    = regular[FEATURE_NAMES.index("right_elbow")]
    left_shoulder  = regular[FEATURE_NAMES.index("left_shoulder")]
    right_shoulder = regular[FEATURE_NAMES.index("right_shoulder")]
    elbow_high,    elbow_low    = sorted([left_elbow,    right_elbow],   reverse=True)
    shoulder_high, shoulder_low = sorted([left_shoulder, right_shoulder], reverse=True)

    # Symmetric legs
    left_hip          = regular[FEATURE_NAMES.index("left_hip")]
    right_hip         = regular[FEATURE_NAMES.index("right_hip")]
    left_knee         = regular[FEATURE_NAMES.index("left_knee")]
    right_knee        = regular[FEATURE_NAMES.index("right_knee")]
    left_ankle        = regular[FEATURE_NAMES.index("left_ankle")]
    right_ankle       = regular[FEATURE_NAMES.index("right_ankle")]
    left_hip_abduct   = regular[FEATURE_NAMES.index("left_hip_abduct")]
    right_hip_abduct  = regular[FEATURE_NAMES.index("right_hip_abduct")]
    hip_high,       hip_low       = sorted([left_hip,         right_hip],        reverse=True)
    knee_high,      knee_low      = sorted([left_knee,        right_knee],       reverse=True)
    ankle_high,     ankle_low     = sorted([left_ankle,       right_ankle],      reverse=True)
    hip_abduct_high, hip_abduct_low = sorted([left_hip_abduct, right_hip_abduct], reverse=True)

    return np.concatenate([
        regular,
        [elbow_high, elbow_low, shoulder_high, shoulder_low],
        [hip_high, hip_low, knee_high, knee_low, ankle_high, ankle_low, hip_abduct_high, hip_abduct_low],
    ])


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Scale joint angles from [0, 180] to [0, 1].
    Useful before feeding into neural networks.
    """
    return features / 180.0
