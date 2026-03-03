"""
pose_extractor.py
-----------------
Runs MediaPipe Pose on a single image and returns 33 3D landmarks.
This is the first step in the pipeline: raw image → keypoint vector.
"""

import mediapipe as mp
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PoseResult:
    """Holds the output of a single pose extraction."""
    keypoints: np.ndarray          # shape: (33, 3) — x, y, z per landmark
    visibility: np.ndarray         # shape: (33,)  — confidence per landmark
    image_path: str
    success: bool
    failure_reason: Optional[str] = None


class PoseExtractor:
    """
    Wraps MediaPipe Pose for static image extraction.

    Usage:
        extractor = PoseExtractor()
        result = extractor.extract("path/to/image.jpg")
        if result.success:
            print(result.keypoints.shape)  # (33, 3)
    """

    # MediaPipe landmark indices for reference
    LANDMARKS = {
        "nose": 0,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13,    "right_elbow": 14,
        "left_wrist": 15,    "right_wrist": 16,
        "left_hip": 23,      "right_hip": 24,
        "left_knee": 25,     "right_knee": 26,
        "left_ankle": 27,    "right_ankle": 28,
        "left_heel": 29,     "right_heel": 30,
        "left_foot_index": 31, "right_foot_index": 32,
    }

    def __init__(self, min_detection_confidence: float = 0.5):
        self.min_detection_confidence = min_detection_confidence
        self._mp_pose = mp.solutions.pose

    def extract(self, image_path: str) -> PoseResult:
        """Extract pose keypoints from a single image file."""
        image = cv2.imread(image_path)
        if image is None:
            return PoseResult(
                keypoints=np.zeros((33, 3)),
                visibility=np.zeros(33),
                image_path=image_path,
                success=False,
                failure_reason=f"Could not read image: {image_path}"
            )

        return self.extract_from_array(image, image_path)

    def extract_from_array(self, image: np.ndarray, image_path: str = "") -> PoseResult:
        """Extract pose keypoints from a numpy image array (BGR format from OpenCV)."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with self._mp_pose.Pose(
            static_image_mode=True,                        # KEY: treats each image independently
            min_detection_confidence=self.min_detection_confidence,
            model_complexity=2                             # 0=lite, 1=full, 2=heavy (most accurate)
        ) as pose:
            results = pose.process(rgb)

        if not results.pose_landmarks:
            return PoseResult(
                keypoints=np.zeros((33, 3)),
                visibility=np.zeros(33),
                image_path=image_path,
                success=False,
                failure_reason="MediaPipe could not detect a pose"
            )

        keypoints = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.pose_landmarks.landmark
        ])  # shape: (33, 3)

        visibility = np.array([
            lm.visibility
            for lm in results.pose_landmarks.landmark
        ])  # shape: (33,)

        return PoseResult(
            keypoints=keypoints,
            visibility=visibility,
            image_path=image_path,
            success=True
        )
