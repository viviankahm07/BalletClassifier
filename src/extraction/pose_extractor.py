"""
pose_extractor.py
-----------------
Runs MediaPipe Pose on a single image and returns 33 3D landmarks.
This is the first step in the pipeline: raw image → keypoint vector.

Uses the MediaPipe Tasks API (mediapipe 0.10+).
The pose landmarker model is downloaded automatically on first run.
"""

import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_heavy.task")


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe pose model (~25MB, one-time download)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


@dataclass
class PoseResult:
    """Holds the output of a single pose extraction."""
    keypoints: np.ndarray      # shape: (33, 3) — x, y, z per landmark
    visibility: np.ndarray     # shape: (33,)  — confidence per landmark
    image_path: str
    success: bool
    failure_reason: Optional[str] = None


class PoseExtractor:
    """
    Wraps MediaPipe PoseLandmarker for static image extraction.

    Usage:
        extractor = PoseExtractor()
        result = extractor.extract("path/to/image.jpg")
        if result.success:
            print(result.keypoints.shape)  # (33, 3)
    """

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
        _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            min_pose_detection_confidence=min_detection_confidence,
        )
        self._detector = mp_vision.PoseLandmarker.create_from_options(options)

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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        if not result.pose_landmarks:
            return PoseResult(
                keypoints=np.zeros((33, 3)),
                visibility=np.zeros(33),
                image_path=image_path,
                success=False,
                failure_reason="MediaPipe could not detect a pose"
            )

        landmarks = result.pose_landmarks[0]  # first person detected

        keypoints = np.array([
            [lm.x, lm.y, lm.z] for lm in landmarks
        ])  # shape: (33, 3)

        visibility = np.array([
            getattr(lm, "visibility", 1.0) for lm in landmarks
        ])  # shape: (33,)

        return PoseResult(
            keypoints=keypoints,
            visibility=visibility,
            image_path=image_path,
            success=True
        )
