"""
detection.py - Face detection, eye landmark detection, and drowsiness analysis.

Uses MediaPipe Face Mesh to:
- Detect faces in each frame
- Extract eye landmarks to calculate Eye Aspect Ratio (EAR)
- Detect yawning via Mouth Aspect Ratio (MAR)
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist


# --- EAR and MAR Thresholds ---
EAR_THRESHOLD = 0.25          # Below this = eyes closed
MAR_THRESHOLD = 0.75          # Above this = yawning
CONSEC_FRAMES_THRESHOLD = 20  # Frames below EAR to trigger drowsiness

# --- MediaPipe Face Mesh landmark indices ---
# Left eye: 6 key points for EAR calculation
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
# Right eye: 6 key points for EAR calculation
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
# Mouth inner: 8 key points for MAR calculation
MOUTH_INDICES = [78, 81, 13, 311, 308, 402, 14, 178]


class DrowsinessDetector:
    """Detects drowsiness using Eye Aspect Ratio and yawning via Mouth Aspect Ratio."""

    def __init__(self, predictor_path=None):
        """
        Initialize the detector with MediaPipe Face Mesh.

        Args:
            predictor_path: Unused (kept for API compatibility).
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Frame counter for consecutive eye closures
        self.frame_counter = 0

        # Drowsiness state
        self.is_drowsy = False

    @staticmethod
    def compute_ear(eye_points):
        """
        Calculate Eye Aspect Ratio (EAR).

        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

        When the eye is open, EAR is relatively constant (~0.3).
        When the eye closes, EAR drops toward 0.

        Args:
            eye_points: numpy array of 6 (x, y) coordinates for one eye.

        Returns:
            float: Eye Aspect Ratio value.
        """
        # Vertical distances
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])

        # Horizontal distance
        C = dist.euclidean(eye_points[0], eye_points[3])

        # EAR formula
        ear = (A + B) / (2.0 * C)
        return ear

    @staticmethod
    def compute_mar(mouth_points):
        """
        Calculate Mouth Aspect Ratio (MAR) for yawning detection.

        MAR = (||p2-p8|| + ||p3-p7|| + ||p4-p6||) / (2 * ||p1-p5||)

        When the mouth is open wide (yawning), MAR increases significantly.

        Args:
            mouth_points: numpy array of 8 (x, y) coordinates for inner mouth.

        Returns:
            float: Mouth Aspect Ratio value.
        """
        # Vertical distances
        A = dist.euclidean(mouth_points[1], mouth_points[7])
        B = dist.euclidean(mouth_points[2], mouth_points[6])
        C = dist.euclidean(mouth_points[3], mouth_points[5])

        # Horizontal distance
        D = dist.euclidean(mouth_points[0], mouth_points[4])

        # MAR formula
        mar = (A + B + C) / (2.0 * D)
        return mar

    def detect(self, frame):
        """
        Process a single frame for drowsiness detection.

        Args:
            frame: BGR image (numpy array) from video stream.

        Returns:
            dict: {
                'is_drowsy': bool,
                'is_yawning': bool,
                'ear': float or None,
                'mar': float or None,
                'frame_counter': int,
                'face_detected': bool
            }
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        result = {
            'is_drowsy': self.is_drowsy,
            'is_yawning': False,
            'ear': None,
            'mar': None,
            'frame_counter': self.frame_counter,
            'face_detected': False
        }

        if not results.multi_face_landmarks:
            return result

        result['face_detected'] = True
        landmarks = results.multi_face_landmarks[0].landmark

        def get_points(indices):
            return np.array(
                [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices],
                dtype=np.int32
            )

        left_eye = get_points(LEFT_EYE_INDICES)
        right_eye = get_points(RIGHT_EYE_INDICES)
        mouth = get_points(MOUTH_INDICES)

        # --- Eye Aspect Ratio ---
        left_ear = self.compute_ear(left_eye)
        right_ear = self.compute_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        result['ear'] = round(ear, 3)

        # --- Mouth Aspect Ratio (Yawning) ---
        mar = self.compute_mar(mouth)
        result['mar'] = round(mar, 3)

        if mar > MAR_THRESHOLD:
            result['is_yawning'] = True

        # --- Drowsiness Logic ---
        if ear < EAR_THRESHOLD:
            self.frame_counter += 1
            if self.frame_counter >= CONSEC_FRAMES_THRESHOLD:
                self.is_drowsy = True
        else:
            self.frame_counter = 0
            self.is_drowsy = False

        result['is_drowsy'] = self.is_drowsy
        result['frame_counter'] = self.frame_counter

        # Draw eye and mouth contours on frame for visualization
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [mouth], True, (0, 255, 255), 1)

        return result
