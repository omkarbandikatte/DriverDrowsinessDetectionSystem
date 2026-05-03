"""
detection.py - Face detection, eye landmark detection, and drowsiness analysis.

Uses dlib's 68-point facial landmark predictor to:
- Detect faces in each frame
- Extract eye landmarks to calculate Eye Aspect Ratio (EAR)
- Detect yawning via Mouth Aspect Ratio (MAR)
"""

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist


# --- EAR and MAR Thresholds ---
EAR_THRESHOLD = 0.25          # Below this = eyes closed
MAR_THRESHOLD = 0.75          # Above this = yawning
CONSEC_FRAMES_THRESHOLD = 20  # Frames below EAR to trigger drowsiness

# --- Facial landmark indices (dlib 68-point model) ---
# Left eye: points 42-47, Right eye: points 36-41
LEFT_EYE_START = 42
LEFT_EYE_END = 48
RIGHT_EYE_START = 36
RIGHT_EYE_END = 42

# Mouth: points 60-67 (inner lip)
MOUTH_START = 60
MOUTH_END = 68


class DrowsinessDetector:
    """Detects drowsiness using Eye Aspect Ratio and yawning via Mouth Aspect Ratio."""

    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        """
        Initialize the detector with dlib's face detector and landmark predictor.

        Args:
            predictor_path: Path to dlib's 68-point shape predictor model file.
        """
        # dlib's HOG-based face detector (fast and accurate for frontal faces)
        self.face_detector = dlib.get_frontal_face_detector()

        # 68-point facial landmark predictor
        self.landmark_predictor = dlib.shape_predictor(predictor_path)

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
        # Convert to grayscale for faster face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = self.face_detector(gray, 0)

        result = {
            'is_drowsy': self.is_drowsy,
            'is_yawning': False,
            'ear': None,
            'mar': None,
            'frame_counter': self.frame_counter,
            'face_detected': False
        }

        if len(faces) == 0:
            return result

        # Process the first (largest/closest) face detected
        face = faces[0]
        result['face_detected'] = True

        # Get 68 facial landmarks
        landmarks = self.landmark_predictor(gray, face)
        landmarks_np = self._shape_to_np(landmarks)

        # --- Eye Aspect Ratio ---
        left_eye = landmarks_np[LEFT_EYE_START:LEFT_EYE_END]
        right_eye = landmarks_np[RIGHT_EYE_START:RIGHT_EYE_END]

        left_ear = self.compute_ear(left_eye)
        right_ear = self.compute_ear(right_eye)

        # Average EAR of both eyes
        ear = (left_ear + right_ear) / 2.0
        result['ear'] = round(ear, 3)

        # --- Mouth Aspect Ratio (Yawning) ---
        mouth = landmarks_np[MOUTH_START:MOUTH_END]
        mar = self.compute_mar(mouth)
        result['mar'] = round(mar, 3)

        # Check for yawning
        if mar > MAR_THRESHOLD:
            result['is_yawning'] = True

        # --- Drowsiness Logic ---
        if ear < EAR_THRESHOLD:
            # Eyes are closed - increment counter
            self.frame_counter += 1

            if self.frame_counter >= CONSEC_FRAMES_THRESHOLD:
                self.is_drowsy = True
        else:
            # Eyes are open - reset counter
            self.frame_counter = 0
            self.is_drowsy = False

        result['is_drowsy'] = self.is_drowsy
        result['frame_counter'] = self.frame_counter

        # Draw eye contours on frame for visualization
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [mouth], True, (0, 255, 255), 1)

        return result

    @staticmethod
    def _shape_to_np(shape):
        """
        Convert dlib shape object to numpy array of (x, y) coordinates.

        Args:
            shape: dlib full_object_detection (68 points)

        Returns:
            numpy array of shape (68, 2)
        """
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
