"""
video_stream.py - Handles video capture from IP Webcam or local camera.

Supports:
- IP Webcam URL (e.g., http://192.168.1.5:8080/video)
- Local webcam (index 0, 1, etc.)
"""

import cv2


class VideoStream:
    """Manages video stream from IP Webcam or local camera."""

    def __init__(self, source=0):
        """
        Initialize the video stream.

        Args:
            source: IP Webcam URL string or integer camera index (default: 0)
        """
        self.source = source
        self.cap = None

    def start(self):
        """Open the video capture stream."""
        self.cap = cv2.VideoCapture(self.source)

        # Optimize buffer size for real-time performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise ConnectionError(
                f"Cannot open video source: {self.source}. "
                "Check the URL or camera index."
            )
        return self

    def read_frame(self):
        """
        Read a single frame from the stream.

        Returns:
            tuple: (success: bool, frame: numpy array or None)
        """
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            return False, None

        # Resize for faster processing while maintaining aspect ratio
        frame = cv2.resize(frame, (640, 480))
        return True, frame

    def stop(self):
        """Release the video capture resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_opened(self):
        """Check if the stream is still open."""
        return self.cap is not None and self.cap.isOpened()
"""
video_stream.py - Handles video capture from IP Webcam or local camera.

Supports:
- IP Webcam URL (e.g., http://192.168.1.100:8080/video)
- Local webcam (index 0, 1, etc.)
"""

import cv2


class VideoStream:
    """Manages video capture from an IP Webcam or local camera."""

    def __init__(self, source=0):
        """
        Initialize video stream.

        Args:
            source: IP Webcam URL string or integer for local camera index.
        """
        self.source = source
        self.cap = None

    def start(self):
        """Open the video capture stream."""
        self.cap = cv2.VideoCapture(self.source)

        # Optimize capture buffer size for real-time performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise ConnectionError(
                f"Cannot open video source: {self.source}. "
                "Check if IP Webcam is running and URL is correct."
            )
        return self

    def read_frame(self):
        """
        Read a single frame from the video stream.

        Returns:
            tuple: (success: bool, frame: numpy array or None)
        """
        if self.cap is None:
            return False, None
        return self.cap.read()

    def is_opened(self):
        """Check if the video stream is currently open."""
        return self.cap is not None and self.cap.isOpened()

    def stop(self):
        """Release the video capture resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
