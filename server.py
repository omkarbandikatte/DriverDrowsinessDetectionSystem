"""
server.py - FastAPI backend for the Driver Drowsiness Detection System.

Provides:
- REST API for system control (start/stop, reset, settings)
- MJPEG video stream endpoint
- WebSocket for real-time metrics
"""

import asyncio
import time
import threading
import cv2
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from video_stream import VideoStream
from detection import DrowsinessDetector
from alert import AlertSystem
from escalation import EscalationManager
import detection as detection_module


# --- Request / Response Models ---

class StartRequest(BaseModel):
    source: str = "0"
    ear_threshold: float = 0.25
    mar_threshold: float = 0.75


class SettingsUpdate(BaseModel):
    ear_threshold: Optional[float] = None
    mar_threshold: Optional[float] = None
    alarm_enabled: Optional[bool] = None
    sms_enabled: Optional[bool] = None
    call_enabled: Optional[bool] = None


# --- Global System State ---

class SystemState:
    """Holds all runtime state for the monitoring system."""

    def __init__(self):
        self.stream: Optional[VideoStream] = None
        self.detector: Optional[DrowsinessDetector] = None
        self.alert: Optional[AlertSystem] = None
        self.escalation: Optional[EscalationManager] = None
        self.is_running = False
        self.latest_frame_jpeg: Optional[bytes] = None
        self.latest_metrics: dict = {}
        self._processing_thread: Optional[threading.Thread] = None
        self.settings = {
            "ear_threshold": 0.25,
            "mar_threshold": 0.75,
            "alarm_enabled": True,
            "sms_enabled": True,
            "call_enabled": True,
            "source": "0",
        }
        self.fps = 0.0
        self._frame_lock = threading.Lock()
        self._metrics_lock = threading.Lock()

    def start_monitoring(self, source: str, ear_thresh: float, mar_thresh: float):
        if self.is_running:
            raise RuntimeError("System is already running")

        # Parse source: integer for local camera, string for URL
        try:
            parsed_source = int(source)
        except ValueError:
            parsed_source = source

        # Initialize all modules
        self.stream = VideoStream(parsed_source)
        self.stream.start()

        self.detector = DrowsinessDetector()
        self.alert = AlertSystem()
        self.escalation = EscalationManager()

        # Apply thresholds
        detection_module.EAR_THRESHOLD = ear_thresh
        detection_module.MAR_THRESHOLD = mar_thresh
        self.settings.update({
            "ear_threshold": ear_thresh,
            "mar_threshold": mar_thresh,
            "source": source,
        })

        self.is_running = True
        self._processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self._processing_thread.start()

    def stop_monitoring(self):
        self.is_running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=3)
        if self.stream:
            self.stream.stop()
        if self.alert:
            self.alert.stop_alarm()
            self.alert.cleanup()
        self.latest_frame_jpeg = None
        self.latest_metrics = {}

    def reset_state(self):
        if self.detector:
            self.detector.frame_counter = 0
            self.detector.is_drowsy = False
        if self.alert:
            self.alert.stop_alarm()
        if self.escalation:
            self.escalation._reset()

    def _processing_loop(self):
        """Main frame processing loop (runs in background thread)."""
        frame_count = 0
        fps_start = time.time()

        while self.is_running:
            if not self.stream or not self.stream.is_opened():
                time.sleep(0.1)
                continue

            success, frame = self.stream.read_frame()
            if not success:
                time.sleep(0.05)
                continue

            # Resize for consistent processing
            frame = cv2.resize(frame, (640, 480))

            # Run drowsiness detection
            result = self.detector.detect(frame)

            # Update escalation
            escalation_level = self.escalation.update(result["is_drowsy"])

            # Trigger alerts based on settings
            if self.settings.get("alarm_enabled", True):
                if escalation_level >= 1:
                    self.alert.play_alarm(level=escalation_level)
                else:
                    self.alert.stop_alarm()
            else:
                self.alert.stop_alarm()

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                self.fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            # Draw red border when drowsy
            if result.get("is_drowsy"):
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

            # Encode frame to JPEG
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._frame_lock:
                self.latest_frame_jpeg = jpeg.tobytes()

            # Build metrics dict
            metrics = {
                "ear": result.get("ear"),
                "mar": result.get("mar"),
                "is_drowsy": result.get("is_drowsy", False),
                "is_yawning": result.get("is_yawning", False),
                "face_detected": result.get("face_detected", False),
                "frame_counter": result.get("frame_counter", 0),
                "escalation_level": escalation_level,
                "escalation_status": self.escalation.get_status_text(),
                "drowsy_duration": round(self.escalation.get_drowsy_duration(), 1),
                "sms_sent": self.escalation.sms_sent,
                "call_made": self.escalation.call_made,
                "fps": round(self.fps, 1),
                "timestamp": time.time(),
            }
            with self._metrics_lock:
                self.latest_metrics = metrics


state = SystemState()


# --- FastAPI Application ---

app = FastAPI(title="Driver Drowsiness Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/start")
async def api_start(req: StartRequest):
    try:
        state.start_monitoring(req.source, req.ear_threshold, req.mar_threshold)
        return {"status": "started", "source": req.source}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/stop")
async def api_stop():
    state.stop_monitoring()
    return {"status": "stopped"}


@app.post("/api/reset")
async def api_reset():
    state.reset_state()
    return {"status": "reset"}


@app.get("/api/status")
async def api_status():
    return {
        "is_running": state.is_running,
        "settings": state.settings,
        "metrics": state.latest_metrics,
    }


@app.put("/api/settings")
async def api_settings(update: SettingsUpdate):
    if update.ear_threshold is not None:
        state.settings["ear_threshold"] = update.ear_threshold
        detection_module.EAR_THRESHOLD = update.ear_threshold
    if update.mar_threshold is not None:
        state.settings["mar_threshold"] = update.mar_threshold
        detection_module.MAR_THRESHOLD = update.mar_threshold
    if update.alarm_enabled is not None:
        state.settings["alarm_enabled"] = update.alarm_enabled
        if not update.alarm_enabled and state.alert:
            state.alert.stop_alarm()
    if update.sms_enabled is not None:
        state.settings["sms_enabled"] = update.sms_enabled
    if update.call_enabled is not None:
        state.settings["call_enabled"] = update.call_enabled
    return {"status": "updated", "settings": state.settings}


@app.get("/api/video-feed")
async def api_video_feed():
    """MJPEG video stream endpoint."""
    async def generate_frames():
        while state.is_running:
            with state._frame_lock:
                frame = state.latest_frame_jpeg
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            await asyncio.sleep(1 / 30)

    if not state.is_running:
        raise HTTPException(status_code=400, detail="Monitoring is not running")

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/metrics")
async def ws_metrics(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics."""
    await websocket.accept()
    try:
        while True:
            with state._metrics_lock:
                metrics = state.latest_metrics.copy()
            if metrics:
                await websocket.send_json(metrics)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
