# Driver Drowsiness Detection System

A real-time driver drowsiness detection system built with **computer vision** and **deep learning** techniques. The system monitors a driver's facial features through a video feed, computes physiological indicators of fatigue, and triggers a multi-stage escalating alert mechanism — from audible alarms to automated SMS and phone calls via Twilio.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Introduction](#introduction)
- [Objective](#objective)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Computer Vision Techniques — In Detail](#computer-vision-techniques--in-detail)
  - [1. Face Detection with MediaPipe Face Mesh](#1-face-detection-with-mediapipe-face-mesh)
  - [2. Eye Aspect Ratio (EAR)](#2-eye-aspect-ratio-ear)
  - [3. Mouth Aspect Ratio (MAR)](#3-mouth-aspect-ratio-mar)
  - [4. Temporal Frame Analysis](#4-temporal-frame-analysis)
  - [5. Real-Time Video Processing Pipeline](#5-real-time-video-processing-pipeline)
- [Implementation Details](#implementation-details)
  - [Backend Modules](#backend-modules)
  - [Frontend Dashboard](#frontend-dashboard)
  - [Escalation Pipeline](#escalation-pipeline)
  - [API Endpoints](#api-endpoints)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Controls](#controls)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)

---

## Quick Start

Run the entire system (backend + frontend) with a single command:

```bash
bash start.sh
```

This will:
1. Activate your virtual environment (`venv/` or `.venv/`) if one exists.
2. Install Python dependencies (`pip install -r requirements.txt`) if missing.
3. Install frontend npm dependencies if `frontend/node_modules/` is absent.
4. Start the **FastAPI backend** on `http://localhost:8000`.
5. Start the **React frontend** on `http://localhost:5173`.
6. Open the dashboard in your default browser automatically.

Press **Ctrl+C** to shut down both servers cleanly.

**Options:**

| Flag | Description |
|------|-------------|
| `--no-browser` | Skip automatic browser launch |

```bash
bash start.sh --no-browser
```

> **Prerequisites:** Python 3.8+, Node.js 18+, and a virtual environment with dependencies installed (see [Installation](#installation)).

---

## Introduction

Drowsy driving is a leading contributor to road accidents worldwide. According to the National Highway Traffic Safety Administration (NHTSA), drowsy driving accounts for an estimated 100,000 crashes, 71,000 injuries, and 1,550 fatalities annually in the United States alone. The challenge lies in detecting the onset of drowsiness **before** an accident occurs.

This project implements a **non-intrusive, vision-based drowsiness detection system** that uses a standard camera (webcam or IP camera) to continuously monitor a driver's face. By leveraging facial landmark detection and geometric ratio analysis, the system can reliably determine whether the driver's eyes are closing or the driver is yawning — two of the most reliable physiological indicators of fatigue.

The system operates entirely in real-time, processing video frames at interactive frame rates, and features a **three-tier escalation mechanism** that progressively increases alert intensity — starting from an audible alarm and escalating to SMS notifications and automated emergency phone calls if the driver fails to respond.

---

## Objective

The primary objectives of this project are:

1. **Real-Time Drowsiness Detection** — Develop a system capable of detecting driver drowsiness in real-time using computer vision, specifically by monitoring eye closure patterns (via EAR) and yawning behaviour (via MAR).

2. **Non-Intrusive Monitoring** — Use only a standard camera feed (no wearable sensors or physiological devices) to make the system practical and easy to deploy.

3. **Escalating Alert Mechanism** — Implement a multi-level alert pipeline that responds proportionally to the severity and duration of detected drowsiness:
   - **Level 1 (0–10 s):** Audible alarm at normal volume.
   - **Level 2 (10–20 s):** High-intensity alarm.
   - **Level 3 (20+ s):** SMS and automated phone call to an emergency contact via Twilio.

4. **Remote Monitoring via Web Dashboard** — Provide a React-based frontend dashboard that allows remote monitoring of the driver's status, including a live MJPEG video feed, real-time metrics (EAR, MAR, FPS), and system controls.

5. **Flexible Input Sources** — Support both local webcams and IP camera streams (e.g., from a smartphone running IP Webcam), enabling deployment across varied hardware configurations.

---

## System Architecture

The system follows a modular, layered architecture comprising a **computer vision backend**, a **FastAPI REST/WebSocket server**, and a **React frontend dashboard**.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React + Vite)                      │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │ Live Video │  │  Metrics     │  │  Controls   │  │  Settings  │  │
│  │  (MJPEG)   │  │  Dashboard   │  │  (Start/    │  │  (EAR/MAR  │  │
│  │            │  │  EAR/MAR/FPS │  │   Stop/     │  │  Thresholds│  │
│  │            │  │              │  │   Reset)    │  │  Toggles)  │  │
│  └─────┬──────┘  └──────┬───────┘  └──────┬──────┘  └─────┬──────┘  │
│        │                │                 │               │          │
└────────┼────────────────┼─────────────────┼───────────────┼──────────┘
         │ GET            │ WebSocket       │ POST          │ PUT
         │ /api/video-feed│ /ws/metrics     │ /api/start    │ /api/settings
         │                │                 │ /api/stop     │
┌────────┼────────────────┼─────────────────┼───────────────┼──────────┐
│        ▼                ▼                 ▼               ▼          │
│                    FASTAPI SERVER  (server.py)                       │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │                      SystemState                              │   │
│  │   - Manages lifecycle of all modules                          │   │
│  │   - Background processing thread for frame analysis           │   │
│  │   - Thread-safe access to latest frame & metrics              │   │
│  └───────┬──────────────┬──────────────────┬─────────────────────┘   │
│          │              │                  │                          │
│          ▼              ▼                  ▼                          │
│  ┌──────────────┐ ┌───────────────┐ ┌───────────────┐               │
│  │ VideoStream  │ │ Drowsiness    │ │ AlertSystem   │               │
│  │ (video_      │ │ Detector      │ │ (alert.py)    │               │
│  │ stream.py)   │ │ (detection.py)│ │  - pygame     │               │
│  │  - OpenCV    │ │  - MediaPipe  │ │  - winsound   │               │
│  │  - IP Cam    │ │  - EAR / MAR  │ │  - 3 levels   │               │
│  │  - Webcam    │ │  - Frame cnt  │ │               │               │
│  └──────────────┘ └───────────────┘ └───────────────┘               │
│                                            │                         │
│                                            ▼                         │
│                                   ┌─────────────────┐               │
│                                   │ Escalation      │               │
│                                   │ Manager          │               │
│                                   │ (escalation.py)  │               │
│                                   │  - Twilio SMS    │               │
│                                   │  - Twilio Call   │               │
│                                   │  - Time-based    │               │
│                                   │    thresholds    │               │
│                                   └─────────────────┘               │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **VideoStream** captures frames from a webcam or IP camera via OpenCV.
2. Each frame is passed to the **DrowsinessDetector**, which uses MediaPipe Face Mesh to extract 468 facial landmarks and computes EAR and MAR.
3. If the EAR drops below the threshold for a sustained number of consecutive frames, the **is_drowsy** flag is set.
4. The **EscalationManager** tracks drowsiness duration and determines the appropriate escalation level.
5. The **AlertSystem** plays audible alarms at the corresponding intensity.
6. At critical levels, the **EscalationManager** dispatches SMS and phone calls via the Twilio API.
7. The **FastAPI server** exposes all of this through REST endpoints and a WebSocket for real-time frontend consumption.

---

## Project Structure

```
DriverDrowsinessDetectionSystem/
├── start.sh                # Single-command launcher (backend + frontend)
├── main.py                 # CLI entry point — orchestrates all modules
├── server.py               # FastAPI backend — REST API, MJPEG stream, WebSocket
├── detection.py            # Core CV logic — MediaPipe Face Mesh, EAR, MAR
├── video_stream.py         # Video capture abstraction (webcam / IP camera)
├── alert.py                # Alarm sound system with intensity escalation
├── escalation.py           # Twilio SMS/call escalation logic
├── requirements.txt        # Python dependencies
├── .env.example            # Template for Twilio credentials
├── alarm.wav               # Auto-generated alarm sound file
│
└── frontend/               # React + Vite web dashboard
    ├── package.json
    ├── index.html
    ├── vite.config.js
    └── src/
        ├── main.jsx        # React entry point
        ├── App.jsx         # Main dashboard component
        ├── App.css         # Component-level styles
        └── index.css       # Global design tokens and styles
```

---

## Computer Vision Techniques — In Detail

### 1. Face Detection with MediaPipe Face Mesh

The system uses **Google's MediaPipe Face Mesh** for real-time face detection and facial landmark estimation. Unlike traditional approaches that use dlib's 68-point shape predictor (which requires a separate `.dat` model file), MediaPipe provides:

- **468 3D facial landmarks** with sub-pixel accuracy.
- **Built-in face detection** — no separate face detector (like Haar cascades or HOG) is needed.
- **GPU-accelerated inference** via TensorFlow Lite, running at 30+ FPS on standard hardware.
- **Landmark refinement** for iris and lip regions when `refine_landmarks=True` is enabled.

**Configuration used:**

```python
self.face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,              # Single driver
    refine_landmarks=True,        # Enable iris refinement
    min_detection_confidence=0.5, # Detection threshold
    min_tracking_confidence=0.5   # Tracking threshold
)
```

The model outputs normalised (x, y, z) coordinates for each of the 468 landmarks. These are converted to pixel coordinates for EAR and MAR computation:

```python
point = (int(landmark.x * frame_width), int(landmark.y * frame_height))
```

**Why MediaPipe over dlib?**
- No external model file download required.
- Faster inference (optimised for mobile/edge devices).
- More landmarks (468 vs 68), enabling finer-grained analysis.
- Cross-platform support without CMake/C++ build dependencies.

---

### 2. Eye Aspect Ratio (EAR)

The **Eye Aspect Ratio** is a scalar value that characterises the degree of eye opening. It was introduced by Soukupová and Čech (2016) and is computed from 6 landmark points per eye.

**Landmark indices used (MediaPipe Face Mesh):**

| Eye   | Indices                          |
|-------|----------------------------------|
| Left  | 362, 385, 387, 263, 373, 380    |
| Right | 33, 160, 158, 133, 153, 144     |

**Formula:**

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
```

Where:
- `p1, p4` are the lateral (horizontal) corner points of the eye.
- `p2, p3` are the upper eyelid points.
- `p5, p6` are the lower eyelid points.
- `|| · ||` denotes the Euclidean distance (computed via `scipy.spatial.distance.euclidean`).

**Geometric interpretation:**
- The numerator measures the vertical opening of the eye (two vertical distances summed).
- The denominator measures the horizontal width of the eye.
- The ratio is roughly constant (~0.25–0.35) when the eye is open and drops toward 0 when the eye closes.

**The final EAR is the average of both eyes:**

```python
ear = (left_ear + right_ear) / 2.0
```

**Thresholds:**

| Condition    | EAR Value       |
|-------------|-----------------|
| Eyes open   | ≈ 0.25 – 0.35   |
| Eyes closed | < 0.20           |
| **Default threshold** | **0.25** |

This averaging compensates for partial occlusions or asymmetric blinks.

---

### 3. Mouth Aspect Ratio (MAR)

The **Mouth Aspect Ratio** detects yawning by measuring the vertical opening of the mouth relative to its width, analogous to EAR but for the oral region.

**Landmark indices used (MediaPipe Face Mesh):**

```
Inner mouth: 78, 81, 13, 311, 308, 402, 14, 178
```

These 8 points trace the inner lip contour.

**Formula:**

```
MAR = (||p2 - p8|| + ||p3 - p7|| + ||p4 - p6||) / (2 × ||p1 - p5||)
```

Where:
- `p1, p5` are the left and right mouth corners (horizontal span).
- `p2–p4` are upper inner lip points; `p6–p8` are their lower counterparts.
- Three vertical distances are summed for robustness.

**Thresholds:**

| Condition     | MAR Value        |
|--------------|-----------------|
| Mouth closed | ≈ 0.2 – 0.4     |
| Yawning      | > 0.75           |
| **Default threshold** | **0.75** |

Yawning is treated as a supplementary fatigue indicator and is displayed on the HUD but does not independently trigger escalation (only eye closure does).

---

### 4. Temporal Frame Analysis

A single frame with a low EAR could simply be a blink. To distinguish genuine drowsiness from natural blinks, the system uses **consecutive frame counting**:

```python
CONSEC_FRAMES_THRESHOLD = 20  # ~0.67 seconds at 30 FPS
```

**Logic:**

```python
if ear < EAR_THRESHOLD:
    self.frame_counter += 1
    if self.frame_counter >= CONSEC_FRAMES_THRESHOLD:
        self.is_drowsy = True       # Sustained eye closure → drowsy
else:
    self.frame_counter = 0          # Eyes opened → reset
    self.is_drowsy = False
```

- A normal blink lasts 100–400 ms (~3–12 frames at 30 FPS).
- The threshold of 20 consecutive frames (~667 ms) filters out all natural blinks.
- If the driver's eyes remain closed beyond this threshold, drowsiness is positively identified.

This temporal filtering is critical for minimising false positives while maintaining sensitivity to genuine fatigue episodes.

---

### 5. Real-Time Video Processing Pipeline

Each frame passes through the following pipeline:

```
Camera → OpenCV Capture → BGR→RGB Conversion → MediaPipe Face Mesh
    → Landmark Extraction → EAR/MAR Computation → Temporal Analysis
    → Drowsiness Decision → Escalation Update → Alert Trigger
    → HUD Overlay Rendering → JPEG Encoding → MJPEG Stream / Display
```

**Key optimisations:**

| Technique | Purpose |
|-----------|---------|
| Frame resizing to 640×480 | Reduces computation per frame |
| `CAP_PROP_BUFFERSIZE = 1` | Minimises latency on IP camera streams |
| Single-face mode (`max_num_faces=1`) | Avoids unnecessary multi-face overhead |
| JPEG quality 80 | Balances stream quality vs. bandwidth |
| Background thread processing | Prevents frame analysis from blocking the API server |
| Thread locks on frame/metric access | Ensures thread-safe reads in the async server |

---

## Implementation Details

### Backend Modules

#### `video_stream.py` — Video Capture

Wraps OpenCV's `cv2.VideoCapture` to support both local cameras (by integer index) and IP camera URLs (as strings). Sets `CAP_PROP_BUFFERSIZE = 1` to ensure the latest frame is always read, preventing stale-frame issues common with network streams.

#### `detection.py` — Drowsiness Detector

The core computer vision module. Initialises MediaPipe Face Mesh, extracts relevant landmark subsets, computes EAR and MAR using Euclidean distances, and applies the consecutive-frame threshold to produce a binary drowsiness classification per frame.

Contour visualisation is also performed: eye and mouth landmarks are drawn as polylines on the frame for real-time visual feedback.

#### `alert.py` — Alert System

Manages audible alarms with three intensity levels. Uses **pygame.mixer** as the primary audio backend, with fallbacks to **winsound** (Windows) and terminal bell (`\a`). Can auto-generate a pulsing 880 Hz alarm tone (WAV) if no alarm file exists.

| Level | Volume | Loops | Description       |
|-------|--------|-------|-------------------|
| 1     | 0.7    | 1     | Normal alarm beep |
| 2     | 0.9    | 2     | High-intensity    |
| 3     | 1.0    | 3     | Critical alarm    |

#### `escalation.py` — Escalation Manager

Tracks drowsiness duration and triggers external notifications at configurable time thresholds:

| Duration    | Escalation Level | Action                          |
|------------|------------------|---------------------------------|
| 0–10 s     | 1 (Normal)       | Audible alarm                   |
| 10–20 s    | 2 (High)         | High-intensity alarm            |
| 20–25 s    | 3 (Critical)     | SMS sent via Twilio             |
| 25+ s      | 3 (Critical)     | Automated phone call via Twilio |
| Recovery   | 0 (None)         | All alerts reset                |

SMS and call operations are dispatched in background threads to avoid blocking the detection loop. Twilio credentials are loaded from environment variables via `python-dotenv`.

#### `server.py` — FastAPI Server

Exposes the entire system through a REST API and WebSocket:

- **`SystemState`** class manages the lifecycle of all modules and runs the processing loop in a daemon thread.
- Thread locks (`_frame_lock`, `_metrics_lock`) ensure safe concurrent access between the processing thread and async FastAPI handlers.

#### `main.py` — CLI Entry Point

A standalone CLI application that orchestrates all modules with an OpenCV `imshow` window. Provides an on-screen HUD with EAR, MAR, FPS, drowsiness status, and escalation state. Supports command-line arguments for source selection, threshold tuning, and headless mode.

---

### Frontend Dashboard

A **React 19 + Vite** single-page application providing a web-based monitoring interface:

- **Live Video Feed** — Displays the MJPEG stream from `/api/video-feed`.
- **Real-Time Metrics** — EAR, MAR, FPS, escalation status via WebSocket (`/ws/metrics`).
- **System Controls** — Start/Stop/Reset monitoring.
- **Settings Panel** — Adjust EAR/MAR thresholds, toggle alarm/SMS/call features.
- **Status Indicators** — Visual alerts for drowsiness, yawning, and escalation level.

---

### API Endpoints

| Method | Endpoint           | Description                                |
|--------|--------------------|--------------------------------------------|
| POST   | `/api/start`       | Start monitoring (accepts source, thresholds) |
| POST   | `/api/stop`        | Stop monitoring and release resources      |
| POST   | `/api/reset`       | Reset drowsiness and escalation state      |
| GET    | `/api/status`      | Get current system status and metrics      |
| PUT    | `/api/settings`    | Update thresholds and toggle features      |
| GET    | `/api/video-feed`  | MJPEG video stream                         |
| WS     | `/ws/metrics`      | Real-time metrics via WebSocket (100ms interval) |

---

## Features

- **Real-time face and eye tracking** using MediaPipe Face Mesh (468 landmarks)
- **Eye Aspect Ratio (EAR)** calculation for accurate eye closure detection
- **Mouth Aspect Ratio (MAR)** for yawning detection
- **Temporal frame analysis** to distinguish blinks from drowsiness
- **Three-level escalating alert system:**
  - Level 1 (0–10 s) → Normal audible alarm
  - Level 2 (10–20 s) → High-intensity alarm
  - Level 3 (20+ s) → SMS + automated phone call via Twilio
- **Web dashboard** (React + Vite) with live video, metrics, and controls
- **REST API + WebSocket** backend via FastAPI
- **IP Webcam support** for remote/mobile camera streams
- **On-screen HUD** showing EAR, MAR, FPS, frame counter, and escalation status
- **Headless mode** for server-only operation without GUI
- **Auto-generated alarm sound** (no external audio file required)
- **Thread-safe architecture** for concurrent video processing and API serving

---

## Installation

### Prerequisites

- **Python 3.8+**
- **Node.js 18+** (for the frontend dashboard)
- A webcam (built-in or USB) **OR** IP Webcam app on your phone
- *(Optional)* Twilio account for SMS/call escalation

### Step 1: Clone and Navigate

```bash
git clone <repository-url>
cd DriverDrowsinessDetectionSystem
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # Linux/Mac
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Twilio (Optional)

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # Linux/Mac
```

Edit `.env` with your Twilio credentials from [https://console.twilio.com/](https://console.twilio.com/).

### Step 5: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 6: Set Up IP Webcam (Optional)

1. Install the **"IP Webcam"** app on your Android phone.
2. Open the app → **Start Server**.
3. Note the URL shown (e.g., `http://192.168.1.5:8080/video`).
4. Ensure your phone and computer are on the **same Wi-Fi network**.

---

## Usage

### Single-Command Launch (Recommended)

```bash
bash start.sh
```

Starts the FastAPI backend and React frontend together. Open `http://localhost:5173` in your browser.

---

### CLI Mode (with OpenCV window)

```bash
# Local webcam (default — camera index 0)
python main.py

# IP Webcam
python main.py --source http://192.168.1.5:8080/video

# Custom EAR threshold
python main.py --source 0 --ear-threshold 0.22

# Headless mode (no GUI window)
python main.py --no-display
```

### Web Dashboard Mode

```bash
# Terminal 1 — Start the FastAPI backend
python server.py
# Server runs at http://localhost:8000

# Terminal 2 — Start the React frontend
cd frontend
npm run dev
# Dashboard runs at http://localhost:5173
```

Then open the dashboard in your browser and use the **Start** button to begin monitoring.

---

## Controls

### CLI Mode

| Key | Action                             |
|-----|------------------------------------|
| `q` | Quit the application               |
| `r` | Reset drowsiness state manually    |

### Web Dashboard

| Control       | Action                                      |
|--------------|---------------------------------------------|
| Start        | Begin monitoring with configured source      |
| Stop         | Stop monitoring and release resources        |
| Reset        | Clear drowsiness state and escalation timers |
| EAR Slider   | Adjust eye closure sensitivity               |
| MAR Slider   | Adjust yawning detection sensitivity         |
| Alarm Toggle | Enable/disable audible alarm                 |
| SMS Toggle   | Enable/disable SMS escalation                |
| Call Toggle  | Enable/disable phone call escalation         |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Cannot open video source" | Verify the camera index or IP Webcam URL. Ensure devices are on the same network. |
| `mediapipe` install fails | Ensure Python 3.8–3.11. Try `pip install mediapipe==0.10.14`. |
| No sound playing | Install pygame: `pip install pygame`. Check system audio is not muted. |
| Low FPS | Reduce resolution, use `--no-display`, or close other camera-using applications. |
| Twilio not sending | Verify `.env` credentials. Ensure phone numbers include country codes (e.g., `+1`). |
| WebSocket not connecting | Confirm the FastAPI server is running on port 8000. Check for CORS issues. |
| Frontend not loading | Run `npm install` in the `frontend/` directory. Ensure Node.js 18+ is installed. |

---

## Dependencies

### Python (Backend)

| Package          | Purpose                                          |
|-----------------|--------------------------------------------------|
| `opencv-python` | Video capture, frame processing, image encoding  |
| `mediapipe`     | Face Mesh — 468-point facial landmark detection  |
| `numpy`         | Numerical operations and array handling          |
| `scipy`         | Euclidean distance computation for EAR/MAR       |
| `pygame`        | Cross-platform audio playback for alarms         |
| `twilio`        | SMS and voice call API for emergency escalation  |
| `python-dotenv` | Environment variable loading from `.env` files   |
| `fastapi`       | Async REST API and WebSocket server framework    |
| `uvicorn`       | ASGI server for running FastAPI                  |
| `websockets`    | WebSocket protocol support                       |

### Frontend

| Package     | Purpose                               |
|------------|---------------------------------------|
| `react`    | UI component framework (v19)          |
| `react-dom`| DOM rendering for React               |
| `vite`     | Development server and build tooling  |

---

## References

1. Soukupová, T., & Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks.* 21st Computer Vision Winter Workshop.
2. Lugaresi, C., et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines.* arXiv:1906.08172.
3. MediaPipe Face Mesh — [https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)

---

> **Note:** This project is developed as a Computer Vision mini-project for academic purposes. For production deployment in vehicles, additional sensor fusion (e.g., steering patterns, vehicle telemetry) and rigorous safety validation would be required.