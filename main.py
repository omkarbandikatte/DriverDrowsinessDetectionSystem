"""
main.py - Entry point for the Driver Drowsiness Detection System.

Orchestrates all modules:
- VideoStream: Captures frames from IP Webcam or local camera
- DrowsinessDetector: Analyzes frames for drowsiness indicators
- AlertSystem: Plays alarm sounds at varying intensity
- EscalationManager: Handles SMS/call escalation for prolonged drowsiness

Usage:
    python main.py                          # Use local webcam (camera 0)
    python main.py --source 1               # Use camera index 1
    python main.py --source http://192.168.1.5:8080/video  # IP Webcam URL

Controls:
    Press 'q' to quit
    Press 'r' to reset drowsiness state
"""

import argparse
import time
import cv2

from video_stream import VideoStream
from detection import DrowsinessDetector
from alert import AlertSystem
from escalation import EscalationManager


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Driver Drowsiness Detection System with Escalation Alerts"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: camera index (0, 1) or IP Webcam URL "
             "(e.g., http://192.168.1.5:8080/video). Default: 0"
    )
    parser.add_argument(
        "--predictor",
        default="shape_predictor_68_face_landmarks.dat",
        help="Path to dlib's 68-point shape predictor model file."
    )
    parser.add_argument(
        "--alarm",
        default="alarm.wav",
        help="Path to alarm sound file (WAV format)."
    )
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=0.25,
        help="EAR threshold for eye closure detection. Default: 0.25"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without GUI display (headless mode)."
    )
    return parser.parse_args()


def draw_status_overlay(frame, detection_result, escalation_status, fps):
    """
    Draw status information overlay on the video frame.

    Args:
        frame: The video frame to draw on.
        detection_result: Dict from DrowsinessDetector.detect()
        escalation_status: Status text from EscalationManager
        fps: Current frames per second
    """
    h, w = frame.shape[:2]

    # Background for status panel
    cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (w, 120), (50, 50, 50), 1)

    # EAR value display
    ear = detection_result.get('ear')
    ear_text = f"EAR: {ear:.3f}" if ear is not None else "EAR: N/A"
    ear_color = (0, 255, 0) if ear and ear >= 0.25 else (0, 0, 255)
    cv2.putText(frame, ear_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, ear_color, 2)

    # MAR value display (yawning)
    mar = detection_result.get('mar')
    mar_text = f"MAR: {mar:.3f}" if mar is not None else "MAR: N/A"
    mar_color = (0, 255, 255) if mar and mar >= 0.75 else (0, 255, 0)
    cv2.putText(frame, mar_text, (200, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, mar_color, 2)

    # FPS display
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Drowsiness status
    if detection_result.get('is_drowsy'):
        cv2.putText(frame, "!! DROWSINESS DETECTED !!", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif not detection_result.get('face_detected'):
        cv2.putText(frame, "No face detected", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Driver is alert", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Yawning indicator
    if detection_result.get('is_yawning'):
        cv2.putText(frame, "YAWNING!", (w - 150, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # Escalation status
    cv2.putText(frame, escalation_status, (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Frame counter
    fc = detection_result.get('frame_counter', 0)
    cv2.putText(frame, f"Closed frames: {fc}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Red border flash when drowsy
    if detection_result.get('is_drowsy'):
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)


def main():
    """Main application loop."""
    args = parse_args()

    # Determine video source (integer for local camera, string for URL)
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass  # Keep as string (URL)

    print("=" * 60)
    print("   DRIVER DROWSINESS DETECTION SYSTEM")
    print("=" * 60)
    print(f"  Video source: {source}")
    print(f"  Predictor: {args.predictor}")
    print(f"  EAR threshold: {args.ear_threshold}")
    print(f"  Controls: 'q' = quit, 'r' = reset")
    print("=" * 60)

    # --- Initialize all modules ---
    print("\n[INIT] Starting video stream...")
    stream = VideoStream(source)
    try:
        stream.start()
    except ConnectionError as e:
        print(f"[ERROR] {e}")
        return

    print("[INIT] Loading face detector and landmark predictor...")
    try:
        detector = DrowsinessDetector(predictor_path=args.predictor)
    except RuntimeError as e:
        print(f"[ERROR] Cannot load predictor model: {e}")
        print("  Download it from: "
              "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        stream.stop()
        return

    # Update detection threshold if custom value provided
    from detection import EAR_THRESHOLD
    if args.ear_threshold != 0.25:
        import detection
        detection.EAR_THRESHOLD = args.ear_threshold

    print("[INIT] Initializing alert system...")
    alert = AlertSystem(alarm_sound_path=args.alarm)

    print("[INIT] Initializing escalation manager...")
    escalation = EscalationManager()

    print("\n[RUNNING] System active. Monitoring for drowsiness...\n")

    # --- Main processing loop ---
    fps = 0
    frame_count = 0
    fps_start_time = time.time()

    try:
        while True:
            # Read frame from video stream
            success, frame = stream.read_frame()
            if not success:
                print("[WARNING] Failed to read frame. Retrying...")
                time.sleep(0.1)
                continue

            # Run drowsiness detection on the frame
            result = detector.detect(frame)

            # Update escalation state based on detection result
            escalation_level = escalation.update(result['is_drowsy'])

            # Trigger alerts based on escalation level
            if escalation_level >= 1:
                alert.play_alarm(level=escalation_level)
            else:
                alert.stop_alarm()

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            # Display frame with overlay (unless headless mode)
            if not args.no_display:
                status_text = escalation.get_status_text()
                draw_status_overlay(frame, result, status_text, fps)
                cv2.imshow("Driver Drowsiness Detection", frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[EXIT] Quit signal received.")
                    break
                elif key == ord('r'):
                    # Manual reset
                    detector.frame_counter = 0
                    detector.is_drowsy = False
                    alert.stop_alarm()
                    escalation._reset()
                    print("[RESET] Drowsiness state reset manually.")

    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted by user.")

    finally:
        # Cleanup all resources
        print("[CLEANUP] Releasing resources...")
        stream.stop()
        alert.stop_alarm()
        alert.cleanup()
        cv2.destroyAllWindows()
        print("[DONE] System shut down cleanly.")


if __name__ == "__main__":
    main()
