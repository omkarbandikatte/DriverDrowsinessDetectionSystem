"""
alert.py - Alarm sound system for drowsiness alerts.

Plays an alarm sound when drowsiness is detected.
Supports intensity escalation (louder/more urgent alerts over time).
"""

import os
import threading
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False


class AlertSystem:
    """Manages alarm sounds with escalating intensity levels."""

    # Alert intensity levels
    LEVEL_NORMAL = 1    # Initial drowsiness detected
    LEVEL_HIGH = 2      # Drowsy for 10+ seconds
    LEVEL_CRITICAL = 3  # Drowsy for 20+ seconds (triggers escalation)

    def __init__(self, alarm_sound_path="alarm.wav"):
        """
        Initialize the alert system.

        Args:
            alarm_sound_path: Path to the alarm WAV file.
        """
        self.alarm_sound_path = alarm_sound_path
        self.is_playing = False
        self.current_level = 0
        self._lock = threading.Lock()

        # Initialize pygame mixer for audio playback
        if PYGAME_AVAILABLE:
            pygame.mixer.init()
            self._use_pygame = True
        else:
            self._use_pygame = False

        # Generate alarm sound if file doesn't exist
        if not os.path.exists(self.alarm_sound_path):
            self._generate_alarm_sound()

    def _generate_alarm_sound(self):
        """Generate a simple alarm WAV file if none exists."""
        try:
            import wave
            import struct

            # Generate a beeping alarm tone
            sample_rate = 44100
            duration = 2.0  # seconds
            frequency = 880  # Hz (A5 note - urgent sounding)

            samples = []
            for i in range(int(sample_rate * duration)):
                t = i / sample_rate
                # Create pulsing effect (on/off every 0.2 seconds)
                envelope = 1.0 if (t % 0.4) < 0.2 else 0.0
                sample = envelope * np.sin(2 * np.pi * frequency * t)
                samples.append(int(sample * 32767))

            # Write WAV file
            with wave.open(self.alarm_sound_path, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                for sample in samples:
                    wav_file.writeframes(struct.pack('<h', sample))

            print(f"[ALERT] Generated alarm sound: {self.alarm_sound_path}")
        except Exception as e:
            print(f"[ALERT] Could not generate alarm sound: {e}")

    def play_alarm(self, level=LEVEL_NORMAL):
        """
        Play alarm sound at the specified intensity level.

        Args:
            level: Alert intensity (LEVEL_NORMAL, LEVEL_HIGH, LEVEL_CRITICAL)
        """
        with self._lock:
            if self.is_playing and self.current_level >= level:
                return  # Already playing at same or higher intensity

            self.current_level = level
            self.is_playing = True

        # Play sound in a separate thread to avoid blocking video processing
        thread = threading.Thread(target=self._play_sound, args=(level,), daemon=True)
        thread.start()

    def _play_sound(self, level):
        """Internal method to play sound based on intensity level."""
        try:
            if self._use_pygame:
                # Use pygame for cross-platform audio
                pygame.mixer.music.load(self.alarm_sound_path)

                # Set volume based on level
                volume = min(0.5 + (level * 0.2), 1.0)
                pygame.mixer.music.set_volume(volume)

                # Loop count based on intensity
                loops = level  # 1, 2, or 3 loops
                pygame.mixer.music.play(loops)

            elif WINSOUND_AVAILABLE:
                # Windows fallback: system beep
                frequency = 800 + (level * 200)  # Higher pitch = more urgent
                duration_ms = 500 * level
                winsound.Beep(frequency, duration_ms)

            else:
                # Last resort: terminal bell
                print('\a' * level)

        except Exception as e:
            print(f"[ALERT] Error playing alarm: {e}")
            # Fallback to terminal bell
            print('\a')
        finally:
            with self._lock:
                self.is_playing = False

    def stop_alarm(self):
        """Stop any currently playing alarm."""
        with self._lock:
            self.is_playing = False
            self.current_level = 0

        if self._use_pygame:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

    def cleanup(self):
        """Clean up audio resources."""
        if self._use_pygame:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
