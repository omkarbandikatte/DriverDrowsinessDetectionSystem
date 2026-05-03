"""
escalation.py - Escalation logic for prolonged drowsiness.

Escalation timeline:
- 0-10 seconds drowsy: Normal alarm (beep)
- 10-20 seconds drowsy: High intensity alarm
- 20-30 seconds drowsy: Send SMS + Make phone call via Twilio
"""

import os
import time
import threading
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("[ESCALATION] Warning: twilio not installed. SMS/Call features disabled.")


class EscalationManager:
    """Manages escalation logic based on drowsiness duration."""

    # Escalation time thresholds (in seconds)
    THRESHOLD_HIGH_ALERT = 10     # Increase alert intensity
    THRESHOLD_SMS = 20            # Send SMS notification
    THRESHOLD_CALL = 25           # Make automated phone call

    def __init__(self):
        """Initialize escalation manager with Twilio credentials from environment."""
        # Twilio credentials from .env
        self.twilio_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.twilio_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.twilio_from = os.getenv("TWILIO_PHONE_NUMBER", "")
        self.emergency_contact = os.getenv("EMERGENCY_CONTACT_NUMBER", "")

        # Twilio client (initialized lazily)
        self._twilio_client = None

        # State tracking
        self.drowsy_start_time = None
        self.sms_sent = False
        self.call_made = False
        self.current_level = 0

        # Lock for thread safety
        self._lock = threading.Lock()

    def _get_twilio_client(self):
        """Lazily initialize and return Twilio client."""
        if self._twilio_client is None and TWILIO_AVAILABLE:
            if self.twilio_sid and self.twilio_token:
                self._twilio_client = TwilioClient(self.twilio_sid, self.twilio_token)
            else:
                print("[ESCALATION] Twilio credentials not configured in .env")
        return self._twilio_client

    def update(self, is_drowsy):
        """
        Update escalation state based on current drowsiness status.

        Args:
            is_drowsy: bool - whether drowsiness is currently detected.

        Returns:
            int: Current escalation level (0=none, 1=normal, 2=high, 3=critical)
        """
        with self._lock:
            if is_drowsy:
                # Start tracking drowsiness duration
                if self.drowsy_start_time is None:
                    self.drowsy_start_time = time.time()

                # Calculate how long the driver has been drowsy
                elapsed = time.time() - self.drowsy_start_time

                # Determine escalation level based on duration
                if elapsed >= self.THRESHOLD_SMS:
                    self.current_level = 3  # CRITICAL

                    # Send SMS (only once per episode)
                    if not self.sms_sent:
                        self._send_sms_async()
                        self.sms_sent = True

                    # Make call (only once per episode)
                    if elapsed >= self.THRESHOLD_CALL and not self.call_made:
                        self._make_call_async()
                        self.call_made = True

                elif elapsed >= self.THRESHOLD_HIGH_ALERT:
                    self.current_level = 2  # HIGH

                else:
                    self.current_level = 1  # NORMAL

            else:
                # Driver recovered - reset all escalation state
                self._reset()

        return self.current_level

    def _reset(self):
        """Reset escalation state when driver recovers."""
        self.drowsy_start_time = None
        self.sms_sent = False
        self.call_made = False
        self.current_level = 0

    def get_drowsy_duration(self):
        """
        Get how long the driver has been drowsy (seconds).

        Returns:
            float: Duration in seconds, or 0 if not drowsy.
        """
        if self.drowsy_start_time is None:
            return 0.0
        return time.time() - self.drowsy_start_time

    def _send_sms_async(self):
        """Send SMS alert in a background thread (non-blocking)."""
        thread = threading.Thread(target=self._send_sms, daemon=True)
        thread.start()

    def _send_sms(self):
        """Send SMS notification via Twilio."""
        client = self._get_twilio_client()
        if client is None:
            print("[ESCALATION] SMS ALERT (Twilio not configured) - "
                  "Driver may be drowsy. Please check.")
            return

        try:
            message = client.messages.create(
                body="⚠️ DROWSINESS ALERT: Driver may be drowsy. "
                     "No recovery detected for 20+ seconds. Please check immediately.",
                from_=self.twilio_from,
                to=self.emergency_contact
            )
            print(f"[ESCALATION] SMS sent successfully. SID: {message.sid}")
        except Exception as e:
            print(f"[ESCALATION] Failed to send SMS: {e}")

    def _make_call_async(self):
        """Make phone call in a background thread (non-blocking)."""
        thread = threading.Thread(target=self._make_call, daemon=True)
        thread.start()

    def _make_call(self):
        """Make automated phone call via Twilio."""
        client = self._get_twilio_client()
        if client is None:
            print("[ESCALATION] CALL ALERT (Twilio not configured) - "
                  "Would call emergency contact now.")
            return

        try:
            # TwiML instructs Twilio to speak the alert message
            twiml = (
                '<Response>'
                '<Say voice="alice" language="en-US">'
                'Alert! The driver may be drowsy. Please check on them immediately. '
                'This is an automated safety alert.'
                '</Say>'
                '<Pause length="1"/>'
                '<Say voice="alice" language="en-US">'
                'Repeating: The driver may be drowsy. Please check immediately.'
                '</Say>'
                '</Response>'
            )

            call = client.calls.create(
                twiml=twiml,
                from_=self.twilio_from,
                to=self.emergency_contact
            )
            print(f"[ESCALATION] Phone call initiated. SID: {call.sid}")
        except Exception as e:
            print(f"[ESCALATION] Failed to make call: {e}")

    def get_status_text(self):
        """
        Get human-readable status text for display.

        Returns:
            str: Status message describing current escalation state.
        """
        duration = self.get_drowsy_duration()
        if self.current_level == 0:
            return "Status: AWAKE"
        elif self.current_level == 1:
            return f"Status: DROWSY ({duration:.1f}s)"
        elif self.current_level == 2:
            return f"Status: HIGH ALERT ({duration:.1f}s)"
        else:
            return f"Status: CRITICAL - SMS/CALL SENT ({duration:.1f}s)"
