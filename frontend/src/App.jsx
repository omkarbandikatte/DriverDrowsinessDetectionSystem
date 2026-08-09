import { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';


const API_BASE = window.location.hostname === 'localhost'
  ? 'http://localhost:8000'
  : 'https://driverguard-backend.onrender.com';
const WS_BASE = API_BASE.replace(/^https/, 'wss').replace(/^http/, 'ws');

function App() {
  // --- State ---
  const [isRunning, setIsRunning] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [liveFrame, setLiveFrame] = useState(null);
  const [earThreshold, setEarThreshold] = useState(0.25);
  const [marThreshold, setMarThreshold] = useState(0.75);
  const [alarmEnabled, setAlarmEnabled] = useState(true);
  const [smsEnabled, setSmsEnabled] = useState(true);
  const [callEnabled, setCallEnabled] = useState(true);
  const [metrics, setMetrics] = useState({});
  const [error, setError] = useState('');
  const [fps, setFps] = useState(0);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamWsRef = useRef(null);
  const captureIntervalRef = useRef(null);
  const lastAlarmRef = useRef(null);
  const frameCountRef = useRef(0);

  // Client-side FPS counter
  useEffect(() => {
    const id = setInterval(() => {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
    }, 1000);
    return () => clearInterval(id);
  }, []);

  // --- Browser alarm via Web Audio API ---
  const playAlarm = useCallback((level) => {
    if (!alarmEnabled) return;
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type = 'square';
      osc.frequency.value = level >= 2 ? 1100 : 880;
      gain.gain.setValueAtTime(level >= 2 ? 0.25 : 0.12, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.45);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.45);
    } catch (e) { /* Audio not available */ }
  }, [alarmEnabled]);

  // --- Monitoring control ---
  const startMonitoring = async () => {
    setIsLoading(true);
    setError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
      });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      const canvas = canvasRef.current;
      canvas.width = 640;
      canvas.height = 480;
      const ctx2d = canvas.getContext('2d');

      const ws = new WebSocket(`${WS_BASE}/ws/stream`);
      streamWsRef.current = ws;

      ws.onopen = () => {
        setIsRunning(true);
        setIsLoading(false);
        captureIntervalRef.current = setInterval(() => {
          if (ws.readyState !== WebSocket.OPEN) return;
          ctx2d.drawImage(videoRef.current, 0, 0, 640, 480);
          canvas.toBlob(blob => {
            if (blob && ws.readyState === WebSocket.OPEN) ws.send(blob);
          }, 'image/jpeg', 0.75);
        }, 100); // 10 fps
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.frame) setLiveFrame(`data:image/jpeg;base64,${data.frame}`);
        setMetrics(data);
        frameCountRef.current += 1;
        if (data.is_drowsy) {
          const now = Date.now();
          if (!lastAlarmRef.current || now - lastAlarmRef.current > 2000) {
            playAlarm(data.escalation_level || 1);
            lastAlarmRef.current = now;
          }
        }
      };

      ws.onerror = () => setError('Connection error. Retrying...');
      ws.onclose = () => clearInterval(captureIntervalRef.current);
    } catch (err) {
      const msg = err.name === 'NotAllowedError'
        ? 'Camera access denied. Please allow camera permission and try again.'
        : err.message;
      setError(msg);
      setIsLoading(false);
    }
  };

  const stopMonitoring = () => {
    clearInterval(captureIntervalRef.current);
    if (streamWsRef.current) { streamWsRef.current.close(); streamWsRef.current = null; }
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop());
      videoRef.current.srcObject = null;
    }
    setIsRunning(false);
    setLiveFrame(null);
    setMetrics({});
  };

  const resetState = () => setMetrics({});

  const updateSettings = async (key, value) => {
    try {
      await fetch(`${API_BASE}/api/settings`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [key]: value }),
      });
    } catch (err) { /* fail silently for settings */ }
  };

  // --- Derived values ---
  const ear = metrics.ear ?? null;
  const mar = metrics.mar ?? null;
  const isDrowsy = metrics.is_drowsy ?? false;
  const isYawning = metrics.is_yawning ?? false;
  const faceDetected = metrics.face_detected ?? false;
  const escalationLevel = metrics.escalation_level ?? 0;
  const drowsyDuration = metrics.drowsy_duration ?? 0;

  const statusLabel =
    !isRunning ? 'OFFLINE' :
    !faceDetected ? 'NO FACE' :
    escalationLevel === 3 ? 'CRITICAL' :
    escalationLevel === 2 ? 'HIGH ALERT' :
    isDrowsy ? 'DROWSY' : 'AWAKE';

  const statusColor =
    statusLabel === 'AWAKE' ? 'green' :
    statusLabel === 'DROWSY' ? 'yellow' :
    statusLabel === 'HIGH ALERT' || statusLabel === 'CRITICAL' ? 'red' :
    'indigo';

  return (
    <div className="app">
      {/* Hidden elements for browser webcam capture */}
      <video ref={videoRef} style={{ display: 'none' }} playsInline muted />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      {/* --- Header --- */}
      <header className="header">
        <div className="header__brand">
          <div className="header__icon">🚗</div>
          <div>
            <h1 className="header__title">DriverGuard</h1>
            <p className="header__subtitle">Drowsiness Detection System</p>
          </div>
        </div>
        <div className="header__status">
          <span className={`badge badge--${statusColor}`}>
            <span className={`status-dot status-dot--${statusColor}`} />
            {statusLabel}
          </span>
          {isRunning && <span className="header__fps">{fps} FPS</span>}
        </div>
      </header>

      {/* --- Main Content --- */}
      <main className="main">
        {/* Left: Video + Source Controls */}
        <section className="main__left">
          {/* Video Feed */}
          <div className={`video-card card ${isDrowsy && isRunning ? 'video-card--alert' : ''}`}>
            <div className="video-card__header">
              <span className="video-card__label">
                {isRunning ? (
                  <><span className="live-dot" /> Live Feed</>
                ) : (
                  'Camera Preview'
                )}
              </span>
              {isRunning && isDrowsy && (
                <span className="badge badge--red pulse">⚠ DROWSY</span>
              )}
              {isRunning && isYawning && (
                <span className="badge badge--yellow">😮 YAWN</span>
              )}
            </div>
            <div className="video-card__feed">
              {isRunning ? (
                <img
                  src={liveFrame}
                  alt="Live camera feed"
                  className="video-card__img"
                />
              ) : (
                <div className="video-card__placeholder">
                  <div className="video-card__placeholder-icon">📷</div>
                  <p>Configure source below and start monitoring</p>
                </div>
              )}
            </div>
          </div>

          {/* Controls */}
          <div className="card source-card animate-in">
            <h3 className="card__title">📡 Browser Webcam</h3>
            <p className="card__hint">Uses your device camera directly — works on any device, no setup needed.</p>
            <div className="source-actions">
              {!isRunning ? (
                <button
                  className="btn btn--primary btn--lg"
                  onClick={startMonitoring}
                  disabled={isLoading}
                >
                  {isLoading ? '⏳ Requesting camera...' : '▶ Start Monitoring'}
                </button>
              ) : (
                <>
                  <button className="btn btn--danger" onClick={stopMonitoring}>⏹ Stop</button>
                  <button className="btn btn--ghost" onClick={resetState}>↺ Reset</button>
                </>
              )}
            </div>
            {error && <p className="error-text">⚠ {error}</p>}
          </div>
        </section>

        {/* Right: Metrics + Settings */}
        <section className="main__right">
          {/* Metrics Cards */}
          <div className="metrics-grid animate-in">
            <MetricCard
              label="EAR"
              value={ear !== null ? ear.toFixed(3) : '—'}
              sub="Eye Aspect Ratio"
              color={ear !== null && ear < earThreshold ? 'red' : 'green'}
              threshold={earThreshold}
              current={ear}
              type="below"
            />
            <MetricCard
              label="MAR"
              value={mar !== null ? mar.toFixed(3) : '—'}
              sub="Mouth Aspect Ratio"
              color={mar !== null && mar > marThreshold ? 'yellow' : 'green'}
              threshold={marThreshold}
              current={mar}
              type="above"
            />
            <MetricCard
              label="Closed"
              value={metrics.frame_counter ?? 0}
              sub="Consecutive Frames"
              color={metrics.frame_counter > 15 ? 'red' : 'indigo'}
            />
            <MetricCard
              label="Duration"
              value={drowsyDuration > 0 ? `${drowsyDuration}s` : '0s'}
              sub="Drowsy Duration"
              color={drowsyDuration > 10 ? 'red' : drowsyDuration > 0 ? 'yellow' : 'green'}
            />
          </div>

          {/* Escalation Timeline */}
          <div className="card escalation-card animate-in" style={{ animationDelay: '0.1s' }}>
            <h3 className="card__title">⏱ Escalation Timeline</h3>
            <EscalationTimeline level={escalationLevel} duration={drowsyDuration} />
            <div className="escalation-indicators">
              <IndicatorPill label="SMS Sent" active={metrics.sms_sent} />
              <IndicatorPill label="Call Made" active={metrics.call_made} />
            </div>
          </div>

          {/* Settings Panel */}
          <div className="card settings-card animate-in" style={{ animationDelay: '0.2s' }}>
            <h3 className="card__title">⚙ Detection Settings</h3>

            <div className="setting-row">
              <div className="setting-info">
                <span className="setting-label">EAR Threshold</span>
                <span className="setting-value mono">{earThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                className="slider"
                min="0.15"
                max="0.40"
                step="0.01"
                value={earThreshold}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  setEarThreshold(v);
                  if (isRunning) updateSettings('ear_threshold', v);
                }}
              />
            </div>

            <div className="setting-row">
              <div className="setting-info">
                <span className="setting-label">MAR Threshold</span>
                <span className="setting-value mono">{marThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                className="slider"
                min="0.50"
                max="1.00"
                step="0.01"
                value={marThreshold}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  setMarThreshold(v);
                  if (isRunning) updateSettings('mar_threshold', v);
                }}
              />
            </div>

            <div className="setting-toggle-row">
              <span className="setting-label">🔔 Alarm Sound</span>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={alarmEnabled}
                  onChange={(e) => {
                    setAlarmEnabled(e.target.checked);
                    if (isRunning) updateSettings('alarm_enabled', e.target.checked);
                  }}
                />
                <span className="toggle-slider" />
              </label>
            </div>

            <div className="setting-toggle-row">
              <span className="setting-label">📱 SMS Escalation</span>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={smsEnabled}
                  onChange={(e) => {
                    setSmsEnabled(e.target.checked);
                    if (isRunning) updateSettings('sms_enabled', e.target.checked);
                  }}
                />
                <span className="toggle-slider" />
              </label>
            </div>

            <div className="setting-toggle-row">
              <span className="setting-label">📞 Call Escalation</span>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={callEnabled}
                  onChange={(e) => {
                    setCallEnabled(e.target.checked);
                    if (isRunning) updateSettings('call_enabled', e.target.checked);
                  }}
                />
                <span className="toggle-slider" />
              </label>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}


/* --- Sub-components --- */

function MetricCard({ label, value, sub, color, threshold, current, type }) {
  const percent = (threshold && current !== null && current !== undefined)
    ? type === 'below'
      ? Math.min((current / (threshold * 2)) * 100, 100)
      : Math.min((current / (threshold * 1.5)) * 100, 100)
    : null;

  return (
    <div className={`metric-card card metric-card--${color}`}>
      <span className="metric-card__label">{label}</span>
      <span className={`metric-card__value mono`}>{value}</span>
      <span className="metric-card__sub">{sub}</span>
      {percent !== null && (
        <div className="metric-card__bar">
          <div
            className={`metric-card__bar-fill metric-card__bar-fill--${color}`}
            style={{ width: `${percent}%` }}
          />
        </div>
      )}
    </div>
  );
}

function EscalationTimeline({ level, duration }) {
  const stages = [
    { label: 'Normal', threshold: 0, color: 'green' },
    { label: 'High', threshold: 10, color: 'yellow' },
    { label: 'SMS', threshold: 20, color: 'orange' },
    { label: 'Call', threshold: 25, color: 'red' },
  ];

  return (
    <div className="timeline">
      <div className="timeline__track">
        {stages.map((s, i) => (
          <div
            key={i}
            className={`timeline__segment timeline__segment--${s.color} ${
              duration >= s.threshold ? 'timeline__segment--active' : ''
            }`}
          />
        ))}
      </div>
      <div className="timeline__labels">
        {stages.map((s, i) => (
          <span key={i} className="timeline__label">
            {s.threshold}s<br />
            <small>{s.label}</small>
          </span>
        ))}
      </div>
    </div>
  );
}

function IndicatorPill({ label, active }) {
  return (
    <div className={`indicator-pill ${active ? 'indicator-pill--active' : ''}`}>
      <span className={`indicator-dot ${active ? 'indicator-dot--on' : ''}`} />
      {label}
    </div>
  );
}

export default App;
