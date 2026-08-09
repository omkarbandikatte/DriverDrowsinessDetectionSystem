import { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';

const API_BASE = window.location.hostname === 'localhost'
  ? 'http://localhost:8000'
  : 'https://driverguard-backend.onrender.com';
const WS_BASE = API_BASE.replace(/^https/, 'wss').replace(/^http/, 'ws');
const IS_CLOUD = window.location.hostname !== 'localhost';

function App() {
  // --- State ---
  const [isRunning, setIsRunning] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [sourceType, setSourceType] = useState(IS_CLOUD ? 'ip' : 'system');
  const [cameraIndex, setCameraIndex] = useState('0');
  const [ipUrl, setIpUrl] = useState('');
  const [earThreshold, setEarThreshold] = useState(0.25);
  const [marThreshold, setMarThreshold] = useState(0.75);
  const [alarmEnabled, setAlarmEnabled] = useState(true);
  const [smsEnabled, setSmsEnabled] = useState(true);
  const [callEnabled, setCallEnabled] = useState(true);
  const [metrics, setMetrics] = useState({});
  const [error, setError] = useState('');

  const wsRef = useRef(null);

  // --- WebSocket for live metrics ---
  const connectWebSocket = useCallback(() => {
    if (wsRef.current) wsRef.current.close();

    const ws = new WebSocket(`${WS_BASE}/ws/metrics`);
    ws.onmessage = (event) => {
      try {
        setMetrics(JSON.parse(event.data));
      } catch (e) { /* ignore parse errors */ }
    };
    ws.onerror = () => {};
    ws.onclose = () => {
      // Reconnect after 2s if still running
      setTimeout(() => {
        if (isRunning) connectWebSocket();
      }, 2000);
    };
    wsRef.current = ws;
  }, [isRunning]);

  useEffect(() => {
    if (isRunning) {
      connectWebSocket();
    } else if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, [isRunning, connectWebSocket]);

  // --- API calls ---
  const startMonitoring = async () => {
    setIsLoading(true);
    setError('');
    const source = sourceType === 'system' ? cameraIndex : ipUrl;
    if (sourceType === 'ip' && !ipUrl.trim()) {
      setError('Please enter an IP camera URL');
      setIsLoading(false);
      return;
    }
    try {
      const res = await fetch(`${API_BASE}/api/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source,
          ear_threshold: earThreshold,
          mar_threshold: marThreshold,
        }),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || 'Failed to start');
      }
      setIsRunning(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const stopMonitoring = async () => {
    setIsLoading(true);
    try {
      await fetch(`${API_BASE}/api/stop`, { method: 'POST' });
      setIsRunning(false);
      setMetrics({});
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const resetState = async () => {
    try {
      await fetch(`${API_BASE}/api/reset`, { method: 'POST' });
    } catch (err) {
      setError(err.message);
    }
  };

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
  const fps = metrics.fps ?? 0;

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
                  src={`${API_BASE}/api/video-feed`}
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

          {/* Source Selection */}
          <div className="card source-card animate-in">
            <h3 className="card__title">📡 Video Source</h3>
            <div className="source-tabs">
              <button
                className={`source-tab ${sourceType === 'system' ? 'source-tab--active' : ''}`}
                onClick={() => setSourceType('system')}
                disabled={isRunning || IS_CLOUD}
                title={IS_CLOUD ? 'System camera is not available on the cloud deployment' : ''}
              >
                💻 System Camera
              </button>
              <button
                className={`source-tab ${sourceType === 'ip' ? 'source-tab--active' : ''}`}
                onClick={() => setSourceType('ip')}
                disabled={isRunning}
              >
                📱 IP Camera
              </button>
            </div>
            {IS_CLOUD && (
              <p className="error-text" style={{ marginTop: '8px', color: 'var(--accent-yellow)' }}>
                ⚠ Cloud deployment: only IP Camera is supported. Use the <a href="https://play.google.com/store/apps/details?id=com.pas.webcam" target="_blank" rel="noreferrer" style={{ color: 'inherit' }}>IP Webcam</a> app and enter your phone&rsquo;s stream URL below.
              </p>
            )}

            {sourceType === 'system' ? (
              <div className="source-field">
                <label className="field-label">Camera Index</label>
                <select
                  className="select-field"
                  value={cameraIndex}
                  onChange={(e) => setCameraIndex(e.target.value)}
                  disabled={isRunning}
                >
                  <option value="0">Camera 0 (Default)</option>
                  <option value="1">Camera 1</option>
                  <option value="2">Camera 2</option>
                </select>
              </div>
            ) : (
              <div className="source-field">
                <label className="field-label">IP Camera URL</label>
                <input
                  className="input-field"
                  type="text"
                  placeholder="http://192.168.1.5:8080/video"
                  value={ipUrl}
                  onChange={(e) => setIpUrl(e.target.value)}
                  disabled={isRunning}
                />
              </div>
            )}

            {/* Action Buttons */}
            <div className="source-actions">
              {!isRunning ? (
                <button
                  className="btn btn--primary btn--lg"
                  onClick={startMonitoring}
                  disabled={isLoading}
                >
                  {isLoading ? '⏳ Connecting...' : '▶ Start Monitoring'}
                </button>
              ) : (
                <>
                  <button
                    className="btn btn--danger"
                    onClick={stopMonitoring}
                    disabled={isLoading}
                  >
                    ⏹ Stop
                  </button>
                  <button className="btn btn--ghost" onClick={resetState}>
                    ↺ Reset
                  </button>
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
