import { useState, useEffect, useRef } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import './App.css';

const API_BASE = window.location.hostname === 'localhost'
  ? 'http://localhost:8000'
  : 'https://driverguard-backend.onrender.com';

// Detection constants — identical to Python backend
const EAR_THRESH_DEFAULT = 0.25;
const MAR_THRESH_DEFAULT = 0.75;
const CONSEC_FRAMES = 20;
const LEFT_EYE  = [362, 385, 387, 263, 373, 380];
const RIGHT_EYE = [33, 160, 158, 133, 153, 144];
const MOUTH     = [78, 81, 13, 311, 308, 402, 14, 178];

function dist(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}
function computeEAR(lms, idx) {
  const p = idx.map(i => lms[i]);
  return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2 * dist(p[0], p[3]));
}
function computeMAR(lms, idx) {
  const p = idx.map(i => lms[i]);
  return (dist(p[1], p[7]) + dist(p[2], p[6]) + dist(p[3], p[5])) / (2 * dist(p[0], p[4]));
}
function drawContour(ctx, lms, indices, W, H, color) {
  const pts = indices.map(i => [lms[i].x * W, lms[i].y * H]);
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
  ctx.closePath();
  ctx.stroke();
}
function getEscalationLevel(sec) {
  if (sec <= 0) return 0;
  if (sec >= 20) return 3;
  if (sec >= 10) return 2;
  return 1;
}

function App() {
  const [isRunning,      setIsRunning]      = useState(false);
  const [isLoading,      setIsLoading]      = useState(false);
  const [modelReady,     setModelReady]     = useState(false);
  const [earThreshold,   setEarThreshold]   = useState(EAR_THRESH_DEFAULT);
  const [marThreshold,   setMarThreshold]   = useState(MAR_THRESH_DEFAULT);
  const [alarmEnabled,   setAlarmEnabled]   = useState(true);
  const [smsEnabled,     setSmsEnabled]     = useState(true);
  const [callEnabled,    setCallEnabled]    = useState(true);
  const [metrics,        setMetrics]        = useState({});
  const [error,          setError]          = useState('');
  const [fps,            setFps]            = useState(0);

  const videoRef        = useRef(null);
  const canvasRef       = useRef(null);
  const landmarkerRef   = useRef(null);
  const animRef         = useRef(null);
  const isRunningRef    = useRef(false);
  const frameCountRef   = useRef(0);
  const isDrowsyRef     = useRef(false);
  const drowsyStartRef  = useRef(null);
  const smsSentRef      = useRef(false);
  const callMadeRef     = useRef(false);
  const lastAlarmRef    = useRef(null);
  const fpsCountRef     = useRef(0);
  const earThreshRef    = useRef(EAR_THRESH_DEFAULT);
  const marThreshRef    = useRef(MAR_THRESH_DEFAULT);
  const alarmEnabledRef = useRef(true);
  const smsEnabledRef   = useRef(true);

  useEffect(() => { earThreshRef.current   = earThreshold;   }, [earThreshold]);
  useEffect(() => { marThreshRef.current   = marThreshold;   }, [marThreshold]);
  useEffect(() => { alarmEnabledRef.current = alarmEnabled;  }, [alarmEnabled]);
  useEffect(() => { smsEnabledRef.current  = smsEnabled;     }, [smsEnabled]);

  // Client-side FPS counter
  useEffect(() => {
    const id = setInterval(() => {
      setFps(fpsCountRef.current);
      fpsCountRef.current = 0;
    }, 1000);
    return () => clearInterval(id);
  }, []);

  // Load MediaPipe FaceLandmarker model once on mount
  useEffect(() => {
    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
        );
        landmarkerRef.current = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numFaces: 1,
          minFaceDetectionConfidence: 0.5,
          minFacePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        setModelReady(true);
      } catch (e) {
        setError('Failed to load face detection model. Please refresh.');
      }
    })();
  }, []);

  // Browser alarm via Web Audio API
  const playAlarm = (level) => {
    if (!alarmEnabledRef.current) return;
    try {
      const ac   = new (window.AudioContext || window.webkitAudioContext)();
      const osc  = ac.createOscillator();
      const gain = ac.createGain();
      osc.connect(gain);
      gain.connect(ac.destination);
      osc.type = 'square';
      osc.frequency.value = level >= 2 ? 1100 : 880;
      gain.gain.setValueAtTime(level >= 2 ? 0.25 : 0.12, ac.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ac.currentTime + 0.45);
      osc.start();
      osc.stop(ac.currentTime + 0.45);
    } catch (_) {}
  };

  // Detection loop — stored in a ref so requestAnimationFrame always calls the latest closure
  const detectFnRef = useRef(null);
  detectFnRef.current = (ts) => {
    if (!isRunningRef.current) return;
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !landmarkerRef.current || video.readyState < 2) {
      animRef.current = requestAnimationFrame(t => detectFnRef.current(t));
      return;
    }

    const W = canvas.width;
    const H = canvas.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, W, H);

    const results = landmarkerRef.current.detectForVideo(video, ts);

    let m = {
      face_detected: false, ear: null, mar: null,
      is_drowsy: isDrowsyRef.current, is_yawning: false,
      frame_counter: frameCountRef.current, drowsy_duration: 0,
      escalation_level: 0, sms_sent: smsSentRef.current, call_made: callMadeRef.current,
    };

    if (results.faceLandmarks?.length > 0) {
      const lms = results.faceLandmarks[0];
      m.face_detected = true;

      const ear = (computeEAR(lms, LEFT_EYE) + computeEAR(lms, RIGHT_EYE)) / 2;
      const mar = computeMAR(lms, MOUTH);
      m.ear = parseFloat(ear.toFixed(3));
      m.mar = parseFloat(mar.toFixed(3));
      m.is_yawning = mar > marThreshRef.current;

      if (ear < earThreshRef.current) {
        frameCountRef.current++;
        if (frameCountRef.current >= CONSEC_FRAMES) {
          isDrowsyRef.current = true;
          if (!drowsyStartRef.current) drowsyStartRef.current = Date.now();
        }
      } else {
        frameCountRef.current = 0;
        isDrowsyRef.current   = false;
        drowsyStartRef.current = null;
        smsSentRef.current    = false;
        callMadeRef.current   = false;
      }

      m.is_drowsy     = isDrowsyRef.current;
      m.frame_counter = frameCountRef.current;

      const drowsySec      = drowsyStartRef.current ? Math.round((Date.now() - drowsyStartRef.current) / 1000) : 0;
      m.drowsy_duration    = drowsySec;
      m.escalation_level   = getEscalationLevel(drowsySec);

      // Draw eye and mouth contours
      drawContour(ctx, lms, LEFT_EYE,  W, H, '#22c55e');
      drawContour(ctx, lms, RIGHT_EYE, W, H, '#22c55e');
      drawContour(ctx, lms, MOUTH,     W, H, '#06b6d4');

      // HUD text
      ctx.font      = 'bold 13px monospace';
      ctx.fillStyle = ear < earThreshRef.current ? '#ef4444' : '#22c55e';
      ctx.fillText(`EAR: ${ear.toFixed(3)}`, 8, 22);
      ctx.fillStyle = mar > marThreshRef.current ? '#f59e0b' : '#22c55e';
      ctx.fillText(`MAR: ${mar.toFixed(3)}`, 8, 40);

      if (isDrowsyRef.current) {
        ctx.font      = 'bold 20px monospace';
        ctx.fillStyle = '#ef4444';
        ctx.fillText('\u26a0 DROWSY', W / 2 - 60, 34);

        const now = Date.now();
        if (!lastAlarmRef.current || now - lastAlarmRef.current > 2000) {
          playAlarm(m.escalation_level || 1);
          lastAlarmRef.current = now;
        }

        // POST to backend for Twilio SMS when escalation level 3
        if (m.escalation_level >= 3 && smsEnabledRef.current && !smsSentRef.current) {
          smsSentRef.current = true;
          m.sms_sent = true;
          fetch(`${API_BASE}/api/escalate`, { method: 'POST' }).catch(() => {});
        }
      }
    }

    setMetrics(m);
    fpsCountRef.current++;
    animRef.current = requestAnimationFrame(t => detectFnRef.current(t));
  };

  const startMonitoring = async () => {
    if (!modelReady) { setError('Face detection model still loading, please wait…'); return; }
    setIsLoading(true);
    setError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
      });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      const canvas  = canvasRef.current;
      canvas.width  = videoRef.current.videoWidth  || 640;
      canvas.height = videoRef.current.videoHeight || 480;

      frameCountRef.current  = 0;
      isDrowsyRef.current    = false;
      drowsyStartRef.current = null;
      smsSentRef.current     = false;
      callMadeRef.current    = false;
      lastAlarmRef.current   = null;

      isRunningRef.current = true;
      setIsRunning(true);
      setIsLoading(false);
      animRef.current = requestAnimationFrame(t => detectFnRef.current(t));
    } catch (err) {
      setError(
        err.name === 'NotAllowedError'
          ? 'Camera access denied. Please allow camera permission and try again.'
          : err.message
      );
      setIsLoading(false);
    }
  };

  const stopMonitoring = () => {
    isRunningRef.current = false;
    cancelAnimationFrame(animRef.current);
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop());
      videoRef.current.srcObject = null;
    }
    setIsRunning(false);
    setMetrics({});
  };

  const resetState = () => {
    frameCountRef.current  = 0;
    isDrowsyRef.current    = false;
    drowsyStartRef.current = null;
    smsSentRef.current     = false;
    callMadeRef.current    = false;
    setMetrics({});
  };

  // Derived display values
  const ear             = metrics.ear ?? null;
  const mar             = metrics.mar ?? null;
  const isDrowsy        = metrics.is_drowsy ?? false;
  const isYawning       = metrics.is_yawning ?? false;
  const faceDetected    = metrics.face_detected ?? false;
  const escalationLevel = metrics.escalation_level ?? 0;
  const drowsyDuration  = metrics.drowsy_duration ?? 0;

  const statusLabel =
    !isRunning       ? (modelReady ? 'READY' : 'LOADING…')  :
    !faceDetected    ? 'NO FACE'    :
    escalationLevel === 3 ? 'CRITICAL'  :
    escalationLevel === 2 ? 'HIGH ALERT':
    isDrowsy         ? 'DROWSY'     : 'AWAKE';

  const statusColor =
    statusLabel === 'AWAKE'                                     ? 'green'  :
    statusLabel === 'DROWSY'                                    ? 'yellow' :
    statusLabel === 'HIGH ALERT' || statusLabel === 'CRITICAL'  ? 'red'    :
    'indigo';

  return (
    <div className="app">
      <video ref={videoRef} style={{ display: 'none' }} playsInline muted />

      <header className="header">
        <div className="header__brand">
          <div className="header__icon">\ud83d\ude97</div>
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

      <main className="main">
        <section className="main__left">
          <div className={`video-card card ${isDrowsy && isRunning ? 'video-card--alert' : ''}`}>
            <div className="video-card__header">
              <span className="video-card__label">
                {isRunning ? <><span className="live-dot" /> Live Feed</> : 'Camera Preview'}
              </span>
              {isRunning && isDrowsy  && <span className="badge badge--red pulse">\u26a0 DROWSY</span>}
              {isRunning && isYawning && <span className="badge badge--yellow">\ud83d\ude2e YAWN</span>}
            </div>
            <div className="video-card__feed">
              <canvas
                ref={canvasRef}
                className="video-card__img"
                style={{ display: isRunning ? 'block' : 'none' }}
              />
              {!isRunning && (
                <div className="video-card__placeholder">
                  <div className="video-card__placeholder-icon">\ud83d\udcf7</div>
                  <p>{modelReady ? 'Click Start Monitoring to begin' : 'Loading face detection model\u2026'}</p>
                </div>
              )}
            </div>
          </div>

          <div className="card source-card animate-in">
            <h3 className="card__title">\ud83d\udce1 Browser Webcam</h3>
            <p className="card__hint">
              Runs entirely in your browser \u2014 real-time, no setup, works on any device worldwide.
            </p>
            <div className="source-actions">
              {!isRunning ? (
                <button
                  className="btn btn--primary btn--lg"
                  onClick={startMonitoring}
                  disabled={isLoading || !modelReady}
                >
                  {isLoading ? '\u23f3 Requesting camera\u2026' : !modelReady ? '\u23f3 Loading model\u2026' : '\u25b6 Start Monitoring'}
                </button>
              ) : (
                <>
                  <button className="btn btn--danger" onClick={stopMonitoring}>\u23f9 Stop</button>
                  <button className="btn btn--ghost"  onClick={resetState}>\u21ba Reset</button>
                </>
              )}
            </div>
            {error && <p className="error-text">\u26a0 {error}</p>}
          </div>
        </section>

        <section className="main__right">
          <div className="metrics-grid animate-in">
            <MetricCard
              label="EAR" value={ear !== null ? ear.toFixed(3) : '\u2014'} sub="Eye Aspect Ratio"
              color={ear !== null && ear < earThreshold ? 'red' : 'green'}
              threshold={earThreshold} current={ear} type="below"
            />
            <MetricCard
              label="MAR" value={mar !== null ? mar.toFixed(3) : '\u2014'} sub="Mouth Aspect Ratio"
              color={mar !== null && mar > marThreshold ? 'yellow' : 'green'}
              threshold={marThreshold} current={mar} type="above"
            />
            <MetricCard
              label="Closed" value={metrics.frame_counter ?? 0} sub="Consecutive Frames"
              color={metrics.frame_counter > 15 ? 'red' : 'indigo'}
            />
            <MetricCard
              label="Duration" value={drowsyDuration > 0 ? `${drowsyDuration}s` : '0s'} sub="Drowsy Duration"
              color={drowsyDuration > 10 ? 'red' : drowsyDuration > 0 ? 'yellow' : 'green'}
            />
          </div>

          <div className="card escalation-card animate-in" style={{ animationDelay: '0.1s' }}>
            <h3 className="card__title">\u23f1 Escalation Timeline</h3>
            <EscalationTimeline level={escalationLevel} duration={drowsyDuration} />
            <div className="escalation-indicators">
              <IndicatorPill label="SMS Sent"  active={metrics.sms_sent}  />
              <IndicatorPill label="Call Made" active={metrics.call_made} />
            </div>
          </div>

          <div className="card settings-card animate-in" style={{ animationDelay: '0.2s' }}>
            <h3 className="card__title">\u2699 Detection Settings</h3>
            <div className="setting-row">
              <div className="setting-info">
                <span className="setting-label">EAR Threshold</span>
                <span className="setting-value mono">{earThreshold.toFixed(2)}</span>
              </div>
              <input type="range" className="slider" min="0.15" max="0.40" step="0.01"
                value={earThreshold} onChange={e => setEarThreshold(parseFloat(e.target.value))} />
            </div>
            <div className="setting-row">
              <div className="setting-info">
                <span className="setting-label">MAR Threshold</span>
                <span className="setting-value mono">{marThreshold.toFixed(2)}</span>
              </div>
              <input type="range" className="slider" min="0.50" max="1.00" step="0.01"
                value={marThreshold} onChange={e => setMarThreshold(parseFloat(e.target.value))} />
            </div>
            <div className="setting-toggle-row">
              <span className="setting-label">\ud83d\udd14 Alarm Sound</span>
              <label className="toggle">
                <input type="checkbox" checked={alarmEnabled} onChange={e => setAlarmEnabled(e.target.checked)} />
                <span className="toggle-slider" />
              </label>
            </div>
            <div className="setting-toggle-row">
              <span className="setting-label">\ud83d\udcf1 SMS Escalation</span>
              <label className="toggle">
                <input type="checkbox" checked={smsEnabled} onChange={e => setSmsEnabled(e.target.checked)} />
                <span className="toggle-slider" />
              </label>
            </div>
            <div className="setting-toggle-row">
              <span className="setting-label">\ud83d\udcde Call Escalation</span>
              <label className="toggle">
                <input type="checkbox" checked={callEnabled} onChange={e => setCallEnabled(e.target.checked)} />
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
      <span className="metric-card__value mono">{value}</span>
      <span className="metric-card__sub">{sub}</span>
      {percent !== null && (
        <div className="metric-card__bar">
          <div className={`metric-card__bar-fill metric-card__bar-fill--${color}`} style={{ width: `${percent}%` }} />
        </div>
      )}
    </div>
  );
}

function EscalationTimeline({ level, duration }) {
  const stages = [
    { label: 'Normal', threshold: 0,  color: 'green'  },
    { label: 'High',   threshold: 10, color: 'yellow' },
    { label: 'SMS',    threshold: 20, color: 'orange' },
    { label: 'Call',   threshold: 25, color: 'red'    },
  ];
  return (
    <div className="timeline">
      <div className="timeline__track">
        {stages.map((s, i) => (
          <div key={i} className={`timeline__segment timeline__segment--${s.color} ${duration >= s.threshold ? 'timeline__segment--active' : ''}`} />
        ))}
      </div>
      <div className="timeline__labels">
        {stages.map((s, i) => (
          <span key={i} className="timeline__label">{s.threshold}s<br /><small>{s.label}</small></span>
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
