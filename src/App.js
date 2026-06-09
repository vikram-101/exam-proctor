import { useState, useEffect, useRef, useCallback } from "react";

const STATUS = {
  IDLE: "idle",
  LOADING: "loading",
  READY: "ready",
  MONITORING: "monitoring",
  ALERT: "alert",
};

const ALERTS = {
  MULTIPLE_FACES: "Multiple faces detected",
  NO_FACE: "Face not visible",
  LOOKING_AWAY: "Looking away from screen",
};

export default function ExamProctor() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const modelRef = useRef(null);
  const animFrameRef = useRef(null);
  const audioCtxRef = useRef(null);
  const alarmIntervalRef = useRef(null);

  const [status, setStatus] = useState(STATUS.IDLE);
  const [alertReason, setAlertReason] = useState("");
  const [isAlarming, setIsAlarming] = useState(false);
  const [logs, setLogs] = useState([]);
  const [faceCount, setFaceCount] = useState(0);
  const [lookingAway, setLookingAway] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [noFaceSeconds, setNoFaceSeconds] = useState(0);
  const [loadError, setLoadError] = useState("");
  const isAlarmingRef = useRef(false);

  const addLog = useCallback((msg, type = "info") => {
    const time = new Date().toLocaleTimeString();
    setLogs((prev) => [{ msg, type, time }, ...prev].slice(0, 15));
  }, []);

  useEffect(() => {
    setStatus(STATUS.LOADING);
    setLoadError("");
    const tfScript = document.createElement("script");
    tfScript.src = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js";
    tfScript.onload = () => {
      const bfScript = document.createElement("script");
      bfScript.src = "https://cdn.jsdelivr.net/npm/@tensorflow-models/blazeface@0.0.7/dist/blazeface.min.js";
      bfScript.onload = async () => {
        const model = await window.blazeface.load();
        modelRef.current = model;
        setModelLoaded(true);
        setStatus(STATUS.READY);
        addLog("AI model ready", "success");
      };
      bfScript.onerror = () => {
        const msg = "Failed to load BlazeFace model script.";
        setLoadError(msg);
        setStatus(STATUS.IDLE);
        addLog(msg, "alert");
      };
      document.body.appendChild(bfScript);
    };
    tfScript.onerror = () => {
      const msg = "Failed to load TensorFlow.js script.";
      setLoadError(msg);
      setStatus(STATUS.IDLE);
      addLog(msg, "alert");
    };
    document.body.appendChild(tfScript);
  }, [addLog]);

  const playBeep = useCallback(() => {
    try {
      if (!audioCtxRef.current)
        audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
      const ctx = audioCtxRef.current;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type = "square";
      osc.frequency.setValueAtTime(900, ctx.currentTime);
      osc.frequency.setValueAtTime(600, ctx.currentTime + 0.15);
      gain.gain.setValueAtTime(0.25, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.4);
    } catch (e) {}
  }, []);

  const triggerAlarm = useCallback((reason) => {
    if (isAlarmingRef.current) return;
    isAlarmingRef.current = true;
    setAlertReason(reason);
    setIsAlarming(true);
    setStatus(STATUS.ALERT);
    addLog(reason, "alert");
    playBeep();
    alarmIntervalRef.current = setInterval(playBeep, 900);
  }, [addLog, playBeep]);

  const stopAlarm = useCallback(() => {
    isAlarmingRef.current = false;
    setIsAlarming(false);
    setAlertReason("");
    setStatus(STATUS.MONITORING);
    if (alarmIntervalRef.current) {
      clearInterval(alarmIntervalRef.current);
      alarmIntervalRef.current = null;
    }
    addLog("Alert dismissed", "info");
  }, [addLog]);

  const isLookingAway = (face) => {
    const lm = face.landmarks;
    if (!lm || lm.length < 3) return false;
    const cx = (face.topLeft[0] + face.bottomRight[0]) / 2;
    const fw = face.bottomRight[0] - face.topLeft[0];
    const nose = lm[2];
    return Math.abs((nose[0] - cx) / fw) > 0.25;
  };

  const noFaceRef = useRef(0);

  const detect = useCallback(async () => {
    if (!modelRef.current || !videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    const preds = await modelRef.current.estimateFaces(video, false);
    setFaceCount(preds.length);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0);

    preds.forEach((face) => {
      const [x1, y1] = face.topLeft;
      const [x2, y2] = face.bottomRight;
      const w = x2 - x1, h = y2 - y1;
      const away = isLookingAway(face);
      const multi = preds.length > 1;

      // Glow effect
      ctx.shadowColor = away || multi ? "#ef4444" : "#6366f1";
      ctx.shadowBlur = 12;
      ctx.strokeStyle = away || multi ? "#ef4444" : "#6366f1";
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, w, h);
      ctx.shadowBlur = 0;

      // Corner brackets
      const cs = 14;
      ctx.strokeStyle = away || multi ? "#fca5a5" : "#a5b4fc";
      ctx.lineWidth = 3;
      [[x1,y1],[x2-cs,y1],[x1,y2-cs],[x2-cs,y2-cs]].forEach(([bx,by],i) => {
        ctx.beginPath();
        ctx.moveTo(bx + (i%2===0?0:cs), by);
        ctx.lineTo(bx + (i%2===0?cs:0), by);
        ctx.moveTo(bx, by + (i<2?cs:0));
        ctx.lineTo(bx, by + (i<2?0:cs));
        ctx.stroke();
      });

      if (face.landmarks) {
        face.landmarks.forEach(([lx, ly], i) => {
          ctx.fillStyle = i === 2 ? "#f59e0b" : "#818cf8";
          ctx.beginPath();
          ctx.arc(lx, ly, 2.5, 0, 2 * Math.PI);
          ctx.fill();
        });
      }
    });

    if (!isAlarmingRef.current) {
      if (preds.length > 1) {
        triggerAlarm(ALERTS.MULTIPLE_FACES);
      } else if (preds.length === 1 && isLookingAway(preds[0])) {
        setLookingAway(true);
        triggerAlarm(ALERTS.LOOKING_AWAY);
      } else {
        setLookingAway(false);
      }
      if (preds.length === 0) {
        noFaceRef.current += 1;
        setNoFaceSeconds(noFaceRef.current);
        if (noFaceRef.current >= 3) triggerAlarm(ALERTS.NO_FACE);
      } else {
        noFaceRef.current = 0;
        setNoFaceSeconds(0);
      }
    }

    animFrameRef.current = requestAnimationFrame(detect);
  }, [triggerAlarm]);

  const startMonitoring = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setStatus(STATUS.MONITORING);
      addLog("Monitoring started", "success");
      animFrameRef.current = requestAnimationFrame(detect);
    } catch (e) {
      addLog("Camera access denied", "alert");
    }
  };

  const stopMonitoring = () => {
    cancelAnimationFrame(animFrameRef.current);
    if (videoRef.current?.srcObject)
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
    if (alarmIntervalRef.current) clearInterval(alarmIntervalRef.current);
    isAlarmingRef.current = false;
    setStatus(STATUS.READY);
    setIsAlarming(false);
    setFaceCount(0);
    setNoFaceSeconds(0);
    noFaceRef.current = 0;
    addLog("Monitoring stopped", "info");
  };

  const isMonitoring = status === STATUS.MONITORING || status === STATUS.ALERT;

  return (
    <div style={{
      minHeight: "100vh",
      background: "#07070f",
      color: "#e2e8f0",
      fontFamily: "'Inter', 'Segoe UI', sans-serif",
      display: "flex",
      flexDirection: "column",
    }}>
      {/* Top navbar */}
      <nav style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "14px 28px",
        borderBottom: "1px solid #1e1e2e",
        background: "#0d0d1a",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: "linear-gradient(135deg,#6366f1,#8b5cf6)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16,
          }}>🎓</div>
          <span style={{ fontWeight: 700, fontSize: 16, letterSpacing: 0.5 }}>ExamShield</span>
          <span style={{
            fontSize: 10, padding: "2px 8px", borderRadius: 99,
            background: "#1e1e2e", color: "#6366f1", letterSpacing: 1,
          }}>AI PROCTOR</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{
            width: 8, height: 8, borderRadius: "50%",
            background: isAlarming ? "#ef4444" : isMonitoring ? "#22c55e" : "#374151",
            boxShadow: isMonitoring ? "0 0 6px #22c55e" : "none",
          }} />
          <span style={{ fontSize: 12, color: "#64748b" }}>
            {isAlarming ? "ALERT" : isMonitoring ? "LIVE" : status === STATUS.LOADING ? "Loading..." : "Standby"}
          </span>
        </div>
      </nav>

      {loadError && (
        <div style={{
          background: "#7f1d1d",
          color: "#fde2e2",
          padding: "10px 28px",
          borderBottom: "1px solid #991b1b",
          fontSize: 13,
        }}>
          {loadError}
        </div>
      )}

      {/* Alert strip */}
      {isAlarming && (
        <div style={{
          background: "linear-gradient(90deg,#450a0a,#7f1d1d,#450a0a)",
          borderBottom: "1px solid #991b1b",
          padding: "10px 28px",
          display: "flex", alignItems: "center", justifyContent: "space-between",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 18 }}>🚨</span>
            <div>
              <div style={{ fontWeight: 700, color: "#fca5a5", fontSize: 13 }}>VIOLATION DETECTED</div>
              <div style={{ color: "#fecaca", fontSize: 12 }}>{alertReason}</div>
            </div>
          </div>
          <button onClick={stopAlarm} style={{
            background: "#ef4444", color: "#fff", border: "none",
            borderRadius: 6, padding: "6px 16px", cursor: "pointer",
            fontWeight: 700, fontSize: 12, letterSpacing: 0.5,
          }}>DISMISS</button>
        </div>
      )}

      {/* Main content */}
      <div style={{
        flex: 1, display: "grid",
        gridTemplateColumns: "1fr 280px",
        gap: 0,
        maxWidth: 960, width: "100%",
        margin: "0 auto", padding: "24px",
        boxSizing: "border-box",
        alignItems: "start",
      }}>
        {/* Camera area */}
        <div style={{ paddingRight: 20 }}>
          <div style={{
            position: "relative",
            borderRadius: 16,
            overflow: "hidden",
            background: "#0d0d1a",
            border: `1px solid ${isAlarming ? "#7f1d1d" : "#1e1e2e"}`,
            aspectRatio: "16/10",
            boxShadow: isAlarming
              ? "0 0 40px rgba(239,68,68,0.15)"
              : isMonitoring
              ? "0 0 40px rgba(99,102,241,0.08)"
              : "none",
            transition: "box-shadow 0.4s, border-color 0.4s",
          }}>
            <video ref={videoRef} style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              objectFit: "cover",
              display: isMonitoring ? "block" : "none",
            }} muted playsInline />
            <canvas ref={canvasRef} style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              objectFit: "cover",
              pointerEvents: "none",
              display: isMonitoring ? "block" : "none",
            }} />

            {!isMonitoring && (
              <div style={{
                position: "absolute", inset: 0,
                display: "flex", flexDirection: "column",
                alignItems: "center", justifyContent: "center", gap: 12,
              }}>
                <div style={{
                  width: 72, height: 72, borderRadius: "50%",
                  background: "#1e1e2e",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 28,
                }}>📷</div>
                <div style={{ color: "#374151", fontSize: 13, letterSpacing: 1 }}>
                  {status === STATUS.LOADING ? "Loading AI model..." : "Camera offline"}
                </div>
                {status === STATUS.LOADING && (
                  <div style={{
                    width: 120, height: 3, background: "#1e1e2e", borderRadius: 3, overflow: "hidden",
                  }}>
                    <div style={{
                      height: "100%", width: "40%",
                      background: "linear-gradient(90deg,#6366f1,#8b5cf6)",
                      borderRadius: 3,
                      animation: "slide 1.2s ease-in-out infinite",
                    }} />
                  </div>
                )}
              </div>
            )}

            {/* Face count badge */}
            {isMonitoring && (
              <div style={{
                position: "absolute", top: 12, right: 12,
                background: "rgba(0,0,0,0.75)",
                backdropFilter: "blur(8px)",
                border: `1px solid ${faceCount === 1 ? "#6366f1" : "#ef4444"}`,
                borderRadius: 8, padding: "6px 12px",
                fontSize: 12, display: "flex", alignItems: "center", gap: 6,
              }}>
                <span style={{ color: faceCount === 1 ? "#818cf8" : "#fca5a5" }}>👤</span>
                <span style={{ color: faceCount === 1 ? "#c7d2fe" : "#fca5a5", fontWeight: 600 }}>
                  {faceCount} {faceCount === 1 ? "face" : "faces"}
                </span>
              </div>
            )}
          </div>

          {/* Controls */}
          <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
            <button
              onClick={startMonitoring}
              disabled={!modelLoaded || isMonitoring}
              style={{
                flex: 1, padding: "12px 0",
                background: modelLoaded && !isMonitoring
                  ? "linear-gradient(135deg,#6366f1,#8b5cf6)"
                  : "#1e1e2e",
                color: modelLoaded && !isMonitoring ? "#fff" : "#374151",
                border: "none", borderRadius: 10,
                cursor: modelLoaded && !isMonitoring ? "pointer" : "not-allowed",
                fontWeight: 700, fontSize: 13, letterSpacing: 0.5,
                transition: "opacity 0.2s",
              }}
            >
              {status === STATUS.LOADING ? "Loading model..." : "▶  Start Monitoring"}
            </button>
            <button
              onClick={stopMonitoring}
              disabled={!isMonitoring}
              style={{
                padding: "12px 20px",
                background: isMonitoring ? "#1e1e2e" : "#12121a",
                color: isMonitoring ? "#f87171" : "#374151",
                border: `1px solid ${isMonitoring ? "#7f1d1d" : "#1e1e2e"}`,
                borderRadius: 10, cursor: isMonitoring ? "pointer" : "not-allowed",
                fontWeight: 700, fontSize: 13,
              }}
            >
              ■ Stop
            </button>
          </div>
        </div>

        {/* Right sidebar */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {/* Stats cards */}
          <div style={{
            background: "#0d0d1a", border: "1px solid #1e1e2e",
            borderRadius: 14, padding: 16,
          }}>
            <div style={{ fontSize: 10, color: "#4b5563", letterSpacing: 2, marginBottom: 12, fontWeight: 600 }}>
              LIVE METRICS
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {[
                {
                  label: "Faces Detected",
                  value: faceCount,
                  icon: "👤",
                  ok: faceCount === 1,
                  bad: faceCount !== 1 && isMonitoring,
                },
                {
                  label: "Gaze Direction",
                  value: lookingAway ? "Away" : "Screen",
                  icon: "👁️",
                  ok: !lookingAway,
                  bad: lookingAway,
                },
                {
                  label: "No-Face Timer",
                  value: `${noFaceSeconds}s / 3s`,
                  icon: "⏱️",
                  ok: noFaceSeconds === 0,
                  bad: noFaceSeconds >= 2,
                },
              ].map(({ label, value, icon, ok, bad }) => (
                <div key={label} style={{
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                  padding: "10px 12px",
                  background: "#12121a",
                  border: `1px solid ${bad ? "#7f1d1d" : ok ? "#14532d" : "#1e1e2e"}`,
                  borderRadius: 8,
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ fontSize: 14 }}>{icon}</span>
                    <span style={{ fontSize: 12, color: "#94a3b8" }}>{label}</span>
                  </div>
                  <span style={{
                    fontSize: 13, fontWeight: 700,
                    color: bad ? "#f87171" : ok ? "#4ade80" : "#64748b",
                  }}>{value}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Rules */}
          <div style={{
            background: "#0d0d1a", border: "1px solid #1e1e2e",
            borderRadius: 14, padding: 16,
          }}>
            <div style={{ fontSize: 10, color: "#4b5563", letterSpacing: 2, marginBottom: 12, fontWeight: 600 }}>
              RULES
            </div>
            {[
              { icon: "👥", text: "Multiple faces → Alarm" },
              { icon: "👁️", text: "Looking away → Alarm" },
              { icon: "🚫", text: "No face for 3s → Alarm" },
            ].map(({ icon, text }) => (
              <div key={text} style={{
                display: "flex", gap: 10, alignItems: "center",
                fontSize: 12, color: "#64748b",
                padding: "7px 0",
                borderBottom: "1px solid #12121a",
              }}>
                <span>{icon}</span><span>{text}</span>
              </div>
            ))}
          </div>

          {/* Log */}
          <div style={{
            background: "#0d0d1a", border: "1px solid #1e1e2e",
            borderRadius: 14, padding: 16,
          }}>
            <div style={{ fontSize: 10, color: "#4b5563", letterSpacing: 2, marginBottom: 10, fontWeight: 600 }}>
              EVENT LOG
            </div>
            <div style={{ maxHeight: 180, overflowY: "auto", display: "flex", flexDirection: "column", gap: 4 }}>
              {logs.length === 0 && (
                <div style={{ fontSize: 12, color: "#1f2937" }}>No events yet...</div>
              )}
              {logs.map((log, i) => (
                <div key={i} style={{
                  display: "flex", gap: 8, alignItems: "flex-start",
                  fontSize: 11, padding: "4px 0",
                  borderBottom: "1px solid #12121a",
                }}>
                  <span style={{ color: "#374151", flexShrink: 0 }}>{log.time}</span>
                  <span style={{
                    color: log.type === "alert" ? "#fca5a5"
                      : log.type === "success" ? "#86efac"
                      : "#64748b",
                  }}>{log.msg}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes slide {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(400%); }
        }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #12121a; }
        ::-webkit-scrollbar-thumb { background: #2d2d3d; border-radius: 2px; }
      `}</style>
    </div>
  );
}
