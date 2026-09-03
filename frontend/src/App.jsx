import { useEffect, useRef, useState } from "react";
import * as api from "./api";
import Evaluation from "./Evaluation.jsx";

const fmtPct = (v) => `${Math.round(v * 100)}%`;

function PathologyRow({ label, value, active, positive, onClick }) {
  return (
    <button
      className={`prow ${active ? "active" : ""} ${positive ? "pos" : ""}`}
      onClick={onClick}
      title="Show Grad-CAM for this finding"
    >
      <div className="prow-top">
        <span className="prow-label">{label}</span>
        <span className="prow-pct">{fmtPct(value)}</span>
      </div>
      <div className="prow-track">
        <div className="prow-fill" style={{ width: `${value * 100}%` }} />
      </div>
    </button>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [analysis, setAnalysis] = useState(null);

  // Grad-CAM viewer
  const [camLabel, setCamLabel] = useState(null);
  const [camOverlay, setCamOverlay] = useState(null);
  const [camBusy, setCamBusy] = useState(false);
  const [opacity, setOpacity] = useState(0.55);

  const [report, setReport] = useState(null);
  const [busy, setBusy] = useState(false);
  const [reportBusy, setReportBusy] = useState(false);
  const [error, setError] = useState(null);
  const [showEval, setShowEval] = useState(false);
  const [rejected, setRejected] = useState(null);
  const [chatOpen, setChatOpen] = useState(true);
  const fileInput = useRef(null);

  // chat
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatBusy, setChatBusy] = useState(false);
  const chatLogRef = useRef(null);

  useEffect(() => {
    api.health().catch(() => setError("Backend not reachable — start FastAPI on :8010"));
  }, []);

  function scrollChatToBottom() {
    requestAnimationFrame(() => {
      const el = chatLogRef.current;
      if (el) el.scrollTop = el.scrollHeight;
    });
  }

  // keep the newest message in view as it streams in
  useEffect(scrollChatToBottom, [messages, chatBusy]);

  function pickFile(f) {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setAnalysis(null); setReport(null); setError(null); setRejected(null);
    setCamLabel(null); setCamOverlay(null); setMessages([]);
  }

  function clearFile() {
    setFile(null); setPreview(null); setAnalysis(null); setReport(null);
    setCamLabel(null); setCamOverlay(null); setMessages([]); setRejected(null);
    if (fileInput.current) fileInput.current.value = "";
  }

  async function runAnalyze() {
    if (!file) return;
    setBusy(true); setError(null); setReport(null); setRejected(null);
    try {
      const a = await api.analyze(file);
      setAnalysis(a);
      setCamLabel(a.overlay_label);
      setCamOverlay(a.overlay);
    } catch (e) {
      if (e.rejected) {
        setRejected(e.message);
        setAnalysis(null); setPreview(null); setFile(null);
        setCamLabel(null); setCamOverlay(null);
      } else {
        setError(e.message);
      }
    }
    setBusy(false);
  }

  async function showCam(label) {
    if (label === camLabel) return;
    setCamLabel(label);
    if (label === analysis?.overlay_label) { setCamOverlay(analysis.overlay); return; }
    setCamBusy(true);
    try {
      const res = await api.gradcam(file, label);
      setCamOverlay(res.overlay);
    } catch (e) { setError(e.message); }
    setCamBusy(false);
  }

  async function runReport() {
    if (!analysis) return;
    setReportBusy(true); setError(null);
    try {
      const res = await api.report(analysis.findings, "ollama");
      setReport(res.report);
    } catch (e) { setError(e.message); }
    setReportBusy(false);
  }

  async function sendChat(preset) {
    const text = (preset ?? chatInput).trim();
    if (!text || chatBusy) return;
    const history = messages.map(({ role, content }) => ({ role, content }));
    const next = [...messages, { role: "user", content: text }];
    setMessages(next);
    setChatInput("");
    setChatBusy(true);
    try {
      const res = await api.chat(text, analysis?.findings || {}, history);
      setMessages([...next, { role: "assistant", content: res.answer, sources: res.sources }]);
    } catch (e) {
      setMessages([...next, { role: "assistant", content: `⚠️ ${e.message}`, error: true }]);
    }
    setChatBusy(false);
    scrollChatToBottom();
  }

  async function downloadPdf() {
    const blob = await api.pdf(report);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "chest_xray_report.pdf"; a.click();
    URL.revokeObjectURL(url);
  }

  const probsSorted = analysis?.probs
    ? Object.entries(analysis.probs).sort((a, b) => b[1] - a[1])
    : [];
  const threshold = analysis?.findings?.threshold ?? 0.5;
  const positives = probsSorted.filter(([, v]) => v >= threshold);
  const considered = probsSorted.filter(([, v]) => v < threshold);
  const top = probsSorted[0];

  const sizeMb = file ? (file.size / 1024 / 1024).toFixed(1) : null;

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>Chest X-ray Analysis</h1>
          <div className="sub">Disease classification · Grad-CAM · AI radiology assistant — by Anagha</div>
        </div>
        <button className="btn ghost" onClick={() => setShowEval(true)}>📊 Evaluation</button>
      </header>

      {showEval && <Evaluation onClose={() => setShowEval(false)} />}

      <div className="disclaimer">
        ⚠️ Research / education demo — <strong>not a diagnostic tool</strong>.
      </div>

      <div className="layout">
        {/* ---------------- LEFT: upload ---------------- */}
        <section className="panel upload-col">
          <div className="panel-head col">
            <h2>Upload Chest X-ray</h2>
            <div className="muted sm">Drag and drop your image here, or click to browse</div>
          </div>

          {rejected && (
            <div className="guard-reject">
              ❌ <strong>Image rejected.</strong> {rejected}
            </div>
          )}

          <div
            className="dropzone"
            onClick={() => fileInput.current?.click()}
            onDrop={(e) => { e.preventDefault(); pickFile(e.dataTransfer.files[0]); }}
            onDragOver={(e) => e.preventDefault()}
          >
            <input ref={fileInput} type="file" accept="image/*" hidden
              onChange={(e) => pickFile(e.target.files[0])} />
            <div className="dz-icon">🫁</div>
            <div className="dz-title">Drag &amp; Drop Chest X-ray</div>
            <div className="hint">PNG / JPG · frontal chest X-ray only</div>
            <span className="btn ghost sm dz-btn">Browse Files</span>
          </div>

          {preview && (
            <div className="uploaded">
              <div className="uploaded-head">Uploaded Image</div>
              <div className="uploaded-row">
                <img src={preview} alt="x-ray thumbnail" className="thumb" />
                <div className="uploaded-meta">
                  <div className="fname" title={file?.name}>{file?.name}</div>
                  <div className="muted sm">{sizeMb} MB</div>
                  <div className="ok sm">✓ Upload complete</div>
                </div>
                <button className="icon-btn" title="Remove image" onClick={clearFile}>🗑</button>
              </div>
              <button className="btn accent full" onClick={runAnalyze} disabled={busy}>
                {busy ? <span className="spinner" /> : analysis ? "Re-analyze" : "Analyze X-ray"}
              </button>
            </div>
          )}
        </section>

        {/* ---------------- RIGHT: results ---------------- */}
        <section className="panel results-col">
          <div className="panel-head">
            <h2>AI Analysis Results</h2>
            {analysis && <span className="chip sm">Analyzed just now</span>}
          </div>

          {!analysis && (
            <div className="empty tall">
              {preview
                ? <>Click <b>Analyze X-ray</b> to detect pathologies.</>
                : <>Upload a chest X-ray to see results here.</>}
            </div>
          )}

          {analysis && (
            <>
              {/* headline: top finding + confidence */}
              <div className="headline">
                <div className="headline-cell">
                  <div className="hl-label">Top Finding</div>
                  <div className={`hl-value ${top[1] >= threshold ? "pos" : ""}`}>
                    {top[0]}{top[1] >= threshold ? " Detected" : " (below threshold)"}
                  </div>
                </div>
                <div className="headline-cell">
                  <div className="hl-label">Confidence Score</div>
                  <div className="hl-conf">
                    <span className="hl-pct">{fmtPct(top[1])}</span>
                    <div className="conf-track"><div className="conf-fill" style={{ width: `${top[1] * 100}%` }} /></div>
                  </div>
                  <div className="conf-scale"><span>0%</span><span>50%</span><span>100%</span></div>
                </div>
              </div>

              {/* summary */}
              <div className="summary">
                <div className="hl-label">Summary</div>
                <p>
                  {positives.length > 0 ? (
                    <>The model flagged <strong>{positives.length}</strong>{" "}
                      {positives.length === 1 ? "finding" : "findings"} at or above the{" "}
                      {fmtPct(threshold)} threshold:{" "}
                      <strong>{positives.map(([k]) => k).join(", ")}</strong>. Select any pathology
                      below to see where the model looked. Correlate clinically before drawing
                      conclusions.</>
                  ) : (
                    <>No pathology reached the {fmtPct(threshold)} threshold. The highest score was{" "}
                      <strong>{top[0]}</strong> at {fmtPct(top[1])}. Select any pathology below to
                      inspect its Grad-CAM heatmap.</>
                  )}
                </p>
              </div>

              {/* side-by-side viewer */}
              <div className="viewer-grid">
                <figure className="vpane">
                  <figcaption>Original X-ray</figcaption>
                  <div className="img-frame"><img src={preview} alt="original x-ray" className="base-img" /></div>
                </figure>
                <figure className="vpane">
                  <figcaption>Grad-CAM Heatmap {camLabel && <span className="muted">· {camLabel}</span>}</figcaption>
                  <div className="img-frame">
                    <img src={preview} alt="x-ray" className="base-img" />
                    {camOverlay && <img src={camOverlay} alt="grad-cam" className="cam-img" style={{ opacity }} />}
                    {camBusy && <div className="cam-loading"><span className="spinner" /></div>}
                  </div>
                  <div className="opacity-ctl">
                    <span className="muted sm">Heatmap</span>
                    <input type="range" min="0" max="1" step="0.05" value={opacity}
                      onChange={(e) => setOpacity(parseFloat(e.target.value))} />
                    <span className="muted sm">{Math.round(opacity * 100)}%</span>
                  </div>
                </figure>
              </div>

              {/* all pathologies */}
              <div className="findings-block">
                <div className="group-label pos">Positive ({positives.length})</div>
                {positives.length === 0
                  ? <div className="muted sm pad">None above threshold.</div>
                  : positives.map(([k, v]) => (
                      <PathologyRow key={k} label={k} value={v} positive
                        active={k === camLabel} onClick={() => showCam(k)} />
                    ))}
                <div className="group-label">Considered</div>
                {considered.map(([k, v]) => (
                  <PathologyRow key={k} label={k} value={v}
                    active={k === camLabel} onClick={() => showCam(k)} />
                ))}
              </div>

              {/* report */}
              <div className="report-inline">
                <button className="btn accent full" onClick={runReport} disabled={reportBusy}>
                  {reportBusy ? <span className="spinner" /> : report ? "Regenerate report" : "📄 Generate Radiology Report"}
                </button>
                {report != null && (
                  <>
                    <textarea className="report-edit" value={report}
                      onChange={(e) => setReport(e.target.value)} rows={8} />
                    <div className="row end">
                      <button className="btn ghost sm" onClick={downloadPdf}>⬇️ Download Analysis Report (PDF)</button>
                    </div>
                  </>
                )}
              </div>

              {/* assistant */}
              <div className="assistant">
                <button className="assistant-head" onClick={() => setChatOpen((o) => !o)}>
                  <span className="ah-title">✦ AI Assistant</span>
                  <span className="ah-meta muted sm">RAG · nomic-embed · llama3.1</span>
                  <span className="ah-caret">{chatOpen ? "⌃" : "⌄"}</span>
                </button>

                {chatOpen && (
                  <>
                    <div className="chat-log" ref={chatLogRef}>
                      <div className="msg assistant">
                        <div className="msg-body">Hello! I'm your AI assistant. Ask me anything about this X-ray.</div>
                      </div>
                      {messages.length === 0 && (
                        <div className="chat-suggest">
                          {["What are the signs of the top finding in this X-ray?",
                            "How confident is the model in this prediction?",
                            "What other conditions can cause similar patterns?"].map((q) => (
                            <button key={q} className="suggest" onClick={() => sendChat(q)}>{q}</button>
                          ))}
                        </div>
                      )}
                      {messages.map((m, i) => (
                        <div key={i} className={`msg ${m.role} ${m.error ? "err" : ""}`}>
                          <div className="msg-body">{m.content}</div>
                          {m.sources?.length > 0 && (
                            <div className="msg-sources">
                              {m.sources.map((s) => (
                                <span key={s.pathology + s.section} className="src-chip">
                                  {s.pathology} · {s.section}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                      {chatBusy && <div className="msg assistant"><span className="spinner" /></div>}
                    </div>

                    <div className="chat-input">
                      <input value={chatInput} onChange={(e) => setChatInput(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && sendChat()}
                        placeholder="Type your question…" />
                      <button className="btn icon-send" onClick={() => sendChat()}
                        disabled={chatBusy || !chatInput.trim()} title="Send">→</button>
                    </div>
                  </>
                )}
              </div>
            </>
          )}
        </section>
      </div>

      <footer className="foot">
        🛡 AI analysis for educational and informational purposes only. Not a substitute for professional medical advice.
      </footer>

      {error && <div className="toast error">{error}</div>}
    </div>
  );
}
