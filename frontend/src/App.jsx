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
  const fileInput = useRef(null);

  // chat
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatBusy, setChatBusy] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    api.health().catch(() => setError("Backend not reachable — start FastAPI on :8010"));
  }, []);

  function pickFile(f) {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setAnalysis(null); setReport(null); setError(null); setRejected(null);
    setCamLabel(null); setCamOverlay(null); setMessages([]);
  }

  const [rejected, setRejected] = useState(null);

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
        // not a chest X-ray — block: drop the image, show rejection, no results
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

  async function sendChat() {
    const text = chatInput.trim();
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
    setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
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

      {!preview && (
        <>
          {rejected && (
            <div className="guard-reject">
              ❌ <strong>Image rejected.</strong> {rejected}
            </div>
          )}
          <div className="dropzone big" onClick={() => fileInput.current?.click()}
            onDrop={(e) => { e.preventDefault(); pickFile(e.dataTransfer.files[0]); }}
            onDragOver={(e) => e.preventDefault()}>
            <input ref={fileInput} type="file" accept="image/*"
              onChange={(e) => pickFile(e.target.files[0])} />
            <div className="dz-icon">🫁</div>
            <div className="dz-title">Drop a chest X-ray, or click to browse</div>
            <div className="hint">PNG / JPG · frontal chest X-ray only</div>
          </div>
        </>
      )}

      {preview && (
        <>
          {/* TOP STRIP — compact imaging + findings */}
          <div className="strip">
            <section className="panel viewer compact">
              <div className="panel-head">
                <h2>Imaging</h2>
                <div className="row gap">
                  <button className="btn ghost sm" onClick={() => fileInput.current?.click()}>Change</button>
                  <input ref={fileInput} type="file" accept="image/*" hidden
                    onChange={(e) => pickFile(e.target.files[0])} />
                  <button className="btn sm" onClick={runAnalyze} disabled={busy}>
                    {busy ? <span className="spinner" /> : analysis ? "Re-analyze" : "Analyze"}
                  </button>
                </div>
              </div>
              <div className="img-frame">
                <img src={preview} alt="x-ray" className="base-img" />
                {camOverlay && <img src={camOverlay} alt="grad-cam" className="cam-img" style={{ opacity }} />}
                {camBusy && <div className="cam-loading"><span className="spinner" /></div>}
                {camLabel && <div className="cam-tag">Grad-CAM · {camLabel}</div>}
              </div>
              {analysis && (
                <div className="opacity-ctl">
                  <span className="muted sm">Heatmap</span>
                  <input type="range" min="0" max="1" step="0.05" value={opacity}
                    onChange={(e) => setOpacity(parseFloat(e.target.value))} />
                  <span className="muted sm">{Math.round(opacity * 100)}%</span>
                </div>
              )}
            </section>

            <section className="panel findings compact">
              <div className="panel-head"><h2>Findings</h2>
                {analysis && <span className="muted sm">threshold {fmtPct(threshold)}</span>}
              </div>
              {!analysis && <div className="empty">Run <b>Analyze</b> to detect pathologies.</div>}
              {analysis && (
                <div className="findings-scroll">
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
                  <div className="report-inline">
                    <button className="btn accent sm full" onClick={runReport} disabled={reportBusy}>
                      {reportBusy ? <span className="spinner" /> : report ? "Regenerate report" : "Generate report"}
                    </button>
                    {report != null && (
                      <>
                        <textarea className="report-edit" value={report}
                          onChange={(e) => setReport(e.target.value)} rows={6} />
                        <div className="row end"><button className="btn ghost sm" onClick={downloadPdf}>⬇️ PDF</button></div>
                      </>
                    )}
                  </div>
                </div>
              )}
            </section>
          </div>

          {/* MAIN — full-width chat */}
          <section className="panel chat-main">
            <div className="panel-head">
              <h2>Ask about this X-ray</h2>
              <span className="chip sm">RAG · nomic-embed · llama3.1</span>
            </div>

            <div className="chat-log big">
              {messages.length === 0 && (
                <div className="chat-hint">
                  <div className="muted">Ask about the findings or the pathologies. Try:</div>
                  <div className="chat-suggest row">
                    {["Which findings were flagged and why?",
                      "What is pleural effusion?",
                      "What does the Grad-CAM heatmap show?",
                      "Should the top finding be a concern?"].map((q) => (
                      <button key={q} className="suggest" onClick={() => setChatInput(q)}>{q}</button>
                    ))}
                  </div>
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
              <div ref={chatEndRef} />
            </div>

            <div className="chat-input">
              <input value={chatInput} onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendChat()}
                placeholder={analysis ? "Ask a question about this X-ray…" : "Analyze an image first, then ask…"} />
              <button className="btn" onClick={sendChat} disabled={chatBusy || !chatInput.trim()}>Send</button>
            </div>
          </section>
        </>
      )}

      {error && <div className="toast error">{error}</div>}
    </div>
  );
}
