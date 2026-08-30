import { useEffect, useState } from "react";
import * as api from "./api";

// validated colorblind-safe categorical palette (dark surface): blue/aqua/yellow/violet/red
const COLORS = ["#3987e5", "#199e70", "#c98500", "#9085e9", "#e66767"];

// ---- ROC curves (multi-series line chart) ----
function RocChart({ evalData }) {
  const W = 360, H = 360, pad = 44;
  const x = (fpr) => pad + fpr * (W - pad - 12);
  const y = (tpr) => H - pad - tpr * (H - pad - 12);
  const labels = evalData.labels;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="eval-svg" role="img"
      aria-label="ROC curves per pathology">
      {/* grid + axes */}
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={x(t)} y1={y(0)} x2={x(t)} y2={y(1)} className="grid" />
          <line x1={x(0)} y1={y(t)} x2={x(1)} y2={y(t)} className="grid" />
          <text x={x(t)} y={H - pad + 16} className="axis-lbl" textAnchor="middle">{t}</text>
          <text x={pad - 8} y={y(t) + 3} className="axis-lbl" textAnchor="end">{t}</text>
        </g>
      ))}
      {/* chance diagonal */}
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} className="diag" />
      {/* curves */}
      {labels.map((lab, i) => {
        const roc = evalData.per_label[lab].roc;
        if (!roc?.length) return null;
        const d = roc.map(([f, t], k) => `${k ? "L" : "M"}${x(f).toFixed(1)},${y(t).toFixed(1)}`).join(" ");
        return <path key={lab} d={d} fill="none" stroke={COLORS[i]} strokeWidth="2" />;
      })}
      <text x={W / 2} y={H - 8} className="axis-title" textAnchor="middle">False Positive Rate</text>
      <text x={14} y={H / 2} className="axis-title" textAnchor="middle"
        transform={`rotate(-90 14 ${H / 2})`}>True Positive Rate</text>
    </svg>
  );
}

// ---- AUROC bars (magnitude, single accent, direct-labeled) ----
function AurocBars({ evalData }) {
  const labels = evalData.labels;
  const rows = labels.map((l, i) => ({ label: l, auc: evalData.per_label[l].auroc, color: COLORS[i] }));
  const max = 1.0;
  return (
    <div className="auroc-bars">
      {rows.map((r) => (
        <div className="abar" key={r.label}>
          <div className="abar-label">{r.label}</div>
          <div className="abar-track">
            <div className="abar-fill" style={{ width: `${(r.auc / max) * 100}%`, background: r.color }} />
          </div>
          <div className="abar-val">{r.auc.toFixed(3)}</div>
        </div>
      ))}
      <div className="abar mean">
        <div className="abar-label"><b>Mean</b></div>
        <div className="abar-track">
          <div className="abar-fill mean" style={{ width: `${evalData.mean_auroc * 100}%` }} />
        </div>
        <div className="abar-val"><b>{evalData.mean_auroc.toFixed(3)}</b></div>
      </div>
    </div>
  );
}

// ---- metrics table (confusion @ threshold) ----
function MetricsTable({ evalData }) {
  const thr = evalData.threshold;
  return (
    <div className="table-scroll">
      <table className="metrics-table">
        <thead>
          <tr>
            <th>Pathology</th><th>AUROC</th><th>Sens.</th><th>Spec.</th><th>Acc.</th>
            <th>TP</th><th>FP</th><th>TN</th><th>FN</th>
          </tr>
        </thead>
        <tbody>
          {evalData.labels.map((l, i) => {
            const p = evalData.per_label[l], c = p.confusion;
            return (
              <tr key={l}>
                <td><span className="dot" style={{ background: COLORS[i] }} />{l}</td>
                <td>{p.auroc.toFixed(3)}</td>
                <td>{c.sensitivity.toFixed(2)}</td>
                <td>{c.specificity.toFixed(2)}</td>
                <td>{c.accuracy.toFixed(2)}</td>
                <td>{c.tp}</td><td>{c.fp}</td><td>{c.tn}</td><td>{c.fn}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="muted sm" style={{ marginTop: 8 }}>
        Sensitivity/specificity computed at threshold {thr}. Low sensitivity at 0.5 is
        expected for imbalanced data — AUROC (threshold-independent) is the primary metric.
      </div>
    </div>
  );
}

export default function Evaluation({ onClose }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    api.evaluation().then(setData).catch((e) => setErr(e.message));
  }, []);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h2>Model Evaluation</h2>
          <button className="btn ghost sm" onClick={onClose}>✕ Close</button>
        </div>

        {err && <div className="empty">⚠️ {err}</div>}
        {!data && !err && <div className="empty"><span className="spinner" /> loading…</div>}

        {data && (
          <div className="eval-body">
            <div className="eval-summary">
              <div className="big-metric">
                <div className="bm-value">{data.mean_auroc.toFixed(3)}</div>
                <div className="bm-label">mean AUROC</div>
              </div>
              <div className="eval-meta">
                <div><b>{data.model.arch}</b> · {data.model.trained_on}</div>
                <div className="muted sm">{data.model.epochs} epochs · evaluated on {data.n_valid} validation images</div>
                <div className="muted sm">5 thoracic pathologies · CheXpert validation set</div>
              </div>
            </div>

            <div className="eval-grid">
              <div className="eval-card">
                <h3>ROC curves</h3>
                <div className="legend">
                  {data.labels.map((l, i) => (
                    <span key={l} className="leg"><span className="dot" style={{ background: COLORS[i] }} />{l}</span>
                  ))}
                </div>
                <RocChart evalData={data} />
              </div>
              <div className="eval-card">
                <h3>AUROC per pathology</h3>
                <AurocBars evalData={data} />
              </div>
            </div>

            <div className="eval-card wide">
              <h3>Per-pathology metrics</h3>
              <MetricsTable evalData={data} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
