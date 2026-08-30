// All requests go through Vite's /api proxy -> FastAPI at :8000
const base = "/api";

export async function health() {
  const r = await fetch(`${base}/health`);
  if (!r.ok) throw new Error("backend not reachable");
  return r.json();
}

export async function model() {
  const r = await fetch(`${base}/model`);
  if (!r.ok) throw new Error("model info unavailable");
  return r.json();
}

export async function evaluation() {
  const r = await fetch(`${base}/evaluation`);
  if (!r.ok) throw new Error("evaluation data unavailable — run evaluate_local.py");
  return r.json();
}

export async function analyze(file) {
  const form = new FormData();
  form.append("file", file);
  const r = await fetch(`${base}/analyze`, { method: "POST", body: form });
  if (!r.ok) {
    // 422 = rejected non-X-ray; detail is our structured {error, message, guard}
    let msg = `analyze failed (${r.status})`;
    try {
      const body = await r.json();
      const d = body.detail;
      if (d && typeof d === "object" && d.error === "not_a_chest_xray") {
        const err = new Error(d.message);
        err.rejected = true;
        throw err;
      }
      msg = d?.message || d || msg;
    } catch (e) {
      if (e.rejected) throw e;
    }
    throw new Error(msg);
  }
  return r.json();
}

export async function gradcam(file, label) {
  const form = new FormData();
  form.append("file", file);
  form.append("label", label);
  const r = await fetch(`${base}/gradcam`, { method: "POST", body: form });
  if (!r.ok) throw new Error(`gradcam failed (${r.status})`);
  return r.json();
}

export async function report(findings, provider) {
  const r = await fetch(`${base}/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ findings, provider }),
  });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    throw new Error(detail.detail || `report failed (${r.status})`);
  }
  return r.json();
}

export async function chat(message, findings, history) {
  const r = await fetch(`${base}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, findings, history }),
  });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    throw new Error(detail.detail || `chat failed (${r.status})`);
  }
  return r.json();
}

export async function pdf(reportText) {
  const r = await fetch(`${base}/pdf`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ report: reportText }),
  });
  if (!r.ok) throw new Error("pdf failed");
  return r.blob();
}
