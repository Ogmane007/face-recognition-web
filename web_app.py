import base64
import io
import time
from typing import List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image

from app.config import settings
from app.db import FaceDB
from app.face_engine import FaceEngine

from dataclasses import dataclass
from threading import Lock


# -----------------------
# Models
# -----------------------

class FrameIn(BaseModel):
    image_b64: str  # data URL or plain base64
    threshold: Optional[float] = None

class EnrollIn(BaseModel):
    name: str
    image_b64: str
    client_id: str

# -----------------------
# Utils
# -----------------------

def b64_to_bgr(image_b64: str) -> np.ndarray:
    """
    Accepts either:
    - "data:image/jpeg;base64,...."
    - raw base64 "...."
    Returns BGR numpy image for OpenCV/InsightFace.
    """
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    img_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)
    bgr = rgb[:, :, ::-1].copy()
    return bgr

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# -----------------------
# App init
# -----------------------

app = FastAPI(title="Face Recognition Web App")

db = FaceDB(settings.db_path)
engine = FaceEngine(settings.model_name)

# In-memory stable tracking per browser session would need session IDs.
# For demo simplicity we do per-frame recognition only in API.
# If you want the SAME 5-frame stability in web mode too, tell me and I'll add it.

# -----------------------
# Log table (simple)
# -----------------------

import sqlite3

def init_logs():
    with sqlite3.connect(settings.db_path) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            name TEXT NOT NULL,
            score REAL NOT NULL
        );
        """)
        con.commit()

def add_log(name: str, score: float):
    with sqlite3.connect(settings.db_path) as con:
        con.execute("INSERT INTO logs(ts, name, score) VALUES(?, ?, ?);",
                    (int(time.time()), name, float(score)))
        con.commit()

def read_logs(limit: int = 200):
    with sqlite3.connect(settings.db_path) as con:
        cur = con.execute(
            "SELECT ts, name, score FROM logs ORDER BY id DESC LIMIT ?;",
            (int(limit),)
        )
        rows = cur.fetchall()
    return [{"ts": ts, "name": name, "score": score} for (ts, name, score) in rows]

init_logs()

# -----------------------
# Frontend (served by FastAPI)
# -----------------------

INDEX_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Face Recognition Web</title>
  <style>
    body { font-family: system-ui, Arial; margin: 0; background:#0b1220; color:#e7eefc; }
    header { padding: 14px 18px; background:#0f1a33; display:flex; gap:14px; align-items:center; }
    header a { color:#8fb7ff; text-decoration:none; }
    .wrap { display:grid; grid-template-columns: 1.2fr 0.8fr; gap: 16px; padding: 16px; }
    .card { background:#0f1a33; border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:14px; }
    .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
    input, button { border-radius:12px; padding:10px 12px; border:1px solid rgba(255,255,255,.15); background:#0b1220; color:#e7eefc; }
    button { cursor:pointer; }
    button.primary { background:#2c6bff; border-color:#2c6bff; }
    canvas { width:100%; border-radius:16px; background:#000; }
    small { opacity:.8; }
    .pill { padding:6px 10px; border-radius:999px; background:rgba(255,255,255,.08); }
    table { width:100%; border-collapse: collapse; font-size: 14px; }
    td, th { border-bottom:1px solid rgba(255,255,255,.08); padding:8px 6px; text-align:left; }
  </style>
</head>
<body>
<header>
  <div style="font-weight:700;">Face Recognition Web</div>
  <a href="/">Live</a>
  <a href="/users">Users</a>
  <a href="/logs">Logs</a>
</header>

<div class="wrap">
  <div class="card">
    <div class="row" style="justify-content:space-between;">
      <div class="row">
        <span class="pill" id="status">Idle</span>
        <span class="pill">FPS: <span id="fps">0</span></span>
        <span class="pill">Faces: <span id="faces">0</span></span>
      </div>
      <div class="row">
        <label><small>Threshold</small><br/>
          <input id="thr" type="number" step="0.01" value="0.45" style="width:100px;">
        </label>
        <button id="start" class="primary">Start</button>
        <button id="stop">Stop</button>
      </div>
    </div>

    <div style="margin-top:12px;">
      <video id="video" autoplay playsinline style="display:none;"></video>
      <canvas id="canvas" width="1280" height="720"></canvas>
    </div>

    <div class="row" style="margin-top:12px;">
      <input id="name" placeholder="Name to enroll" style="width:220px;">
      <button id="enroll" class="primary">Enroll (snap)</button>
      <small id="msg"></small>
    </div>
  </div>

  <div class="card">
    <div style="font-weight:700; margin-bottom:10px;">Last results</div>
    <div id="results"><small>No results yet.</small></div>
  </div>
</div>

<script>
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const fpsEl = document.getElementById("fps");
const facesEl = document.getElementById("faces");
const resultsEl = document.getElementById("results");
const msgEl = document.getElementById("msg");

let running = false;
let lastT = performance.now();
let frames = 0;

function setStatus(t){ statusEl.textContent = t; }

async function startCam(){
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
  video.srcObject = stream;
  await new Promise(r => video.onloadedmetadata = r);
}

function dataUrlFromCanvas(){
  // JPEG reduces payload size
  return canvas.toDataURL("image/jpeg", 0.75);
}

function drawBoxes(dets){
  // Draw overlay on top of current canvas
  for (const d of dets){
    const [x1,y1,x2,y2] = d.bbox;
    const name = d.name;
    const score = d.score.toFixed(2);

    // box
    ctx.lineWidth = 3;
    ctx.strokeStyle = (name !== "UNKNOWN") ? "lime" : "red";
    ctx.strokeRect(x1, y1, x2-x1, y2-y1);

    // label background
    const label = `${name} ${score}`;
    ctx.font = "20px system-ui";
    const tw = ctx.measureText(label).width;
    const th = 22;
    ctx.fillStyle = (name !== "UNKNOWN") ? "rgba(0,255,0,0.9)" : "rgba(255,0,0,0.9)";
    ctx.fillRect(x1, Math.max(0, y1 - th - 10), tw + 14, th + 10);

    // label text
    ctx.fillStyle = "black";
    ctx.fillText(label, x1 + 7, Math.max(20, y1 - 10));
  }
}

function renderResults(dets){
  if (!dets.length){
    resultsEl.innerHTML = "<small>No face detected.</small>";
    return;
  }
  resultsEl.innerHTML = dets.map(d => `
    <div class="pill" style="margin:6px 0; display:block;">
      <b>${d.name}</b> — ${d.score.toFixed(2)}
      <small style="opacity:.8;"> (bbox: ${d.bbox.join(", ")})</small>
    </div>
  `).join("");
}

async function tick(){
  if (!running) return;

  // draw current frame to canvas
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // send to backend
  const thr = parseFloat(document.getElementById("thr").value || "0.45");
  const payload = { image_b64: dataUrlFromCanvas(), threshold: thr };

  try{
    const res = await fetch("/api/recognize", {
      method:"POST",
      headers:{ "Content-Type":"application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    const dets = data.detections || [];
    facesEl.textContent = dets.length;
    renderResults(dets);
    drawBoxes(dets);
    setStatus("Running");
  }catch(e){
    setStatus("Error (check server)");
    console.error(e);
  }

  // FPS
  frames++;
  const now = performance.now();
  if (now - lastT >= 1000){
    fpsEl.textContent = frames;
    frames = 0;
    lastT = now;
  }

  requestAnimationFrame(tick);
}

document.getElementById("start").onclick = async () => {
  if (running) return;
  try{
    await startCam();
    running = true;
    tick();
  }catch(e){
    setStatus("Camera error");
    console.error(e);
  }
};

document.getElementById("stop").onclick = () => {
  running = false;
  if (video.srcObject){
    video.srcObject.getTracks().forEach(t => t.stop());
  }
  setStatus("Stopped");
};

document.getElementById("enroll").onclick = async () => {
  const name = document.getElementById("name").value.trim();
  if (!name){
    msgEl.textContent = "Enter a name first.";
    return;
  }

  const image_b64 = dataUrlFromCanvas();
  const client_id = "web-ui";

  try{
    const res = await fetch("/api/enroll", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, image_b64, client_id })
    });
    const data = await res.json();
    if (data.error){
      msgEl.textContent = data.error;
    } else {
      msgEl.textContent = data.message || "Enrolled!";
      document.getElementById("name").value = "";
    }
  }catch(e){
    msgEl.textContent = "Enroll failed.";
    console.error(e);
  }
};
</script>
</body>
</html>
"""

USERS_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Users - Face Recognition</title>
  <style>
    body { font-family: system-ui, Arial; margin: 0; background:#0b1220; color:#e7eefc; }
    header { padding: 14px 18px; background:#0f1a33; display:flex; gap:14px; align-items:center; }
    header a { color:#8fb7ff; text-decoration:none; }
    .wrap { max-width: 900px; margin: 20px auto; padding: 0 16px; }
    .card { background:#0f1a33; border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:14px; }
    table { width:100%; border-collapse: collapse; }
    td, th { border-bottom:1px solid rgba(255,255,255,.08); padding:8px 6px; text-align:left; }
    button, input { border-radius:12px; padding:10px 12px; border:1px solid rgba(255,255,255,.15); background:#0b1220; color:#e7eefc; }
    button { cursor:pointer; }
  </style>
</head>
<body>
<header>
  <div style="font-weight:700;">Face Recognition Web</div>
  <a href="/">Live</a>
  <a href="/users">Users</a>
  <a href="/logs">Logs</a>
</header>
<div class="wrap">
  <div class="card">
    <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
      <h3 style="margin:0;">Enrolled Users</h3>
      <button id="refresh">Refresh</button>
    </div>
    <table id="tbl" style="margin-top:10px;">
      <thead><tr><th>Name</th><th>Samples</th><th>Action</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>
<script>
async function loadUsers(){
  const res = await fetch("/api/users");
  const data = await res.json();
  const tbody = document.querySelector("#tbl tbody");
  tbody.innerHTML = "";
  for (const u of data.users){
    const tr = document.createElement("tr");
    const delBtn = document.createElement("button");
    delBtn.textContent = "Delete";
    delBtn.onclick = async () => {
      if (!confirm(`Delete '${u.name}' ?`)) return;
      await fetch(`/api/users/${encodeURIComponent(u.name)}`, { method: "DELETE" });
      loadUsers();
    };
    const td1 = document.createElement("td");
    td1.textContent = u.name;
    const td2 = document.createElement("td");
    td2.textContent = u.samples;
    const td3 = document.createElement("td");
    td3.appendChild(delBtn);
    tr.appendChild(td1);
    tr.appendChild(td2);
    tr.appendChild(td3);
    tbody.appendChild(tr);
  }
}
document.getElementById("refresh").onclick = loadUsers;
loadUsers();
</script>
</body>
</html>
"""

LOGS_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Logs - Face Recognition</title>
  <style>
    body { font-family: system-ui, Arial; margin: 0; background:#0b1220; color:#e7eefc; }
    header { padding: 14px 18px; background:#0f1a33; display:flex; gap:14px; align-items:center; }
    header a { color:#8fb7ff; text-decoration:none; }
    .wrap { max-width: 900px; margin: 20px auto; padding: 0 16px; }
    .card { background:#0f1a33; border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:14px; }
    table { width:100%; border-collapse: collapse; }
    td, th { border-bottom:1px solid rgba(255,255,255,.08); padding:8px 6px; text-align:left; }
    button, input { border-radius:12px; padding:10px 12px; border:1px solid rgba(255,255,255,.15); background:#0b1220; color:#e7eefc; }
    button { cursor:pointer; }
  </style>
</head>
<body>
<header>
  <div style="font-weight:700;">Face Recognition Web</div>
  <a href="/">Live</a>
  <a href="/users">Users</a>
  <a href="/logs">Logs</a>
</header>
<div class="wrap">
  <div class="card">
    <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
      <h3 style="margin:0;">Recognition Logs</h3>
      <span style="opacity:.8;">(latest first)</span>
      <button id="refresh">Refresh</button>
    </div>
    <table id="tbl" style="margin-top:10px;">
      <thead><tr><th>Time</th><th>Name</th><th>Score</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>
<script>
function fmt(ts){
  const d = new Date(ts*1000);
  return d.toLocaleString();
}
async function loadLogs(){
  const res = await fetch("/api/logs?limit=200");
  const data = await res.json();
  const tbody = document.querySelector("#tbl tbody");
  tbody.innerHTML = "";
  for (const r of data.logs){
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${fmt(r.ts)}</td><td>${r.name}</td><td>${r.score.toFixed(2)}</td>`;
    tbody.appendChild(tr);
  }
}
document.getElementById("refresh").onclick = loadLogs;
loadLogs();
</script>
</body>
</html>
"""

# -----------------------
# Routes (Pages)
# -----------------------

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(INDEX_HTML)

@app.get("/users", response_class=HTMLResponse)
def users_page():
    return HTMLResponse(USERS_HTML)

@app.get("/logs", response_class=HTMLResponse)
def logs_page():
    return HTMLResponse(LOGS_HTML)


# -----------------------
# 5-frame stability state
# -----------------------

def iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = a1 + a2 - inter + 1e-9
    return inter / union

def assign_tracks(tracks, det_bboxes, iou_thresh=0.30):
    """
    Greedy assignment: best IoU first
    tracks: dict[int, Track]
    det_bboxes: list[(x1,y1,x2,y2)]
    """
    if not tracks:
        return [], list(range(len(det_bboxes)))

    pairs = []
    for tid, tr in tracks.items():
        for di, bb in enumerate(det_bboxes):
            pairs.append((iou(tr.bbox, bb), tid, di))

    pairs.sort(reverse=True, key=lambda x: x[0])

    used_tracks = set()
    used_dets = set()
    assignments = []

    for score, tid, di in pairs:
        if score < iou_thresh:
            break
        if tid in used_tracks or di in used_dets:
            continue
        used_tracks.add(tid)
        used_dets.add(di)
        assignments.append((tid, di))

    unmatched = [i for i in range(len(det_bboxes)) if i not in used_dets]
    return assignments, unmatched

@dataclass
class Track:
    bbox: tuple
    miss: int = 0
    locked_name: Optional[str] = None
    locked_score: float = 0.0
    candidate_name: Optional[str] = None
    streak: int = 0
    last_score: float = 0.0

    def update_bbox(self, bbox):
        self.bbox = bbox
        self.miss = 0

    def mark_missed(self):
        self.miss += 1

    def update_identity(self, name: str, score: float, lock_frames: int = 5):
        self.last_score = float(score)

        if self.locked_name is not None:
            return

        if name == "UNKNOWN":
            self.candidate_name = None
            self.streak = 0
            return

        if name == self.candidate_name:
            self.streak += 1
        else:
            self.candidate_name = name
            self.streak = 1

        if self.streak >= lock_frames:
            self.locked_name = self.candidate_name
            self.locked_score = float(score)

# Per-client tracking store (in-memory)
_client_lock = Lock()
_client_state = {}  # client_id -> {"tracks": {tid: Track}, "next_id": int, "last_seen": float}

LOCK_FRAMES = 5
IOU_THRESH = 0.30
MAX_MISSES = 10          # frames to keep a disappearing face track
CLIENT_TTL_SEC = 60      # remove client state if not seen for 60s

def get_client_state(client_id: str):
    now = time.time()
    with _client_lock:
        # cleanup old clients
        stale = [cid for cid, st in _client_state.items() if now - st["last_seen"] > CLIENT_TTL_SEC]
        for cid in stale:
            del _client_state[cid]

        if client_id not in _client_state:
            _client_state[client_id] = {"tracks": {}, "next_id": 1, "last_seen": now}
        else:
            _client_state[client_id]["last_seen"] = now

        return _client_state[client_id]

# -----------------------
# Routes (API)
# -----------------------

@app.post("/api/recognize")
def api_recognize(payload: FrameIn):
    threshold = payload.threshold if payload.threshold is not None else settings.match_threshold

    try:
        frame = b64_to_bgr(payload.image_b64)
    except Exception as e:
        return JSONResponse({"error": f"Invalid image_b64: {str(e)}"}, status_code=400)

    faces = engine.detect_faces(frame)
    known = db.fetch_all()

    dets = []
    h, w = frame.shape[:2]

    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        x1 = clamp(x1, 0, w - 1)
        x2 = clamp(x2, 0, w - 1)
        y1 = clamp(y1, 0, h - 1)
        y2 = clamp(y2, 0, h - 1)

        name, score = engine.recognize_one(f.embedding, known, threshold)

        # Log only recognized hits (optional; change if you want to log UNKNOWN too)
        if name != "UNKNOWN":
            add_log(name, score)

        dets.append({
            "bbox": [x1, y1, x2, y2],
            "name": name,
            "score": float(score),
            "det_score": float(getattr(f, "det_score", 0.0))
        })

    return {"detections": dets}

@app.post("/api/enroll")
def api_enroll(payload: EnrollIn):
    name = payload.name.strip()
    if not name:
        return JSONResponse({"error": "Name is required"}, status_code=400)

    try:
        frame = b64_to_bgr(payload.image_b64)
    except Exception as e:
        return JSONResponse({"error": f"Invalid image_b64: {str(e)}"}, status_code=400)

    faces = engine.detect_faces(frame)
    if not faces:
        return JSONResponse({"error": "No face detected"}, status_code=400)

    # pick largest face
    def area(face):
        x1, y1, x2, y2 = face.bbox
        return (x2 - x1) * (y2 - y1)

    best = max(faces, key=area)
    db.add(name, np.asarray(best.embedding, dtype=np.float32))
    total = db.count_samples(name)

    return {"message": f"Enrolled '{name}'. Samples: {total}"}

@app.get("/api/users")
def api_users():
    users = db.list_users()
    return {
        "users": [{"name": u, "samples": db.count_samples(u)} for u in users]
    }

@app.delete("/api/users/{name}")
def api_delete_user(name: str):
    n = db.delete_user(name)
    return {"deleted_samples": n}

@app.get("/api/logs")
def api_logs(limit: int = 200):
    limit = max(1, min(int(limit), 1000))
    return {"logs": read_logs(limit)}