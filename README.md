# Face Recognition Web App (FastAPI + Browser Webcam)

A real-time face recognition system that runs in the browser using your webcam and performs recognition on a FastAPI backend using InsightFace + ONNX Runtime.

✅ Browser webcam streaming  
✅ Face bounding boxes + name + confidence  
✅ **5-frame stability lock** (reduces flicker)  
✅ Enroll users from the web UI  
✅ Users dashboard + recognition logs  

## Demo Features

### Live Recognition
- Open the Live page
- Click Start
- Faces are detected and labeled:
  - **SEARCHING** (red)  
  - **STABLE 1/5 → 5/5** (yellow)  
  - **LOCKED** (green)

### Enroll
- Type a name
- Click Enroll (snap)
- Repeat 3–10 times per person for best results

### Dashboard
- **/users**: list users + sample counts + delete
- **/logs**: show recognized (LOCKED) events

## Tech Stack
- **Python 3.11 (64-bit)**
- **FastAPI** (backend API + pages)
- **InsightFace** (FaceAnalysis: detection + embeddings)
- **ONNX Runtime** (model execution)
- **OpenCV** (local scripts support)
- **SQLite** (embeddings + logs)

## Requirements (Windows Recommended)
> This project works best on **Python 3.11**.  
Newer Python versions (e.g., 3.14) may not have compatible wheels yet.

- Windows 10/11
- Python 3.11.x (64-bit)
- Microsoft Visual C++ Redistributable 2015–2022 (x64) *(recommended)*


## Installation

### 1) Clone
git clone https://github.com/YOUR_USERNAME/face-recognition-web.git
cd face-recognition-web
