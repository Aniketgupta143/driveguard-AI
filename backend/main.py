
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from datetime import datetime
import asyncio
import threading

from ai_engine import run_camera

app = FastAPI(
    title="DriveGuard AI API",
    description="Backend API for the DriveGuard AI driver monitoring system",
    version="1.0.0"
)

# Shared AI monitoring state
latest_state = {
    "camera_status": "Starting",
    "face_detected": False,
    "eye_status": "Unknown",
    "ear": 0.0,
    "drowsy": False,
    "yawning": False,
    "mar": 0.0,
    "head_status": "Unknown",
    "pitch_change": 0.0,
    "yaw_change": 0.0,
    "perclos": 0.0,
    "perclos_observed_seconds": 0.0,
    "last_updated": None
}

state_lock = threading.Lock()
camera_stop_event = threading.Event()
camera_thread = None


# Called by the AI engine whenever it produces new data
def update_ai_state(new_state):
    global latest_state

    with state_lock:
        latest_state = new_state.copy()


@app.on_event("startup")
def start_ai_camera():
    global camera_thread

    camera_stop_event.clear()

    camera_thread = threading.Thread(
        target=run_camera,
        args=(camera_stop_event, update_ai_state),
        kwargs={"camera_index": 0},
        daemon=True
    )

    camera_thread.start()
    print("DriveGuard AI camera thread started.")


@app.on_event("shutdown")
def stop_ai_camera():
    camera_stop_event.set()

    if camera_thread and camera_thread.is_alive():
        camera_thread.join(timeout=3)

    print("DriveGuard AI camera stopped.")


@app.get("/")
def home():
    return {
        "message": "Welcome to DriveGuard AI Backend!",
        "status": "Running",
        "project": "DriveGuard AI"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/ai/status")
def ai_status():
    return {
        "ai_engine": "DriveGuard AI",
        "status": "Running",
        "features": [
            "Face Detection",
            "Eye Closure Detection",
            "Drowsiness Detection",
            "Yawning Detection",
            "Head Pose Detection",
            "PERCLOS Estimation"
        ]
    }


@app.get("/ai/live")
def get_live_ai_data():
    with state_lock:
        return latest_state.copy()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Mobile/web client connected.")

    try:
        while True:
            with state_lock:
                current_state = latest_state.copy()

            await websocket.send_json(current_state)
            await asyncio.sleep(0.2)

    except WebSocketDisconnect:
        print("Mobile/web client disconnected.")

    except Exception as error:
        print("WebSocket error:", error)
