# DriveGuard AI

## AI-Powered Driver Drowsiness Monitoring and Accident-Prevention Prototype

> **Project type:** Academic / major project\
> **Primary goal:** Monitor driver behaviour using computer vision,
> detect possible signs of fatigue, and display monitoring results in a
> mobile dashboard.\
> **Architecture:** Laptop webcam → Python AI engine → FastAPI/WebSocket
> backend → React Native (Expo) mobile application.\
> **Project folder:** `drishyaguard Ai`

------------------------------------------------------------------------

## 1. Project overview

DriveGuard AI is an experimental driver-monitoring system. It uses a
laptop webcam as a dash-camera-style input, processes facial landmarks
with computer-vision techniques, and sends the latest monitoring state
to a mobile application.

The Python engine analyses eye closure, mouth opening/yawning, head
orientation, and an estimated PERCLOS measure. It can play local audio
warnings for selected events. FastAPI exposes HTTP endpoints and a
WebSocket stream. The React Native application displays the live state
in a dashboard.

The current design is:

-   **Laptop:** runs the webcam and AI processing.
-   **Phone:** provides the monitoring interface.
-   **Backend:** runs on the laptop and coordinates the AI engine and
    client connections.
-   **Local connection:** phone and laptop can communicate over the same
    Wi-Fi network.
-   **Remote connection:** ngrok was installed/configured and a tunnel
    was brought online. End-to-end remote mobile operation still needs
    to be re-tested and secured.

This project is a prototype for learning and demonstration. It is **not
a certified driver-safety device**, and its detections must not be
relied on as the sole means of preventing accidents.

------------------------------------------------------------------------

## 2. Project objectives

1.  Detect a driver's face and obtain facial landmarks.
2.  Estimate whether the eyes are open or closed.
3.  Detect prolonged eye closure and trigger a drowsiness warning.
4.  Detect yawning using mouth landmarks.
5.  Estimate head orientation and flag sustained abnormal head position.
6.  Calculate an approximate PERCLOS value over a rolling observation
    window.
7.  Send the current AI state to a mobile dashboard in near real time.
8.  Support local-network operation and develop secure remote access.
9.  Keep the design extensible for future GPS, emergency notification,
    database, and hardware modules.

------------------------------------------------------------------------

## 3. Implemented functionality

The following features have been implemented and tested during
development:

  -----------------------------------------------------------------------
  Feature                 Implementation /        Status
                          behaviour               
  ----------------------- ----------------------- -----------------------
  Webcam capture          OpenCV camera capture,  Implemented
                          normally camera index   
                          `0`                     

  Face landmarks          MediaPipe Face Mesh     Implemented

  Eye state               Eye Aspect Ratio (EAR)  Implemented

  Drowsiness              Prolonged eye-closure   Implemented
                          timer; configured       
                          threshold was over 2    
                          seconds in the          
                          standalone engine       

  Drowsiness sound        Repeating warning while Implemented
                          prolonged closure       
                          remains active; stops   
                          when eyes reopen        

  Yawning                 Mouth Aspect Ratio      Implemented
                          (MAR)                   

  Yawn sound              One short beep for a    Implemented
                          detected yawn event     

  Head pose               Facial landmarks and    Implemented
                          OpenCV `solvePnP`, with 
                          neutral-position        
                          calibration and         
                          smoothing               

  Head-position sound     Two beeps after a       Implemented
                          sustained head-position 
                          warning (configured     
                          threshold was over 5    
                          seconds)                

  PERCLOS                 Approximate rolling     Implemented
                          60-second eye-closure   
                          estimate                

  FastAPI                 HTTP API and            Implemented
                          application lifecycle   

  WebSocket               Live state broadcast to Implemented
                          connected clients       

  Mobile dashboard        React Native / Expo     Implemented and locally
                          screen displaying live  tested
                          monitoring state        

  Same-Wi-Fi phone access Phone successfully      Tested
                          requested the laptop's  
                          `/ai/live` endpoint     

  ngrok tunnel            Public HTTPS forwarding Tunnel online; remote
                          tunnel was brought      app flow requires
                          online                  re-test

  Authentication          Backend authentication  Not implemented
                          for public access       

  GPS / SOS / database /  Future work             Planned
  hardware                                        
  -----------------------------------------------------------------------

**Measurement note:** EAR, MAR, head-pose values, and PERCLOS depend on
camera placement, lighting, face position, calibration, and thresholds.
The current PERCLOS is an experimental estimate, not a medically
validated measurement.

------------------------------------------------------------------------

## 4. Technology stack

### Backend and computer vision

-   **Python 3.12.4** --- backend and AI engine language.
-   **OpenCV (`cv2`)** --- webcam capture, image processing, and
    head-pose calculation.
-   **MediaPipe 0.10.21** --- face mesh landmarks. This version was
    selected because the installed MediaPipe 1.2.0 package did not
    expose the `mp.solutions` API used by this project.
-   **NumPy 1.26.4** --- numerical calculations.
-   **Pygame 2.6.1** --- audio warning playback.
-   **FastAPI 0.162.0** --- HTTP API and WebSocket endpoint.
-   **Uvicorn 0.54.0** --- ASGI server.
-   **Python threading** --- run camera processing outside FastAPI's
    request/event loop.
-   **WebSockets** --- real-time updates to the mobile client.

### Mobile and networking

-   **React Native** and **Expo** --- mobile application.
-   **JavaScript** --- mobile UI logic.
-   **Expo Go** --- development testing on Android.
-   **ngrok** --- exposes the local FastAPI service through a public
    HTTPS forwarding address. The mobile WebSocket should use `wss://`
    through the HTTPS tunnel.

### Development environment

-   Windows, PowerShell, Visual Studio Code.
-   Python `3.12.4`.
-   Node.js `v24.18.0`.
-   npm `12.0.1`.
-   Git for version control.

These are the versions recorded during development. Verify installed
versions when setting up on another computer.

------------------------------------------------------------------------

## 5. Repository structure

The project has been organised approximately as follows:

``` text
drishyaguard Ai/
│
├── backend/
│   ├── venv/                 # Python virtual environment (do not commit)
│   ├── main.py               # FastAPI app, camera thread, HTTP + WebSocket
│   ├── ai_engine.py          # AI engine integrated with FastAPI
│   ├── camera.py             # Standalone webcam AI implementation
│   └── camera_backup.py      # Backup of camera.py
│
├── mobile/
│   ├── App.js                # React Native monitoring dashboard
│   ├── package.json
│   ├── package-lock.json
│   ├── assets/
│   └── node_modules/         # Installed dependencies (do not commit)
│
├── hardware/                 # Reserved for future hardware work
├── documentation/            # Reserved for project documentation
└── README.md
```

The exact contents may evolve. Keep `camera.py` and `camera_backup.py`
safe: the standalone camera implementation was working before the
integrated `ai_engine.py` workflow was added.

------------------------------------------------------------------------

## 6. AI processing concepts

### 6.1 Face Mesh

MediaPipe Face Mesh detects facial landmarks. The eye and mouth
calculations use selected landmark coordinates. If the face is not
detected, the engine should publish a state indicating that no face is
currently visible.

### 6.2 Eye Aspect Ratio (EAR)

EAR measures the relative vertical opening of the eye compared with its
horizontal width. A lower EAR generally corresponds to a more closed
eye.

A commonly used form is:

\[ EAR = `\frac{d(p_2,p_6)+d(p_3,p_5)}{2d(p_1,p_4)}`{=tex} \]

where `d` is Euclidean distance and the points represent selected eye
landmarks. The exact landmark indices and threshold are defined in the
source code.

The engine uses EAR over time rather than treating one low reading as
sufficient evidence of drowsiness. The standalone prototype was
configured to trigger after eye closure lasting more than approximately
two seconds.

### 6.3 Mouth Aspect Ratio (MAR)

MAR compares mouth opening height with mouth width. A larger value can
indicate an open mouth or yawn. The actual threshold and event logic are
implementation parameters in the AI engine.

### 6.4 Head pose

Head orientation is estimated from facial landmarks using OpenCV's
`solvePnP` method. The engine performs a neutral-position calibration at
startup (approximately the first three seconds in the tested
implementation), then compares later pose estimates with the calibrated
position.

**Known item to review:** During one API test, the mobile state showed
`pitch_change: -354.66`. This may indicate angle wrapping, coordinate
convention, or calibration handling. The head-position feature had
previously been tested as working, but this extreme value should be
investigated before relying on the numerical angle or alert threshold.
Do not interpret it as a realistic physical head rotation.

### 6.5 PERCLOS

PERCLOS describes the proportion of time the eyes are closed over an
observation period. This project calculates an approximate EAR-based
value over a rolling 60-second window. It is a prototype estimate and
has not been validated against a clinical or research-grade PERCLOS
implementation.

------------------------------------------------------------------------

## 7. Backend architecture

The FastAPI application starts the AI camera engine in a background
thread during application startup. A callback publishes the latest AI
state into shared application state. A lock is used when reading or
updating that state. The camera stop event is set during application
shutdown.

This design keeps the camera loop from blocking FastAPI's asynchronous
WebSocket handling.

### HTTP endpoints

  Endpoint          Method   Purpose
  ----------------- -------- ---------------------------------------------
  `/`               GET      Backend welcome/status response
  `/health`         GET      Health check and timestamp
  `/ai/status`      GET      AI engine status and supported feature list
  `/ai/live`        GET      Latest driver-monitoring state
  `/docs`           GET      Interactive FastAPI documentation
  `/openapi.json`   GET      OpenAPI schema

### WebSocket endpoint

  -----------------------------------------------------------------------
  Endpoint                Protocol                Purpose
  ----------------------- ----------------------- -----------------------
  `/ws`                   WebSocket               Sends current
                                                  monitoring state to
                                                  connected clients

  -----------------------------------------------------------------------

The integrated implementation sends a state update approximately every
`0.2` seconds (about five updates per second).

### Example live state

This is an example of the data shape returned during development. Values
are illustrative and change with camera input:

``` json
{
  "camera_status": "running",
  "face_detected": true,
  "eye_status": "EYES OPEN",
  "ear": 0.2304,
  "drowsy": false,
  "yawning": false,
  "mar": 0.0196,
  "head_status": "head_position_alert",
  "pitch_change": -354.66,
  "yaw_change": 2.91,
  "perclos": 23.83,
  "perclos_observed_seconds": 60.0,
  "last_updated": 1793516646.039289
}
```

The extreme pitch value in this historical sample is a diagnostic
example, not an expected or correct result.

------------------------------------------------------------------------

## 8. Mobile application

The Expo application connects to the backend's WebSocket endpoint and
updates its screen when new JSON messages arrive.

The dashboard currently displays:

-   Backend connection status.
-   Camera status.
-   Face detection status.
-   Eye status and EAR.
-   Yawning status and MAR.
-   Drowsiness warning state.
-   Head-position status and pitch/yaw values.
-   PERCLOS estimate and observation duration.

The mobile app does not itself perform the webcam-based AI processing.
The laptop backend performs that work and sends the resulting state to
the phone.

### WebSocket address

For same-network testing, the WebSocket URL uses the laptop's LAN IP,
for example:

``` javascript
const WS_URL = "ws://192.168.1.75:8000/ws";
```

For remote access through ngrok, use the current ngrok hostname with
secure WebSockets:

``` javascript
const WS_URL = "wss://YOUR-CURRENT-NGROK-HOST/ws";
```

Replace the placeholder with the actual hostname shown by the running
ngrok process. Do not treat a temporary tunnel hostname as permanent.

------------------------------------------------------------------------

## 9. Setup instructions (Windows)

### Prerequisites

Install Python 3.12, Node.js and npm, Visual Studio Code, Expo Go on an
Android phone, and ngrok if remote access is needed.

### 9.1 Backend setup

Open PowerShell in the project root:

``` powershell
cd "C:\Users\Swift Go\Desktop\drishyaguard Ai"
cd backend
```

Create the backend virtual environment only if it does not already
exist:

``` powershell
python -m venv venv
```

Activate it:

``` powershell
.\venv\Scripts\Activate.ps1
```

Install the packages used by the project:

``` powershell
python -m pip install --upgrade pip
pip install fastapi uvicorn opencv-python mediapipe==0.10.21 numpy==1.26.4 pygame==2.6.1 websockets
```

If a maintained `requirements.txt` is added later, install from it:

``` powershell
pip install -r requirements.txt
```

### 9.2 Mobile setup

Open a second terminal:

``` powershell
cd "C:\Users\Swift Go\Desktop\drishyaguard Ai\mobile"
npm install
```

------------------------------------------------------------------------

## 10. Run the project

Run each service in a separate terminal. Keep the terminals open while
using the application.

### Terminal 1 --- FastAPI and AI engine

``` powershell
cd "C:\Users\Swift Go\Desktop\drishyaguard Ai\backend"
.\venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Expected behaviour:

-   Uvicorn starts on port `8000`.
-   FastAPI startup completes.
-   The AI camera thread starts.
-   The webcam opens and the AI engine begins processing.

Open the local API documentation:

``` text
http://127.0.0.1:8000/docs
```

Check the latest state:

``` text
http://127.0.0.1:8000/ai/live
```

### Terminal 2 --- ngrok for remote backend access (optional)

Authenticate ngrok once on the computer using the token from the ngrok
dashboard. Run this locally and never put the token in source control:

``` powershell
ngrok config add-authtoken YOUR_NGROK_AUTH_TOKEN
```

Then start the tunnel:

``` powershell
ngrok http 8000
```

Copy the HTTPS forwarding hostname displayed by ngrok. Use that hostname
in the mobile WebSocket URL with `wss://`.

Keep ngrok running while remote clients need to connect. The laptop must
remain powered on, online, and running the backend. A tunnel is not the
same as deploying the application to a permanently hosted server.

### Terminal 3 --- Expo mobile application

``` powershell
cd "C:\Users\Swift Go\Desktop\drishyaguard Ai\mobile"
npx expo start
```

Scan the QR code with Expo Go.

For local LAN mode, the phone and laptop should be on the same Wi-Fi
network. For remote backend data, the mobile WebSocket URL must point to
the active ngrok tunnel. Expo's development-server connection and the
backend ngrok tunnel are separate connections.

If Expo has a cache-related issue, stop it with `Ctrl+C` and try:

``` powershell
npx expo start --clear
```

Expo's own tunnel may require Expo account sign-in and its own tunnel
dependency; this is separate from ngrok's backend tunnel.

------------------------------------------------------------------------

## 11. Network and connectivity

### Local network

The laptop and phone must be connected to the same Wi-Fi network. The
laptop's LAN IP recorded during testing was `192.168.1.75`; this address
can change, so check `ipconfig` if local connection fails.

The local WebSocket address used during testing was:

``` text
ws://192.168.1.75:8000/ws
```

### Remote network

For a phone on mobile data or a different Wi-Fi network:

1.  Start the FastAPI backend on the laptop.
2.  Start `ngrok http 8000`.
3.  Copy the current HTTPS forwarding hostname.
4.  Configure the app to use `wss://HOSTNAME/ws`.
5.  Keep the laptop and both processes running.

  Purpose            URL pattern
  ------------------ -------------------------------------
  Public HTTP API    `https://HOSTNAME/ai/live`
  Public WebSocket   `wss://HOSTNAME/ws`
  Local HTTP API     `http://LAPTOP_LAN_IP:8000/ai/live`
  Local WebSocket    `ws://LAPTOP_LAN_IP:8000/ws`

### Remote-access security warning

The prototype backend was developed without authentication. Do **not**
expose sensitive or real driver-monitoring information to the public
internet until authentication and access controls have been added.
Anyone who obtains an unprotected public URL may be able to access its
endpoints.

Before public or production use, add authentication for HTTP and
WebSocket connections, HTTPS/WSS-only remote access, input validation,
rate limiting, logging, and a policy for handling and retaining driver
data.

Never publish ngrok auth tokens, passwords, private keys, or other
secrets in this README or in Git.

------------------------------------------------------------------------

## 12. Troubleshooting

### Backend does not start

-   Confirm the terminal is in the `backend` directory.
-   Activate `backend\venv`.
-   Check that Python dependencies are installed in that environment.
-   Check whether port `8000` is already in use.
-   Read the first traceback/error in the terminal.

### Webcam does not open

-   Close other programs using the webcam.
-   Confirm the correct camera index (the current setup uses `0`).
-   Check Windows camera permissions.
-   Test standalone `camera.py` only when the integrated backend is
    stopped, so two processes do not compete for the same camera.

### Phone cannot reach local backend

-   Confirm both devices are on the same Wi-Fi.
-   Check the laptop IP with `ipconfig`.
-   Open `http://LAPTOP_IP:8000/ai/live` in the phone browser.
-   Allow Python/Uvicorn through Windows Firewall on private networks.
-   Confirm Uvicorn was started with `--host 0.0.0.0`.

### Mobile dashboard says disconnected

-   Confirm the backend is running.
-   Confirm the WebSocket URL is correct.
-   For local access use `ws://LAPTOP_IP:8000/ws`.
-   For ngrok HTTPS forwarding use `wss://CURRENT_HOSTNAME/ws`.
-   Check that ngrok is online and forwarding to port `8000`.
-   Remember that the ngrok hostname can change.

### Expo QR code appears but the app does not load

-   Confirm Expo Go is installed and compatible with the project's Expo
    SDK.
-   Try restarting with `npx expo start --clear`.
-   For LAN mode, verify phone and laptop are on the same Wi-Fi.
-   Check the Expo terminal for the actual JavaScript or bundler error.
-   Do not confuse Expo's development-server tunnel with the ngrok
    tunnel used for FastAPI.

### MediaPipe import error

The project uses MediaPipe `0.10.21` because the earlier `1.2.0`
installation did not provide the `mp.solutions` API expected by the
original code. Verify the active virtual environment and installed
version:

``` powershell
python -c "import mediapipe as mp; print(mp.__version__); print(hasattr(mp, 'solutions'))"
```

------------------------------------------------------------------------

## 13. Current project status and roadmap

### Completed

-   [x] Project architecture planned.
-   [x] Laptop webcam capture.
-   [x] MediaPipe face landmark detection.
-   [x] EAR-based eye closure detection.
-   [x] Drowsiness timer and warning sound.
-   [x] MAR-based yawning detection and beep.
-   [x] Head-pose estimation and calibration.
-   [x] Sustained head-position alert and double beep.
-   [x] Approximate rolling PERCLOS calculation.
-   [x] FastAPI backend and health/status endpoints.
-   [x] AI engine integrated with FastAPI using a background thread.
-   [x] Live AI state endpoint.
-   [x] WebSocket live-state broadcasting.
-   [x] Expo / React Native mobile dashboard.
-   [x] Local phone-to-laptop API connectivity tested successfully.
-   [x] ngrok installation and an online forwarding tunnel demonstrated.

### To verify or improve

-   [ ] Re-test the complete Expo mobile experience after the last
    reported Expo tunnel/startup issue.
-   [ ] Confirm end-to-end remote mobile WebSocket connection over
    mobile data.
-   [ ] Add authentication before exposing the backend publicly.
-   [ ] Investigate head-pose pitch angle wrapping (historical `-354.66`
    reading).
-   [ ] Improve head-pose calibration and angle normalization.
-   [ ] Add automated tests and a stable dependency/requirements file.
-   [ ] Add robust reconnect/error handling and connection status in the
    app.
-   [ ] Evaluate AI thresholds under varied lighting, camera placement,
    and driver positions.

### Planned future modules

-   [ ] GPS location display.
-   [ ] Emergency SOS workflow and notification integration.
-   [ ] MongoDB storage for monitoring history.
-   [ ] ESP32 integration.
-   [ ] MPU6050 accelerometer/gyroscope integration.
-   [ ] Hardware buzzer.
-   [ ] Sensor-based accident-event detection.
-   [ ] Testing, documentation, and deployment improvements.

These future modules are plans, not current implemented capabilities.

------------------------------------------------------------------------

## 14. Development notes

-   Use `backend\venv` for backend work. A separate root `.venv` was
    also present during development; avoid accidentally installing
    packages into the wrong environment.
-   Preserve the working standalone `camera.py` and `camera_backup.py`.
-   The integrated backend imports `run_camera` from `ai_engine.py`.
-   Start the backend before testing the mobile dashboard.
-   Do not run multiple camera processes at once if they compete for the
    same webcam.
-   Use the current ngrok hostname; do not assume an old free-tunnel URL
    remains valid.
-   Do not commit `venv/`, `node_modules/`, `.env`, credentials, or
    tokens.

Suggested `.gitignore` entries:

``` gitignore
# Python
__pycache__/
*.py[cod]
venv/
.venv/
.env

# Node / Expo
node_modules/
.expo/
dist/
web-build/

# Logs and local files
*.log
.DS_Store
Thumbs.db
```

------------------------------------------------------------------------

## 15. Conclusion

DriveGuard AI demonstrates how computer vision, facial landmark
analysis, Python backend development, real-time WebSocket communication,
and mobile application development can be combined into a
driver-monitoring prototype.

The system currently processes webcam input on a laptop and presents its
analysis on a mobile dashboard. The next major engineering priorities
are improving measurement robustness, securing remote access, validating
the complete remote mobile workflow, and developing the planned GPS,
SOS, database, and hardware modules.

**DriveGuard AI --- Intelligent Driver Monitoring Prototype.**
