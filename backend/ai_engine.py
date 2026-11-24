import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
from collections import deque


# =========================================================
# DriveGuard AI Engine
# Runs the laptop webcam and publishes the latest detection
# results through a callback supplied by the FastAPI backend.
# =========================================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_POINTS = [13, 14, 78, 308]

EAR_THRESHOLD = 0.22
DROWSY_SECONDS = 2.0

MAR_THRESHOLD = 0.60
YAWN_SECONDS = 1.5

CALIBRATION_SECONDS = 3.0
YAW_THRESHOLD = 20
PITCH_THRESHOLD = 18
YAW_RESET = 12
PITCH_RESET = 12
HEAD_AWAY_SECONDS = 5.0

PERCLOS_WINDOW_SECONDS = 60.0
MAX_SAMPLE_GAP_SECONDS = 0.5


def create_tone(frequency, duration):
    sample_rate = 44100
    samples = np.arange(int(sample_rate * duration))
    wave = 0.5 * np.sin(
        2 * np.pi * frequency * samples / sample_rate
    )
    audio = (wave * 32767).astype(np.int16)
    stereo_audio = np.column_stack((audio, audio))
    return pygame.sndarray.make_sound(stereo_audio)


def calculate_ear(points):
    vertical1 = np.linalg.norm(points[1] - points[5])
    vertical2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal == 0:
        return 0.0

    return float((vertical1 + vertical2) / (2.0 * horizontal))


def calculate_mar(landmarks, width, height):
    points = []
    for index in MOUTH_POINTS:
        x = int(landmarks.landmark[index].x * width)
        y = int(landmarks.landmark[index].y * height)
        points.append([x, y])

    points = np.array(points)
    vertical = np.linalg.norm(points[0] - points[1])
    horizontal = np.linalg.norm(points[2] - points[3])

    if horizontal == 0:
        return 0.0

    return float(vertical / horizontal)


def calculate_head_pose(landmarks, width, height):
    landmark_ids = [1, 152, 33, 263, 61, 291]
    face_2d = []

    for index in landmark_ids:
        x = int(landmarks.landmark[index].x * width)
        y = int(landmarks.landmark[index].y * height)
        face_2d.append([x, y])

    face_2d = np.array(face_2d, dtype=np.float64)

    face_3d = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1]
    ], dtype=np.float64)

    focal_length = width
    center = (width / 2, height / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        face_3d,
        face_2d,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.decomposeProjectionMatrix(
        np.hstack((rotation_matrix, translation_vector))
    )[6]

    pitch = float(angles[0][0])
    yaw = float(angles[1][0])
    roll = float(angles[2][0])

    return pitch, yaw, roll


def update_perclos(samples, current_time, eyes_closed):
    samples.append((current_time, eyes_closed))
    window_start = current_time - PERCLOS_WINDOW_SECONDS

    while len(samples) >= 2 and samples[1][0] <= window_start:
        samples.popleft()

    closed_duration = 0.0
    observed_duration = 0.0
    sample_list = list(samples)

    for index in range(len(sample_list) - 1):
        sample_time, was_closed = sample_list[index]
        next_time = sample_list[index + 1][0]
        interval = next_time - sample_time

        # Ignore long gaps, for example when the camera freezes.
        if interval <= 0 or interval > MAX_SAMPLE_GAP_SECONDS:
            continue

        overlap_start = max(sample_time, window_start)
        overlap_end = min(next_time, current_time)
        overlap = max(0.0, overlap_end - overlap_start)

        observed_duration += overlap
        if was_closed:
            closed_duration += overlap

    if observed_duration <= 0:
        return 0.0, 0.0

    return (closed_duration / observed_duration) * 100.0, observed_duration


def run_camera(stop_event, publish_callback, camera_index=0):
    """
    Start webcam monitoring until stop_event is set.

    publish_callback receives a dictionary containing the latest
    driver-monitoring measurements. The FastAPI app can expose this
    dictionary to connected phones.
    """

    pygame_ready = False
    cap = None

    eye_closed_start = None
    alarm_playing = False

    yawn_start = None
    yawn_detected = False

    head_away_start = None
    head_alert_played = False

    calibration_start = None
    calibration_samples = []
    neutral_pitch = None
    neutral_yaw = None
    smooth_pitch = None
    smooth_yaw = None

    perclos_samples = deque()

    state = {
        "camera_status": "starting",
        "face_detected": False,
        "eye_status": "unknown",
        "ear": None,
        "drowsy": False,
        "yawning": False,
        "mar": None,
        "head_status": "not_calibrated",
        "pitch_change": None,
        "yaw_change": None,
        "perclos": 0.0,
        "perclos_observed_seconds": 0.0,
        "last_updated": None
    }

    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2)
        pygame_ready = True

        alarm_sound = create_tone(1000, 0.5)
        yawn_beep = create_tone(1200, 0.25)
        head_beep = create_tone(1500, 0.20)
        head_channel = pygame.mixer.Channel(1)

        mp_face_mesh = mp.solutions.face_mesh
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        print("DriveGuard AI camera engine started.")
        publish_callback({**state, "camera_status": "running"})

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            while not stop_event.is_set():
                success, frame = cap.read()

                if not success:
                    state["camera_status"] = "camera_read_error"
                    publish_callback(dict(state))
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                current_time = time.time()

                # Reset per-frame values before filling them.
                state["last_updated"] = current_time
                state["face_detected"] = bool(results.multi_face_landmarks)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    height, width, _ = frame.shape

                    left_eye = np.array([
                        [
                            int(face_landmarks.landmark[i].x * width),
                            int(face_landmarks.landmark[i].y * height)
                        ]
                        for i in LEFT_EYE
                    ])

                    right_eye = np.array([
                        [
                            int(face_landmarks.landmark[i].x * width),
                            int(face_landmarks.landmark[i].y * height)
                        ]
                        for i in RIGHT_EYE
                    ])

                    ear = (
                        calculate_ear(left_eye) +
                        calculate_ear(right_eye)
                    ) / 2.0

                    eyes_closed = ear < EAR_THRESHOLD

                    if eyes_closed:
                        if eye_closed_start is None:
                            eye_closed_start = current_time

                        closed_duration = current_time - eye_closed_start
                        drowsy = closed_duration >= DROWSY_SECONDS

                        if drowsy and not alarm_playing:
                            alarm_sound.play(loops=-1)
                            alarm_playing = True

                        eye_status = (
                            "DROWSINESS ALERT" if drowsy else "EYES CLOSED"
                        )
                    else:
                        eye_closed_start = None
                        closed_duration = 0.0
                        drowsy = False
                        eye_status = "EYES OPEN"

                        if alarm_playing:
                            alarm_sound.stop()
                            alarm_playing = False

                    perclos_value, observed_seconds = update_perclos(
                        perclos_samples,
                        current_time,
                        eyes_closed
                    )

                    mar = calculate_mar(face_landmarks, width, height)
                    if mar > MAR_THRESHOLD:
                        if yawn_start is None:
                            yawn_start = current_time

                        if (
                            current_time - yawn_start >= YAWN_SECONDS
                            and not yawn_detected
                        ):
                            yawn_detected = True
                            yawn_beep.play()
                    else:
                        yawn_start = None
                        yawn_detected = False

                    pose = calculate_head_pose(
                        face_landmarks, width, height
                    )

                    if pose is not None:
                        pitch, yaw, _ = pose

                        if calibration_start is None:
                            calibration_start = current_time

                        if neutral_pitch is None:
                            calibration_samples.append([pitch, yaw])

                            if (
                                current_time - calibration_start
                                >= CALIBRATION_SECONDS
                                and len(calibration_samples) >= 10
                            ):
                                neutral_pitch = float(np.median(
                                    [sample[0] for sample in calibration_samples]
                                ))
                                neutral_yaw = float(np.median(
                                    [sample[1] for sample in calibration_samples]
                                ))
                                smooth_pitch = 0.0
                                smooth_yaw = 0.0
                                print("Head calibration completed.")
                        else:
                            pitch_delta = pitch - neutral_pitch
                            yaw_delta = yaw - neutral_yaw

                            smooth_pitch = (
                                0.8 * smooth_pitch + 0.2 * pitch_delta
                            )
                            smooth_yaw = (
                                0.8 * smooth_yaw + 0.2 * yaw_delta
                            )

                            head_away = (
                                abs(smooth_yaw) > YAW_THRESHOLD
                                or abs(smooth_pitch) > PITCH_THRESHOLD
                            )
                            head_returned = (
                                abs(smooth_yaw) < YAW_RESET
                                and abs(smooth_pitch) < PITCH_RESET
                            )

                            if head_away:
                                if head_away_start is None:
                                    head_away_start = current_time

                                away_duration = current_time - head_away_start

                                if (
                                    away_duration >= HEAD_AWAY_SECONDS
                                    and not head_alert_played
                                ):
                                    head_channel.play(head_beep)
                                    head_channel.queue(head_beep)
                                    head_alert_played = True
                            elif head_returned:
                                head_away_start = None
                                head_alert_played = False

                    if neutral_pitch is None:
                        head_status = "calibrating"
                    elif head_away_start is not None:
                        if current_time - head_away_start >= HEAD_AWAY_SECONDS:
                            head_status = "head_position_alert"
                        else:
                            head_status = "head_away"
                    else:
                        head_status = "forward"

                    state.update({
                        "camera_status": "running",
                        "face_detected": True,
                        "eye_status": eye_status,
                        "ear": round(float(ear), 4),
                        "drowsy": bool(drowsy),
                        "yawning": bool(yawn_detected),
                        "mar": round(float(mar), 4),
                        "head_status": head_status,
                        "pitch_change": (
                            round(float(smooth_pitch), 2)
                            if smooth_pitch is not None else None
                        ),
                        "yaw_change": (
                            round(float(smooth_yaw), 2)
                            if smooth_yaw is not None else None
                        ),
                        "perclos": round(float(perclos_value), 2),
                        "perclos_observed_seconds": round(
                            float(observed_seconds), 2
                        ),
                        "last_updated": current_time
                    })

                    # Keep the laptop preview window as a local dash-camera view.
                    cv2.putText(
                        frame, f"EAR: {ear:.3f}", (25, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
                    cv2.putText(
                        frame, eye_status, (25, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 0, 255) if drowsy else (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame, f"PERCLOS: {perclos_value:.1f}%", (25, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        "YAWNING DETECTED" if yawn_detected else "Yawn: No",
                        (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 165, 255) if yawn_detected else (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame, f"Head: {head_status}", (25, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )

                else:
                    eye_closed_start = None
                    yawn_start = None
                    yawn_detected = False
                    head_away_start = None
                    head_alert_played = False
                    perclos_samples.clear()

                    if alarm_playing:
                        alarm_sound.stop()
                        alarm_playing = False

                    state.update({
                        "camera_status": "running",
                        "face_detected": False,
                        "eye_status": "no_face",
                        "ear": None,
                        "drowsy": False,
                        "yawning": False,
                        "mar": None,
                        "head_status": "no_face",
                        "pitch_change": None,
                        "yaw_change": None,
                        "perclos": None,
                        "perclos_observed_seconds": 0.0,
                        "last_updated": current_time
                    })

                    cv2.putText(
                        frame, "NO FACE DETECTED", (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                    )

                publish_callback(dict(state))
                cv2.imshow("DriveGuard AI - Laptop Dash Camera", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        state["camera_status"] = "stopped"
        publish_callback(dict(state))

    except Exception as error:
        state["camera_status"] = "error"
        state["error"] = str(error)
        publish_callback(dict(state))
        print("DriveGuard AI engine error:", error)

    finally:
        if alarm_playing and pygame_ready:
            pygame.mixer.stop()

        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        if pygame_ready:
            pygame.mixer.quit()

        print("DriveGuard AI camera engine stopped.")
