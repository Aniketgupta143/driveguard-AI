
import cv2
import mediapipe as mp
import numpy as np
import time
import pygame

# ==========================================
# 1. INITIALIZATION
# ==========================================

pygame.mixer.init(frequency=44100, size=-16, channels=2)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_POINTS = [13, 14, 78, 308]

EAR_THRESHOLD = 0.22
DROWSY_SECONDS = 2.0

MAR_THRESHOLD = 0.60
YAWN_SECONDS = 1.5

# Head movement settings
CALIBRATION_SECONDS = 3.0

YAW_THRESHOLD = 20
PITCH_THRESHOLD = 18

# Lower reset thresholds prevent timer flickering
YAW_RESET = 12
PITCH_RESET = 12

HEAD_AWAY_SECONDS = 5.0

eye_closed_start = None
alarm_playing = False

yawn_start = None
yawn_detected = False

head_away_start = None
head_alert_played = False

# Head calibration
calibration_start = None
calibration_samples = []
neutral_pitch = None
neutral_yaw = None

# Smoothed head angles
smooth_pitch = None
smooth_yaw = None

# ==========================================
# 2. SOUND GENERATION
# ==========================================

def create_tone(frequency, duration):

    sample_rate = 44100
    samples = np.arange(int(sample_rate * duration))

    wave = 0.5 * np.sin(
        2 * np.pi * frequency * samples / sample_rate
    )

    audio = (wave * 32767).astype(np.int16)

    stereo_audio = np.column_stack((audio, audio))

    return pygame.sndarray.make_sound(stereo_audio)


alarm_sound = create_tone(1000, 0.5)
yawn_beep = create_tone(1200, 0.25)
head_beep = create_tone(1500, 0.20)

# Dedicated channel for the two head-warning beeps
head_channel = pygame.mixer.Channel(1)

# ==========================================
# 3. EYE ASPECT RATIO
# ==========================================

def calculate_ear(points):

    vertical1 = np.linalg.norm(points[1] - points[5])
    vertical2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal == 0:
        return 0

    return (vertical1 + vertical2) / (2.0 * horizontal)


# ==========================================
# 4. MOUTH ASPECT RATIO
# ==========================================

def calculate_mar(landmarks, width, height):

    points = []

    for i in MOUTH_POINTS:

        x = int(landmarks.landmark[i].x * width)
        y = int(landmarks.landmark[i].y * height)

        points.append([x, y])

    points = np.array(points)

    vertical = np.linalg.norm(points[0] - points[1])
    horizontal = np.linalg.norm(points[2] - points[3])

    if horizontal == 0:
        return 0

    return vertical / horizontal


# ==========================================
# 5. HEAD POSE ESTIMATION
# ==========================================

def calculate_head_pose(landmarks, width, height):

    landmark_ids = [1, 152, 33, 263, 61, 291]

    face_2d = []

    for idx in landmark_ids:

        x = int(landmarks.landmark[idx].x * width)
        y = int(landmarks.landmark[idx].y * height)

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


# ==========================================
# 6. START CAMERA
# ==========================================

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

print("DriveGuard AI Started")
print("Look straight during the first 3 seconds.")
print("Press Q to exit.")

try:

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            results = face_mesh.process(rgb_frame)

            current_time = time.time()

            if results.multi_face_landmarks:

                face_landmarks = results.multi_face_landmarks[0]

                h, w, _ = frame.shape

                # ==================================
                # 7. EYE DETECTION
                # ==================================

                left_eye = np.array([
                    [
                        int(face_landmarks.landmark[i].x * w),
                        int(face_landmarks.landmark[i].y * h)
                    ]
                    for i in LEFT_EYE
                ])

                right_eye = np.array([
                    [
                        int(face_landmarks.landmark[i].x * w),
                        int(face_landmarks.landmark[i].y * h)
                    ]
                    for i in RIGHT_EYE
                ])

                ear = (
                    calculate_ear(left_eye) +
                    calculate_ear(right_eye)
                ) / 2

                cv2.putText(
                    frame,
                    f"EAR: {ear:.3f}",
                    (30, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                if ear < EAR_THRESHOLD:

                    if eye_closed_start is None:
                        eye_closed_start = current_time

                    closed_duration = current_time - eye_closed_start

                    if closed_duration >= DROWSY_SECONDS:

                        eye_status = "DROWSINESS ALERT!"
                        eye_color = (0, 0, 255)

                        if not alarm_playing:
                            alarm_sound.play(loops=-1)
                            alarm_playing = True

                    else:
                        eye_status = "EYES CLOSED"
                        eye_color = (0, 165, 255)

                else:

                    eye_closed_start = None
                    closed_duration = 0

                    eye_status = "EYES OPEN"
                    eye_color = (0, 255, 0)

                    if alarm_playing:
                        alarm_sound.stop()
                        alarm_playing = False

                cv2.putText(
                    frame,
                    eye_status,
                    (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    eye_color,
                    2
                )

                # ==================================
                # 8. YAWNING DETECTION
                # ==================================

                mar = calculate_mar(face_landmarks, w, h)

                cv2.putText(
                    frame,
                    f"MAR: {mar:.2f}",
                    (30, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                if mar > MAR_THRESHOLD:

                    if yawn_start is None:
                        yawn_start = current_time

                    yawn_duration = current_time - yawn_start

                    if (
                        yawn_duration >= YAWN_SECONDS
                        and not yawn_detected
                    ):

                        yawn_detected = True
                        yawn_beep.play()

                else:

                    yawn_start = None
                    yawn_detected = False

                if yawn_detected:

                    cv2.putText(
                        frame,
                        "YAWNING DETECTED!",
                        (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2
                    )

                # ==================================
                # 9. HEAD POSE DETECTION
                # ==================================

                pose = calculate_head_pose(
                    face_landmarks, w, h
                )

                if pose is not None:

                    pitch, yaw, roll = pose

                    # Start calibration timer
                    if calibration_start is None:
                        calibration_start = current_time

                    if neutral_pitch is None:

                        calibration_samples.append(
                            [pitch, yaw]
                        )

                        elapsed = current_time - calibration_start

                        cv2.putText(
                            frame,
                            "KEEP HEAD STRAIGHT - CALIBRATING",
                            (30, 190),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Calibration: {min(elapsed, CALIBRATION_SECONDS):.1f}/3.0s",
                            (30, 225),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )

                        if elapsed >= CALIBRATION_SECONDS:

                            if len(calibration_samples) >= 10:

                                neutral_pitch = float(
                                    np.median(
                                        [sample[0] for sample in calibration_samples]
                                    )
                                )

                                neutral_yaw = float(
                                    np.median(
                                        [sample[1] for sample in calibration_samples]
                                    )
                                )

                                smooth_pitch = 0
                                smooth_yaw = 0

                                print("Head calibration completed.")
                                print("Neutral pitch:", neutral_pitch)
                                print("Neutral yaw:", neutral_yaw)

                    else:

                        # Calculate movement relative to neutral
                        pitch_delta = pitch - neutral_pitch
                        yaw_delta = yaw - neutral_yaw

                        # Smooth noisy angle readings
                        smooth_pitch = (
                            0.8 * smooth_pitch +
                            0.2 * pitch_delta
                        )

                        smooth_yaw = (
                            0.8 * smooth_yaw +
                            0.2 * yaw_delta
                        )

                        cv2.putText(
                            frame,
                            f"Pitch change: {smooth_pitch:.1f}",
                            (30, 190),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Yaw change: {smooth_yaw:.1f}",
                            (30, 220),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )

                        # Trigger threshold
                        head_away = (
                            abs(smooth_yaw) > YAW_THRESHOLD
                            or abs(smooth_pitch) > PITCH_THRESHOLD
                        )

                        # Reset threshold (hysteresis)
                        head_returned = (
                            abs(smooth_yaw) < YAW_RESET
                            and abs(smooth_pitch) < PITCH_RESET
                        )

                        if head_away:

                            if head_away_start is None:
                                head_away_start = current_time

                            away_duration = (
                                current_time - head_away_start
                            )

                            if away_duration >= HEAD_AWAY_SECONDS:

                                cv2.putText(
                                    frame,
                                    "HEAD POSITION ALERT!",
                                    (30, 265),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 0, 255),
                                    2
                                )

                                if not head_alert_played:

                                    head_channel.play(head_beep)
                                    head_channel.queue(head_beep)

                                    head_alert_played = True

                            else:

                                cv2.putText(
                                    frame,
                                    f"Head away: {away_duration:.1f}s",
                                    (30, 265),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 165, 255),
                                    2
                                )

                        elif head_returned:

                            # Reset timer when head returns to neutral
                            head_away_start = None
                            head_alert_played = False

                            cv2.putText(
                                frame,
                                "HEAD FORWARD",
                                (30, 265),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2
                            )

                        else:

                            # Between trigger and reset thresholds:
                            # retain the current timer state
                            if head_away_start is not None:

                                away_duration = (
                                    current_time - head_away_start
                                )

                                cv2.putText(
                                    frame,
                                    f"Head away: {away_duration:.1f}s",
                                    (30, 265),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 165, 255),
                                    2
                                )

                            else:

                                cv2.putText(
                                    frame,
                                    "HEAD FORWARD",
                                    (30, 265),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 0),
                                    2
                                )

            else:

                eye_closed_start = None
                yawn_start = None
                yawn_detected = False

                head_away_start = None
                head_alert_played = False

                if alarm_playing:
                    alarm_sound.stop()
                    alarm_playing = False

                cv2.putText(
                    frame,
                    "NO FACE DETECTED",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            # ==================================
            # 10. DISPLAY
            # ==================================

            cv2.imshow(
                "DriveGuard AI - Driver Monitoring",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:

    if alarm_playing:
        alarm_sound.stop()

    head_channel.stop()

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

    print("DriveGuard AI Stopped")
