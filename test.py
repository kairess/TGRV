import cv2
import mediapipe as mp
import time
from datetime import datetime

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture("/Users/brad/Movies/tennis/C0302.MP4")

FAST_SPEED = 30

frame_rate = cap.get(cv2.CAP_PROP_FPS)
speed = 1
is_recording = False
out = None
current_state = "Normal Playback"

prev_time = 0
curr_time = 0

while cap.isOpened():
    if speed != 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + speed)

    success, image = cap.read()
    if not success:
        print("Failed to read frame")
        break

    # inference
    if speed == 1:
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(image, f'Status: {current_state}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    image_resized = cv2.resize(image, (1280, 720))
    cv2.imshow('TGRV', image_resized)

    if is_recording:
        out.write(image)

    key = cv2.waitKey(30)

    if key == ord('q'):
        break

    if key == ord('a'):
        speed = speed - FAST_SPEED
        current_state = f"Fast Rewind {speed}"
    elif key == ord('d'):
        speed = speed + FAST_SPEED
        current_state = f"Fast Forward {speed}"
    elif key == ord('s'):
        speed = 1
        current_state = "Normal Playback"
    elif key == ord('r'):
        if not is_recording:
            filename = f"output/{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, frame_rate, (int(cap.get(3)), int(cap.get(4))))
            is_recording = True
            current_state = "Recording"
        else:
            out.release()
            is_recording = False
            current_state = "Normal Playback"

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
