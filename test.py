import cv2
import mediapipe as mp
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from helper import calculate_arm_speed, create_shorts_video

VIDEO_PATH = "/Users/brad/Movies/tennis/test1.MP4"
FAST_SPEED = 30
SHORTS_SIZE = (720, 1280)

SPEED_THRESHOLD = 0.01
HIGHLIGHT_DURATION = 5 # frames


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(VIDEO_PATH)

frame_rate = cap.get(cv2.CAP_PROP_FPS)
speed = 1
is_recording = False
out = None
out_shorts = None  # 쇼츠 영상을 위한 새로운 VideoWriter 객체
current_state = "Normal Playback"

prev_time = 0
curr_time = 0

prev_landmarks = None
arm_speed = 0
highlight_frames = 0
is_highlight = False

arm_speed_history = []

recording_indicator = np.zeros((30, 30, 3), dtype=np.uint8)


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

        shorts_image = create_shorts_video(image, results, SHORTS_SIZE)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

            arm_speed = calculate_arm_speed(results.pose_landmarks.landmark, prev_landmarks)

            # 하이라이트 감지
            if arm_speed > SPEED_THRESHOLD:
                highlight_frames += 1
                if highlight_frames >= HIGHLIGHT_DURATION:
                    if not is_highlight:
                        print("하이라이트 시작")
                        is_highlight = True
                        # 여기에 녹화 시작 로직 추가
            else:
                if is_highlight:
                    print("하이라이트 종료")
                    is_highlight = False
                    # 여기에 녹화 종료 로직 추가
                highlight_frames = 0

            prev_landmarks = results.pose_landmarks.landmark

            arm_speed_history.append(arm_speed)

        if shorts_image is not None:
            cv2.imshow('TGRV Shorts', shorts_image)
        else:
            cv2.destroyWindow('TGRV Shorts')
    else:
        cv2.destroyWindow('TGRV Shorts')

    if is_recording:
        out.write(image)
        if shorts_image is not None:
            out_shorts.write(shorts_image)
        cv2.circle(recording_indicator, (15, 15), 10, (0, 0, 255), -1)
    else:
        recording_indicator.fill(0)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    image_resized = cv2.resize(image, (1280, 720))

    cv2.putText(image_resized, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image_resized, f'Status: {current_state}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image_resized, f'Speed: {arm_speed:.2f}', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    image_resized[10:40, image_resized.shape[1]-40:image_resized.shape[1]-10] = recording_indicator

    cv2.imshow('TGRV', image_resized)

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
            dt = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"output/{dt}.mp4"
            filename_shorts = f"output/{dt}_shorts.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, frame_rate, (int(cap.get(3)), int(cap.get(4))))
            out_shorts = cv2.VideoWriter(filename_shorts, fourcc, frame_rate, SHORTS_SIZE)  # 쇼츠 비디오 크기로 설정
            is_recording = True
        else:
            out.release()
            out_shorts.release()
            is_recording = False
            current_state = "Normal Playback"

cap.release()
if out:
    out.release()
if out_shorts:
    out_shorts.release()
cv2.destroyAllWindows()

plt.figure(figsize=(16, 9))
plt.plot(arm_speed_history)
plt.title('Arm Speed History')
plt.xlabel('Frame')
plt.ylabel('Speed')
plt.grid(True)
plt.savefig(f'output/{datetime.now().strftime("%Y%m%d_%H%M%S")}_arm_speed_history.png')
plt.show()
