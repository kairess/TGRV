import cv2
import mediapipe as mp
import time
from datetime import datetime
import numpy as np

VIDEO_PATH = "/Users/brad/Movies/tennis/C0302.MP4"
FAST_SPEED = 30
SHORTS_SIZE = (720, 1280)

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

recording_indicator = np.zeros((30, 30, 3), dtype=np.uint8)

def create_shorts_video(image, results):
    if results.pose_landmarks:
        h, w = image.shape[:2]
        landmarks = results.pose_landmarks.landmark

        # 몸의 중심점 계산
        center_x = int(sum([lm.x for lm in landmarks]) / len(landmarks) * w)

        # 세로 비디오 크기 설정 (9:16 비율)
        shorts_width = int(h * 9 / 16)

        # 잘라낼 영역 계산
        left = max(0, center_x - shorts_width // 2)
        right = min(w, left + shorts_width)

        # 이미지 자르기
        cropped = image[:, left:right]

        # 패딩 추가
        if cropped.shape[1] < shorts_width:
            pad_left = (shorts_width - cropped.shape[1]) // 2
            pad_right = shorts_width - cropped.shape[1] - pad_left
            cropped = cv2.copyMakeBorder(cropped, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return cv2.resize(cropped, SHORTS_SIZE)  # 9:16 비율로 리사이즈

    return None

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

        shorts_image = create_shorts_video(image, results)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

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
    cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(image, f'Status: {current_state}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    image[10:40, image.shape[1]-40:image.shape[1]-10] = recording_indicator

    image_resized = cv2.resize(image, (1280, 720))
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
            filename = f"output/{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            filename_shorts = f"output/{datetime.now().strftime('%Y%m%d_%H%M%S')}_shorts.mp4"
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
