import cv2
import mediapipe as mp
import time
from datetime import datetime
import numpy as np

VIDEO_PATH = "/Users/brad/Movies/tennis/C0302.MP4"
FAST_SPEED = 30
SHORTS_SIZE = (720, 1280)

SPEED_THRESHOLD = 0.02
HIGHLIGHT_DURATION = 30 # frames


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

recording_indicator = np.zeros((30, 30, 3), dtype=np.uint8)

frame_count = 0  # 프레임 카운터 추가

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

def calculate_arm_speed(current_landmarks, prev_landmarks):
    if not prev_landmarks:
        return 0

    # 오른쪽 손목과 팔꿈치의 랜드마크 인덱스
    RIGHT_WRIST = 16
    RIGHT_ELBOW = 14
    
    # 왼쪽 손목과 팔꿈치의 랜드마크 인덱스
    LEFT_WRIST = 15
    LEFT_ELBOW = 13

    def calculate_speed(current, prev):
        return np.sqrt((current.x - prev.x)**2 + (current.y - prev.y)**2 + (current.z - prev.z)**2)

    right_wrist_speed = calculate_speed(current_landmarks[RIGHT_WRIST], prev_landmarks[RIGHT_WRIST])
    right_elbow_speed = calculate_speed(current_landmarks[RIGHT_ELBOW], prev_landmarks[RIGHT_ELBOW])
    left_wrist_speed = calculate_speed(current_landmarks[LEFT_WRIST], prev_landmarks[LEFT_WRIST])
    left_elbow_speed = calculate_speed(current_landmarks[LEFT_ELBOW], prev_landmarks[LEFT_ELBOW])

    # 오른쪽과 왼쪽 팔의 평균 속도 계산
    right_arm_speed = (right_wrist_speed + right_elbow_speed) / 2
    left_arm_speed = (left_wrist_speed + left_elbow_speed) / 2

    # 더 빠른 쪽의 팔 속도를 반환
    return max(right_arm_speed, left_arm_speed)


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

            # 3프레임마다 속도 계산
            if frame_count % 3 == 0:
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

            frame_count += 1

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
