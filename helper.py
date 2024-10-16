import numpy as np
import cv2

def create_shorts_video(image, results, size):
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

        return cv2.resize(cropped, size)  # 9:16 비율로 리사이즈

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

