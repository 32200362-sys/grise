"""
D.I.G 실습 - 실시간 빠름/느림 추론 프로그램

목적:
  - 웹캠으로 손을 추적하고 손 속도(hand_speed_px_per_s)를 계산
  - 학습된 model.pkl로 매 프레임 빠름/느림을 예측
  - 화면에 '빠름' 또는 '느림' 텍스트를 출력

중요 - 피처 일관성:
  속도 계산 방식은 data_collector.py와 반드시 동일해야 한다.
  (학습 데이터와 추론 입력이 다른 방식으로 만들어지면 모델이 오작동한다)
  data_collector.py의 hand_speed_px_per_s 계산 로직을 그대로 옮겼다.

깜빡임 방지:
  프레임마다 속도가 튀므로 단일 프레임 예측은 텍스트가 빠르게 깜빡인다.
  최근 N프레임 속도의 이동평균을 내서 안정적으로 판단한다.

사용법:
  1) train.py로 model.pkl을 먼저 생성한다
  2) model.pkl과 이 스크립트를 같은 폴더에 둔다
  3) python realtime_predict.py 실행
  4) 손을 움직이면 화면 상단에 빠름/느림이 표시된다
  5) 'q' 키로 종료
"""

import pickle
import time
import os
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

# ---------------------------------------------------------
# 설정값 (data_collector.py와 동일하게 유지)
# ---------------------------------------------------------
MODEL_PATH = "model.pkl"
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# 이동평균 윈도우 크기 (클수록 안정적이지만 반응이 느려짐)
SMOOTH_WINDOW = 5

# 한글 라벨 매핑 (모델이 출력하는 'safe'/'danger'를 화면 문구로 변환)
#   - safe   = 빠름 (s로 수집)
#   - danger = 느림 (d로 수집)
LABEL_TO_TEXT = {
    "safe": "FAST",     # 빠름
    "danger": "SLOW",   # 느림
}

# ---------------------------------------------------------
# 모델 로드
# ---------------------------------------------------------
if not os.path.isfile(MODEL_PATH):
    raise RuntimeError(f"{MODEL_PATH} 가 없습니다. 먼저 train.py를 실행하세요.")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
feature_columns = bundle["feature_columns"]
print(f"[로드] 모델 로드 완료. 사용 피처: {feature_columns}")

# 이 추론 코드는 hand_speed_px_per_s 하나만 계산하므로,
# 학습 때 다른 피처를 썼다면 입력이 맞지 않는다. 미리 확인한다.
if feature_columns != ["hand_speed_px_per_s"]:
    raise RuntimeError(
        f"이 추론 코드는 ['hand_speed_px_per_s']만 지원합니다. "
        f"모델 피처는 {feature_columns} 입니다. train.py의 FEATURE_COLUMNS를 맞추세요."
    )

# ---------------------------------------------------------
# MediaPipe 초기화 (data_collector.py와 동일하게 Holistic 사용)
# ---------------------------------------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다. CAMERA_INDEX 값을 확인하세요.")

# ---------------------------------------------------------
# 상태 변수
# ---------------------------------------------------------
prev_hand_pos = None
prev_time = None
speed_history = deque(maxlen=SMOOTH_WINDOW)  # 최근 속도 저장 (이동평균용)

print("=" * 50)
print("실시간 빠름/느림 추론 시작")
print("손을 움직이세요. 'q' = 종료")
print("=" * 50)

# ---------------------------------------------------------
# 메인 루프
# ---------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # data_collector.py와 동일하게 좌우 반전
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    current_time = time.time()

    result = holistic.process(rgb_frame)

    # --- 손 위치 추출 (data_collector.py와 동일: 손목 landmark[0]) ---
    hand_x_px, hand_y_px = None, None
    if result.left_hand_landmarks:
        lw = result.left_hand_landmarks.landmark[0]
        hand_x_px, hand_y_px = lw.x * FRAME_WIDTH, lw.y * FRAME_HEIGHT
        mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    elif result.right_hand_landmarks:
        rw = result.right_hand_landmarks.landmark[0]
        hand_x_px, hand_y_px = rw.x * FRAME_WIDTH, rw.y * FRAME_HEIGHT
        mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    if hand_x_px is not None:
        cv2.circle(frame, (int(hand_x_px), int(hand_y_px)), 8, (0, 255, 0), -1)

    # --- 손 속도 계산 (data_collector.py와 완전히 동일한 방식) ---
    hand_speed_px_per_s = None
    if hand_x_px is not None and prev_hand_pos is not None and prev_time is not None:
        dt = current_time - prev_time
        if dt > 0:
            hand_speed_px_per_s = float(np.hypot(
                hand_x_px - prev_hand_pos[0], hand_y_px - prev_hand_pos[1]
            ) / dt)

    # --- 예측 ---
    pred_text = "..."  # 손이 없거나 속도 계산 전이면 표시할 기본값
    if hand_speed_px_per_s is not None:
        speed_history.append(hand_speed_px_per_s)
        # 이동평균으로 깜빡임 완화
        smoothed_speed = float(np.mean(speed_history))

        # 모델 입력은 학습 때와 동일한 형태 (피처 1개, shape (1,1))
        X = np.array([[smoothed_speed]], dtype=np.float32)
        pred_label = model.predict(X)[0]  # 'safe' 또는 'danger'
        pred_text = LABEL_TO_TEXT.get(pred_label, pred_label)

    # --- 화면 표시 ---
    color = (0, 200, 255) if pred_text == "FAST" else (0, 255, 0) if pred_text == "SLOW" else (200, 200, 200)
    cv2.putText(frame, pred_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    if hand_speed_px_per_s is not None:
        cv2.putText(frame, f"speed: {np.mean(speed_history):.0f} px/s", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("D.I.G Realtime Predict", frame)

    # --- 이전 프레임 값 갱신 (data_collector.py와 동일) ---
    if hand_x_px is not None:
        prev_hand_pos = (hand_x_px, hand_y_px)
    prev_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------------------------------------
# 종료 처리
# ---------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
holistic.close()
print("추론 종료.")
