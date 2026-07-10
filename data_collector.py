"""
D.I.G 프로젝트 - 손/팔/물체 동작 데이터 수집 + 라벨링 도구 (프로토타입)

목적:1
  - 웹캠으로 손+팔(MediaPipe Holistic)과 컵 역할 ArUco 마커를 동시에 추적
  - 매 프레임마다 특징값(거리, 속도, 팔꿈치/어깨 위치 등)을 계산
  - 사용자가 키보드로 실시간 라벨을 달면서 동작을 녹화
  - 결과를 CSV로 저장 (이후 학습 단계에서 그대로 사용 가능)

추적 랜드마크 (Pose 기준):
  - 11: 왼쪽 어깨 / 12: 오른쪽 어깨
  - 13: 왼쪽 팔꿈치 / 14: 오른쪽 팔꿈치
  - 15: 왼쪽 손목 / 16: 오른쪽 손목

조작 방법 (실행 중):
  - 's' 키: 정상(safe) 라벨로 녹화 시작/종료 토글
  - 'd' 키: 위험(danger) 라벨로 녹화 시작/종료 토글
  - 'q' 키: 종료 및 CSV 저장
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import os

# ---------------------------------------------------------
# 설정값 (본인 환경에 맞게 조정)
# ---------------------------------------------------------
MARKER_SIZE_CM = 5.0
ARUCO_DICT = cv2.aruco.DICT_4X4_50
OUTPUT_CSV = "motion_dataset.csv"
CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FOCAL_LENGTH_PX = FRAME_WIDTH
CAMERA_MATRIX = np.array([
    [FOCAL_LENGTH_PX, 0, FRAME_WIDTH / 2],
    [0, FOCAL_LENGTH_PX, FRAME_HEIGHT / 2],
    [0, 0, 1]
], dtype=np.float32)
DIST_COEFFS = np.zeros((4, 1))

# ---------------------------------------------------------
# 초기화 - Holistic으로 손 + 팔/포즈 동시 추적
# ---------------------------------------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

holistic = mp_holistic.Holistic(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Pose 랜드마크 인덱스
POSE_LEFT_SHOULDER  = mp_pose.PoseLandmark.LEFT_SHOULDER
POSE_RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER
POSE_LEFT_ELBOW     = mp_pose.PoseLandmark.LEFT_ELBOW
POSE_RIGHT_ELBOW    = mp_pose.PoseLandmark.RIGHT_ELBOW
POSE_LEFT_WRIST     = mp_pose.PoseLandmark.LEFT_WRIST
POSE_RIGHT_WRIST    = mp_pose.PoseLandmark.RIGHT_WRIST

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다. CAMERA_INDEX 값을 확인하세요.")

# ---------------------------------------------------------
# CSV 준비 - 팔 랜드마크 컬럼 추가
# ---------------------------------------------------------
file_exists = os.path.isfile(OUTPUT_CSV)
csv_file = open(OUTPUT_CSV, mode="a", newline="")
csv_writer = csv.writer(csv_file)
if not file_exists:
    csv_writer.writerow([
        "timestamp", "label",
        # 손 위치 (손목 기준)
        "hand_x_px", "hand_y_px",
        # 팔 랜드마크 (우세한 쪽 - 마커에 더 가까운 손 기준)
        "shoulder_x_px", "shoulder_y_px",
        "elbow_x_px", "elbow_y_px",
        "wrist_x_px", "wrist_y_px",
        # 팔 각도 (어깨-팔꿈치-손목)
        "elbow_angle_deg",
        # 마커 정보
        "marker_x_px", "marker_y_px",
        "marker_distance_cm",
        "hand_marker_dist_px",
        # 속도
        "hand_speed_px_per_s",
        "approach_speed_cm_per_s",
    ])

# ---------------------------------------------------------
# 유틸 함수
# ---------------------------------------------------------
def landmark_to_px(landmark):
    return landmark.x * FRAME_WIDTH, landmark.y * FRAME_HEIGHT

def calc_angle(a, b, c):
    """세 점 a-b-c에서 b를 꼭짓점으로 하는 각도(도) 계산"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

# ---------------------------------------------------------
# 상태 변수
# ---------------------------------------------------------
recording_label = None
prev_hand_pos = None
prev_time = None
prev_marker_distance = None
sample_count = {"safe": 0, "danger": 0}

print("=" * 50)
print("D.I.G 데이터 수집 도구 시작 (손 + 팔 추적)")
print("'s' = 정상(safe) 라벨 토글, 'd' = 위험(danger) 라벨 토글, 'q' = 종료")
print(f"마커 실제 크기 설정값: {MARKER_SIZE_CM}cm")
print("=" * 50)

# ---------------------------------------------------------
# 메인 루프
# ---------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    current_time = time.time()

    # --- Holistic 추적 (손 + 포즈) ---
    result = holistic.process(rgb_frame)

    # 손 랜드마크 그리기
    if result.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if result.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # 포즈(팔) 랜드마크 그리기 - 팔(어깨/팔꿈치/손목)만 직접 그림
    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark
        ARM_POINTS = [
            POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER,
            POSE_LEFT_ELBOW, POSE_RIGHT_ELBOW,
            POSE_LEFT_WRIST, POSE_RIGHT_WRIST,
        ]
        ARM_CONNECTIONS = [
            (POSE_LEFT_SHOULDER, POSE_LEFT_ELBOW),
            (POSE_LEFT_ELBOW, POSE_LEFT_WRIST),
            (POSE_RIGHT_SHOULDER, POSE_RIGHT_ELBOW),
            (POSE_RIGHT_ELBOW, POSE_RIGHT_WRIST),
            (POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER),
        ]
        # 연결선
        for a, b in ARM_CONNECTIONS:
            if lm[a].visibility > 0.3 and lm[b].visibility > 0.3:
                pa = (int(lm[a].x * FRAME_WIDTH), int(lm[a].y * FRAME_HEIGHT))
                pb = (int(lm[b].x * FRAME_WIDTH), int(lm[b].y * FRAME_HEIGHT))
                cv2.line(frame, pa, pb, (245, 117, 66), 3)
        # 관절 점
        for p in ARM_POINTS:
            if lm[p].visibility > 0.3:
                pt = (int(lm[p].x * FRAME_WIDTH), int(lm[p].y * FRAME_HEIGHT))
                cv2.circle(frame, pt, 6, (245, 66, 230), -1)

    # --- 손 위치 추출 (두 손 중 마커에 더 가까운 쪽 선택) ---
    hand_x_px, hand_y_px = None, None
    dominant_side = None  # "left" or "right"

    candidates = []
    if result.left_hand_landmarks:
        lw = result.left_hand_landmarks.landmark[0]
        candidates.append(("left", lw.x * FRAME_WIDTH, lw.y * FRAME_HEIGHT))
    if result.right_hand_landmarks:
        rw = result.right_hand_landmarks.landmark[0]
        candidates.append(("right", rw.x * FRAME_WIDTH, rw.y * FRAME_HEIGHT))

    if candidates:
        # 마커 위치가 있으면 더 가까운 손, 없으면 첫 번째 손
        dominant_side, hand_x_px, hand_y_px = candidates[0]
        cv2.circle(frame, (int(hand_x_px), int(hand_y_px)), 8, (0, 255, 0), -1)

    # --- 팔 랜드마크 추출 (dominant_side 기준) ---
    shoulder_x_px = shoulder_y_px = None
    elbow_x_px = elbow_y_px = None
    wrist_x_px = wrist_y_px = None
    elbow_angle_deg = None

    if result.pose_landmarks and dominant_side is not None:
        lm = result.pose_landmarks.landmark
        if dominant_side == "left":
            sh = lm[POSE_LEFT_SHOULDER]
            el = lm[POSE_LEFT_ELBOW]
            wr = lm[POSE_LEFT_WRIST]
        else:
            sh = lm[POSE_RIGHT_SHOULDER]
            el = lm[POSE_RIGHT_ELBOW]
            wr = lm[POSE_RIGHT_WRIST]

        shoulder_x_px, shoulder_y_px = landmark_to_px(sh)
        elbow_x_px, elbow_y_px = landmark_to_px(el)
        wrist_x_px, wrist_y_px = landmark_to_px(wr)

        elbow_angle_deg = calc_angle(
            (shoulder_x_px, shoulder_y_px),
            (elbow_x_px, elbow_y_px),
            (wrist_x_px, wrist_y_px)
        )

        # 팔꿈치 각도 화면 표시
        cv2.putText(frame, f"elbow: {elbow_angle_deg:.0f}deg",
                    (int(elbow_x_px) + 10, int(elbow_y_px)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    # --- ArUco 마커 추적 ---
    corners, ids, _ = aruco_detector.detectMarkers(frame)
    marker_x_px, marker_y_px, marker_distance_cm = None, None, None
    if ids is not None and len(corners) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        marker_corners = corners[0][0]
        marker_x_px = float(np.mean(marker_corners[:, 0]))
        marker_y_px = float(np.mean(marker_corners[:, 1]))

        marker_obj_points = np.array([
            [-MARKER_SIZE_CM / 2, MARKER_SIZE_CM / 2, 0],
            [MARKER_SIZE_CM / 2, MARKER_SIZE_CM / 2, 0],
            [MARKER_SIZE_CM / 2, -MARKER_SIZE_CM / 2, 0],
            [-MARKER_SIZE_CM / 2, -MARKER_SIZE_CM / 2, 0],
        ], dtype=np.float32)
        success, rvec, tvec = cv2.solvePnP(
            marker_obj_points, marker_corners, CAMERA_MATRIX, DIST_COEFFS
        )
        if success:
            marker_distance_cm = float(np.linalg.norm(tvec))

    # --- 특징값 계산 ---
    hand_marker_dist_px = None
    hand_speed_px_per_s = 0.0
    approach_speed_cm_per_s = 0.0

    if hand_x_px is not None and marker_x_px is not None:
        hand_marker_dist_px = float(np.hypot(hand_x_px - marker_x_px, hand_y_px - marker_y_px))

    if hand_x_px is not None and prev_hand_pos is not None and prev_time is not None:
        dt = current_time - prev_time
        if dt > 0:
            hand_speed_px_per_s = float(np.hypot(
                hand_x_px - prev_hand_pos[0], hand_y_px - prev_hand_pos[1]
            ) / dt)

    if marker_distance_cm is not None and prev_marker_distance is not None and prev_time is not None:
        dt = current_time - prev_time
        if dt > 0:
            approach_speed_cm_per_s = float((prev_marker_distance - marker_distance_cm) / dt)

    # --- 화면 표시 ---
    status_text = f"LABEL: {recording_label.upper() if recording_label else 'NONE'}"
    status_color = (0, 0, 255) if recording_label == "danger" else (0, 255, 0) if recording_label == "safe" else (200, 200, 200)
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, f"safe:{sample_count['safe']}  danger:{sample_count['danger']}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if marker_distance_cm is not None:
        cv2.putText(frame, f"marker_dist: {marker_distance_cm:.1f}cm", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if hand_marker_dist_px is not None:
        cv2.putText(frame, f"hand-marker px: {hand_marker_dist_px:.0f}", (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("D.I.G Data Collector", frame)

    # --- 라벨이 켜져 있으면 매 프레임 기록 ---
    if recording_label is not None and hand_x_px is not None:
        csv_writer.writerow([
            current_time, recording_label,
            hand_x_px, hand_y_px,
            shoulder_x_px, shoulder_y_px,
            elbow_x_px, elbow_y_px,
            wrist_x_px, wrist_y_px,
            elbow_angle_deg,
            marker_x_px, marker_y_px,
            marker_distance_cm,
            hand_marker_dist_px,
            hand_speed_px_per_s,
            approach_speed_cm_per_s,
        ])
        sample_count[recording_label] += 1

    # --- 이전 프레임 값 갱신 ---
    if hand_x_px is not None:
        prev_hand_pos = (hand_x_px, hand_y_px)
    if marker_distance_cm is not None:
        prev_marker_distance = marker_distance_cm
    prev_time = current_time

    # --- 키 입력 처리 ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        recording_label = None if recording_label == "safe" else "safe"
        print(f"[라벨 변경] -> {recording_label}")
    elif key == ord('d'):
        recording_label = None if recording_label == "danger" else "danger"
        print(f"[라벨 변경] -> {recording_label}")

# ---------------------------------------------------------
# 종료 처리
# ---------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
csv_file.close()
holistic.close()

print("=" * 50)
print(f"수집 종료. 저장 파일: {OUTPUT_CSV}")
print(f"수집된 샘플 수 - safe: {sample_count['safe']}, danger: {sample_count['danger']}")
print("=" * 50)
