"""
D.I.G - 전역 설정

모든 튜닝 값을 여기 한 곳에 모은다.
계층별 코드는 이 모듈만 import 하고, 서로의 설정을 직접 참조하지 않는다.

좌표계 정의 (매우 중요 - 모든 계층이 이 규약을 따른다)
---------------------------------------------------------
  * 카메라는 작업 공간을 내려다보는 천장(오버헤드) 카메라를 가정한다.
  * 월드 좌표 = 이미지 평면을 그대로 쓰되, 단위는 cm, Y축만 위로 뒤집는다.
        world_x_cm = px_x / px_per_cm
        world_y_cm = (FRAME_HEIGHT - px_y) / px_per_cm
    -> X: 오른쪽, Y: 위쪽인 오른손 좌표계. 각도는 반시계(CCW) 방향이 양수.
  * px_per_cm은 매 프레임 ArUco 로봇 마커의 실제 픽셀 크기로부터 추정한다.
    (마커가 안 보이면 아래 DEFAULT_PX_PER_CM으로 폴백)
"""

# =========================================================
# 1) 카메라 / 좌표 변환
# =========================================================
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

# 오버헤드 카메라는 좌우 반전하지 않는다.
# (반전하면 월드 좌표계가 왼손계가 되어 heading/회전 방향이 전부 뒤집힌다)
FLIP_HORIZONTAL = False

# 마커가 안 잡힐 때 쓸 기본 스케일. 처음 한 번 실측해서 맞춰둘 것.
DEFAULT_PX_PER_CM = 8.0

# =========================================================
# 2) 인식 계층 - YOLO (Roboflow 학습 모델)
# =========================================================
# backend 선택:
#   "ultralytics" : Roboflow에서 YOLOv8/v11 .pt 로 export 해서 로컬 추론 (권장, 가장 빠름)
#   "roboflow"    : roboflow inference SDK / 호스팅 API 사용
#   "stub"        : 모델 없이 파이프라인만 돌려볼 때 (항상 빈 검출)
YOLO_BACKEND = "ultralytics"

# backend="ultralytics" 일 때
YOLO_WEIGHTS = "models/dig_yolo.pt"

# backend="roboflow" 일 때
ROBOFLOW_MODEL_ID = "your-workspace/your-project/1"
ROBOFLOW_API_KEY = ""  # 환경변수 ROBOFLOW_API_KEY 가 있으면 그쪽이 우선

YOLO_CONF_THRESHOLD = 0.45
YOLO_IMG_SIZE = 640

# Roboflow에서 라벨링한 클래스 이름을 역할에 매핑한다.
# 여기 이름만 실제 데이터셋에 맞게 바꾸면 나머지 코드는 손댈 필요 없다.
CLASS_CUP = "cup"
CLASS_OBSTACLE_NAMES = ("obstacle", "box", "bottle", "chair")

# YOLO는 매 프레임 돌릴 필요가 없다. N프레임마다 1회 추론하고 사이는 직전 결과를 유지.
YOLO_EVERY_N_FRAMES = 2

# =========================================================
# 3) 인식 계층 - MediaPipe Pose (사람)
# =========================================================
# ★ Tasks API를 쓴다. legacy `mp.solutions.pose`는 mediapipe 0.10.30+ 에서 제거됨.
#   모델 파일이 별도로 필요하다:  python tools/download_pose_model.py
#   lite(가장 빠름) / full(균형) / heavy(가장 정확) 중 선택.
POSE_MODEL_PATH = "models/pose_landmarker_lite.task"

POSE_NUM_POSES = 1           # 추적할 사람 수. 1명이면 훨씬 빠르다.
POSE_MIN_DETECTION_CONF = 0.5
POSE_MIN_PRESENCE_CONF = 0.5
POSE_MIN_TRACKING_CONF = 0.5
POSE_MIN_VISIBILITY = 0.4    # 이 값 미만인 랜드마크는 없는 것으로 취급

# =========================================================
# 3-b) 인식 계층 - 손 스켈레톤 / 그립 모양 (HandLandmarker)
# =========================================================
# Pose는 손목 1점만 준다. 손가락 21점은 이 모델이 따로 필요하다.
#   python tools/download_models.py
USE_HAND = True
HAND_MODEL_PATH = "models/hand_landmarker.task"
HAND_NUM_HANDS = 2
HAND_MIN_DETECTION_CONF = 0.5
HAND_MIN_PRESENCE_CONF = 0.5
HAND_MIN_TRACKING_CONF = 0.5

# 그립 판정 임계 (hand_world_landmarks 기준, 손 크기로 정규화된 무차원 값)
#   aperture = 엄지끝-검지끝 거리 / 손크기   (편 손 ~1.5-2.5, 집기 ~0.1-0.3)
#   openness = 손가락끝-손목 평균거리 / 손크기 (편 손 ~2.5-3.0, 주먹 ~1.2-1.5)
# 사람마다 다르므로 설정 패널에서 실제 값을 보며 맞출 것.
GRIP_APERTURE_MIN = 0.55
GRIP_APERTURE_MAX = 1.70
GRIP_OPENNESS_MAX = 2.60

# =========================================================
# 3-c) 인식 계층 - 시선(머리 방향) (FaceLandmarker)
# =========================================================
# 홍채가 아니라 머리 방향을 쓴다. 1~2m에서 홍채는 노이즈가 너무 크다.
USE_GAZE = True
FACE_MODEL_PATH = "models/face_landmarker.task"
FACE_MIN_DETECTION_CONF = 0.5
FACE_MIN_PRESENCE_CONF = 0.5
FACE_MIN_TRACKING_CONF = 0.5

# 얼굴 정면이 표준모델 +Z인지 -Z인지는 버전에 따라 다를 수 있다.
# 화면의 시선 화살표가 반대로 나오면 -1로 바꿀 것. (설정 패널에도 있음)
GAZE_FORWARD_SIGN = 1.0

# 머리 방향과 '머리->컵' 방향의 각도가 이 안이면 "컵을 보고 있다"로 판정
GAZE_CONE_DEG = 35.0

# =========================================================
# 4) 인식 계층 - ArUco (로봇 위치 + heading)
# =========================================================
# cv2 상수 이름을 문자열로 둔다. robot_tracker.py가 실제 값으로 변환한다.
# (config가 cv2를 import하면 판단/경로계산 계층까지 OpenCV에 묶여서
#  하드웨어 없이 로직만 테스트하는 게 불가능해진다)
ARUCO_DICT_NAME = "DICT_4X4_50"
ROBOT_MARKER_ID = 0          # 로봇 위에 붙인 마커 ID
MARKER_SIZE_CM = 8.0         # 마커 한 변의 실제 길이(cm) - 반드시 실측할 것

# 마커의 위쪽 변(TL -> TR) 방향을 로봇의 정면(+X_robot)으로 정의한다.
# 실제 로봇에서 정면이 다르면 이 오프셋(도 단위)으로 보정한다.
MARKER_HEADING_OFFSET_DEG = 0.0

# 검출이 튀는 것을 막기 위한 지수이동평균 계수 (0=고정, 1=필터 없음)
ROBOT_POSE_EMA_ALPHA = 0.5

# =========================================================
# 4-b) 테스트 모드 - ArUco로 컵/장애물 인식 (YOLO 대체)
# =========================================================
# True면 YOLO 대신 마커로 물체를 인식한다.
# 설정 패널의 "모드" 탭에서 실행 중에도 켜고 끌 수 있다.
# YOLO 모델이 없거나, 경로계산을 결정적으로 튜닝할 때 쓴다.
TEST_MODE_ARUCO = False

# ★ ID -> (역할, 실제 반경 cm) ★
#
# 반경은 반드시 '물체의 실제 크기'를 적어야 한다. 마커 크기가 아니다.
# 마커는 위치만 알려줄 뿐 물체가 얼마나 큰지는 모른다.
# 30cm 상자에 5cm 마커를 붙였는데 반경을 2.5cm로 두면
# 로봇이 상자를 긁고 지나간다. potential field의 척력이 이 값을 쓴다.
#
# 역할은 "cup"(목표) 또는 "obstacle"(회피) 둘 중 하나.
MARKER_OBJECTS = {
    1:  ("cup", 4.0),        # 컵 반지름 약 4cm
    10: ("obstacle", 15.0),  # 상자 등
    11: ("obstacle", 15.0),
    12: ("obstacle", 10.0),
    13: ("obstacle", 10.0),
}

# 마커가 가려진 동안 마지막 위치를 유지할 시간 [s].
# ArUco는 귀퉁이만 가려도 완전히 사라지는데, 하필 손이 컵으로 갈 때
# 컵 마커를 덮기 쉽다. 그 순간 목표가 사라지면 로봇이 멈춰버린다.
MARKER_OBJECT_HOLD_S = 0.8

# =========================================================
# 4-c) 로봇 위 컵 적재 감지
# =========================================================
# 로봇 마커와 컵 사이 거리가 이 안쪽이면 "컵이 로봇 위에 올라갔다"로 판정한다.
# 로봇 반경 + 컵 반경보다 조금 크게 잡아 여유를 둔다.
CUP_ON_ROBOT_DIST_CM = 12.0

# 경계값 근처에서 판정이 깜빡이지 않도록 짧게 유지한다.
CUP_ON_ROBOT_HOLD_S = 0.3

# =========================================================
# 5) 판단 계층 - 위험 판정
# =========================================================
# 손목-컵 거리 기준 (cm)
RISK_DANGER_DIST_CM = 15.0   # 이 안쪽이면 즉시 DANGER
RISK_WARN_DIST_CM = 35.0     # 이 안쪽이면 WARN

# 접근 속도 기준 (cm/s, 양수 = 가까워지는 중)
RISK_APPROACH_SPEED_CM_S = 25.0

# 충돌 예상 시간(TTC) 기준 (s). 접근 속도가 임계 이상이고 TTC가 이보다 짧으면 DANGER.
RISK_TTC_DANGER_S = 0.8
RISK_TTC_WARN_S = 1.6

# 거리/속도 스무딩 (프레임 단위 노이즈 제거)
RISK_DIST_EMA_ALPHA = 0.4
RISK_SPEED_EMA_ALPHA = 0.3

# 채터링 방지: 한번 DANGER가 뜨면 최소 이 시간만큼 유지
RISK_DANGER_HOLD_S = 0.7
RISK_WARN_HOLD_S = 0.4

# ---------------------------------------------------------
# 위기 상황(DANGER) 스크린샷
#   DANGER 에피소드가 시작되는 순간(직전 프레임까지 DANGER가 아니었을 때)의
#   HUD 오버레이 포함 프레임을 INCIDENT_LOG_DIR에 저장한다.
#   같은 에피소드가 계속 이어지는 동안은 다시 찍지 않는다.
# ---------------------------------------------------------
INCIDENT_LOG_ENABLED = True
INCIDENT_LOG_DIR = "incidents"
INCIDENT_LOG_COOLDOWN_S = 5.0   # 연속 저장 최소 간격

# ---------------------------------------------------------
# 의도(intent) 신호 - 그립 모양 / 시선
#
# ★ 설계 원칙 ★
#   의도 신호는 임계값을 넓히는 역할만 한다. 하드 정지를 직접 트리거하지 않는다.
#   이유:
#     1) 신뢰도가 낮다. 사람이 잠깐 쳐다봤다고 로봇이 서면 쓸 수 없는 시스템이 된다.
#     2) fail-safe가 아니다. 얼굴이 가려지거나 손이 안 잡히면 '위험 없음'으로
#        조용히 읽힌다. 안전 기능이 침묵으로 실패하는 건 최악의 형태다.
#   그래서 운동학(거리·속도)만 실제 판정을 내리고, 의도는 그 임계값을 키워
#   '더 일찍' 반응하게만 만든다. 절대 위험도를 낮추지 않는다.
# ---------------------------------------------------------
INTENT_USE_GRIP = True       # 그립 모양을 의도 신호로 쓸지
INTENT_USE_GAZE = True       # 시선(머리 방향)을 의도 신호로 쓸지

# 의도가 감지되면 거리/TTC 임계값에 곱할 배수 (1.0 = 효과 없음)
INTENT_DIST_BOOST = 1.5      # 예: WARN 35cm -> 52.5cm 에서 반응
INTENT_TTC_BOOST = 1.4

# 의도 신호도 잠깐 끊길 수 있으므로 이 시간만큼 유지한다
INTENT_HOLD_S = 1.0

# =========================================================
# 6) 경로계산 계층 - Potential Field
# =========================================================
# --- 인력 (목표 컵) ---
# 인력의 최대 크기 = PF_K_ATTRACT * PF_ATTRACT_MAX_CM = 72.
# 아래 척력 게인들은 모두 이 값(72)과 비교해서 읽으면 된다.
PF_K_ATTRACT = 1.2
PF_ATTRACT_MAX_CM = 60.0     # 이보다 멀면 인력을 일정하게 (conic well)
PF_GOAL_TOLERANCE_CM = 8.0   # 이 안에 들어오면 도착으로 간주

# --- 척력 ---
# 척력은 정규화 거리 d^ = d/d0 로 계산한다:  F = k * (1/d^ - 1) / d^^2
# 이렇게 하면 게인 k가 영향 반경(d0)과 무관해지고, 인력 크기(72)와 직접 비교된다.
#   d^=0.75 -> 0.59k   d^=0.5 -> 4k   d^=0.33 -> 18k   d^=0.2 -> 100k
#
# (원래 형태인 k*(1/d - 1/d0)/d^2 는 d0에 세제곱으로 의존해서,
#  게인을 900으로 놔도 30cm에서 척력이 0.011밖에 안 나온다. 인력 72에 완전히 묻힌다.)

# 척력 (장애물): d^=0.5(=22.5cm)에서 100 -> 인력 72를 넘어 확실히 밀어낸다
PF_K_REPULSE_OBSTACLE = 25.0
PF_OBSTACLE_INFLUENCE_CM = 45.0

# 척력 (사람): 장애물보다 크게, 더 멀리서부터 회피
PF_K_REPULSE_HUMAN = 60.0
PF_HUMAN_INFLUENCE_CM = 70.0

# 수치 폭주 방지: d^ 가 이보다 작으면 이 값으로 고정 (1/d^^2 발산 방지)
PF_MIN_DIST_RATIO = 0.05

# 위험 등급에 따라 사람 척력을 증폭
PF_HUMAN_GAIN_BY_RISK = {"SAFE": 1.0, "WARN": 1.8, "DANGER": 3.0}

# --- 힘 -> 속도 변환 ---
# 합력에 이 값을 곱해 cm/s를 얻고 MAX_LINEAR_SPEED_CM_S로 클램프한다.
# 0.5면 인력 최대(72)일 때 36cm/s가 되어 최대속도에 딱 맞는다.
# 힘을 항상 최대속도로 정규화하면 안 된다 -- 장애물 때문에 합력이 줄어도
# 여전히 전속력으로 돌진하게 된다.
PF_FORCE_TO_SPEED = 0.5

# 지역 최소점(local minima) 탈출: 합력이 이보다 작고 목표에 도달 못했으면
# 목표 방향의 수직 성분을 섞어 흔들어준다. (힘 2.0 = 약 1cm/s)
PF_LOCAL_MINIMA_FORCE = 2.0
PF_ESCAPE_GAIN = 0.6

# 출력 속도 제한
MAX_LINEAR_SPEED_CM_S = 35.0
MAX_ANGULAR_SPEED_RAD_S = 1.8

# 가속도 제한 (cm/s^2) - 급격한 명령 변화로 로봇이 튀는 것 방지
MAX_LINEAR_ACCEL_CM_S2 = 90.0

# 위험 등급별 속도 스케일
SPEED_SCALE_BY_RISK = {"SAFE": 1.0, "WARN": 0.4, "DANGER": 0.0}

# =========================================================
# 7) 자세 제어 (heading)
# =========================================================
# 진행 방향을 정면으로 맞추는 P 게인
HEADING_KP = 2.0
HEADING_DEADBAND_RAD = 0.08  # 이 안이면 회전 명령 0

# =========================================================
# 8) 전송 계층 - UDP
# =========================================================
ESP32_IP = "192.168.0.50"    # ESP32의 IP (고정 IP 권장)
ESP32_PORT = 8888
UDP_SEND_HZ = 30             # 전송 주기. ESP32 타임아웃보다 충분히 빨라야 한다.

# =========================================================
# 9) 디버그 표시
# =========================================================
SHOW_WINDOW = True
DRAW_POSE = True
DRAW_DETECTIONS = True
DRAW_FIELD_VECTOR = True
PRINT_HZ = 2.0               # 콘솔 로그 주기
