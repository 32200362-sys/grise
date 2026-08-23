"""
D.I.G - PC측 메인 파이프라인 (인식 -> 판단 -> 경로계산 -> 전송)

    [1] 인식   : YOLO(컵/장애물) + MediaPipe Pose(사람) + ArUco(로봇 위치/heading)
    [2] 판단   : 손목-컵 거리 + 접근속도 -> SAFE / WARN / DANGER
    [3] 경로계산: potential field -> 월드 속도벡터 -> 로봇 좌표계로 회전변환
    [4] 전송   : {vx, vy, w, status} JSON -> UDP -> ESP32
    [5] 제어   : ESP32 (esp32/ 폴더의 펌웨어가 담당)

실행:
    python main.py

키:
    q / ESC : 종료 (종료 시 STOP 명령을 여러 번 전송)
    스페이스 : 일시정지 토글 (로봇 정지, 인식은 계속)

안전 원칙:
    - 어떤 예외가 나든 finally에서 STOP을 보낸다.
    - 로봇 마커를 놓치면 속도를 만들지 않는다 (위치를 모르는 채로 움직이면 안 됨).
    - DANGER면 속도를 0으로 만든다 (config.SPEED_SCALE_BY_RISK).
    - ESP32는 별도로 수신 타임아웃 정지를 갖는다. PC가 죽어도 로봇은 선다.
"""

from __future__ import annotations

import math
import sys
import time
import traceback

if sys.platform == "win32":
    # Windows 콘솔 기본 코드페이지(cp949)는 이모지/em-dash 등 일부 유니코드 문자를
    # 인코딩하지 못해 print()에서 UnicodeEncodeError로 죽는다. UTF-8로 강제한다.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import cv2

import config
from comm import UdpSender
from decision import RiskEvaluator
from perception import (
    Camera,
    GazeTracker,
    HandTracker,
    MarkerObjectDetector,
    MarkerScanner,
    ObjectDetector,
    PoseTracker,
    RobotTracker,
)
from planning import PotentialField, heading_command, world_to_robot
from safety import IncidentLogger
from ui import SettingsPanel, load_tuning

# 위험 등급 -> ESP32에 보낼 status
STATUS_BY_RISK = {"SAFE": "RUN", "WARN": "SLOW", "DANGER": "STOP"}

RISK_COLOR = {
    "SAFE": (0, 220, 0),
    "WARN": (0, 200, 255),
    "DANGER": (0, 0, 255),
}

# 화면 우상단 "SETTINGS" 버튼 영역 (x1, y1, x2, y2)
# cv2.putText는 한글을 못 그리므로 영문으로 표기한다.
BTN_RECT = (config.FRAME_WIDTH - 150, 8, config.FRAME_WIDTH - 10, 44)


def _in_rect(x, y, rect) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_settings_button(frame, hovered: bool) -> None:
    x1, y1, x2, y2 = BTN_RECT
    bg = (70, 70, 70) if not hovered else (110, 110, 110)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
    cv2.putText(frame, "SETTINGS [T]", (x1 + 10, y2 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_hud(frame, world, robot, cup, obstacles, human, risk, field,
             vx_r, vy_r, w, status, fps, cup_on_robot=False):
    """디버그 오버레이."""
    h, wpx = frame.shape[:2]

    # --- 검출 박스 ---
    if config.DRAW_DETECTIONS:
        for d in [cup] if cup else []:
            x1 = int(d.cx_px - d.w_px / 2); y1 = int(d.cy_px - d.h_px / 2)
            x2 = int(d.cx_px + d.w_px / 2); y2 = int(d.cy_px + d.h_px / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"CUP {d.confidence:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        for d in obstacles:
            x1 = int(d.cx_px - d.w_px / 2); y1 = int(d.cy_px - d.h_px / 2)
            x2 = int(d.cx_px + d.w_px / 2); y2 = int(d.cy_px + d.h_px / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 255), 2)
            cv2.putText(frame, d.label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1)

    # --- 손목 강조 + 손목-컵 선 ---
    for j in human.wrists:
        cv2.circle(frame, (int(j.px[0]), int(j.px[1])), 10, (255, 255, 0), 2)
    if cup and human.wrists:
        nearest = min(
            human.wrists,
            key=lambda j: math.hypot(j.x_cm - cup.x_cm, j.y_cm - cup.y_cm),
        )
        cv2.line(frame,
                 (int(nearest.px[0]), int(nearest.px[1])),
                 (int(cup.cx_px), int(cup.cy_px)),
                 RISK_COLOR[risk.level], 2)

    # --- 컵 적재 표시 (로봇 위치에 링) ---
    if cup_on_robot and robot.detected:
        cv2.circle(frame, (int(robot.px[0]), int(robot.px[1])), 26, (0, 255, 0), 3)

    # --- 속도 벡터 (월드 -> 픽셀) ---
    if config.DRAW_FIELD_VECTOR and robot.detected and field.speed > 0.5:
        start = (int(robot.px[0]), int(robot.px[1]))
        scale_px = world.px_per_cm * 1.0  # 1초 뒤 예상 이동량을 그린다
        end = (
            int(start[0] + field.vx_world * scale_px),
            int(start[1] - field.vy_world * scale_px),
        )
        cv2.arrowedLine(frame, start, end, (0, 255, 0), 3, tipLength=0.25)

    # --- 텍스트 패널 ---
    panel = [
        (f"STATUS {status}", RISK_COLOR[risk.level], 0.9),
        (f"RISK   {risk.level}  ({risk.reason})", RISK_COLOR[risk.level], 0.55),
    ]
    if risk.distance_cm is not None:
        ttc = f"{risk.ttc_s:.2f}s" if risk.ttc_s is not None else "-"
        panel.append((
            f"hand-cup {risk.distance_cm:5.1f}cm  approach {risk.approach_speed_cm_s:+6.1f}cm/s  TTC {ttc}",
            (255, 255, 255), 0.5,
        ))
    panel.append((
        f"cmd  vx {vx_r:+6.1f}  vy {vy_r:+6.1f} cm/s   w {w:+5.2f} rad/s",
        (255, 255, 255), 0.5,
    ))
    panel.append((
        f"robot {'OK ' if robot.detected else 'LOST'}  "
        f"heading {math.degrees(robot.heading_rad):+6.1f}deg  "
        f"scale {world.px_per_cm:.2f}px/cm  {fps:4.1f}fps",
        (200, 200, 200), 0.45,
    ))
    if field.in_local_minima:
        panel.append(("LOCAL MINIMA - escaping", (0, 200, 255), 0.5))
    if not field.has_goal:
        panel.append(("NO CUP DETECTED", (0, 165, 255), 0.5))
    elif field.goal_reached:
        panel.append(("GOAL REACHED", (0, 255, 0), 0.6))
    if cup_on_robot:
        panel.append(("CUP ON ROBOT", (0, 255, 0), 0.6))

    y = 30
    for text, color, size in panel:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    size, color, 2 if size >= 0.6 else 1)
        y += int(28 * max(size, 0.6))

    cv2.putText(frame, "q=quit  space=pause", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


def main() -> None:
    print("=" * 60)
    print("D.I.G 통합 파이프라인 시작")
    print("=" * 60)

    # 저장해둔 튜닝 값이 있으면 먼저 적용한다 (config 기본값 위에 덮어씀).
    load_tuning()

    camera = Camera()
    pose_tracker = PoseTracker()
    robot_tracker = RobotTracker()

    # 마커 스캔은 프레임당 한 번만. 로봇 추적과 테스트 모드 물체 검출이 공유한다.
    scanner = MarkerScanner()
    marker_detector = MarkerObjectDetector()

    # YOLO는 가중치가 없어도 죽지 않게 한다. 테스트 모드로 계속 쓸 수 있어야 하므로.
    yolo_detector = None
    try:
        yolo_detector = ObjectDetector()
    except RuntimeError as e:
        print(f"[YOLO] 비활성화: {str(e).splitlines()[0]}")
        print("       테스트 모드(ArUco)로 진행합니다. 설정 패널 '모드' 탭에서 전환 가능.")
        config.TEST_MODE_ARUCO = True

    # 의도 신호(그립/시선)는 선택 사항이다. 모델이 없으면 끄고 계속 진행한다.
    # 이 둘은 안전 판정을 직접 내리지 않으므로 없어도 시스템은 정상 동작한다.
    hand_tracker = None
    if config.USE_HAND:
        try:
            hand_tracker = HandTracker()
        except RuntimeError as e:
            print(f"[Hand] 비활성화: {str(e).splitlines()[0]}")

    gaze_tracker = None
    if config.USE_GAZE:
        try:
            gaze_tracker = GazeTracker()
        except RuntimeError as e:
            print(f"[Gaze] 비활성화: {str(e).splitlines()[0]}")

    risk_eval = RiskEvaluator()
    field_planner = PotentialField()
    sender = UdpSender()
    panel = SettingsPanel()
    incident_logger = IncidentLogger()

    paused = False
    fps = 0.0
    last_frame_t = time.time()
    last_print_t = 0.0

    # 마우스로 SETTINGS 버튼을 누를 수 있게 한다.
    mouse_state = {"hover": False}

    # 로봇 위 컵 적재 판정: 경계값 근처에서 깜빡이지 않도록 hold를 둔다.
    cup_on_robot_state = {"last_true_t": -1e9}

    def on_mouse(event, x, y, flags, _param):
        mouse_state["hover"] = _in_rect(x, y, BTN_RECT)
        if event == cv2.EVENT_LBUTTONDOWN and mouse_state["hover"]:
            panel.toggle()

    if config.SHOW_WINDOW:
        cv2.namedWindow("D.I.G Pipeline")
        cv2.setMouseCallback("D.I.G Pipeline", on_mouse)

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("[경고] 프레임을 읽지 못했습니다. 재시도합니다.")
                sender.send(0, 0, 0, "STOP", force=True)
                time.sleep(0.05)
                continue

            now = time.time()
            dt = now - last_frame_t
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            last_frame_t = now

            world = camera.world
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---------------- [1] 인식 계층 ----------------
            # ArUco를 먼저 처리해야 px_per_cm 스케일이 갱신되고,
            # 뒤이은 YOLO/Pose의 월드 변환이 올바른 스케일을 쓴다.
            # 마커는 한 번만 스캔해서 로봇 추적과 물체 검출이 나눠 쓴다.
            scan = scanner.scan(frame)
            robot = robot_tracker.process(frame, world, scan)

            # 테스트 모드면 YOLO 대신 ArUco로 컵/장애물을 인식한다.
            # 설정 패널에서 실행 중에 전환할 수 있으므로 매 프레임 확인한다.
            use_marker = config.TEST_MODE_ARUCO or yolo_detector is None
            det_source = marker_detector if use_marker else yolo_detector
            detections = det_source.detect(frame, world, scan=scan, now=now)

            human = pose_tracker.process(rgb, world, now)

            cup = ObjectDetector.pick_cup(detections)
            obstacles = ObjectDetector.pick_obstacles(detections)

            # 로봇 마커 <-> 컵 거리가 가까우면 "컵이 로봇 위에 올라갔다"로 본다.
            if robot.detected and cup:
                dist_robot_cup = math.hypot(
                    robot.x_cm - cup.x_cm, robot.y_cm - cup.y_cm
                )
                if dist_robot_cup < config.CUP_ON_ROBOT_DIST_CM:
                    cup_on_robot_state["last_true_t"] = now
            cup_on_robot = (
                now - cup_on_robot_state["last_true_t"] < config.CUP_ON_ROBOT_HOLD_S
            )

            # 의도 신호 (그립 모양 / 시선). 없으면 None -> 판정에 영향 없음.
            hands = hand_tracker.process(rgb, world, now) if hand_tracker else None
            gaze = None
            if gaze_tracker is not None:
                target = (cup.x_cm, cup.y_cm) if cup else None
                gaze = gaze_tracker.process(rgb, world, target, now)

            # ---------------- [2] 판단 계층 ----------------
            # hands/gaze는 임계값을 넓히는 데만 쓰인다. 하드 정지는 거리·속도만 트리거한다.
            risk = risk_eval.evaluate(human, cup, now, hands=hands, gaze=gaze)

            # ---------------- [3] 경로계산 계층 ----------------
            field = field_planner.compute(
                robot=robot,
                cup=cup,
                obstacles=obstacles,
                human_pose=human,
                risk_level=risk.level,
                now=now,
            )

            # 월드 -> 로봇 좌표계 회전변환
            if robot.detected:
                vx_r, vy_r = world_to_robot(
                    field.vx_world, field.vy_world, robot.heading_rad
                )
                w_cmd = heading_command(
                    field.vx_world, field.vy_world, robot.heading_rad
                )
            else:
                # 로봇 위치를 모르면 어떤 방향으로 보낼지 알 수 없다. 무조건 정지.
                vx_r = vy_r = w_cmd = 0.0

            status = STATUS_BY_RISK[risk.level]
            if not robot.detected:
                status = "STOP"
            if paused:
                vx_r = vy_r = w_cmd = 0.0
                status = "STOP"

            # ---------------- [4] 전송 계층 ----------------
            # DANGER/정지 상황은 주기를 기다리지 않고 즉시 보낸다.
            urgent = status == "STOP"
            sender.send(vx_r, vy_r, w_cmd, status, force=urgent)

            # ---------------- 디버그 ----------------
            if now - last_print_t > (1.0 / config.PRINT_HZ):
                last_print_t = now
                d_txt = f"{risk.distance_cm:.1f}cm" if risk.distance_cm is not None else "-"
                print(
                    f"[{status:4}] risk={risk.level:6} d={d_txt:>8} "
                    f"v=({vx_r:+6.1f},{vy_r:+6.1f})cm/s w={w_cmd:+5.2f} "
                    f"robot={'O' if robot.detected else 'X'} "
                    f"cup={'O' if cup else 'X'} obs={len(obstacles)} "
                    f"[{'ArUco' if use_marker else 'YOLO'}] {fps:.1f}fps"
                )

            # 설정 패널: 바뀐 값을 config에 반영하고 현재 측정값을 보낸다.
            # 다음 프레임부터 새 값이 그대로 적용된다.
            #
            # ★ 키/값 모두 영문으로만 작성할 것 ★
            # 이 dict는 ui/settings_panel.py의 readout 라벨에 font=("Consolas", 9)로
            # 그려지는데, Consolas에는 한글 글리프가 없어서 한글이 섞이면 깨진 기호로
            # 나온다. cv2.putText가 한글을 못 그리는 것과 같은 종류의 문제다.
            panel.pump({
                "hand-cup": (f"{risk.distance_cm:.1f} cm"
                             if risk.distance_cm is not None else "-"),
                "approach": f"{risk.approach_speed_cm_s:+.1f} cm/s",
                "TTC": f"{risk.ttc_s:.2f} s" if risk.ttc_s is not None else "-",
                "risk": f"{risk.level}  ({risk.reason})",
                "intent": ("grip " if risk.intent_grip else "")
                          + ("gaze" if risk.intent_gaze else "")
                          or "none",
                "robot vel": f"vx {vx_r:+.1f}  vy {vy_r:+.1f} cm/s",
                "angular": f"{w_cmd:+.2f} rad/s",
                "goal dist": (f"{field.goal_distance_cm:.1f} cm"
                              if field.goal_distance_cm is not None else "-"),
                "status": status,
                "detector": ("ArUco(test)" if use_marker else "YOLO")
                            + f"  cup={'O' if cup else 'X'} obstacle={len(obstacles)}",
                "cup on robot": "YES" if cup_on_robot else "no",
                "FPS": f"{fps:.1f}",
            })

            if config.SHOW_WINDOW:
                if config.DRAW_POSE:
                    pose_tracker.draw(frame, human)
                if hands is not None:
                    hand_tracker.draw(frame, hands)
                if gaze is not None:
                    gaze_tracker.draw(frame, gaze, world)
                if use_marker:
                    # 마커 외곽선 + '실제 회피 반경'을 그린다.
                    # 마커 크기와 회피 반경은 다르다는 걸 눈으로 확인하기 위함.
                    scanner.draw(frame, scan)
                    marker_detector.draw(frame, world, detections)
                robot_tracker.draw(frame, robot, world)
                draw_hud(frame, world, robot, cup, obstacles, human, risk, field,
                         vx_r, vy_r, w_cmd, status, fps, cup_on_robot=cup_on_robot)
                draw_settings_button(frame, mouse_state["hover"])
                if paused:
                    cv2.putText(frame, "PAUSED", (config.FRAME_WIDTH // 2 - 90, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                # HUD까지 다 그려진 프레임을 넘긴다 - DANGER 시작 순간만 저장한다.
                incident_logger.update(frame, risk, now)

                cv2.imshow("D.I.G Pipeline", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord(' '):
                    paused = not paused
                    field_planner.reset()
                    print(f"[일시정지] {paused}")
                if key == ord('t'):
                    panel.toggle()

    except KeyboardInterrupt:
        print("\n[중단] Ctrl+C")
    except Exception:
        print("\n[예외 발생] 로봇을 정지시킵니다.")
        traceback.print_exc()
    finally:
        # 어떤 경로로 끝나든 반드시 정지 명령을 보낸다.
        panel.close()
        sender.close()
        camera.release()
        for t in (pose_tracker, hand_tracker, gaze_tracker):
            if t is not None:
                t.close()
        cv2.destroyAllWindows()
        print("종료 완료.")


if __name__ == "__main__":
    main()
