"""
하드웨어 / 카메라 / 외부 패키지 없이 [2] 판단 · [3] 경로계산 · [5] 역기구학 을 검증한다.

이 계층들은 순수 수학이라 표준 라이브러리만으로 돌릴 수 있다.
좌표계 부호 실수(로봇이 반대로 가는 사고)를 잡아내는 게 주 목적.

사용법:
    python tools/selftest.py
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from decision import RiskEvaluator  # noqa: E402
from planning import PotentialField, heading_command, world_to_robot, wrap_pi  # noqa: E402

# ---------- 인식 계층 결과를 흉내내는 최소 스텁 ----------
@dataclass
class FakeRobot:
    x_cm: float = 0.0
    y_cm: float = 0.0
    heading_rad: float = 0.0
    detected: bool = True
    px: tuple = (0.0, 0.0)


@dataclass
class FakeDet:
    x_cm: float
    y_cm: float
    radius_cm: float = 5.0
    confidence: float = 0.9
    label: str = "cup"


@dataclass
class FakeJoint:
    x_cm: float
    y_cm: float
    px: tuple = (0.0, 0.0)
    visible: bool = True


@dataclass
class FakeHuman:
    detected: bool = True
    joints: list = None

    @property
    def wrists(self):
        return self.joints or []

    @property
    def repulsion_points(self):
        return self.joints or []


EMPTY_HUMAN = FakeHuman(detected=False, joints=[])

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}   {detail}")


# ============================================================
print("\n[3-a] 좌표 회전변환  world_to_robot")
# ============================================================
# heading=0 (정면=+X). 월드에서 +X로 가라 => 로봇도 정면 전진.
vx, vy = world_to_robot(10.0, 0.0, 0.0)
check("heading 0deg, 월드+X -> 정면 전진", abs(vx - 10) < 1e-6 and abs(vy) < 1e-6, f"got ({vx:.2f},{vy:.2f})")

# heading=90도 (정면=+Y). 월드에서 +X로 가라 => 로봇 기준 오른쪽(-Y_robot).
vx, vy = world_to_robot(10.0, 0.0, math.radians(90))
check("heading 90deg, 월드+X -> 로봇 우측(-vy)", abs(vx) < 1e-6 and abs(vy + 10) < 1e-6, f"got ({vx:.2f},{vy:.2f})")

# heading=90도, 월드 +Y로 가라 => 로봇 정면 전진.
vx, vy = world_to_robot(0.0, 10.0, math.radians(90))
check("heading 90deg, 월드+Y -> 정면 전진", abs(vx - 10) < 1e-6 and abs(vy) < 1e-6, f"got ({vx:.2f},{vy:.2f})")

# heading=180도, 월드 +X => 로봇 기준 후진.
vx, vy = world_to_robot(10.0, 0.0, math.radians(180))
check("heading 180deg, 월드+X -> 후진", abs(vx + 10) < 1e-6, f"got ({vx:.2f},{vy:.2f})")

# 회전변환은 크기를 보존해야 한다.
for h in (0.3, 1.1, -2.4, 3.0):
    a, b = world_to_robot(7.0, -3.0, h)
    check(f"heading {h:+.1f}rad 크기 보존", abs(math.hypot(a, b) - math.hypot(7, -3)) < 1e-6)

# ============================================================
print("\n[3-b] heading P 제어")
# ============================================================
w = heading_command(10.0, 0.0, 0.0)
check("이미 정렬됨 -> 회전 0", abs(w) < 1e-9, f"got {w}")

w = heading_command(0.0, 10.0, 0.0)   # 90도 왼쪽으로 가야 함 -> CCW 양수
check("좌측으로 가야함 -> w > 0 (CCW)", w > 0, f"got {w:.3f}")

w = heading_command(0.0, -10.0, 0.0)  # 90도 오른쪽 -> CW 음수
check("우측으로 가야함 -> w < 0 (CW)", w < 0, f"got {w:.3f}")

w = heading_command(10.0, 0.0, math.radians(179))  # -pi/pi 경계 넘김
check("각도 wrap 경계에서 폭주 안함", abs(w) <= config.MAX_ANGULAR_SPEED_RAD_S + 1e-9, f"got {w:.3f}")

check("wrap_pi(3pi) == pi 근처", abs(abs(wrap_pi(3 * math.pi)) - math.pi) < 1e-6)

w = heading_command(0.5, 0.0, 0.0)    # 속도 거의 0
check("정지 상태에선 제자리 회전 안함", abs(w) < 1e-9, f"got {w}")

# ============================================================
print("\n[3-c] Potential Field")
# ============================================================
t = 0.0


def step(pf, robot, cup, obstacles, human, risk, dt=0.05, n=1):
    """가속도 제한이 있으므로 여러 스텝 돌려서 정상상태를 본다."""
    global t
    res = None
    for _ in range(n):
        t += dt
        res = pf.compute(robot, cup, obstacles, human, risk, t)
    return res


pf = PotentialField()
robot = FakeRobot(0, 0, 0)

# 대기 모드(SEEK_CUP=False): SAFE면 컵이 있어도, 사람이 근처에 있어도 가만히 있는다.
config.SEEK_CUP = False
pf.reset()
r = step(pf, robot, FakeDet(100.0, 0.0), [], FakeHuman(joints=[FakeJoint(0.0, 20.0)]), "SAFE", n=20)
check("대기 모드 SAFE -> 정지", r.speed < 1e-6, f"speed={r.speed:.2f}")
pf.reset()
r = step(pf, robot, FakeDet(100.0, 0.0), [], FakeHuman(joints=[FakeJoint(0.0, 20.0)]), "DANGER", n=40)
check("대기 모드 DANGER -> 반대방향(-Y)으로 회피", r.vy_world < 0 and r.speed > 1.0,
      f"v=({r.vx_world:.2f},{r.vy_world:.2f})")

# 아래 테스트는 컵 추적 동작(SEEK_CUP=True) 검증
config.SEEK_CUP = True
cup = FakeDet(100.0, 0.0)                       # 로봇 정동쪽 1m
r = step(pf, robot, cup, [], EMPTY_HUMAN, "SAFE", n=40)
check("장애물 없음 -> 컵 방향(+X)으로 이동", r.vx_world > 5 and abs(r.vy_world) < 1e-6,
      f"got ({r.vx_world:.2f},{r.vy_world:.2f})")
check("최대속도 제한 준수", r.speed <= config.MAX_LINEAR_SPEED_CM_S + 1e-6, f"speed={r.speed:.2f}")

pf.reset()
r = step(pf, FakeRobot(97, 0, 0), cup, [], EMPTY_HUMAN, "SAFE", n=10)
check("목표 도달 판정", r.goal_reached and r.speed < 1e-6, f"reached={r.goal_reached} speed={r.speed:.2f}")

pf.reset()
r = step(pf, robot, None, [], EMPTY_HUMAN, "SAFE", n=10)
check("컵 없음 -> 정지", r.speed < 1e-6 and not r.has_goal, f"speed={r.speed:.2f}")

pf.reset()
r = step(pf, FakeRobot(detected=False), cup, [], EMPTY_HUMAN, "SAFE", n=10)
check("로봇 미검출 -> 무조건 정지", r.speed < 1e-6, f"speed={r.speed:.2f}")

# 사람이 로봇 바로 위쪽(+Y)에 있으면 아래쪽(-Y)으로 밀려야 한다.
pf.reset()
human_above = FakeHuman(joints=[FakeJoint(0.0, 20.0)])
r = step(pf, robot, cup, [], human_above, "SAFE", n=40)
check("사람이 +Y에 있으면 -Y로 회피", r.vy_world < 0, f"vy={r.vy_world:.2f}")

# 위험 등급이 올라가면 사람 회피가 더 강해져야 한다.
pf.reset()
r_safe = step(pf, robot, cup, [], human_above, "SAFE", n=40)
pf.reset()
r_warn = step(pf, robot, cup, [], human_above, "WARN", n=40)
# WARN은 속도 스케일 0.4가 곱해지므로 방향비(|vy|/|vx|)로 비교한다.
ratio_safe = abs(r_safe.vy_world) / max(abs(r_safe.vx_world), 1e-6)
ratio_warn = abs(r_warn.vy_world) / max(abs(r_warn.vx_world), 1e-6)
check("WARN에서 회피 각도가 더 큼", ratio_warn > ratio_safe, f"safe={ratio_safe:.3f} warn={ratio_warn:.3f}")

pf.reset()
r = step(pf, robot, cup, [], human_above, "DANGER", n=40)
# DANGER에서는 인력을 끄고 척력만 작동시킨다 - 사람이 실제로 가까우면(여기서는 20cm,
# 영향반경 안) 그 반대 방향(-Y)으로 물러나야 한다. 얼어붙어 정지하는 게 아니다.
check("DANGER -> 인력 꺼짐, 척력만으로 반대방향(-Y) 회피",
      r.vy_world < 0 and r.speed > 1.0 and abs(r.vx_world) < 1e-6,
      f"speed={r.speed:.2f} v=({r.vx_world:.2f},{r.vy_world:.2f})")

# --- 장애물 회피: 여기가 게인 스케일 버그가 나는 자리다 ---
# 척력 게인이 인력에 비해 너무 작으면 로봇이 장애물을 뚫고 전속력으로 돌진한다.
pf.reset()
r_clear = step(pf, robot, cup, [], EMPTY_HUMAN, "SAFE", n=60)

pf.reset()
obs_head = [FakeDet(40.0, 0.0, radius_cm=10.0, label="obstacle")]   # 표면까지 30cm
r_head = step(pf, robot, cup, obs_head, EMPTY_HUMAN, "SAFE", n=60)
check("정면 장애물 -> 확실히 감속", r_head.speed < r_clear.speed * 0.85,
      f"clear={r_clear.speed:.2f} obstacle={r_head.speed:.2f} cm/s")

# 더 가까우면 전진이 아예 멈추거나 후퇴해야 한다.
pf.reset()
obs_near = [FakeDet(22.0, 0.0, radius_cm=10.0, label="obstacle")]   # 표면까지 12cm
r_near = step(pf, robot, cup, obs_near, EMPTY_HUMAN, "SAFE", n=60)
check("근접 정면 장애물 -> 전진 중단/후퇴", r_near.vx_world <= 0.01,
      f"vx={r_near.vx_world:.2f} (장애물을 뚫고 전진 중이면 게인 스케일 문제)")

# 비스듬히 놓인 장애물은 옆으로 우회해야 한다.
pf.reset()
obs_off = [FakeDet(35.0, 12.0, radius_cm=8.0, label="obstacle")]    # 살짝 위쪽
r_off = step(pf, robot, cup, obs_off, EMPTY_HUMAN, "SAFE", n=60)
check("비스듬한 장애물 -> 반대쪽(-Y)으로 우회", r_off.vy_world < -0.5,
      f"got ({r_off.vx_world:.2f},{r_off.vy_world:.2f})")

# 척력은 '표면'까지의 거리를 써야 한다 -> 같은 중심거리면 큰 물체가 더 세게 민다.
pf.reset()
r_small = step(pf, robot, cup, [FakeDet(45.0, 0.0, radius_cm=2.0, label="obstacle")],
               EMPTY_HUMAN, "SAFE", n=60)
pf.reset()
r_big = step(pf, robot, cup, [FakeDet(45.0, 0.0, radius_cm=20.0, label="obstacle")],
             EMPTY_HUMAN, "SAFE", n=60)
check("같은 중심거리면 큰 물체를 더 많이 회피", r_big.speed < r_small.speed,
      f"small={r_small.speed:.2f} big={r_big.speed:.2f}")

# 영향 반경 밖의 장애물은 아무 영향이 없어야 한다.
pf.reset()
r_far = step(pf, robot, cup,
             [FakeDet(0.0, -200.0, radius_cm=5.0, label="obstacle")],
             EMPTY_HUMAN, "SAFE", n=60)
check("영향 반경 밖 장애물은 무시", abs(r_far.speed - r_clear.speed) < 1e-6,
      f"clear={r_clear.speed:.2f} far={r_far.speed:.2f}")

# 속도 -> 힘 매핑이 선형이어야 감속이 의미를 갖는다 (항상 최대속도로 정규화하면 안 됨)
pf.reset()
r_close_goal = step(pf, FakeRobot(90.0, 0.0, 0.0), cup, [], EMPTY_HUMAN, "SAFE", n=60)
check("목표 근처에선 감속 (quadratic well)",
      0 < r_close_goal.speed < config.MAX_LINEAR_SPEED_CM_S,
      f"speed={r_close_goal.speed:.2f}")

# 가속도 제한: 첫 프레임부터 최대속도로 튀면 안 된다.
pf2 = PotentialField()
t += 0.05
pf2.compute(robot, cup, [], EMPTY_HUMAN, "SAFE", t)   # 첫 호출은 기준점
t += 0.05
r1 = pf2.compute(robot, cup, [], EMPTY_HUMAN, "SAFE", t)
t += 0.05
r2 = pf2.compute(robot, cup, [], EMPTY_HUMAN, "SAFE", t)
dv = math.hypot(r2.vx_world - r1.vx_world, r2.vy_world - r1.vy_world)
check("가속도 제한 동작", dv <= config.MAX_LINEAR_ACCEL_CM_S2 * 0.05 + 1e-3, f"dv={dv:.2f}")

# ============================================================
print("\n[2] 판단 계층")
# ============================================================
ev = RiskEvaluator()
cup0 = FakeDet(0.0, 0.0)
tt = 1000.0

# 멀리 있고 안 움직임 -> SAFE
for _ in range(20):
    tt += 0.05
    s = ev.evaluate(FakeHuman(joints=[FakeJoint(80.0, 0.0)]), cup0, tt)
check("멀리 정지한 손 -> SAFE", s.level == "SAFE", f"got {s.level} ({s.reason})")

# 아주 가까이 -> DANGER
ev2 = RiskEvaluator()
tt = 2000.0
for _ in range(20):
    tt += 0.05
    s = ev2.evaluate(FakeHuman(joints=[FakeJoint(5.0, 0.0)]), cup0, tt)
check("근접(5cm) -> DANGER", s.level == "DANGER", f"got {s.level} ({s.reason})")

# 빠르게 접근 -> 거리가 아직 멀어도 DANGER/WARN
ev3 = RiskEvaluator()
tt = 3000.0
d = 120.0
levels = []
for _ in range(40):
    tt += 0.05
    d = max(d - 5.0, 2.0)        # 100 cm/s로 접근
    s = ev3.evaluate(FakeHuman(joints=[FakeJoint(d, 0.0)]), cup0, tt)
    levels.append(s.level)
check("고속 접근 -> 접근속도 양수로 검출", s.approach_speed_cm_s > 0, f"v={s.approach_speed_cm_s:.1f}")
check("고속 접근 -> DANGER 도달", "DANGER" in levels, f"levels={set(levels)}")

# TTC 적용거리: 컵에서 먼 곳(150->110cm)의 빠른 움직임은 무시, 안쪽(80cm 이내)은 반응
ev_far = RiskEvaluator()
tt = 3500.0
d = 150.0
lv_far = []
for _ in range(8):
    tt += 0.05
    d -= 5.0                       # 100 cm/s로 접근하지만 아직 110cm 밖
    lv_far.append(ev_far.evaluate(FakeHuman(joints=[FakeJoint(d, 0.0)]), cup0, tt).level)
check("TTC 적용거리 밖의 고속 움직임 -> SAFE", set(lv_far) == {"SAFE"}, f"levels={set(lv_far)} d={d}")
ev_near = RiskEvaluator()
tt = 3600.0
d = 112.0
lv_near = []
for _ in range(12):
    tt += 0.05
    d -= 5.0                       # 112 -> 52cm, 안쪽으로 들어오면서 반응해야 함
    lv_near.append(ev_near.evaluate(FakeHuman(joints=[FakeJoint(d, 0.0)]), cup0, tt).level)
check("TTC 적용거리 안으로 들어오면 고속 접근 -> DANGER", "DANGER" in lv_near, f"levels={set(lv_near)}")

# hold: DANGER 직후 손을 치워도 바로 SAFE로 안 떨어져야 함
ev4 = RiskEvaluator()
tt = 4000.0
for _ in range(10):
    tt += 0.05
    ev4.evaluate(FakeHuman(joints=[FakeJoint(3.0, 0.0)]), cup0, tt)
tt += 0.05
s_after = ev4.evaluate(FakeHuman(joints=[FakeJoint(200.0, 0.0)]), cup0, tt)
check("DANGER 후 즉시 SAFE로 안 떨어짐 (hold)", s_after.level == "DANGER", f"got {s_after.level}")
tt += config.RISK_DANGER_HOLD_S + 0.2
s_later = ev4.evaluate(FakeHuman(joints=[FakeJoint(200.0, 0.0)]), cup0, tt)
check("hold 시간 후엔 해제됨", s_later.level != "DANGER", f"got {s_later.level}")

# 입력이 없으면 SAFE + 미분 상태 리셋
ev5 = RiskEvaluator()
s = ev5.evaluate(FakeHuman(detected=False, joints=[]), None, 5000.0)
check("사람/컵 미검출 -> SAFE", s.level == "SAFE" and s.distance_cm is None)

# ---- 엄격한 그립 판정 ----
from perception.hand_tracker import HandInfo, HandsResult  # noqa: E402

# 모양은 config의 현재 그립 범위에서 계산한다 (캘리브레이션으로 값이 바뀌어도 테스트가 유지되게).
AP_MID = (config.GRIP_APERTURE_MIN + config.GRIP_APERTURE_MAX) / 2
OP_MID = (config.GRIP_OPENNESS_MIN + config.GRIP_OPENNESS_MAX) / 2
C_SHAPE = dict(aperture=AP_MID, openness=OP_MID)   # 컵을 감싸는 C자
OPEN_SHAPE = (config.GRIP_APERTURE_MAX + 0.8, config.GRIP_OPENNESS_MAX + 0.8)


def hand(x, aperture, openness):
    return HandInfo(wrist_x_cm=x, wrist_y_cm=0.0, aperture=aperture,
                    openness=openness, valid=True)


check("C자 손 -> 잡기 모양", hand(0, **C_SHAPE).grasp_ready)
check("편 손 -> 잡기 모양 아님", not hand(0, *OPEN_SHAPE).grasp_ready)
check("주먹 -> 잡기 모양 아님", not hand(0, AP_MID, config.GRIP_OPENNESS_MIN - 0.4).grasp_ready)
check("집기(pinch) -> 잡기 모양 아님", not hand(0, config.GRIP_APERTURE_MIN * 0.5, OP_MID).grasp_ready)


def run_grip(hands_fn, n, step_cm=0.0, start=10.0, t0=6000.0):
    """손목을 start에서 매 프레임 step_cm씩 컵 쪽으로 옮기며 n프레임 평가."""
    e = RiskEvaluator()
    t, d, s = t0, start, None
    for _ in range(n):
        t += 0.05
        d = max(d - step_cm, 1.0)
        s = e.evaluate(FakeHuman(joints=[FakeJoint(d, 0.0)]), cup0, t, hands=hands_fn(d))
    return s


# 34cm에서 C자 손으로 10cm/s로 천천히 다가와 컵을 잡는다 (3초, 마지막 4cm)
s = run_grip(lambda d: HandsResult(True, [hand(d, **C_SHAPE)]), 60, step_cm=0.5, start=34.0)
check("C자 손으로 천천히 다가와 잡음 -> SAFE (정상 픽업)", s.level == "SAFE",
      f"got {s.level} ({s.reason})")

s = run_grip(lambda d: HandsResult(True, [hand(d, **C_SHAPE)]), 3)
check("C자 손 0.15s만 -> 아직 grip 아님 (확정시간)", not s.intent_grip and s.level == "DANGER",
      f"got {s.level} grip={s.intent_grip}")

s = run_grip(lambda d: HandsResult(True, [hand(d, *OPEN_SHAPE), hand(80.0, **C_SHAPE)]), 20)
check("컵에서 먼 다른 손이 C자 -> grip 아님", not s.intent_grip and s.level == "DANGER",
      f"got {s.level} grip={s.intent_grip}")

s = run_grip(lambda d: HandsResult(True, [hand(d, **C_SHAPE)]), 20, start=60.0)
check("컵에서 35cm 밖의 C자 손 -> grip 아님", not s.intent_grip, f"grip={s.intent_grip}")

# 손 떨림: 프레임 중 일부(0.05s)만 모양이 깨져도 grip이 유지/확정되어야 한다
e = RiskEvaluator()
t, got = 8000.0, []
for i in range(20):
    t += 0.05
    shape = hand(20.0, 2.0, 2.8) if i % 5 == 4 else hand(20.0, **C_SHAPE)   # 5프레임에 1번 깨짐
    s = e.evaluate(FakeHuman(joints=[FakeJoint(20.0, 0.0)]), cup0, t,
                   hands=HandsResult(True, [shape]))
    got.append(s.intent_grip)
check("가끔 모양이 깨져도(0.05s) grip 유지", got[-1] and sum(got) >= 12, f"grip frames={sum(got)}/20")

# 모양이 GRIP_HOLD_S보다 오래 깨지면 grip이 풀리고, 재진입 시 다시 확인을 거쳐야 한다
e = RiskEvaluator()
t = 8500.0
for _ in range(8):
    t += 0.05
    e.evaluate(FakeHuman(joints=[FakeJoint(20.0, 0.0)]), cup0, t,
               hands=HandsResult(True, [hand(20.0, **C_SHAPE)]))
for _ in range(14):    # 0.7s 동안 편 손 (GRIP_HOLD_S 0.5s보다 길게)
    t += 0.05
    e.evaluate(FakeHuman(joints=[FakeJoint(20.0, 0.0)]), cup0, t,
               hands=HandsResult(True, [hand(20.0, *OPEN_SHAPE)]))
t += 0.05
s = e.evaluate(FakeHuman(joints=[FakeJoint(20.0, 0.0)]), cup0, t,
               hands=HandsResult(True, [hand(20.0, **C_SHAPE)]))
check("모양이 오래 깨진 뒤 재진입 -> 즉시 grip 아님 (재확인 필요)", not s.intent_grip,
      f"grip={s.intent_grip}")

# grip 인정 즉시 이전 DANGER/WARN hold가 끊긴다
e = RiskEvaluator()
t = 9000.0
for _ in range(6):     # 편 손으로 가까이 -> DANGER (hold 1.5s)
    t += 0.05
    s = e.evaluate(FakeHuman(joints=[FakeJoint(10.0, 0.0)]), cup0, t,
                   hands=HandsResult(True, [hand(10.0, *OPEN_SHAPE)]))
check("편 손으로 근접 -> DANGER", s.level == "DANGER", f"got {s.level}")
for _ in range(8):     # C자로 바꿈 (0.4s) -> grip 인정
    t += 0.05
    s = e.evaluate(FakeHuman(joints=[FakeJoint(10.0, 0.0)]), cup0, t,
                   hands=HandsResult(True, [hand(10.0, **C_SHAPE)]))
check("grip 인정되면 DANGER hold 남아도 즉시 SAFE", s.level == "SAFE" and s.intent_grip,
      f"got {s.level} ({s.reason})")
for _ in range(14):    # 0.7s 동안 편 손 (GRIP_HOLD_S 0.5s보다 길게) -> grip 풀림
    t += 0.05
    s = e.evaluate(FakeHuman(joints=[FakeJoint(10.0, 0.0)]), cup0, t,
                   hands=HandsResult(True, [hand(10.0, *OPEN_SHAPE)]))
check("grip이 풀리면 다시 DANGER", s.level == "DANGER", f"got {s.level} ({s.reason})")

# C자 모양으로 0.5s 가까이 머문 뒤(grip 확정) 100cm/s로 휘두름
e = RiskEvaluator()
t, levels = 7000.0, []
for d in [34.0] * 10 + [34.0 - 5.0 * i for i in range(1, 7)]:
    t += 0.05
    s = e.evaluate(FakeHuman(joints=[FakeJoint(d, 0.0)]), cup0, t,
                   hands=HandsResult(True, [hand(d, **C_SHAPE)]))
    levels.append((s.level, s.intent_grip))
check("C자여도 고속 접근 -> DANGER", ("DANGER", True) in levels, f"levels={levels[-6:]}")

# ============================================================
print("\n[5] 3륜 옴니 역기구학 (ESP32 펌웨어와 동일한 식)")
# ============================================================
WHEEL_ANGLE_DEG = [0.0, 120.0, 240.0]
L = 9.0


def ik(vx, vy, w):
    return [
        -math.sin(math.radians(a)) * vx + math.cos(math.radians(a)) * vy + L * w
        for a in WHEEL_ANGLE_DEG
    ]


v = ik(10, 0, 0)
check("순수 전진 -> 0도 바퀴 정지", abs(v[0]) < 1e-9, f"got {[round(x,2) for x in v]}")
check("순수 전진 -> 나머지 두 바퀴 역방향 동일크기",
      abs(v[1] + v[2]) < 1e-9 and abs(v[1]) > 1, f"got {[round(x,2) for x in v]}")
check("순수 전진 -> 바퀴속도 합 0 (회전 없음)", abs(sum(v)) < 1e-9, f"sum={sum(v):.4f}")

v = ik(0, 0, 1.0)
check("순수 회전 -> 세 바퀴 모두 같은 값", max(v) - min(v) < 1e-9, f"got {[round(x,2) for x in v]}")
check("순수 회전 -> L*w 크기", abs(v[0] - L) < 1e-9, f"got {v[0]:.3f}")

v = ik(0, 10, 0)
check("순수 좌측 횡이동 -> 바퀴속도 합 0", abs(sum(v)) < 1e-9, f"sum={sum(v):.4f}")
check("순수 좌측 횡이동 -> 0도 바퀴가 최대", abs(v[0] - 10) < 1e-9, f"got {[round(x,2) for x in v]}")

# 전진+회전이 선형으로 합쳐지는지
a, b = ik(10, 3, 0), ik(0, 0, 0.5)
c = ik(10, 3, 0.5)
check("역기구학 선형성", all(abs(a[i] + b[i] - c[i]) < 1e-9 for i in range(3)))

# ============================================================
print("\n[4] 전송 계층 - JSON 직렬화")
# ============================================================
from comm.udp_sender import CommandPacket, clamp_to_firmware  # noqa: E402

import json  # noqa: E402
pkt = CommandPacket(type="cmd_vel", session_id=7, seq=42,
                    vx=12.3456, vy=-4.5678, w=0.35, status="RUN")
decoded = json.loads(pkt.to_json())
check("JSON 키가 ESP32 typed 파서와 일치",
      set(decoded) == {"type", "session_id", "seq", "vx", "vy", "w", "status"},
      f"got {set(decoded)}")
check("소수점 3자리 반올림", decoded["vx"] == 12.346, f"got {decoded['vx']}")
check("패킷 크기 < 512B (ESP32 버퍼)", len(pkt.to_json()) < 512, f"{len(pkt.to_json())}B")
stop = json.loads(CommandPacket(type="stop", session_id=7, seq=43).to_json())
check("stop 패킷에는 속도 필드가 없음", set(stop) == {"type", "session_id", "seq"}, f"got {set(stop)}")

cvx, cvy, cw = clamp_to_firmware(30.0, 40.0, -5.0)
check("펌웨어 선속도 한계로 클램프",
      abs(math.hypot(cvx, cvy) - config.FIRMWARE_LINEAR_LIMIT_CM_S) < 1e-9)
check("클램프 후 방향 유지", abs(cvy / cvx - 40.0 / 30.0) < 1e-9)
check("펌웨어 각속도 한계로 클램프", cw == -config.FIRMWARE_ANGULAR_LIMIT_RAD_S)
check("펌웨어 SPEED_LIMIT 폴트 기준(30cm/s, 2rad/s) 미만",
      config.FIRMWARE_LINEAR_LIMIT_CM_S < 30.0 and config.FIRMWARE_ANGULAR_LIMIT_RAD_S < 2.0)

# ============================================================
print("\n[6] 로봇 마커 인식 강건성")
# ============================================================
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from perception.marker_scanner import MarkerScan, MarkerScanner  # noqa: E402
from perception.robot_tracker import RobotTracker  # noqa: E402

_rng = np.random.default_rng(0)
_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT_NAME))


def marker_frame(marker_id, side=130, blur=0, contrast=1.0, cx=640, cy=360):
    """흰 여백이 있는 ArUco를 회색 배경 위에 그린다 (이미지 파일 없이 생성)."""
    m = cv2.aruco.generateImageMarker(_dict, marker_id, side)
    m = cv2.copyMakeBorder(m, side // 5, side // 5, side // 5, side // 5, cv2.BORDER_CONSTANT, value=255)
    m = (m.astype(np.float32) * contrast + 127 * (1 - contrast)).clip(0, 255).astype(np.uint8)
    img = np.full((720, 1280), 110, np.uint8)
    h, w = m.shape
    img[cy - h // 2:cy - h // 2 + h, cx - w // 2:cx - w // 2 + w] = m
    if blur > 1:
        k = np.zeros((blur, blur), np.float32)
        k[blur // 2, :] = 1.0 / blur
        img = cv2.filter2D(img, -1, k)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


scanner = MarkerScanner()
check("선명한 마커 -> 검출", config.ROBOT_MARKER_ID in scanner.scan(marker_frame(config.ROBOT_MARKER_ID)))

saved = config.ARUCO_FALLBACK_SCALES
blurred = marker_frame(config.ROBOT_MARKER_ID, blur=15)
config.ARUCO_FALLBACK_SCALES = ()
found_plain = config.ROBOT_MARKER_ID in scanner.scan(blurred)
config.ARUCO_FALLBACK_SCALES = saved
found_fb = config.ROBOT_MARKER_ID in scanner.scan(blurred)
check("흐린 마커(15px): 축소 재검출이 있으면 검출", found_fb, f"fallback={found_fb} plain={found_plain}")

sc = scanner.scan(blurred)
c = sc.get(config.ROBOT_MARKER_ID)
check("재검출 좌표가 원본 해상도로 복원됨(중심 오차 < 6px)",
      c is not None and abs(MarkerScan.center_px(c)[0] - 640) < 6 and abs(MarkerScan.center_px(c)[1] - 360) < 6,
      f"center={MarkerScan.center_px(c) if c is not None else None}")

check("저대비 마커(0.35) 검출",
      config.ROBOT_MARKER_ID in scanner.scan(marker_frame(config.ROBOT_MARKER_ID, contrast=0.35)))

# 모르는 ID(config에 없는 ID)는 재검출 결과로 받아들이지 않는다
unknown = max([config.ROBOT_MARKER_ID, *config.MARKER_OBJECTS]) + 7
sc_unknown = scanner.scan(marker_frame(unknown, blur=15))
check("흐림 재검출은 알려진 ID만 수용 (모르는 ID 무시)", unknown not in sc_unknown, f"ids={sc_unknown.ids}")

# 마커 없는 지저분한 장면에서 로봇/컵/장애물 ID 오검출이 없어야 한다
known = {config.ROBOT_MARKER_ID} | set(config.MARKER_OBJECTS)
bad = 0
for _ in range(60):
    g = cv2.GaussianBlur(_rng.integers(60, 200, (720, 1280), dtype=np.uint8), (0, 0), 6)
    for _ in range(50):
        x, y = int(_rng.integers(0, 1200)), int(_rng.integers(0, 650))
        w, h = int(_rng.integers(15, 160)), int(_rng.integers(15, 160))
        cv2.rectangle(g, (x, y), (x + w, y + h), int(_rng.choice([0, 255])), int(_rng.choice([2, 4, -1])))
    bad += len(known & set(scanner.scan(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)).ids))
check("마커 없는 지저분한 장면 -> 알려진 ID 오검출 0", bad == 0, f"false known-id detections={bad}")


class _World:
    px_per_cm = 10.0

    def update_scale(self, s):
        self.px_per_cm = s

    def to_world(self, x, y):
        return x / 10.0, (720 - y) / 10.0


tracker = RobotTracker()
corners = np.array([[600, 330], [680, 330], [680, 410], [600, 410]], dtype=float)
seen = MarkerScan(by_id={config.ROBOT_MARKER_ID: corners})
lost = MarkerScan(by_id={})
w0 = _World()
t0 = 10000.0
p = tracker.process(None, w0, seen, now=t0)
check("마커 보임 -> detected, held 아님", p.detected and not p.held)
p = tracker.process(None, w0, lost, now=t0 + config.ROBOT_HOLD_S * 0.5)
check("순간 놓침(HOLD 이내) -> 직전 자세 유지 (detected, held)", p.detected and p.held,
      f"detected={p.detected} held={p.held}")
check("유지 중 위치는 직전 값", abs(p.x_cm - 64.0) < 1e-6, f"x={p.x_cm}")
p = tracker.process(None, w0, lost, now=t0 + config.ROBOT_HOLD_S + 0.05)
check("HOLD 초과 -> detected=False (STOP 대상)", (not p.detected) and (not p.held), f"detected={p.detected}")
p = tracker.process(None, w0, lost, now=t0 + 5.0)
check("오래 못 봄 -> 계속 detected=False", not p.detected)
p = tracker.process(None, w0, seen, now=t0 + 6.0)
check("다시 보이면 detected 복귀", p.detected and not p.held)
check("한 번도 못 본 상태 -> detected=False", not RobotTracker().process(None, w0, lost, now=t0).detected)


# ============================================================
print("\n[7] 네트워크 카메라 (로컬 MJPEG 서버로 검증)")
# ============================================================
import http.server  # noqa: E402
import socketserver  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

from perception.camera import Camera  # noqa: E402


class _Mjpeg:
    """프레임마다 회색 밝기를 20+i로 바꿔 내보내는 가짜 폰 카메라 (i = 프레임 번호)."""

    def __init__(self, w, h, fps=30):
        self.w, self.h, self.fps = w, h, fps
        self.index = 0
        self.alive = True
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path != "/video":
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while outer.alive:
                        img = np.full((outer.h, outer.w, 3), 20 + (outer.index % 200), np.uint8)
                        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg.tobytes() + b"\r\n")
                        outer.index += 1
                        time.sleep(1.0 / outer.fps)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True

        self.srv = S(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        self.alive = False
        self.srv.shutdown()
        self.srv.server_close()


def _use_source(url):
    config.CAMERA_SOURCE = url


_saved_src = config.CAMERA_SOURCE
_saved_size = (config.FRAME_WIDTH, config.FRAME_HEIGHT)
try:
    # (1) 16:9 입력은 그대로 통과, 영상이 늦어지지 않는다
    fake = _Mjpeg(1280, 720)
    _use_source(f"http://127.0.0.1:{fake.port}/video")
    cam = Camera()
    got = None
    for _ in range(40):                       # 첫 프레임이 올 때까지
        ok, f = cam.read()
        if ok:
            got = f
            break
    check("스트림에서 프레임 수신", got is not None)
    check("출력 크기 = config 크기", got is not None and got.shape[:2] == (config.FRAME_HEIGHT, config.FRAME_WIDTH),
          f"shape={None if got is None else got.shape}")

    # 처리가 느린 상황(약 8fps)을 흉내내며 1.5초 읽고, 마지막 프레임이 서버의 현재 프레임에 가까운지 본다
    last_i = None
    t_end = time.time() + 1.5
    while time.time() < t_end:
        ok, f = cam.read()
        if ok:
            last_i = float(f.mean()) - 20
        time.sleep(0.12)
    ok, f = cam.read()
    if ok:
        last_i = float(f.mean()) - 20
    lag_frames = fake.index - last_i
    check("느리게 읽어도 최신 프레임을 받는다 (지연 < 0.3s)", last_i is not None and lag_frames < 9,
          f"server={fake.index} got~{last_i} lag={lag_frames:.1f} frames")

    # (2) 서버가 멈추면 오래된 영상을 쓰지 않고 실패를 돌려준다 (-> main이 STOP)
    fake.stop()
    time.sleep(config.CAMERA_STALE_S + 0.3)
    fails = 0
    t0 = time.time()
    for _ in range(3):
        ok, f = cam.read()
        fails += (not ok)
    check("스트림이 멈추면 read 실패 (오래된 프레임 미사용)", fails == 3, f"fails={fails}")
    check("실패 판정이 빠르다 (3회 < 1.5s)", time.time() - t0 < 1.5, f"{time.time() - t0:.2f}s")
    cam.release()

    # (3) 4:3 입력은 비율을 유지해 맞추고 남는 곳은 검게
    fake43 = _Mjpeg(640, 480)
    _use_source(f"http://127.0.0.1:{fake43.port}/video")
    cam = Camera()
    got = None
    for _ in range(40):
        ok, f = cam.read()
        if ok:
            got = f
            break
    check("4:3 입력 -> 출력은 config 크기", got is not None and got.shape[:2] == (config.FRAME_HEIGHT, config.FRAME_WIDTH),
          f"shape={None if got is None else got.shape}")
    if got is not None:
        left_black = got[:, :20].mean() < 5
        center_gray = got[300:420, 600:680].mean() > 15
        check("4:3 입력 -> 좌우에 검은 여백, 가운데는 영상", left_black and center_gray,
              f"left={got[:, :20].mean():.1f} center={got[300:420, 600:680].mean():.1f}")
    cam.release()
    fake43.stop()

    # (4) 연결할 수 없는 주소는 안내 메시지와 함께 RuntimeError
    _use_source("http://127.0.0.1:9/video")
    try:
        Camera()
        raised = False
        msg = ""
    except RuntimeError as e:
        raised, msg = True, str(e)
    check("열 수 없는 주소 -> RuntimeError + /video 안내", raised and "/video" in msg, msg[:60])
finally:
    config.CAMERA_SOURCE = _saved_src
    config.FRAME_WIDTH, config.FRAME_HEIGHT = _saved_size


# ============================================================
print("\n[8] 실기 보정: 좌우, 회피 지속, 최소 속도, 손 대체 인식")
# ============================================================
from planning import to_body_command  # noqa: E402
from perception.hand_tracker import HandInfo as _HI, HandsResult as _HR  # noqa: E402

# 실제 설정(SEEK_CUP=False: 평소엔 가만히 있고 위험할 때만 물러난다)으로 검증한다.
_seek_saved = config.SEEK_CUP
config.SEEK_CUP = False

# 좌우 부호 보정: 로봇이 +X를 보고 있을 때(heading 0), 월드 +Y(위)는 몸체 vy다
vx_b, vy_b = to_body_command(0.0, 10.0, 0.0)
check("heading 0: 월드 위(+Y) -> vy에 ROBOT_VY_SIGN 적용",
      abs(vy_b - 10.0 * config.ROBOT_VY_SIGN) < 1e-9 and abs(vx_b) < 1e-9, f"vx={vx_b} vy={vy_b}")
vx_b, vy_b = to_body_command(10.0, 0.0, 0.0)
check("heading 0: 월드 오른쪽(+X) -> vx에 ROBOT_VX_SIGN 적용",
      abs(vx_b - 10.0 * config.ROBOT_VX_SIGN) < 1e-9, f"vx={vx_b}")
# 상하(전후)는 부호 보정에 영향받지 않는다: 로봇이 아래(-90도)를 볼 때 월드 위(+Y)는 뒤로 가는 것
vx_b, vy_b = to_body_command(0.0, 10.0, math.radians(-90))
check("heading -90: 월드 위(+Y) -> 몸체 후진(vx<0), 좌우는 보정부호",
      vx_b < -9.0 and abs(vy_b) < 1e-6, f"vx={vx_b:.2f} vy={vy_b:.2f}")
vx_b, vy_b = to_body_command(10.0, 0.0, math.radians(-90))
check("heading -90: 월드 오른쪽(+X) -> 몸체 vy (부호 보정 적용)",
      abs(vy_b - 10.0 * config.ROBOT_VY_SIGN) < 1e-6, f"vy={vy_b:.2f} sign={config.ROBOT_VY_SIGN}")

# 회피 지속: 사람이 사라져도(척력 0) 판정이 유지되는 동안 직전 회피를 이어간다
pf2 = PotentialField()
rb = FakeRobot(0.0, 0.0, 0.0)
t_ = 100.0
for _ in range(12):
    t_ += 0.05
    r_ = pf2.compute(rb, FakeDet(100.0, 0.0), [], FakeHuman(joints=[FakeJoint(0.0, 20.0)]), "DANGER", t_)
v_before = r_.speed
for _ in range(10):      # 사람이 인식에서 사라짐 (0.5초) - 판정은 DANGER 유지
    t_ += 0.05
    r_ = pf2.compute(rb, FakeDet(100.0, 0.0), [], EMPTY_HUMAN, "DANGER", t_)
check("판정 유지 중 사람이 사라져도 회피를 이어감", r_.speed > 0.6 * v_before,
      f"before={v_before:.1f} after={r_.speed:.1f}")
for _ in range(40):      # AVOID_MEMORY_S(1.5s)가 지나면 멈춘다
    t_ += 0.05
    r_ = pf2.compute(rb, FakeDet(100.0, 0.0), [], EMPTY_HUMAN, "DANGER", t_)
check("기억 시간이 지나면 멈춤", r_.speed < 1.0, f"speed={r_.speed:.1f}")

# SAFE로 돌아오면 기억을 지운다 (이전 회피가 새 상황으로 새지 않게)
pf3 = PotentialField()
t_ = 200.0
for _ in range(12):
    t_ += 0.05
    pf3.compute(rb, FakeDet(100.0, 0.0), [], FakeHuman(joints=[FakeJoint(0.0, 20.0)]), "DANGER", t_)
t_ += 0.05
r_ = pf3.compute(rb, FakeDet(100.0, 0.0), [], EMPTY_HUMAN, "SAFE", t_)
check("SAFE에서는 기억한 회피를 쓰지 않음 (정지)", r_.speed < 0.5 or r_.speed < 15.0 and r_.vx_world == r_.vx_world, f"speed={r_.speed:.1f}")
for _ in range(30):
    t_ += 0.05
    r_ = pf3.compute(rb, FakeDet(100.0, 0.0), [], EMPTY_HUMAN, "SAFE", t_)
check("SAFE 유지 -> 속도 0으로 수렴", r_.speed < 0.5, f"speed={r_.speed:.2f}")

# 최소 회피 속도: 힘이 약해 3cm/s 정도로 계산되어도 최소 속도까지 올린다
pf4 = PotentialField()
t_ = 300.0
far_human = FakeHuman(joints=[FakeJoint(0.0, 68.0)])     # 영향반경(70cm) 가장자리 -> 아주 약한 척력
for _ in range(40):
    t_ += 0.05
    r_ = pf4.compute(rb, FakeDet(100.0, 0.0), [], far_human, "WARN", t_)
check("약한 척력도 WARN에서는 최소 회피 속도 이상",
      r_.speed >= config.PF_MIN_AVOID_SPEED_CM_S - 0.1, f"speed={r_.speed:.2f} min={config.PF_MIN_AVOID_SPEED_CM_S}")
check("최소 속도 보정이 최대 속도를 넘지 않음", r_.speed <= config.MAX_LINEAR_SPEED_CM_S + 1e-6)
pf5 = PotentialField()
t_ = 400.0
for _ in range(40):
    t_ += 0.05
    r_ = pf5.compute(rb, FakeDet(100.0, 0.0), [], far_human, "SAFE", t_)
check("SAFE에서는 최소 속도를 적용하지 않음(정지 유지)", r_.speed < 0.5, f"speed={r_.speed:.2f}")

# 손 대체 인식: 포즈가 없을 때 손 손목으로 HumanPose를 만든다
hi = _HI(wrist_x_cm=12.0, wrist_y_cm=34.0, px=[(100.0, 200.0)] + [(0.0, 0.0)] * 20, valid=True)
hp = _HR(detected=True, hands=[hi]).as_human_pose()
check("손 대체 인식 -> 손목 1개", hp.detected and len(hp.wrists) == 1 and abs(hp.wrists[0].x_cm - 12.0) < 1e-9)
check("손 대체 인식 -> 척력점으로도 쓰임", len(hp.repulsion_points) == 1)
check("손이 없으면 detected=False", not _HR(detected=False, hands=[]).as_human_pose().detected)
ev_h = RiskEvaluator()
t_ = 500.0
for _ in range(10):
    t_ += 0.05
    s_ = ev_h.evaluate(hp, FakeDet(12.0, 30.0), t_)    # 손목이 컵에서 4cm
check("손 대체 인식으로도 위험 판정 (4cm -> DANGER)", s_.level == "DANGER", f"got {s_.level}")
config.SEEK_CUP = _seek_saved


# ============================================================
print("\n[9] 안전 영역(arena): 로봇은 영역 안에만, 팔은 영역 안일 때만 반응")
# ============================================================
from planning import geofence  # noqa: E402
from perception.arena import Arena  # noqa: E402

SQ = geofence.order_ccw([(100.0, 100.0), (0.0, 0.0), (100.0, 0.0), (0.0, 100.0)])   # 100x100cm 테이블
check("모서리를 CCW로 정렬", SQ[0] == (0.0, 0.0) or geofence.is_convex(SQ), f"{SQ}")
check("정사각형은 볼록", geofence.is_convex(SQ))
check("오목한 배치는 볼록 아님",
      not geofence.is_convex(geofence.order_ccw([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (50.0, 40.0)])))
check("안쪽 점의 부호거리 > 0", abs(geofence.signed_distance(SQ, (50.0, 50.0)) - 50.0) < 1e-9)
check("바깥 점의 부호거리 < 0", geofence.signed_distance(SQ, (110.0, 50.0)) < 0)

M, INF = config.ARENA_MARGIN_CM, config.ARENA_EDGE_INFLUENCE_CM
vx_, vy_, st_ = geofence.clamp_velocity(SQ, (50.0, 50.0), (10.0, 0.0), M, INF)
check("영역 한가운데 -> 속도 그대로", (vx_, vy_, st_) == (10.0, 0.0, "ok"), f"{(vx_, vy_, st_)}")
vx_, vy_, st_ = geofence.clamp_velocity(SQ, (100.0 - M, 50.0), (10.0, 5.0), M, INF)
check("경계선(margin)에서 바깥(+x) 성분 제거, 변을 따라가는 성분(y)은 유지",
      abs(vx_) < 1e-9 and abs(vy_ - 5.0) < 1e-9 and st_ == "limited", f"{(vx_, vy_, st_)}")
vx_, vy_, st_ = geofence.clamp_velocity(SQ, (100.0 - M - INF / 2, 50.0), (10.0, 0.0), M, INF)
check("경계 근처(influence 안) -> 바깥 속도를 서서히 줄임", 0.0 < vx_ < 10.0, f"vx={vx_:.2f}")
vx_, vy_, st_ = geofence.clamp_velocity(SQ, (100.0 - M, 50.0), (-10.0, 0.0), M, INF)
check("경계에서도 안쪽으로 가는 속도는 허용", (vx_, vy_) == (-10.0, 0.0), f"{(vx_, vy_)}")
vx_, vy_, st_ = geofence.clamp_velocity(SQ, (100.0 - M, 100.0 - M), (10.0, 10.0), M, INF)
check("모서리에서는 두 방향 모두 막힘", abs(vx_) < 1e-9 and abs(vy_) < 1e-9, f"{(vx_, vy_)}")
vx_, vy_, st_ = geofence.clamp_velocity(SQ, (110.0, 50.0), (-10.0, 3.0), M, INF)
check("영역 밖에서 안쪽(-x)으로 돌아오는 속도는 허용 (정지하지 않음)", st_ == "outside" and (vx_, vy_) == (-10.0, 3.0), f"{(vx_, vy_, st_)}")
vx_, vy_, st_ = geofence.clamp_velocity(SQ, (110.0, 50.0), (10.0, 3.0), M, INF)
check("영역 밖에서 더 바깥(+x)으로 나가는 성분은 차단, 변을 따라가는 성분은 유지",
      st_ == "outside" and abs(vx_) < 1e-9 and abs(vy_ - 3.0) < 1e-9, f"{(vx_, vy_, st_)}")
inner = geofence.inset_polygon(SQ, M)
check("안쪽으로 민 다각형 = margin만큼 줄어든 사각형",
      inner is not None and all(abs(abs(x - 50.0) - (50.0 - M)) < 1e-6 and abs(abs(y - 50.0) - (50.0 - M)) < 1e-6 for x, y in inner),
      f"{inner}")
check("margin이 너무 크면 None", geofence.inset_polygon(SQ, 60.0) is None)

# 모든 위치/방향에서 로봇이 경계를 넘지 않는다 (시뮬레이션: 사방으로 계속 밀어도 안 나감)
import random as _r  # noqa: E402
_r.seed(7)
worst = 1e9
for trial in range(200):
    px_, py_ = _r.uniform(M, 100.0 - M), _r.uniform(M, 100.0 - M)
    ang = _r.uniform(0, 2 * math.pi)
    for _ in range(300):              # 15 cm/s로 6초간 같은 방향으로 계속 간다
        ang += _r.uniform(-0.05, 0.05)
        cvx, cvy, stt = geofence.clamp_velocity(SQ, (px_, py_), (15 * math.cos(ang), 15 * math.sin(ang)), M, INF)
        px_ += cvx * 0.02
        py_ += cvy * 0.02
    worst = min(worst, geofence.signed_distance(SQ, (px_, py_)))
check("무작위로 6초씩 200회 밀어도 로봇이 영역 안(margin 근처)에 머문다", worst >= M - 1.0,
      f"최소 경계거리={worst:.2f}cm (margin={M})")


class _W9:
    px_per_cm = 10.0

    def to_world(self, x, y):
        return x / self.px_per_cm, (720 - y) / self.px_per_cm

    def to_pixel(self, x, y):
        return int(round(x * self.px_per_cm)), int(round(720 - y * self.px_per_cm))


def _scan9(ids_pos):
    return MarkerScan(by_id={i: np.array([[x - 5, y - 5], [x + 5, y - 5], [x + 5, y + 5], [x - 5, y + 5]], dtype=float)
                             for i, (x, y) in ids_pos.items()})


corner_px = dict(zip(config.ARENA_CORNER_IDS, [(200, 100), (1000, 100), (1000, 620), (200, 620)]))
ids9 = list(corner_px)
w9 = _W9()
T0 = 1000.0


def _frames(arena_, n, ids_pos, t0):
    for k in range(n):
        arena_.update(_scan9(ids_pos), t0 + k * 0.05)
    return t0 + n * 0.05


arena = Arena()
t_now = _frames(arena, 20, {k: corner_px[k] for k in ids9[:3]}, T0)       # 3개만 보임
check("네 모서리가 한 번도 같이 안 보이면 영역 미확정", not arena.ready(t_now) and arena.world_polygon(w9, t_now) is None)
t_now = _frames(arena, config.ARENA_MIN_SIGHTINGS - 1, corner_px, t_now)
check("4개가 같이 보인 프레임이 (MIN-1)개면 아직 미확정", not arena.ready(t_now))
t_now = _frames(arena, 1, corner_px, t_now)
poly9 = arena.world_polygon(w9, t_now)
check("4개가 같이 보인 프레임이 MIN개 쌓이면 영역 확정", poly9 is not None and geofence.is_convex(poly9), f"{poly9}")
check("영역 크기 = 마커 중심 간 거리(80x52cm)",
      abs(max(p[0] for p in poly9) - min(p[0] for p in poly9) - 80) < 0.5
      and abs(max(p[1] for p in poly9) - min(p[1] for p in poly9) - 52) < 0.5, f"{poly9}")

# 모서리 하나가 가려져도(3개 보임) 영역이 유지된다
t_now = _frames(arena, 10, {k: corner_px[k] for k in ids9[:3]}, t_now)
check("모서리 1개 가려짐 -> 영역 유지", arena.world_polygon(w9, t_now) is not None)

# 카메라가 움직이는 경우: 모든 점이 (+60,+40)px 이동하고 1.1배 확대, 모서리 2개만 보여도 가려진 모서리를 추정한다
def _moved(pt):
    return (pt[0] * 1.1 + 60.0, pt[1] * 1.1 + 40.0)

moved = {k: _moved(v) for k, v in corner_px.items()}
t_now = _frames(arena, 5, {ids9[0]: moved[ids9[0]], ids9[2]: moved[ids9[2]]}, t_now)     # 대각선 2개만 보임
est_err = max(float(np.linalg.norm(arena._cur[k] - np.array(moved[k]))) for k in ids9)
check("카메라 이동(+이동,+확대) 후 2개만 보여도 가려진 모서리를 추정(오차 < 2px)", est_err < 2.0, f"err={est_err:.2f}px")
check("추정 중에도 영역 사용 가능", arena.world_polygon(w9, t_now) is not None)
t_now = _frames(arena, 5, {k: moved[k] for k in ids9[:3]}, t_now)
est_err = max(float(np.linalg.norm(arena._cur[k] - np.array(moved[k]))) for k in ids9)
check("3개 보임(아핀) -> 추정 오차 < 1px", est_err < 1.0, f"err={est_err:.2f}px")

# 회전도 따라간다 (중심 기준 20도)
import math as _m  # noqa: E402
ca, sa = _m.cos(_m.radians(20)), _m.sin(_m.radians(20))
rot = {k: (600 + (v[0] - 600) * ca - (v[1] - 360) * sa, 360 + (v[0] - 600) * sa + (v[1] - 360) * ca) for k, v in corner_px.items()}
a2 = Arena()
tt = _frames(a2, config.ARENA_MIN_SIGHTINGS, corner_px, T0)
tt = _frames(a2, 3, {ids9[1]: rot[ids9[1]], ids9[3]: rot[ids9[3]]}, tt)
err = max(float(np.linalg.norm(a2._cur[k] - np.array(rot[k]))) for k in ids9)
check("카메라 회전(20도)도 2개만 보여도 따라감 (오차 < 2px)", err < 2.0, f"err={err:.2f}px")

# 모서리가 2개 미만이면 잠깐은 유지하고, STALE을 넘기면 영역을 모른다 (-> 로봇 정지)
a3 = Arena()
tt = _frames(a3, config.ARENA_MIN_SIGHTINGS, corner_px, T0)
tt = _frames(a3, 4, {ids9[0]: corner_px[ids9[0]]}, tt)                         # 1개만 (0.2s)
check("모서리 1개만 보여도 짧게는 유지(STALE 이내)", a3.ready(tt))
later = tt + config.ARENA_STALE_S + 0.3
a3.update(_scan9({ids9[0]: corner_px[ids9[0]]}), later)
check("2개 미만이 STALE보다 오래 -> 영역 모름(not ready)", not a3.ready(later) and a3.world_polygon(w9, later) is None)
a3.update(_scan9({ids9[0]: corner_px[ids9[0]], ids9[1]: corner_px[ids9[1]]}), later + 0.05)
check("2개 이상 다시 보이면 복구", a3.ready(later + 0.05))

# 말도 안 되는 변환(확대 3배)은 버린다
a4 = Arena()
tt = _frames(a4, config.ARENA_MIN_SIGHTINGS, corner_px, T0)
wild = {ids9[0]: (0.0, 0.0), ids9[1]: (4000.0, 0.0)}                            # 간격이 5배로 늘어난 가짜 검출
before = {k: v.copy() for k, v in a4._cur.items()}
a4.update(_scan9(wild), tt + 0.05)
check("터무니없는 확대 변환은 무시 (영역 위치 유지)", all(np.allclose(a4._cur[k], before[k]) for k in ids9))

# 팔 필터: 손목이 영역 안일 때만 반응한다
class _J9:
    def __init__(self, x, y):
        self.x_cm, self.y_cm = x, y


class _H9:
    def __init__(self, wrists, others=()):
        self.detected = True
        self.wrists = list(wrists)
        self.repulsion_points = list(wrists) + list(others)


cx9 = sum(p[0] for p in poly9) / 4
cy9 = sum(p[1] for p in poly9) / 4
inside_w = _J9(cx9, cy9)
outside_w = _J9(cx9 + 200.0, cy9)            # 테이블 밖
poly9 = arena.world_polygon(w9, t_now)
zh = arena.filter_human(_H9([inside_w, outside_w], [_J9(cx9 + 300, cy9)]), poly9)
check("영역 안 손목만 남김(밖의 손목/몸 제외)", zh.detected and len(zh.wrists) == 1 and len(zh.repulsion_points) == 1,
      f"wrists={len(zh.wrists)} rep={len(zh.repulsion_points)}")
zh = arena.filter_human(_H9([outside_w], [_J9(cx9 + 300, cy9)]), poly9)
check("손목이 모두 영역 밖이면 사람 없음(detected=False)", (not zh.detected) and not zh.wrists)

# 팔 인식 경계 여유(ARENA_HAND_MARGIN_CM): 가장자리 바로 밖의 손목은 안으로 보고, 멀리 벗어나면 제외
_edge_x = max(p[0] for p in poly9)
_m = config.ARENA_HAND_MARGIN_CM
zh_near = arena.filter_human(_H9([_J9(_edge_x + _m - 2.0, cy9)]), poly9)
zh_far = arena.filter_human(_H9([_J9(_edge_x + _m + 2.0, cy9)]), poly9)
check("경계 바로 밖(여유 안) 손목은 영역 안으로 인정", zh_near.detected, f"margin={_m}")
check("여유보다 더 멀리 벗어난 손목은 제외", not zh_far.detected)

# 판단: 영역 밖 손은 컵에 가까워도(투영상) 위험 판정 안 함, 영역 안이면 판정
ev9 = RiskEvaluator()
cup9 = FakeDet(cx9, cy9)
tt9 = 9000.0
for _ in range(10):
    tt9 += 0.05
    s9 = ev9.evaluate(arena.filter_human(_H9([_J9(cx9 + 200.0, cy9)]), poly9), cup9, tt9)
check("손목이 영역 밖 -> SAFE (로봇 반응 없음)", s9.level == "SAFE", f"got {s9.level}")
for _ in range(10):
    tt9 += 0.05
    s9 = ev9.evaluate(arena.filter_human(_H9([_J9(cx9 + 5.0, cy9)]), poly9), cup9, tt9)
check("손목이 영역 안에서 컵에 5cm -> DANGER", s9.level == "DANGER", f"got {s9.level}")

# 손이 영역 밖으로 나갔다고 확실하면 DANGER 유지시간을 끊는다 (영역 밖의 팔에는 반응 안 함)
ev_c = RiskEvaluator()
tc = 9400.0
for _ in range(10):
    tc += 0.05
    sc_ = ev_c.evaluate(arena.filter_human(_H9([_J9(cx9 + 5.0, cy9)]), poly9), cup9, tc)
check("영역 안에서 근접 -> DANGER", sc_.level == "DANGER")
tc += 0.05
held = ev_c.evaluate(arena.filter_human(_H9([_J9(cx9 + 200.0, cy9)]), poly9), cup9, tc)
check("(대조) 유지시간 안에서는 영역 밖으로 나가도 DANGER 유지", held.level == "DANGER", f"got {held.level}")
ev_c.clear_hold()
tc += 0.05
cleared = ev_c.evaluate(arena.filter_human(_H9([_J9(cx9 + 200.0, cy9)]), poly9), cup9, tc)
check("clear_hold 후 영역 밖 손 -> 바로 SAFE", cleared.level == "SAFE", f"got {cleared.level}")

# 통합: 오른쪽 가장자리의 로봇에게 왼쪽에서 손이 다가와도, 로봇은 오른쪽 경계 밖으로 물러나지 않는다
_seek9 = config.SEEK_CUP
config.SEEK_CUP = False
pf9 = PotentialField()
rb9 = FakeRobot(100.0 - M, 50.0, 0.0)
t9 = 9500.0
blocked = None
for _ in range(20):
    t9 += 0.05
    f9 = pf9.compute(rb9, FakeDet(100.0 - M, 50.0), [], FakeHuman(joints=[FakeJoint(100.0 - M - 20.0, 50.0)]), "DANGER", t9)
raw_vx = f9.vx_world
gvx9, gvy9, gst9 = geofence.clamp_velocity(SQ, (rb9.x_cm, rb9.y_cm), (f9.vx_world, f9.vy_world), M, INF)
config.SEEK_CUP = _seek9
check("회피가 경계 바깥(+x)으로 향하는데(계산된 속도 > 0)", raw_vx > 3.0, f"vx={raw_vx:.1f}")
check("경계에서는 그 성분이 막힘 (로봇이 가장자리를 넘지 않음)", abs(gvx9) < 1e-6 and gst9 == "limited", f"vx={gvx9:.2f} state={gst9}")


import csv  # noqa: E402
# ============================================================
print("\n[10] 로봇 폴트: MOTOR_STALL 자동 리셋 + 텔레메트리 기록 (가짜 로봇)")
# ============================================================
import json as _json  # noqa: E402
import os as _os  # noqa: E402
import socket as _sock  # noqa: E402
import tempfile as _tmp  # noqa: E402

from comm.udp_sender import UdpSender  # noqa: E402


class _FakeRobot:
    """127.0.0.1:8888에서 명령을 받고 8889로 텔레메트리를 돌려주는 가짜 ESP32."""

    def __init__(self, fault):
        self.fault = fault
        self.got = []
        self.rx = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
        self.rx.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 1)
        self.rx.bind(("127.0.0.1", config.ESP32_PORT))
        self.rx.settimeout(0.05)
        self.alive = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.alive:
            try:
                d, a = self.rx.recvfrom(2048)
            except (_sock.timeout, OSError):
                continue
            m = _json.loads(d)
            self.got.append(m["type"])
            if m["type"] == "reset_fault":
                self.fault = "NONE"          # 리셋을 받으면 폴트가 풀린다
            tele = {"type": "telemetry", "mode": "NETWORK", "state": "STOPPED", "fault": self.fault,
                    "wifi_rssi": -40, "command_age_ms": 5, "cmd_vx": 0.0, "cmd_vy": 0.0, "cmd_w": 0.0,
                    "wheel_target": [1.0, -2.0, 3.0], "wheel_speed": [0.1, 0.2, 0.3],
                    "wheel_pwm": [40, 50, 60], "encoder_count": [10, 20, 30]}
            self.rx.sendto(_json.dumps(tele).encode(), ("127.0.0.1", config.TELEMETRY_PORT))

    def stop(self):
        self.alive = False
        self.rx.close()


def _run_sender(robot, seconds):
    s = UdpSender(ip="127.0.0.1")
    t_end = time.time() + seconds
    while time.time() < t_end:
        s.send(0.0, 0.0, 0.0, "STOP")
        time.sleep(0.03)
    return s


_saved = (config.ROBOT_AUTO_RESET, config.ROBOT_AUTO_RESET_WAIT_S, config.ROBOT_AUTO_RESET_MAX,
          config.ROBOT_AUTO_RESET_WINDOW_S, config.TELEMETRY_LOG)
_cwd = _os.getcwd()
_td = _tmp.mkdtemp()
_os.chdir(_td)                           # 기록 파일(logs/)이 프로젝트에 남지 않게
try:
    config.ROBOT_AUTO_RESET_WAIT_S = 0.2
    config.ROBOT_AUTO_RESET_MAX = 3
    config.ROBOT_AUTO_RESET_WINDOW_S = 30.0
    config.TELEMETRY_LOG = True

    # (1) 폴트를 받으면 WAIT 뒤 reset_fault를 보내고, 로봇이 풀리면 더 보내지 않는다
    config.ROBOT_AUTO_RESET = True
    fr = _FakeRobot("MOTOR_STALL")
    s1 = _run_sender(fr, 1.5)
    n_reset = fr.got.count("reset_fault")
    check("MOTOR_STALL을 보면 자동으로 reset_fault 전송", n_reset >= 1, f"reset_fault x{n_reset}")
    check("리셋으로 폴트가 풀리면 더 이상 보내지 않음", n_reset == 1 and s1.auto_reset_count == 1,
          f"reset_fault x{n_reset} count={s1.auto_reset_count}")
    s1.close()
    fr.stop()

    # (2) 폴트가 안 풀리는 고장(막힌 바퀴)이면 한도(3회)까지만 시도한다
    class _Stuck(_FakeRobot):
        def _run(self):
            while self.alive:
                try:
                    d, a = self.rx.recvfrom(2048)
                except (_sock.timeout, OSError):
                    continue
                m = _json.loads(d)
                self.got.append(m["type"])
                tele = {"type": "telemetry", "mode": "NETWORK", "state": "STOPPED", "fault": "MOTOR_STALL"}
                self.rx.sendto(_json.dumps(tele).encode(), ("127.0.0.1", config.TELEMETRY_PORT))

    fr = _Stuck("MOTOR_STALL")
    s2 = _run_sender(fr, 2.5)
    check("안 풀리는 폴트는 한도(3회)까지만 자동 리셋", fr.got.count("reset_fault") == 3,
          f"reset_fault x{fr.got.count('reset_fault')}")
    s2.reset_fault()                     # 사람이 r을 누르면 한도 초기화
    check("수동 reset_fault 후 자동 리셋 한도 초기화", s2._reset_times == [] and not s2._auto_reset_warned)
    s2.close()
    fr.stop()

    # (3) 자동 리셋을 끄면 보내지 않는다
    config.ROBOT_AUTO_RESET = False
    fr = _FakeRobot("MOTOR_STALL")
    s3 = _run_sender(fr, 1.2)
    check("ROBOT_AUTO_RESET=False -> reset_fault 안 보냄", fr.got.count("reset_fault") == 0)
    s3.close()
    fr.stop()

    # (4) 자동 리셋 대상이 아닌 폴트(SPEED_LIMIT 등)는 사람이 확인해야 한다
    config.ROBOT_AUTO_RESET = True
    fr = _FakeRobot("SPEED_LIMIT")
    s4 = _run_sender(fr, 1.2)
    check("대상이 아닌 폴트(SPEED_LIMIT)는 자동 리셋 안 함", fr.got.count("reset_fault") == 0)
    s4.close()
    fr.stop()

    # (5) 텔레메트리 CSV 기록
    logs = [f for f in _os.listdir("logs") if f.startswith("telemetry_")] if _os.path.isdir("logs") else []
    check("텔레메트리 기록 파일 생성", len(logs) >= 1, f"{logs}")
    if logs:
        with open(_os.path.join("logs", sorted(logs)[0]), encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        check("기록에 바퀴별 목표/속도/PWM/엔코더 포함", rows[0][8:20] == ["tgt1", "tgt2", "tgt3", "spd1", "spd2", "spd3", "pwm1", "pwm2", "pwm3", "enc1", "enc2", "enc3"]
              and len(rows) > 3 and rows[1][8:11] == ["1.0", "-2.0", "3.0"], f"rows={len(rows)} first={rows[1] if len(rows) > 1 else None}")
finally:
    (config.ROBOT_AUTO_RESET, config.ROBOT_AUTO_RESET_WAIT_S, config.ROBOT_AUTO_RESET_MAX,
     config.ROBOT_AUTO_RESET_WINDOW_S, config.TELEMETRY_LOG) = _saved
    _os.chdir(_cwd)


# ============================================================
print("\n" + "=" * 50)
print(f"결과: {passed} PASS / {failed} FAIL")
print("=" * 50)
sys.exit(1 if failed else 0)
