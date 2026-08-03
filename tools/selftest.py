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
check("DANGER -> 속도 0", r.speed < 1e-6, f"speed={r.speed:.2f}")

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
from comm.udp_sender import CommandPacket  # noqa: E402

import json  # noqa: E402
pkt = CommandPacket(seq=42, t=1699.123456, vx=12.3456, vy=-4.5678, w=0.35, status="RUN")
decoded = json.loads(pkt.to_json())
check("JSON 키가 ESP32 파서와 일치",
      set(decoded) == {"seq", "t", "vx", "vy", "w", "status"}, f"got {set(decoded)}")
check("소수점 3자리 반올림", decoded["vx"] == 12.346, f"got {decoded['vx']}")
check("패킷 크기 < 512B (ESP32 버퍼)", len(pkt.to_json()) < 512, f"{len(pkt.to_json())}B")

# ============================================================
print("\n" + "=" * 50)
print(f"결과: {passed} PASS / {failed} FAIL")
print("=" * 50)
sys.exit(1 if failed else 0)
