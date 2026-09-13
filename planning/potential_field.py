"""
경로계산 계층 - Potential Field로 월드좌표 속도벡터 산출.

인력 (목표 = 컵)
    거리가 PF_ATTRACT_MAX_CM 이내면 quadratic well (거리에 비례),
    그보다 멀면 conic well (일정 크기)로 전환한다.
    멀리 있을 때 인력이 무한정 커져 척력을 뭉개는 것을 막기 위함.

척력 (장애물 + 사람) - FIRAS 형태
    d < d0 일 때만 작용:
        F = k * (1/d - 1/d0) * (1/d^2) * unit(로봇 - 장애물)
    d는 물체 표면까지의 거리(중심거리 - 반경)로 계산해 크기가 큰 물체를 제대로 피한다.

사람 척력은 판단 계층의 위험 등급에 따라 증폭된다 (PF_HUMAN_GAIN_BY_RISK).
DANGER면 사람 주변을 훨씬 크게 우회한다.

지역 최소점(local minima)
    인력과 척력이 상쇄되어 합력이 거의 0이 되면 로봇이 멈춘다.
    합력 크기가 PF_LOCAL_MINIMA_FORCE 미만이면서 목표에 도달하지 않았으면
    목표 방향의 수직 성분을 섞어 옆으로 빠져나가게 한다.

출력은 월드 좌표계 속도 [cm/s]. 로봇 좌표계 변환은 transform.py가 담당한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import config


@dataclass
class FieldResult:
    vx_world: float = 0.0        # cm/s
    vy_world: float = 0.0        # cm/s
    speed: float = 0.0           # cm/s (크기)
    goal_reached: bool = False
    has_goal: bool = False
    goal_distance_cm: float | None = None
    in_local_minima: bool = False
    # 디버그용 성분 분해
    attract: tuple[float, float] = (0.0, 0.0)
    repulse: tuple[float, float] = (0.0, 0.0)


def _repulsive(rx, ry, ox, oy, k, d0, obstacle_radius=0.0):
    """
    한 장애물이 로봇에 가하는 척력 벡터.

    정규화 거리 d^ = d/d0 를 써서   F = k * (1/d^ - 1) / d^^2   로 계산한다.
    d^가 무차원이라 게인 k가 영향 반경에 의존하지 않고, 인력 크기와 직접 비교된다.
      d^=1(경계) -> 0     d^=0.5 -> 4k     d^=0.25 -> 48k
    """
    dx = rx - ox
    dy = ry - oy
    center_dist = math.hypot(dx, dy)
    if center_dist < 1e-6:
        # 정확히 겹치면 방향을 정할 수 없다. 임의 방향으로 최대 척력.
        return (k / (config.PF_MIN_DIST_RATIO ** 2), 0.0)

    # 중심이 아니라 '표면'까지의 거리를 쓴다 -> 큰 물체를 제대로 피한다.
    d = center_dist - obstacle_radius
    if d >= d0:
        return (0.0, 0.0)

    d_hat = max(d / d0, config.PF_MIN_DIST_RATIO)
    mag = k * (1.0 / d_hat - 1.0) / (d_hat * d_hat)
    ux, uy = dx / center_dist, dy / center_dist
    return (mag * ux, mag * uy)


class PotentialField:
    def __init__(self) -> None:
        # 가속도 제한을 위해 직전 출력 속도를 기억한다.
        self._prev_vx = 0.0
        self._prev_vy = 0.0
        self._prev_t: float | None = None

    def reset(self) -> None:
        self._prev_vx = 0.0
        self._prev_vy = 0.0
        self._prev_t = None

    def compute(
        self,
        robot,          # perception.RobotPose
        cup,            # perception.Detection | None
        obstacles,      # list[perception.Detection]
        human_pose,     # perception.HumanPose
        risk_level: str,
        now: float,
    ) -> FieldResult:
        result = FieldResult()

        if not robot.detected:
            # 로봇을 못 찾으면 명령을 만들 수 없다. 정지.
            self._decay(now)
            return result

        rx, ry = robot.x_cm, robot.y_cm

        # ---------- 인력 ----------
        # IDLE_UNTIL_THREAT 모드에서는 목표로 끌려가지 않는다 - 평소엔 제자리 대기,
        # 아래 척력만으로 위험할 때만 물러난다.
        fx_att = fy_att = 0.0
        if cup is not None:
            result.has_goal = True
            if not config.IDLE_UNTIL_THREAT:
                dx = cup.x_cm - rx
                dy = cup.y_cm - ry
                dist = math.hypot(dx, dy)
                result.goal_distance_cm = dist

                if dist < config.PF_GOAL_TOLERANCE_CM:
                    result.goal_reached = True
                elif dist > 1e-6:
                    if dist <= config.PF_ATTRACT_MAX_CM:
                        # quadratic well: 가까울수록 부드럽게 감속
                        fx_att = config.PF_K_ATTRACT * dx
                        fy_att = config.PF_K_ATTRACT * dy
                    else:
                        # conic well: 멀면 일정한 크기로
                        m = config.PF_K_ATTRACT * config.PF_ATTRACT_MAX_CM
                        fx_att = m * dx / dist
                        fy_att = m * dy / dist

        # ---------- 척력: 장애물 ----------
        fx_rep = fy_rep = 0.0
        for o in obstacles:
            rxf, ryf = _repulsive(
                rx, ry, o.x_cm, o.y_cm,
                config.PF_K_REPULSE_OBSTACLE,
                config.PF_OBSTACLE_INFLUENCE_CM,
                obstacle_radius=o.radius_cm,
            )
            fx_rep += rxf
            fy_rep += ryf

        # ---------- 척력: 사람 ----------
        human_gain = config.PF_HUMAN_GAIN_BY_RISK.get(risk_level, 1.0)
        for j in human_pose.repulsion_points:
            rxf, ryf = _repulsive(
                rx, ry, j.x_cm, j.y_cm,
                config.PF_K_REPULSE_HUMAN * human_gain,
                config.PF_HUMAN_INFLUENCE_CM,
            )
            fx_rep += rxf
            fy_rep += ryf

        result.attract = (fx_att, fy_att)
        result.repulse = (fx_rep, fy_rep)

        # ---------- 합력 ----------
        fx = fx_att + fx_rep
        fy = fy_att + fy_rep
        f_mag = math.hypot(fx, fy)

        # ---------- 지역 최소점 탈출 ----------
        # 목표로 이동하지 않는 IDLE_UNTIL_THREAT 모드에서는 애초에 갇힐 목표가 없다.
        if (
            not config.IDLE_UNTIL_THREAT
            and result.has_goal
            and not result.goal_reached
            and f_mag < config.PF_LOCAL_MINIMA_FORCE
        ):
            result.in_local_minima = True
            gx = cup.x_cm - rx
            gy = cup.y_cm - ry
            gnorm = math.hypot(gx, gy)
            if gnorm > 1e-6:
                # 목표 방향의 왼쪽 수직 벡터로 밀어낸다.
                px, py = -gy / gnorm, gx / gnorm
                push = config.PF_ESCAPE_GAIN * config.MAX_LINEAR_SPEED_CM_S
                fx += px * push
                fy += py * push
                f_mag = math.hypot(fx, fy)

        # ---------- 힘 -> 속도 ----------
        # 고정 게인으로 변환한 뒤 clamp 한다.
        # 힘을 항상 최대속도로 정규화하면, 장애물 척력으로 합력이 줄어들어도
        # 여전히 전속력이 나와서 "감속하며 접근"이라는 동작 자체가 사라진다.
        if result.goal_reached or f_mag < 1e-6:
            vx = vy = 0.0
        else:
            vx = fx * config.PF_FORCE_TO_SPEED
            vy = fy * config.PF_FORCE_TO_SPEED
            sp = math.hypot(vx, vy)
            if sp > config.MAX_LINEAR_SPEED_CM_S:
                k = config.MAX_LINEAR_SPEED_CM_S / sp
                vx *= k
                vy *= k

        # ---------- 위험 등급별 감속 ----------
        # IDLE_UNTIL_THREAT 모드에서는 위험할수록 오히려 더 움직여야(물러나야) 하므로
        # 여기서 속도를 깎지 않는다. 위험도에 따른 긴급도는 이미 위 척력 계산에서
        # PF_HUMAN_GAIN_BY_RISK로 반영됐다 (SAFE 1.0 / WARN 1.8 / DANGER 3.0).
        if not config.IDLE_UNTIL_THREAT:
            risk_scale = config.SPEED_SCALE_BY_RISK.get(risk_level, 0.0)
            vx *= risk_scale
            vy *= risk_scale

        # ---------- 가속도 제한 ----------
        vx, vy = self._limit_accel(vx, vy, now)

        result.vx_world = vx
        result.vy_world = vy
        result.speed = math.hypot(vx, vy)
        return result

    # ---------- 내부 ----------
    def _limit_accel(self, vx: float, vy: float, now: float) -> tuple[float, float]:
        if self._prev_t is None:
            self._prev_t = now
            self._prev_vx, self._prev_vy = vx, vy
            return vx, vy

        dt = max(now - self._prev_t, 1e-3)
        max_dv = config.MAX_LINEAR_ACCEL_CM_S2 * dt

        dvx = vx - self._prev_vx
        dvy = vy - self._prev_vy
        dv = math.hypot(dvx, dvy)
        if dv > max_dv:
            k = max_dv / dv
            vx = self._prev_vx + dvx * k
            vy = self._prev_vy + dvy * k

        self._prev_vx, self._prev_vy = vx, vy
        self._prev_t = now
        return vx, vy

    def _decay(self, now: float) -> None:
        """정지 명령을 낼 때도 가속도 상태를 0으로 수렴시킨다."""
        self._limit_accel(0.0, 0.0, now)
