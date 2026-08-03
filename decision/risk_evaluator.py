"""
판단 계층 - 사람 손이 컵에 위험하게 접근하는지 판정.

판정 입력
  - 손목-컵 거리 d [cm]          : 두 손목 중 더 가까운 쪽
  - 접근 속도 v = -dd/dt [cm/s]  : 양수면 가까워지는 중
  - 충돌 예상 시간 TTC = d / v [s]

판정 규칙 (위에서부터 먼저 걸리는 것 적용)
  DANGER : d < RISK_DANGER_DIST_CM
           또는 (v > 임계) and (TTC < RISK_TTC_DANGER_S)
  WARN   : d < RISK_WARN_DIST_CM
           또는 (v > 임계) and (TTC < RISK_TTC_WARN_S)
  SAFE   : 그 외

노이즈 대책
  - 거리/속도 모두 EMA로 스무딩한다. 원시 프레임 미분은 수십 cm/s씩 튄다.
  - 등급이 올라가면 최소 유지 시간(hold)을 둬서 SAFE<->DANGER 채터링을 막는다.
    등급이 내려가는 것만 지연되고, 올라가는 것은 즉시 반영된다. (안전 방향 우선)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import config

_LEVEL_ORDER = {"SAFE": 0, "WARN": 1, "DANGER": 2}


@dataclass
class RiskState:
    level: str = "SAFE"                  # SAFE | WARN | DANGER
    distance_cm: float | None = None     # 스무딩된 손목-컵 거리
    approach_speed_cm_s: float = 0.0     # 양수 = 접근 중
    ttc_s: float | None = None           # 충돌 예상 시간
    reason: str = "no-data"

    # 의도 신호 (판정 자체를 뒤집지는 않고 임계값만 넓힌다)
    intent: bool = False
    intent_grip: bool = False
    intent_gaze: bool = False

    @property
    def is_danger(self) -> bool:
        return self.level == "DANGER"


class RiskEvaluator:
    def __init__(self) -> None:
        self._dist_ema: float | None = None
        self._speed_ema: float = 0.0
        self._prev_dist: float | None = None
        self._prev_t: float | None = None

        self._held_level = "SAFE"
        self._held_until = 0.0

        # 의도 신호 유지 (깜빡임 방지)
        self._intent_grip_until = 0.0
        self._intent_gaze_until = 0.0

    def reset(self) -> None:
        """컵이나 사람을 놓쳤을 때 미분 상태를 버린다 (재등장 시 속도 폭주 방지)."""
        self._dist_ema = None
        self._speed_ema = 0.0
        self._prev_dist = None
        self._prev_t = None

    def _update_intent(self, hands, gaze, now: float) -> tuple[bool, bool]:
        """
        의도 신호를 갱신한다. 신호가 잠깐 끊겨도 INTENT_HOLD_S 동안 유지한다.
        (손이 몸에 가려지거나 고개를 잠깐 돌리는 것만으로 반응이 사라지면 쓸모가 없다)
        """
        if config.INTENT_USE_GRIP and hands is not None and hands.any_grasp_ready:
            self._intent_grip_until = now + config.INTENT_HOLD_S
        if config.INTENT_USE_GAZE and gaze is not None and gaze.looking_at_target:
            self._intent_gaze_until = now + config.INTENT_HOLD_S

        grip = config.INTENT_USE_GRIP and now < self._intent_grip_until
        gaze_on = config.INTENT_USE_GAZE and now < self._intent_gaze_until
        return grip, gaze_on

    def evaluate(self, human_pose, cup, now: float | None = None,
                 hands=None, gaze=None) -> RiskState:
        """
        human_pose : perception.HumanPose
        cup        : perception.Detection | None  (목표 컵)
        hands      : perception.HandsResult | None  (그립 모양 - 의도 신호)
        gaze       : perception.GazeInfo | None     (머리 방향 - 의도 신호)

        hands/gaze는 임계값을 넓히는 데만 쓰인다. 이것만으로 DANGER가 되지는 않는다.
        """
        now = time.time() if now is None else now

        grip_on, gaze_on = self._update_intent(hands, gaze, now)
        intent = grip_on or gaze_on

        # --- 입력이 없으면 판정 불가. 상태를 리셋하고 SAFE로 둔다. ---
        if cup is None or not human_pose.detected or not human_pose.wrists:
            self.reset()
            s = self._apply_hold(
                RiskState(level="SAFE", reason="사람 또는 컵 미검출"), now
            )
            s.intent, s.intent_grip, s.intent_gaze = intent, grip_on, gaze_on
            return s

        # --- 두 손목 중 컵에 더 가까운 쪽 ---
        raw_dist = min(
            math.hypot(w.x_cm - cup.x_cm, w.y_cm - cup.y_cm)
            for w in human_pose.wrists
        )

        # --- 거리 EMA ---
        a_d = config.RISK_DIST_EMA_ALPHA
        if self._dist_ema is None:
            self._dist_ema = raw_dist
        else:
            self._dist_ema = (1 - a_d) * self._dist_ema + a_d * raw_dist
        dist = self._dist_ema

        # --- 접근 속도 (거리의 감소율) ---
        raw_speed = 0.0
        if self._prev_dist is not None and self._prev_t is not None:
            dt = now - self._prev_t
            if dt > 1e-3:
                raw_speed = (self._prev_dist - dist) / dt
        a_v = config.RISK_SPEED_EMA_ALPHA
        self._speed_ema = (1 - a_v) * self._speed_ema + a_v * raw_speed
        speed = self._speed_ema

        self._prev_dist = dist
        self._prev_t = now

        # --- TTC ---
        ttc = dist / speed if speed > 1.0 else None

        # --- 의도 신호로 임계값을 넓힌다 (판정 규칙 자체는 그대로) ---
        d_boost = config.INTENT_DIST_BOOST if intent else 1.0
        t_boost = config.INTENT_TTC_BOOST if intent else 1.0
        danger_dist = config.RISK_DANGER_DIST_CM * d_boost
        warn_dist = config.RISK_WARN_DIST_CM * d_boost
        ttc_danger = config.RISK_TTC_DANGER_S * t_boost
        ttc_warn = config.RISK_TTC_WARN_S * t_boost

        # --- 규칙 판정 ---
        level, reason = "SAFE", "정상"

        if dist < danger_dist:
            level, reason = "DANGER", f"근접 {dist:.0f}cm"
        elif (
            speed > config.RISK_APPROACH_SPEED_CM_S
            and ttc is not None
            and ttc < ttc_danger
        ):
            level, reason = "DANGER", f"급접근 TTC {ttc:.2f}s"
        elif dist < warn_dist:
            level, reason = "WARN", f"접근 {dist:.0f}cm"
        elif (
            speed > config.RISK_APPROACH_SPEED_CM_S
            and ttc is not None
            and ttc < ttc_warn
        ):
            level, reason = "WARN", f"접근중 TTC {ttc:.2f}s"

        if intent and level != "SAFE":
            tags = []
            if grip_on:
                tags.append("그립")
            if gaze_on:
                tags.append("시선")
            reason += f" +{'/'.join(tags)}"

        state = RiskState(
            level=level,
            distance_cm=dist,
            approach_speed_cm_s=speed,
            ttc_s=ttc,
            reason=reason,
            intent=intent,
            intent_grip=grip_on,
            intent_gaze=gaze_on,
        )
        return self._apply_hold(state, now)

    def _apply_hold(self, state: RiskState, now: float) -> RiskState:
        """등급 상승은 즉시, 하강은 hold 시간이 지난 뒤에만 허용한다."""
        new_rank = _LEVEL_ORDER[state.level]
        held_rank = _LEVEL_ORDER[self._held_level]

        if new_rank >= held_rank:
            # 올라가거나 같으면 그대로 반영하고 유지 타이머를 갱신
            self._held_level = state.level
            hold = (
                config.RISK_DANGER_HOLD_S if state.level == "DANGER"
                else config.RISK_WARN_HOLD_S if state.level == "WARN"
                else 0.0
            )
            self._held_until = now + hold
        else:
            # 내려가려는데 아직 유지 시간이 남았으면 이전 등급을 유지
            if now < self._held_until:
                state.level = self._held_level
                state.reason += f" (hold {self._held_until - now:.1f}s)"
            else:
                self._held_level = state.level

        return state
