"""
판단 계층 - 사람 손이 컵에 "정상적으로 집으러" 오는지, 아니면 "위험하게(치듯)"
접근하는지 판정.

★ 핵심 설계 ★
  그립(집기 자세)이 감지되면 정상적인 픽업으로 보고 회피하지 않는다(SAFE 고정).
  그립이 아닌 채로 접근하면(편 손/주먹으로 빠르게 다가오는 등 - 치려는 행위로
  간주) 아래 운동학 규칙으로 DANGER/WARN을 판정한다.

  이건 "의도 신호는 위험판정을 절대 뒤집지 않는다"는 예전 설계와 정반대다.
  하지만 실패 방향이 안전하다: 그립 인식이 실패하면(손이 가려짐 등) "그립 아님"
  분기로 떨어져서 오히려 더 보수적으로(회피 쪽으로) 판정한다. 즉 그립 인식이
  조용히 실패해도 위험판정이 침묵하는 게 아니라 반대로 더 조심하게 된다.
  시선(gaze)은 그립만큼 신뢰할 수 없어서(치려는 손도 컵을 쳐다볼 수 있음)
  이 판정을 뒤집는 데 쓰지 않고, 기존처럼 임계값만 살짝 넓히는 보조 신호로만 쓴다.

판정 입력 (그립이 아닐 때)
  - 손목-컵 거리 d [cm]          : 두 손목 중 더 가까운 쪽
  - 접근 속도 v = -dd/dt [cm/s]  : 양수면 가까워지는 중
  - 충돌 예상 시간 TTC = d / v [s]

판정 규칙 (위에서부터 먼저 걸리는 것 적용)
  SAFE   : 그립(집기 자세) 인정됨 - 정상 픽업으로 간주
           인정 조건: 컵에 가장 가까운 손 / 컵 GRIP_NEAR_CUP_CM 이내 / C자 모양 /
           GRIP_CONFIRM_S 이상 연속. 그래도 TTC DANGER 수준 고속 접근이면 아래 규칙으로 간다.
  DANGER : (그립 아님) d < RISK_DANGER_DIST_CM
           또는 (v > 임계) and (TTC < RISK_TTC_DANGER_S)
  WARN   : (그립 아님) d < RISK_WARN_DIST_CM
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

    # 의도 신호 (grip은 SAFE로 판정을 뒤집는다, gaze는 임계값만 넓힌다)
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
        self._grip_since: float | None = None   # grip 모양이 (짧은 끊김 허용하며) 이어진 시작 시각
        self._grip_last_shape = -1e9            # grip 모양이 마지막으로 보인 시각

    def reset(self) -> None:
        """컵이나 사람을 놓쳤을 때 미분 상태를 버린다 (재등장 시 속도 폭주 방지)."""
        self._dist_ema = None
        self._speed_ema = 0.0
        self._prev_dist = None
        self._prev_t = None

    @staticmethod
    def grip_shape_near_cup(hands, cup) -> bool:
        """
        컵에 가장 가까운 손이, 컵 근처(GRIP_NEAR_CUP_CM)에서 잡기 모양을 하고 있는가.
        화면의 다른 손(반대 손, 컵에서 먼 손)이 잡기 모양이어도 인정하지 않는다.
        """
        if hands is None or cup is None or not hands.detected:
            return False
        h = hands.nearest_to(cup.x_cm, cup.y_cm)
        if h is None:
            return False
        if math.hypot(h.wrist_x_cm - cup.x_cm, h.wrist_y_cm - cup.y_cm) > config.GRIP_NEAR_CUP_CM:
            return False
        return h.grasp_ready

    def _update_intent(self, hands, gaze, cup, now: float) -> tuple[bool, bool]:
        """
        의도 신호를 갱신한다.
        grip : 컵 근처의 잡기 모양이 GRIP_CONFIRM_S 이상 연속돼야 인정, 인정 후 GRIP_HOLD_S 유지.
        gaze : 감지되면 INTENT_HOLD_S 동안 유지 (고개를 잠깐 돌리는 것만으로 사라지지 않게).
        """
        if config.INTENT_USE_GRIP:
            if self.grip_shape_near_cup(hands, cup):
                if self._grip_since is None:
                    self._grip_since = now
                self._grip_last_shape = now
                if now - self._grip_since >= config.GRIP_CONFIRM_S:
                    self._intent_grip_until = now + config.GRIP_HOLD_S
            elif now - self._grip_last_shape > config.GRIP_GAP_S:
                # 모양이 GRIP_GAP_S보다 오래 깨져야 확인 타이머를 처음부터 다시 시작한다
                self._grip_since = None
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

        grip이 감지되면(집으려는 자세) 무조건 SAFE로 판정해 회피하지 않는다.
        grip이 아닐 때는 gaze가 임계값만 넓히는 보조 신호로 쓰인다.
        """
        now = time.time() if now is None else now

        grip_on, gaze_on = self._update_intent(hands, gaze, cup, now)
        intent = grip_on or gaze_on

        # --- 입력이 없으면 판정 불가. 상태를 리셋하고 SAFE로 둔다. ---
        # reason은 main.py의 cv2.putText HUD에 그대로 그려진다.
        # OpenCV Hershey 폰트는 한글 글리프가 없어 한글을 넣으면 화면에서 깨진다.
        # 그래서 reason은 항상 영문으로만 작성한다 (콘솔 print/설정패널은 별개).
        if cup is None or not human_pose.detected or not human_pose.wrists:
            self.reset()
            s = self._apply_hold(
                RiskState(level="SAFE", reason="no person or cup"), now
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

        # --- 그립(집기 자세) 감지 시: 정상적인 픽업으로 보고 회피하지 않는다 ---
        # 단, 잡기 모양이어도 TTC DANGER 수준으로 빠르게 오면 치는 동작으로 본다.
        # 속도/TTC 규칙은 컵에서 RISK_TTC_MAX_DIST_CM 안에서만 적용한다.
        ttc_zone = dist <= config.RISK_TTC_MAX_DIST_CM
        fast = (ttc_zone and speed > config.RISK_APPROACH_SPEED_CM_S
                and ttc is not None and ttc < config.RISK_TTC_DANGER_S)
        grip_exempt = grip_on and (config.GRIP_ALLOW_FAST_APPROACH or not fast)
        tag = "grasp, too fast" if grip_on else "no grasp"
        if grip_exempt:
            level, reason = "SAFE", "grasp (picking up)"
        else:
            # 그립이 아닐 때만 운동학 규칙으로 판정한다.
            # 시선은 그립만큼 신뢰할 수 없으므로(치려는 손도 컵을 볼 수 있음)
            # 판정을 뒤집진 않고 임계값만 살짝 넓히는 보조 신호로만 쓴다.
            d_boost = config.INTENT_DIST_BOOST if gaze_on else 1.0
            t_boost = config.INTENT_TTC_BOOST if gaze_on else 1.0
            danger_dist = config.RISK_DANGER_DIST_CM * d_boost
            warn_dist = config.RISK_WARN_DIST_CM * d_boost
            ttc_danger = config.RISK_TTC_DANGER_S * t_boost
            ttc_warn = config.RISK_TTC_WARN_S * t_boost

            # --- 규칙 판정 (reason은 영문 고정 - 위 주석 참고) ---
            level, reason = "SAFE", "normal"

            if dist < danger_dist:
                level, reason = "DANGER", f"close {dist:.0f}cm ({tag})"
            elif (
                ttc_zone
                and speed > config.RISK_APPROACH_SPEED_CM_S
                and ttc is not None
                and ttc < ttc_danger
            ):
                level, reason = "DANGER", f"fast approach TTC {ttc:.2f}s ({tag})"
            elif dist < warn_dist:
                level, reason = "WARN", f"approaching {dist:.0f}cm ({tag})"
            elif (
                ttc_zone
                and speed > config.RISK_APPROACH_SPEED_CM_S
                and ttc is not None
                and ttc < ttc_warn
            ):
                level, reason = "WARN", f"approaching TTC {ttc:.2f}s ({tag})"

            if gaze_on and level != "SAFE":
                reason += " +gaze"

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
        if grip_exempt:
            # 정상 픽업으로 인정되면 이전 위험 등급의 유지시간을 끊고 바로 SAFE로 내린다.
            # (고속 접근은 grip_exempt가 아니므로 hold가 그대로 적용된다)
            self._held_level, self._held_until = "SAFE", 0.0
            return state
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
