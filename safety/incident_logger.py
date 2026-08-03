"""
위기 상황(DANGER) 스크린샷 저장.

언제 저장하나
    매 프레임 저장하면 DANGER가 몇 초만 지속돼도 수백 장이 쌓인다.
    그래서 '한 번의 DANGER 에피소드'가 시작되는 순간(직전 프레임이 DANGER가
    아니었다가 DANGER로 바뀌는 전이)에만 저장한다. 판단 계층의 hold 로직 덕분에
    같은 에피소드 안에서는 level이 계속 "DANGER"로 유지되므로 자연히 한 번만
    찍힌다. 혹시 짧은 시간 안에 여러 번 전이되더라도 COOLDOWN_S로 한 번 더 제한한다.

무엇을 저장하나
    main.py가 HUD(위험 등급, 거리, 로봇 속도 등)를 다 그려 넣은 뒤의 프레임을
    받는다. 나중에 파일만 열어봐도 그 순간 상황을 알 수 있게 하려는 목적이다.

파일명에 원인을 넣는 이유
    수십 장이 쌓였을 때 파일 탐색기에서 훑어보는 것만으로 어떤 상황이었는지
    (근접인지 급접근인지, 몇 cm였는지) 바로 알 수 있게 하기 위함.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

import config


class IncidentLogger:
    def __init__(self) -> None:
        self._prev_level = "SAFE"
        self._last_save_t = 0.0
        self._count = 0

    def update(self, frame, risk, now: float | None = None) -> str | None:
        """
        DANGER 에피소드가 막 시작된 프레임이면 frame을 저장한다.
        저장했으면 파일 경로를, 아니면 None을 반환한다.
        """
        now = time.time() if now is None else now

        entering_danger = risk.level == "DANGER" and self._prev_level != "DANGER"
        self._prev_level = risk.level

        if not config.INCIDENT_LOG_ENABLED or not entering_danger:
            return None
        if now - self._last_save_t < config.INCIDENT_LOG_COOLDOWN_S:
            return None

        out_dir = Path(config.INCIDENT_LOG_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 파일명에 시각 + 원인 요약을 넣어 나중에 목록만 보고도 상황을 알 수 있게 한다.
        lt = time.localtime(now)
        ms = int((now % 1) * 1000)
        ts = time.strftime("%Y%m%d_%H%M%S", lt)
        reason = "".join(c if c.isalnum() else "_" for c in risk.reason)[:40]

        self._count += 1
        fname = f"{ts}_{ms:03d}_DANGER_{reason}.png"
        path = out_dir / fname

        ok = cv2.imwrite(str(path), frame)
        if ok:
            self._last_save_t = now
            print(f"[incident] saved #{self._count}: {path}")
            return str(path)

        print(f"[incident] failed to write {path}")
        return None
