"""
안전 영역(arena) - 테이블 네 모서리의 ArUco 마커(config.ARENA_CORNER_IDS)로 만든 사각형.

역할
  1) 로봇이 이 사각형 밖으로 나가지 않게 한다 (planning/geofence.py)
  2) 사람의 손목이 이 사각형 안에 있을 때만 위험 판정/회피를 한다 (filter_human)

카메라가 움직일 수 있어서(손에 든 폰, 흔들리는 거치대) 위치를 고정해 기억하지 않는다.
  - 네 모서리가 한 프레임에 모두 보이면 그 배치를 기준(ref)으로 저장한다.
  - 이후 2개 이상 보이면, 기준 배치 -> 지금 보이는 위치의 이동/회전/확대를 구해
    가려진 모서리를 추정한다 (2개: 닮음변환, 3개 이상: 아핀).
  - 모서리가 2개 미만으로 ARENA_STALE_S보다 오래 안 보이면 영역을 모르는 상태(not ready)다.
기준이 아직 없거나 not ready이면 main은 로봇을 움직이지 않는다 (ARENA_REQUIRED).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import config
from planning import geofence

from .marker_scanner import MarkerScan


@dataclass
class ZoneHuman:
    """영역 안에 있는 관절만 모은 사람. HumanPose와 같은 읽기 인터페이스(detected/wrists/repulsion_points)."""

    detected: bool = False
    wrists: list = field(default_factory=list)
    repulsion_points: list = field(default_factory=list)


def _fit_transform(src: np.ndarray, dst: np.ndarray):
    """src -> dst 변환 (A, t). 점 2개: 닮음변환(이동+회전+균등확대), 3개 이상: 아핀(최소제곱)."""
    k = len(src)
    if k == 2:
        s = complex(*(src[1] - src[0]))
        d = complex(*(dst[1] - dst[0]))
        if abs(s) < 1e-6:
            return None
        z = d / s
        A = np.array([[z.real, -z.imag], [z.imag, z.real]])
        return A, dst[0] - A @ src[0]
    X = np.hstack([src, np.ones((k, 1))])
    sol, *_ = np.linalg.lstsq(X, dst, rcond=None)
    return sol[:2].T, sol[2]


class Arena:
    def __init__(self) -> None:
        self._ref = None            # 네 모서리가 모두 보였던 기준 배치 (px)
        self._cur = {}              # 현재 추정 위치 (px)
        self._visible = set()       # 이번 프레임에 실제로 보인 모서리
        self._full = 0              # 네 모서리가 동시에 보인 프레임 수
        self._good_t = -1e9         # 마지막으로 2개 이상 보인 시각

    # ------------------------------------------------------------------ 학습/추정
    def update(self, scan: MarkerScan, now: float | None = None) -> None:
        now = time.time() if now is None else now
        ids = config.ARENA_CORNER_IDS
        vis = {}
        for mid in ids:
            c = scan.get(mid)
            if c is not None:
                vis[mid] = np.array(MarkerScan.center_px(c), dtype=float)
        self._visible = set(vis)

        if len(vis) == len(ids):
            a = config.ARENA_CORNER_EMA_ALPHA
            if self._cur and all(m in self._cur for m in ids) and \
                    max(float(np.linalg.norm(vis[m] - self._cur[m])) for m in ids) < 4.0:
                cur = {m: (1 - a) * self._cur[m] + a * vis[m] for m in ids}   # 정지 상태: 떨림 완화
            else:
                cur = dict(vis)                                              # 움직였으면 그대로 따라간다
            self._ref = {m: v.copy() for m, v in cur.items()}
            self._cur = cur
            self._full += 1
            self._good_t = now
        elif len(vis) >= 2 and self._ref is not None:
            have = list(vis)
            T = _fit_transform(np.array([self._ref[m] for m in have]), np.array([vis[m] for m in have]))
            if T is not None:
                A, tr = T
                scale = float(np.sqrt(abs(np.linalg.det(A))))
                if 0.5 < scale < 2.0:                      # 터무니없는 변환은 버린다
                    cur = {m: A @ self._ref[m] + tr for m in ids}
                    for m in have:
                        cur[m] = vis[m]                    # 보이는 건 측정값을 그대로 쓴다
                    self._cur = cur
                    self._good_t = now

    def missing_ids(self) -> list:
        """지금 프레임에서 보이지 않는 모서리 ID."""
        return [m for m in config.ARENA_CORNER_IDS if m not in self._visible]

    def ready(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return (self._ref is not None and self._full >= config.ARENA_MIN_SIGHTINGS
                and (now - self._good_t) <= config.ARENA_STALE_S)

    # ------------------------------------------------------------------ 기하
    def world_polygon(self, world, now: float | None = None):
        """CCW 순서의 월드 좌표 다각형. 준비가 안 됐거나 볼록하지 않으면 None."""
        if not self.ready(now):
            return None
        pts = [world.to_world(*self._cur[m]) for m in config.ARENA_CORNER_IDS]
        poly = geofence.order_ccw(pts)
        return poly if geofence.is_convex(poly) else None

    def status_text(self, world, now: float | None = None) -> str:
        """HUD용 한 줄 (영문만)."""
        n = len(config.ARENA_CORNER_IDS)
        if self._ref is None or self._full < config.ARENA_MIN_SIGHTINGS:
            return (f"ARENA need all {n} corners at once ({len(self._visible)} seen, "
                    f"{self._full}/{config.ARENA_MIN_SIGHTINGS})")
        if not self.ready(now):
            return f"ARENA lost (<2 corners for {config.ARENA_STALE_S:.1f}s)"
        if self.world_polygon(world, now) is None:
            return "ARENA invalid (not convex)"
        est = n - len(self._visible)
        return f"ARENA ok ({len(self._visible)} seen, {est} est)" if est else "ARENA ok (4 seen)"

    # ------------------------------------------------------------------ 사람 필터
    def filter_human(self, human, poly) -> ZoneHuman:
        """영역 안(ARENA_HAND_MARGIN_CM 여유 포함)의 관절만 남긴다. 손목이 없으면 detected=False."""
        m = config.ARENA_HAND_MARGIN_CM
        inside = lambda j: geofence.inside(poly, (j.x_cm, j.y_cm), -m)  # noqa: E731
        wrists = [j for j in human.wrists if inside(j)]
        points = [j for j in human.repulsion_points if inside(j)]
        return ZoneHuman(detected=bool(wrists), wrists=wrists, repulsion_points=points)

    # ------------------------------------------------------------------ 그리기
    def draw(self, frame, world, now: float | None = None) -> None:
        poly = self.world_polygon(world, now)
        for m in config.ARENA_CORNER_IDS:
            if m in self._cur:
                x, y = self._cur[m]
                seen = m in self._visible
                cv2.circle(frame, (int(x), int(y)), 7 if seen else 5,
                           (255, 255, 0) if seen else (0, 140, 255), 2)
        if poly is None:
            return
        pts = [world.to_pixel(x, y) for x, y in poly]
        for i in range(len(pts)):
            cv2.line(frame, pts[i], pts[(i + 1) % len(pts)], (255, 255, 0), 2)
        inner = geofence.inset_polygon(poly, config.ARENA_MARGIN_CM)
        if inner is not None:
            ip = [world.to_pixel(x, y) for x, y in inner]
            for i in range(len(ip)):
                cv2.line(frame, ip[i], ip[(i + 1) % len(ip)], (0, 200, 0), 1)
