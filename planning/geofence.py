"""
경로계산 계층 - 안전 영역(geofence): 볼록 다각형(테이블의 4개 모서리 마커) 안에서만 움직이게 한다.

좌표는 월드 좌표(cm, Y 위쪽 +). 다각형은 반시계(CCW) 순서의 볼록 다각형이다.
CCW에서는 각 변의 '왼쪽'이 안쪽이다.

  d_in    : 한 변까지의 안쪽 방향 거리 (양수면 그 변의 안쪽)
  margin  : 로봇 중심이 변에서 이만큼 안쪽에 있어야 한다 (로봇 반지름 + 여유)
  influence : 경계(margin선)에 이 거리까지 다가오면 바깥으로 가는 속도를 서서히 줄인다

바깥으로 향하는 속도 성분만 줄이고 변을 따라가는 성분은 남긴다 -> 경계에서 미끄러지듯 움직인다.
"""

from __future__ import annotations

import math


def order_ccw(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """중심 기준 각도로 정렬해 반시계 순서로 만든다 (월드 좌표, Y 위쪽)."""
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def is_convex(poly: list[tuple[float, float]]) -> bool:
    """CCW 순서의 모든 모서리가 같은 쪽으로 꺾이면 볼록. 면적이 거의 0이면 아님."""
    n = len(poly)
    if n < 3:
        return False
    signs = [_cross(poly[i], poly[(i + 1) % n], poly[(i + 2) % n]) for i in range(n)]
    return all(s > 1e-6 for s in signs)


def _edges(poly):
    """각 변의 (시작점, 안쪽 법선 n_in)."""
    n = len(poly)
    out = []
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        length = math.hypot(ex, ey)
        if length < 1e-9:
            continue
        out.append(((ax, ay), (-ey / length, ex / length)))
    return out


def edge_distances(poly, p) -> list[tuple[float, tuple[float, float]]]:
    """각 변에 대해 (안쪽 거리 d_in, 바깥 법선 n_out)."""
    res = []
    for (ax, ay), (nx, ny) in _edges(poly):
        d_in = (p[0] - ax) * nx + (p[1] - ay) * ny
        res.append((d_in, (-nx, -ny)))
    return res


def signed_distance(poly, p) -> float:
    """다각형 안이면 가장 가까운 변까지의 거리(+), 밖이면 가장 많이 벗어난 변까지의 거리(-)."""
    return min(d for d, _ in edge_distances(poly, p))


def inside(poly, p, margin: float = 0.0) -> bool:
    return signed_distance(poly, p) >= margin


def clamp_velocity(poly, p, v, margin: float, influence: float):
    """
    위치 p의 로봇 속도 v=(vx, vy)에서 '경계 밖으로 향하는 성분'을 줄인다.

    반환: (vx, vy, state)  state = "ok" | "limited" | "outside"
      "outside": 로봇 중심이 다각형(margin 없이) 밖에 있다. 벗어난 변 쪽으로 더 나가는 성분은
                모두 막고, 안쪽으로 돌아오거나 변을 따라가는 성분만 허용한다
                (밖에서 정지해 버리면 위험을 피하지도 못하기 때문).
    """
    vx, vy = v
    if signed_distance(poly, p) < 0.0:
        for d_in, (nox, noy) in edge_distances(poly, p):
            if d_in >= 0.0:
                continue                      # 이 변은 아직 안쪽
            out_comp = vx * nox + vy * noy
            if out_comp > 0.0:                # 벗어난 변 쪽으로 더 나가려는 성분 제거
                vx -= out_comp * nox
                vy -= out_comp * noy
        return vx, vy, "outside"

    limited = False
    for d_in, (nox, noy) in edge_distances(poly, p):
        out_comp = vx * nox + vy * noy
        if out_comp <= 0.0:
            continue
        d = d_in - margin               # margin선까지의 남은 거리
        if d <= 0.0:
            factor = 1.0                # margin선 안쪽(바깥쪽 띠)에서는 바깥 성분을 전부 막는다
        elif d < influence:
            factor = 1.0 - d / influence
        else:
            continue
        vx -= out_comp * factor * nox
        vy -= out_comp * factor * noy
        limited = True
    return vx, vy, ("limited" if limited else "ok")


def inset_polygon(poly, margin: float):
    """각 변을 margin만큼 안쪽으로 민 다각형 (그리기용). 접혀서 사라지면 None."""
    edges = _edges(poly)
    n = len(edges)
    if n < 3:
        return None
    lines = []                           # 안쪽으로 민 변의 (점, 방향)
    for i, ((ax, ay), (nx, ny)) in enumerate(edges):
        bx, by = poly[(i + 1) % n]
        lines.append(((ax + nx * margin, ay + ny * margin), (bx - ax, by - ay)))
    pts = []
    for i in range(n):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-9:
            return None
        t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
        pts.append((p1[0] + d1[0] * t, p1[1] + d1[1] * t))
    if not is_convex(pts):
        return None
    # margin이 너무 크면 변이 서로 지나쳐 뒤집힌 작은 다각형이 된다.
    # 올바른 안쪽 다각형의 꼭짓점은 원래 다각형 변에서 모두 margin 이상 떨어져 있다.
    if any(signed_distance(poly, q) < margin - 1e-6 for q in pts):
        return None
    return pts
