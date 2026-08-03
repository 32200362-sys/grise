"""
로봇에 붙일 ArUco 마커 이미지 생성 (인쇄용).

정확한 물리 크기로 인쇄되도록 DPI를 계산해서 여백까지 넣어준다.
config.MARKER_SIZE_CM 값과 실제 인쇄된 크기가 일치해야 거리 계산이 맞는다.

사용법:
    python tools/make_marker.py                 # ID 0, config.MARKER_SIZE_CM 크기
    python tools/make_marker.py --id 3 --cm 10  # ID 3, 10cm

인쇄할 때 반드시 "실제 크기 / 100% / 배율 조정 안함"으로 출력할 것.
"맞춤(fit to page)"으로 인쇄하면 크기가 달라져서 거리가 전부 틀어진다.
인쇄 후 자로 검은 사각형 한 변을 재서 config.MARKER_SIZE_CM에 실측값을 넣는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

import config

DPI = 300  # 인쇄 해상도
CM_PER_INCH = 2.54


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=config.ROBOT_MARKER_ID)
    ap.add_argument("--cm", type=float, default=config.MARKER_SIZE_CM)
    ap.add_argument("--dict", type=str, default=config.ARUCO_DICT_NAME)
    args = ap.parse_args()

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))

    # 물리 크기 -> 픽셀
    side_px = int(round(args.cm / CM_PER_INCH * DPI))
    marker = cv2.aruco.generateImageMarker(aruco_dict, args.id, side_px)

    # 흰 여백(quiet zone)이 있어야 검출이 안정적이다. 한 변의 25%.
    pad = int(side_px * 0.25)
    canvas = np.full((side_px + 2 * pad, side_px + 2 * pad), 255, np.uint8)
    canvas[pad:pad + side_px, pad:pad + side_px] = marker

    # 하단에 정보 + 실측용 눈금 표시
    canvas = cv2.copyMakeBorder(canvas, 0, 140, 0, 0, cv2.BORDER_CONSTANT, value=255)
    base_y = side_px + 2 * pad

    cv2.putText(canvas, f"{args.dict}  ID={args.id}  {args.cm:.1f}cm",
                (pad, base_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2)
    cv2.putText(canvas, "print at 100% (no fit-to-page), then measure the black square",
                (pad, base_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.62, 0, 1)

    # 마커 폭과 정확히 같은 길이의 눈금자 -> 인쇄 후 이 선을 재면 검증된다
    ry = base_y + 118
    cv2.line(canvas, (pad, ry), (pad + side_px, ry), 0, 3)
    for x in (pad, pad + side_px):
        cv2.line(canvas, (x, ry - 12), (x, ry + 12), 0, 3)

    out = Path(__file__).resolve().parent.parent / f"aruco_id{args.id}_{args.cm:.0f}cm.png"
    cv2.imwrite(str(out), canvas)

    print(f"저장 완료: {out}")
    print(f"  사전(dictionary) : {args.dict}")
    print(f"  마커 ID          : {args.id}   (config.ROBOT_MARKER_ID = {config.ROBOT_MARKER_ID})")
    print(f"  인쇄 크기        : {args.cm:.1f}cm  ({side_px}px @ {DPI}DPI)")
    print()
    print("다음 단계:")
    print("  1) 100% 배율로 인쇄 (fit-to-page 끄기)")
    print("  2) 자로 검은 사각형 한 변을 실측")
    print(f"  3) config.py 의 MARKER_SIZE_CM 을 실측값으로 수정 (현재 {config.MARKER_SIZE_CM})")
    print("  4) 로봇 위에 평평하게 부착 - 마커 위쪽 변이 로봇 정면을 향하도록")
    return 0


if __name__ == "__main__":
    sys.exit(main())
