"""
카메라 캡처 + 픽셀 <-> 월드 좌표 변환.

월드 좌표계 정의는 config.py 상단 주석 참조.
  world_x_cm = px_x / px_per_cm
  world_y_cm = (FRAME_HEIGHT - px_y) / px_per_cm   # Y축을 위로 뒤집는다

px_per_cm은 ArUco 로봇 마커의 실제 픽셀 크기로부터 매 프레임 갱신된다.
RobotTracker가 update_scale()을 호출해 주고, 마커가 안 보이면 직전 값을 유지한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class WorldFrame:
    """픽셀 <-> 월드(cm) 변환기. 프레임마다 스케일이 갱신될 수 있다."""

    px_per_cm: float = config.DEFAULT_PX_PER_CM
    frame_height: int = config.FRAME_HEIGHT

    def update_scale(self, px_per_cm: float) -> None:
        """ArUco 마커로부터 추정한 스케일로 갱신. 튀는 값은 무시한다."""
        if px_per_cm is None or not np.isfinite(px_per_cm) or px_per_cm <= 0:
            return
        # 직전 값 대비 2배 이상 튀면 오검출로 보고 버린다.
        if 0.5 * self.px_per_cm <= px_per_cm <= 2.0 * self.px_per_cm:
            # 완만하게 따라가도록 EMA
            self.px_per_cm = 0.8 * self.px_per_cm + 0.2 * px_per_cm
        else:
            self.px_per_cm = px_per_cm

    # ---- 변환 ----
    def to_world(self, px_x: float, px_y: float) -> tuple[float, float]:
        return (
            px_x / self.px_per_cm,
            (self.frame_height - px_y) / self.px_per_cm,
        )

    def to_pixel(self, world_x: float, world_y: float) -> tuple[int, int]:
        return (
            int(round(world_x * self.px_per_cm)),
            int(round(self.frame_height - world_y * self.px_per_cm)),
        )

    def to_world_vec(self, px_dx: float, px_dy: float) -> tuple[float, float]:
        """변위 벡터 변환 (Y 부호 반전만 적용, 평행이동 없음)."""
        return (px_dx / self.px_per_cm, -px_dy / self.px_per_cm)

    def px_len_to_cm(self, px: float) -> float:
        return px / self.px_per_cm


class Camera:
    """OpenCV VideoCapture 래퍼."""

    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
        # 버퍼가 쌓이면 지연이 생겨 실시간 제어에 치명적이다.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"웹캠을 열 수 없습니다 (CAMERA_INDEX={config.CAMERA_INDEX}). "
                "config.py의 CAMERA_INDEX를 확인하세요."
            )

        self.world = WorldFrame()

    def read(self):
        """(ok, frame) 반환. frame은 BGR."""
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        if config.FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)
        return True, frame

    def release(self) -> None:
        self.cap.release()
