"""
카메라 캡처 + 픽셀 <-> 월드 좌표 변환.

월드 좌표계 정의는 config.py 상단 주석 참조.
  world_x_cm = px_x / px_per_cm
  world_y_cm = (FRAME_HEIGHT - px_y) / px_per_cm   # Y축을 위로 뒤집는다

px_per_cm은 ArUco 로봇 마커의 실제 픽셀 크기로부터 매 프레임 갱신된다.
RobotTracker가 update_scale()을 호출해 주고, 마커가 안 보이면 직전 값을 유지한다.
"""

from __future__ import annotations

import os
import threading
import time
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
    """
    영상 입력. USB 웹캠(CAMERA_INDEX) 또는 네트워크 스트림(CAMERA_SOURCE, 예: 폰 IP Webcam 앱).

    네트워크 스트림은 별도 스레드가 계속 받아 '가장 최근 프레임'만 남긴다.
    처리 속도(약 15fps)가 송출 속도(30fps)보다 느릴 때 오래된 프레임이 쌓여 영상이
    점점 늦어지는 것을 막기 위해서다. 새 프레임이 오래 안 오면 read()가 실패를 돌려주고
    main이 STOP을 보낸다. 스트림이 끊기면 자동으로 다시 연결한다.

    출력 프레임은 항상 config.FRAME_WIDTH x FRAME_HEIGHT다. 입력 해상도가 다르면
    비율을 유지한 채 맞추고 남는 곳은 검게 채운다 (좌표 계산이 이 크기를 전제로 한다).
    """

    def __init__(self) -> None:
        src = config.CAMERA_SOURCE
        self.source = config.CAMERA_INDEX if src is None else src
        self.is_stream = isinstance(self.source, str)
        self.world = WorldFrame()
        self._size_warned = False

        self.cap = self._open()
        if not self.cap.isOpened():
            if self.is_stream:
                raise RuntimeError(
                    f"카메라 스트림을 열 수 없습니다 ({self.source}).\n"
                    "  - 폰 앱에서 서버가 켜져 있는지, PC와 같은 Wi-Fi인지 확인하세요.\n"
                    "  - IP Webcam 앱은 주소 끝에 /video 가 붙어야 합니다 (예: http://폰IP:8080/video)."
                )
            raise RuntimeError(
                f"웹캠을 열 수 없습니다 (CAMERA_INDEX={config.CAMERA_INDEX}). "
                "config.py의 CAMERA_INDEX를 확인하세요."
            )

        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._frame = None
        self._frame_t = 0.0
        self._seq = 0
        self._last_seq = 0
        self._thread = None
        if self.is_stream:
            self._thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
            self._thread.start()
            print(f"[Camera] 스트림 연결: {self.source}")

    # ------------------------------------------------------------------ 열기
    def _open(self):
        if self.is_stream:
            # 버퍼링을 줄이고, 응답이 없으면 5초 뒤 포기한다 (ffmpeg 옵션, OpenCV가 열 때 읽는다).
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                "fflags;nobuffer|flags;low_delay|timeout;5000000",
            )
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
        # 버퍼가 쌓이면 지연이 생겨 실시간 제어에 치명적이다.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    # ------------------------------------------------------------------ 스트림 수신 스레드
    def _reader(self) -> None:
        fails = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if ok and frame is not None:
                fails = 0
                with self._cond:
                    self._frame = frame
                    self._frame_t = time.time()
                    self._seq += 1
                    self._cond.notify_all()
                continue
            fails += 1
            time.sleep(0.05)
            if fails >= config.CAMERA_RECONNECT_AFTER:
                fails = 0
                print("[Camera] 스트림이 끊겼습니다. 다시 연결합니다...")
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = self._open()

    # ------------------------------------------------------------------ 크기 맞추기
    def _fit(self, frame):
        W, H = config.FRAME_WIDTH, config.FRAME_HEIGHT
        h, w = frame.shape[:2]
        if (w, h) == (W, H):
            return frame
        if not self._size_warned:
            self._size_warned = True
            print(f"[Camera] 입력 {w}x{h} -> {W}x{H}로 맞춥니다"
                  + ("" if abs(w / h - W / H) < 0.01 else " (비율이 달라 검은 여백을 채웁니다)"))
        s = min(W / w, H / h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (nw, nh), interpolation=interp)
        if (nw, nh) == (W, H):
            return resized
        canvas = np.zeros((H, W, 3), dtype=frame.dtype)
        x0, y0 = (W - nw) // 2, (H - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas

    # ------------------------------------------------------------------ 읽기
    def read(self):
        """(ok, frame) 반환. frame은 BGR. 새 프레임이 없거나 너무 오래됐으면 (False, None)."""
        if not self.is_stream:
            ok, frame = self.cap.read()
            if not ok:
                return False, None
        else:
            deadline = time.time() + config.CAMERA_READ_TIMEOUT_S
            with self._cond:
                while self._seq == self._last_seq:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return False, None
                    self._cond.wait(remaining)
                self._last_seq = self._seq
                frame, frame_t = self._frame, self._frame_t
            if time.time() - frame_t > config.CAMERA_STALE_S:
                return False, None
        frame = self._fit(frame)
        if config.FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)
        return True, frame

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()
