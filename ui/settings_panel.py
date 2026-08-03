"""
실시간 튜닝 패널.

왜 tkinter인가
    OpenCV의 cv2.createButton은 Qt 백엔드에서만 동작한다.
    이 프로젝트가 쓰는 opencv-python 휠은 WIN32UI 빌드라 버튼을 만들 수 없다.
    cv2.createTrackbar는 정수만 지원해서 0.35 같은 게인을 다루기 불편하다.
    tkinter는 표준 라이브러리라 의존성이 안 늘고, 슬라이더+숫자입력+탭을 다 쓸 수 있다.

★ 왜 스레드가 아니라 별도 프로세스인가 ★
    Tk를 워커 스레드에서 띄우고 메인 스레드가 영상 루프를 도는 구조는 안전하지 않다.
    tkinter 위젯들은 서로 참조 순환을 만들기 때문에 참조 카운트만으로는 해제되지 않고,
    결국 '순환 GC'가 치우게 된다. 그런데 순환 GC는 어느 스레드에서든 돌 수 있어서,
    메인 스레드가 Tcl 객체를 해제하는 순간
        Tcl_AsyncDelete: async handler deleted by the wrong thread
    로 프로세스가 즉시 죽는다. (실제로 gc.collect() 한 번에 재현됨)
    이건 정리 코드를 아무리 잘 짜도 막을 수 없다. GC 타이밍을 통제할 수 없기 때문이다.

    그래서 패널을 자식 프로세스로 완전히 분리했다.
    부모(영상 루프)는 tkinter를 import조차 하지 않는다.

프로세스 간 통신
    자식 -> 부모 : stdout에 JSON 한 줄씩. {"attr": "MAX_LINEAR_SPEED_CM_S", "value": 20.0}
    부모 -> 자식 : stdin에 JSON 한 줄씩 (현재 측정값 표시용)
    부모는 리더 스레드로 stdout을 읽어 큐에 쌓고, 영상 루프가 pump()에서 config에 적용한다.
    큐에 오가는 건 전부 순수 문자열/숫자라 스레드 안전 문제가 없다.

어떻게 즉시 반영되나
    각 계층(risk_evaluator / potential_field / transform)은 매 호출마다 config.XXX를
    읽고 캐싱하지 않는다. 따라서 pump()가 config 속성을 덮어쓰면 다음 프레임부터 적용된다.

저장
    조정한 값은 tuning.json으로 저장되고 다음 실행 때 자동으로 불러온다.
    config.py 원본은 건드리지 않으므로, 파일을 지우면 언제든 기본값으로 돌아온다.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

# 이 파일은 두 가지 방식으로 로드된다.
#   1) 부모: `from ui import SettingsPanel`  -> 프로젝트 루트가 이미 sys.path에 있음
#   2) 자식: `python ui/settings_panel.py`   -> sys.path[0]이 ui/ 라서 config를 못 찾음
# 2번을 위해 프로젝트 루트를 먼저 넣어준다.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402

TUNING_FILE = Path(__file__).resolve().parent.parent / "tuning.json"

# ---------------------------------------------------------
# 조절 대상 파라미터
#   (config 속성명, 표시 이름, 최소, 최대, 소수자리, 설명)
# 물리 실측값(MARKER_SIZE_CM 등)은 일부러 뺐다.
# 그건 튜닝 대상이 아니라 자로 재서 넣어야 하는 값이다.
# ---------------------------------------------------------
PARAM_SPECS: dict[str, list[tuple]] = {
    "판단 (위험 판정)": [
        ("RISK_DANGER_DIST_CM", "DANGER 거리", 3, 60, 1, "손-컵 거리가 이 안이면 즉시 위험"),
        ("RISK_WARN_DIST_CM", "WARN 거리", 5, 120, 1, "이 안이면 경고 (감속)"),
        ("RISK_APPROACH_SPEED_CM_S", "접근속도 임계", 3, 100, 1, "이보다 빠르게 다가오면 TTC 판정"),
        ("RISK_TTC_DANGER_S", "TTC DANGER", 0.1, 3.0, 2, "충돌예상시간이 이보다 짧으면 위험"),
        ("RISK_TTC_WARN_S", "TTC WARN", 0.2, 5.0, 2, "충돌예상시간이 이보다 짧으면 경고"),
        ("RISK_DIST_EMA_ALPHA", "거리 스무딩", 0.05, 1.0, 2, "작을수록 부드럽고 느림"),
        ("RISK_SPEED_EMA_ALPHA", "속도 스무딩", 0.05, 1.0, 2, "작을수록 부드럽고 느림"),
        ("RISK_DANGER_HOLD_S", "DANGER 유지", 0.0, 3.0, 2, "위험 해제까지 최소 유지 시간"),
        ("RISK_WARN_HOLD_S", "WARN 유지", 0.0, 3.0, 2, "경고 해제까지 최소 유지 시간"),
    ],
    "경로계산 (potential field)": [
        ("PF_K_ATTRACT", "인력 게인", 0.05, 5.0, 2, "컵으로 끌리는 힘"),
        ("PF_ATTRACT_MAX_CM", "인력 포화거리", 10, 150, 1, "이보다 멀면 인력이 일정해짐"),
        ("PF_GOAL_TOLERANCE_CM", "도착 판정거리", 2, 40, 1, "이 안에 들어오면 도착"),
        ("PF_K_REPULSE_OBSTACLE", "장애물 척력", 0, 200, 1, "인력 최대치(72)와 비교해서 읽을 것"),
        ("PF_OBSTACLE_INFLUENCE_CM", "장애물 영향반경", 10, 120, 1, "이 밖의 장애물은 무시"),
        ("PF_K_REPULSE_HUMAN", "사람 척력", 0, 400, 1, "장애물보다 크게 두는 게 보통"),
        ("PF_HUMAN_INFLUENCE_CM", "사람 영향반경", 10, 180, 1, "사람은 더 멀리서부터 피한다"),
        ("PF_FORCE_TO_SPEED", "힘→속도 게인", 0.05, 2.0, 2, "0.5면 인력최대(72)에서 36cm/s"),
        ("PF_ESCAPE_GAIN", "지역최소 탈출", 0.0, 2.0, 2, "합력이 0에 가까울 때 옆으로 빠지는 세기"),
    ],
    "속도 · 자세": [
        ("MAX_LINEAR_SPEED_CM_S", "최대 속도", 3, 80, 1, "안전 확인 전에는 낮게 두세요"),
        ("MAX_LINEAR_ACCEL_CM_S2", "최대 가속도", 10, 300, 1, "작을수록 부드럽지만 정지거리 증가"),
        ("MAX_ANGULAR_SPEED_RAD_S", "최대 각속도", 0.1, 5.0, 2, "회전 속도 상한"),
        ("HEADING_KP", "heading P게인", 0.0, 8.0, 2, "진행방향으로 정면을 맞추는 세기"),
        ("HEADING_DEADBAND_RAD", "heading 불감대", 0.0, 0.5, 3, "이 안이면 회전 안함 (떨림 방지)"),
    ],
    "인식": [
        ("YOLO_CONF_THRESHOLD", "YOLO 신뢰도", 0.05, 0.95, 2, "낮추면 많이 잡지만 오탐 증가"),
        ("POSE_MIN_VISIBILITY", "관절 신뢰도", 0.05, 0.95, 2, "이 미만인 관절은 없는 것으로 취급"),
        ("ROBOT_POSE_EMA_ALPHA", "로봇자세 스무딩", 0.05, 1.0, 2, "작을수록 부드럽고 느림"),
        ("YOLO_EVERY_N_FRAMES", "YOLO 주기", 1, 10, 0, "N프레임마다 1회 추론 (크면 빨라짐)"),
    ],
    "그립 · 시선": [
        ("GRIP_APERTURE_MIN", "그립 벌림 하한", 0.0, 3.0, 2, "엄지-검지 / 손크기. 이보다 좁으면 주먹"),
        ("GRIP_APERTURE_MAX", "그립 벌림 상한", 0.0, 3.0, 2, "이보다 넓으면 그냥 편 손"),
        ("GRIP_OPENNESS_MAX", "손 펼침 상한", 0.5, 4.0, 2, "이보다 크면 편 손으로 간주"),
        ("GAZE_CONE_DEG", "시선 원뿔각", 5, 90, 1, "머리방향과 컵방향 각도가 이 안이면 '보는 중'"),
        ("GAZE_FORWARD_SIGN", "시선 부호", -1, 1, 0, "화살표가 반대로 나오면 -1로"),
        ("INTENT_DIST_BOOST", "의도 거리배수", 1.0, 3.0, 2, "의도 감지 시 거리 임계를 이만큼 넓힘"),
        ("INTENT_TTC_BOOST", "의도 TTC배수", 1.0, 3.0, 2, "의도 감지 시 TTC 임계를 이만큼 넓힘"),
        ("INTENT_HOLD_S", "의도 유지시간", 0.0, 5.0, 2, "신호가 끊겨도 이 시간만큼 유지"),
    ],
}

# 위험 등급별 속도 배율. DANGER는 안전 기능이라 UI에 노출하지 않고 0으로 고정한다.
SPEED_SCALE_KEYS = [
    ("SAFE", "SAFE 속도배율", 0.0, 1.0),
    ("WARN", "WARN 속도배율", 0.0, 1.0),
]

# 체크박스로 켜고 끄는 항목.
# 매 프레임 config에서 읽히는 값만 넣을 것. (USE_HAND/USE_GAZE처럼 시작 시점에
# 트래커를 만들지 말지 결정하는 값은 실행 중에 바꿔도 효과가 없다)
BOOL_SPECS = [
    ("TEST_MODE_ARUCO", "테스트 모드 (ArUco로 컵/장애물 인식)",
     "YOLO 대신 마커로 인식한다. 모델 없이 전체 파이프라인을 돌려볼 수 있다."),
    ("INTENT_USE_GRIP", "그립 모양을 의도 신호로 사용",
     "잡을 준비가 된 손 모양이면 거리/TTC 임계를 넓힌다."),
    ("INTENT_USE_GAZE", "시선을 의도 신호로 사용",
     "컵을 보고 있으면 거리/TTC 임계를 넓힌다."),
    ("SHOW_WINDOW", "영상 창 표시", "끄면 화면 없이 헤드리스로 돈다."),
    ("DRAW_POSE", "사람 관절 그리기", ""),
    ("DRAW_DETECTIONS", "검출 박스 그리기", ""),
    ("DRAW_FIELD_VECTOR", "속도 벡터 그리기", ""),
]

SS_PREFIX = "SPEED_SCALE::"

_DEFAULTS: dict[str, float] = {}


def _all_attrs():
    for specs in PARAM_SPECS.values():
        for spec in specs:
            yield spec


def _capture_defaults() -> None:
    """config.py 원본 값을 한 번만 기억해 둔다 (초기화 버튼용)."""
    if _DEFAULTS:
        return
    for attr, *_ in _all_attrs():
        _DEFAULTS[attr] = getattr(config, attr)
    for attr, *_ in BOOL_SPECS:
        _DEFAULTS[attr] = getattr(config, attr)
    for key, *_ in SPEED_SCALE_KEYS:
        _DEFAULTS[SS_PREFIX + key] = config.SPEED_SCALE_BY_RISK[key]


def current_values() -> dict:
    """현재 config에서 튜닝 대상 값만 뽑아온다."""
    d = {attr: getattr(config, attr) for attr, *_ in _all_attrs()}
    d.update({attr: bool(getattr(config, attr)) for attr, *_ in BOOL_SPECS})
    for key, *_ in SPEED_SCALE_KEYS:
        d[SS_PREFIX + key] = config.SPEED_SCALE_BY_RISK[key]
    return d


def apply_value(attr: str, value) -> None:
    """이름 하나를 config에 반영. DANGER 배율은 어떤 경우에도 0을 유지한다."""
    if attr.startswith(SS_PREFIX):
        key = attr[len(SS_PREFIX):]
        if key == "DANGER":
            config.SPEED_SCALE_BY_RISK["DANGER"] = 0.0
        elif key in config.SPEED_SCALE_BY_RISK:
            config.SPEED_SCALE_BY_RISK[key] = float(value)
    elif hasattr(config, attr):
        cur = getattr(config, attr)
        # bool은 int의 하위 클래스라 반드시 int보다 먼저 검사해야 한다.
        # 순서를 바꾸면 True/False가 1/0 정수로 바뀌어버린다.
        if isinstance(cur, bool):
            setattr(config, attr, bool(value))
        elif isinstance(cur, int):
            setattr(config, attr, int(round(float(value))))
        else:
            setattr(config, attr, float(value))


def save_tuning(path: Path = TUNING_FILE) -> None:
    path.write_text(json.dumps(current_values(), indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"[튜닝] 저장: {path}")


def load_tuning(path: Path = TUNING_FILE, verbose: bool = True) -> bool:
    """저장된 값을 config에 적용. 파일이 없으면 아무것도 안 한다."""
    _capture_defaults()
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"[튜닝] 불러오기 실패({e}). 기본값을 사용합니다.")
        return False
    for k, v in data.items():
        apply_value(k, v)
    config.SPEED_SCALE_BY_RISK["DANGER"] = 0.0
    if verbose:
        print(f"[튜닝] {path.name} 에서 {len(data)}개 값 적용")
    return True


def reset_to_defaults() -> None:
    _capture_defaults()
    for k, v in _DEFAULTS.items():
        apply_value(k, v)
    config.SPEED_SCALE_BY_RISK["DANGER"] = 0.0


# =========================================================
# 부모(영상 루프) 쪽
# =========================================================
class SettingsPanel:
    """
    설정 창을 자식 프로세스로 띄우고, 변경된 값을 config에 반영한다.
    이 클래스는 tkinter를 import하지 않는다.

    영상 루프에서 할 일:
        panel.pump({"손-컵 거리": "..."})   # 매 프레임 1회
        panel.toggle()                      # 버튼/키 입력 시
        panel.close()                       # 종료 시
    """

    def __init__(self) -> None:
        _capture_defaults()
        self._proc: subprocess.Popen | None = None
        self._q: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None

    # ---------- 상태 ----------
    def is_open(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ---------- 열기/닫기 ----------
    def toggle(self) -> None:
        self.close() if self.is_open() else self.open()

    def open(self) -> None:
        if self.is_open():
            return
        script = str(Path(__file__).resolve())
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-u", script, json.dumps(current_values())],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None, text=True, encoding="utf-8",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as e:
            print(f"[설정] 패널을 띄우지 못했습니다: {e}")
            self._proc = None
            return

        self._reader = threading.Thread(target=self._read_loop,
                                        args=(self._proc,), daemon=True)
        self._reader.start()

    def close(self) -> None:
        p = self._proc
        if p is None:
            return
        try:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    p.kill()
        except OSError:
            pass
        self._proc = None

    # ---------- 매 프레임 ----------
    def pump(self, readout: dict[str, str] | None = None) -> int:
        """
        자식이 보낸 값 변경을 config에 반영하고, 현재 측정값을 자식에게 보낸다.
        반영한 개수를 반환한다. 영상 루프에서 매 프레임 부르면 된다.
        """
        n = 0
        while True:
            try:
                attr, value = self._q.get_nowait()
            except queue.Empty:
                break
            apply_value(attr, value)
            n += 1

        if readout and self.is_open():
            try:
                self._proc.stdin.write(json.dumps(readout, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
            except (OSError, ValueError, AttributeError):
                pass  # 자식이 이미 죽었으면 무시
        return n

    # ---------- 리더 스레드 (문자열만 다룬다) ----------
    def _read_loop(self, proc: subprocess.Popen) -> None:
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "attr" in msg:
                    self._q.put((msg["attr"], msg["value"]))
        except (OSError, ValueError):
            pass


# =========================================================
# 자식(설정 창) 쪽 - 여기서만 tkinter를 쓴다
# =========================================================
def _run_gui(init_values: dict) -> None:
    import tkinter as tk
    from tkinter import ttk

    for k, v in init_values.items():
        apply_value(k, v)

    out = sys.stdout

    def emit(attr: str, value) -> None:
        out.write(json.dumps({"attr": attr, "value": value}) + "\n")
        out.flush()

    # 부모가 보내는 측정값을 읽는 스레드. Tk를 만지지 않고 dict만 갱신한다.
    readout: dict[str, str] = {}

    def stdin_loop():
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                readout.clear()
                readout.update(d)
        except (OSError, ValueError):
            pass

    threading.Thread(target=stdin_loop, daemon=True).start()

    root = tk.Tk()
    root.title("D.I.G 설정 - 실시간 튜닝")
    root.geometry("580x720")

    syncers: list = []

    def fmt(v: float, dec: int) -> str:
        return f"{v:.0f}" if dec == 0 else f"{v:.{dec}f}"

    def add_row(parent, row, key, label, lo, hi, dec, desc, getter):
        cur = float(getter())
        var = tk.DoubleVar(value=cur)
        evar = tk.StringVar(value=fmt(cur, dec))

        ttk.Label(parent, text=label, width=17).grid(
            row=row * 2, column=0, sticky="w", padx=(8, 4), pady=(6, 0))

        def apply(v, from_entry=False):
            v = max(lo, min(hi, v))
            if dec == 0:
                v = int(round(v))
            apply_value(key, v)
            emit(key, v)
            if not from_entry:
                evar.set(fmt(v, dec))

        def on_slide(_=None):
            apply(var.get())

        def on_entry(_=None):
            try:
                v = float(evar.get())
            except ValueError:
                evar.set(fmt(getter(), dec))
                return
            v = max(lo, min(hi, v))
            var.set(v)
            apply(v, from_entry=True)
            evar.set(fmt(getter(), dec))

        ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal",
                  command=on_slide).grid(row=row * 2, column=1, sticky="ew",
                                         padx=4, pady=(6, 0))
        e = ttk.Entry(parent, textvariable=evar, width=8, justify="right")
        e.grid(row=row * 2, column=2, padx=(4, 8), pady=(6, 0))
        e.bind("<Return>", on_entry)
        e.bind("<FocusOut>", on_entry)

        if desc:
            ttk.Label(parent, text=desc, foreground="#666", font=("", 8)).grid(
                row=row * 2 + 1, column=0, columnspan=3, sticky="w", padx=(10, 8))

        def sync():
            v = float(getter())
            var.set(v)
            evar.set(fmt(v, dec))
        syncers.append(sync)

    def add_check(parent, row, attr, label, desc):
        var = tk.BooleanVar(value=bool(getattr(config, attr)))

        def on_toggle():
            v = bool(var.get())
            apply_value(attr, v)
            emit(attr, v)

        ttk.Checkbutton(parent, text=label, variable=var,
                        command=on_toggle).grid(row=row * 2, column=0, columnspan=3,
                                                sticky="w", padx=8, pady=(10, 0))
        if desc:
            ttk.Label(parent, text=desc, foreground="#666", font=("", 8)).grid(
                row=row * 2 + 1, column=0, columnspan=3, sticky="w", padx=(28, 8))

        def sync():
            var.set(bool(getattr(config, attr)))
        syncers.append(sync)

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=8, pady=(8, 4))

    # --- 모드 탭 (체크박스) ---
    tab = ttk.Frame(nb)
    nb.add(tab, text="모드")
    for row, (attr, label, desc) in enumerate(BOOL_SPECS):
        add_check(tab, row, attr, label, desc)
    ttk.Label(
        tab,
        text="테스트 모드에서 인식하는 마커 ID는 config.MARKER_OBJECTS에 있습니다.\n"
             "각 ID의 '실제 반경'은 반드시 물체의 진짜 크기로 적어야 합니다.\n"
             "마커 크기가 아닙니다 — 작게 적으면 로봇이 물체를 긁고 지나갑니다.",
        foreground="#a60", justify="left",
    ).grid(row=len(BOOL_SPECS) * 2, column=0, columnspan=3,
           sticky="w", padx=8, pady=(18, 4))
    tab.columnconfigure(1, weight=1)

    for group, specs in PARAM_SPECS.items():
        tab = ttk.Frame(nb)
        nb.add(tab, text=group)
        for row, (attr, label, lo, hi, dec, desc) in enumerate(specs):
            add_row(tab, row, attr, label, lo, hi, dec, desc,
                    (lambda a=attr: getattr(config, a)))
        tab.columnconfigure(1, weight=1)

    tab = ttk.Frame(nb)
    nb.add(tab, text="속도배율")
    for row, (key, label, lo, hi) in enumerate(SPEED_SCALE_KEYS):
        add_row(tab, row, SS_PREFIX + key, label, lo, hi, 2, "",
                (lambda k=key: config.SPEED_SCALE_BY_RISK[k]))
    ttk.Label(
        tab,
        text="DANGER 배율은 0으로 고정되어 있습니다.\n"
             "안전 정지 기능이라 UI에서 올릴 수 없게 했습니다.\n"
             "정말 바꿔야 하면 config.py를 직접 수정하세요.",
        foreground="#b00", justify="left",
    ).grid(row=len(SPEED_SCALE_KEYS) * 2, column=0, columnspan=3,
           sticky="w", padx=8, pady=(16, 4))
    tab.columnconfigure(1, weight=1)

    ro_frame = ttk.LabelFrame(root, text="현재 측정값")
    ro_frame.pack(fill="x", padx=8, pady=4)
    ro_label = ttk.Label(ro_frame, text="(영상 창이 실행 중이면 여기에 표시됩니다)",
                         font=("Consolas", 9), justify="left")
    ro_label.pack(anchor="w", padx=8, pady=6)

    def push_all():
        """불러오기/초기화 후 모든 값을 부모에게 다시 알린다."""
        for k, v in current_values().items():
            emit(k, v)
        for s in syncers:
            s()

    def on_load():
        load_tuning()
        push_all()

    def on_reset():
        reset_to_defaults()
        push_all()
        print("[튜닝] 기본값으로 초기화", file=sys.stderr)

    btns = ttk.Frame(root)
    btns.pack(fill="x", padx=8, pady=(0, 8))
    ttk.Button(btns, text="저장", command=save_tuning).pack(side="left")
    ttk.Button(btns, text="불러오기", command=on_load).pack(side="left", padx=4)
    ttk.Button(btns, text="기본값으로 초기화", command=on_reset).pack(side="left", padx=4)
    ttk.Button(btns, text="닫기", command=root.destroy).pack(side="right")

    def pump():
        if readout:
            ro_label.config(text="\n".join(f"{k:<12}{v}" for k, v in readout.items()))
        root.after(120, pump)

    root.after(120, pump)
    root.mainloop()


if __name__ == "__main__":
    _init = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    _run_gui(_init)
