"""
로봇(ESP32) 연결 확인 - 움직임 명령은 절대 보내지 않는다 (STOP / heartbeat만).

    python tools/robot_link_check.py            # config.ESP32_IP (None이면 자동 탐색)
    python tools/robot_link_check.py 172.20.10.3

1) ESP32를 찾고 (브로드캐스트 STOP -> 텔레메트리 응답)
2) 텔레메트리(모드, 상태, 폴트, RSSI, 엔코더)를 1초마다 출력한다.
Ctrl+C로 종료하면 STOP을 보내고 끝난다.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from comm import UdpSender  # noqa: E402


def main() -> None:
    sender = UdpSender(ip=sys.argv[1] if len(sys.argv) > 1 else None)
    start = time.time()
    last_print = 0.0
    try:
        while True:
            # STOP만 보낸다. 세션 유지와 텔레메트리 수신용.
            sender.send(0.0, 0.0, 0.0, "STOP")
            now = time.time()
            if now - last_print >= 1.0:
                last_print = now
                if sender.connected:
                    t = sender.telemetry
                    print(f"[OK] {sender.esp_ip}  mode={t.get('mode')} state={t.get('state')} "
                          f"fault={t.get('fault')} rssi={t.get('wifi_rssi')} "
                          f"enc={t.get('encoder_count')} age={t.get('command_age_ms')}ms")
                elif now - start > 5.0:
                    print("[..] 텔레메트리 없음. 확인할 것: 로봇 전원/핫스팟 연결, "
                          "Windows 방화벽 UDP 8889 인바운드 허용, 또는 IP를 직접 인자로 지정")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()


if __name__ == "__main__":
    main()
