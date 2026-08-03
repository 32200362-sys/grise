"""
ESP32 없이 전송 계층을 검증하는 도구.

ESP32 대신 이 스크립트를 띄워두고 main.py를 실행하면
어떤 명령이 실제로 나가고 있는지 그대로 볼 수 있다.
패킷 주기(Hz)와 유실(seq 건너뜀)도 함께 계산해 준다.

사용법:
    1) config.ESP32_IP = "127.0.0.1" 로 바꾼다
    2) python tools/udp_monitor.py
    3) 다른 터미널에서 python main.py
"""

import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", config.ESP32_PORT))
sock.settimeout(1.0)

print(f"UDP 수신 대기: 0.0.0.0:{config.ESP32_PORT}   (Ctrl+C 종료)")
print("-" * 78)

count = 0
lost = 0
last_seq = None
t_start = time.time()
last_report = t_start

try:
    while True:
        try:
            data, addr = sock.recvfrom(1024)
        except socket.timeout:
            print("  ... 패킷 없음 (ESP32라면 지금 타임아웃 정지 상태)")
            continue

        count += 1
        try:
            pkt = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError:
            print(f"  [파싱 실패] {data!r}")
            continue

        seq = pkt.get("seq")
        if last_seq is not None and seq is not None and seq > last_seq + 1:
            lost += seq - last_seq - 1
        last_seq = seq

        latency_ms = (time.time() - pkt.get("t", time.time())) * 1000

        print(
            f"seq={seq:<6} {pkt.get('status','?'):<5} "
            f"vx={pkt.get('vx',0):+7.2f} vy={pkt.get('vy',0):+7.2f} "
            f"w={pkt.get('w',0):+6.3f}  lat={latency_ms:5.1f}ms"
        )

        now = time.time()
        if now - last_report >= 5.0:
            hz = count / (now - t_start)
            print("-" * 78)
            print(f"  통계: {count}패킷  {hz:.1f}Hz  유실 {lost}개  from {addr[0]}")
            print("-" * 78)
            last_report = now

except KeyboardInterrupt:
    elapsed = time.time() - t_start
    print(f"\n종료. 총 {count}패킷 / {elapsed:.1f}초 "
          f"= {count / max(elapsed, 1e-6):.1f}Hz, 유실 {lost}개")
