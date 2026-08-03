/*
 * D.I.G - [5] 제어 계층 (ESP32, C++)
 * ============================================================
 *  PC(main.py)가 UDP로 보내는 {vx, vy, w, status} JSON을 받아
 *  3륜 옴니휠 역기구학 -> 바퀴별 목표속도 -> 엔코더 PID -> PWM 출력.
 *
 *  수신 패킷:
 *    {"seq":1234,"t":1699.5,"vx":12.3,"vy":-4.5,"w":0.35,"status":"RUN"}
 *      vx : 로봇 정면(+X_robot) 방향 속도 [cm/s]
 *      vy : 로봇 좌측(+Y_robot) 방향 속도 [cm/s]
 *      w  : 각속도 [rad/s], 반시계(CCW) 양수
 *      status : "RUN" | "SLOW" | "STOP"
 *
 *  ★ 필수 안전장치 ★
 *    CMD_TIMEOUT_MS 동안 유효한 패킷이 없으면 즉시 정지한다.
 *    PC가 죽든, WiFi가 끊기든, 케이블이 빠지든 로봇은 반드시 선다.
 *    이 타임아웃은 어떤 상황에서도 우회되면 안 된다.
 *
 *  필요 라이브러리:
 *    - ArduinoJson (v7 이상)     : 라이브러리 매니저에서 설치
 *    - ESP32 Arduino Core        : v2.x / v3.x 모두 지원 (아래 LEDC 호환 처리)
 * ============================================================
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>
#include <math.h>

// ============================================================
// 1. 사용자 설정
// ============================================================
static const char* WIFI_SSID = "YOUR_WIFI_SSID";
static const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";

static const uint16_t UDP_PORT = 8888;

// 고정 IP - PC의 config.ESP32_IP와 반드시 일치시킬 것.
// DHCP를 쓰면 IP가 바뀌어 PC가 엉뚱한 곳으로 전송하게 된다.
static IPAddress LOCAL_IP(192, 168, 0, 50);
static IPAddress GATEWAY (192, 168, 0, 1);
static IPAddress SUBNET  (255, 255, 255, 0);

// --- 안전 타임아웃 (필수) ---
static const uint32_t CMD_TIMEOUT_MS = 300;   // PC는 30Hz(33ms)로 보낸다. 9패킷 유실 = 정지.

// --- 로봇 기구 제원 (실측해서 반드시 수정할 것) ---
static const float WHEEL_RADIUS_CM   = 2.9f;   // 옴니휠 반지름
static const float ROBOT_RADIUS_CM   = 9.0f;   // 로봇 중심 ~ 바퀴 접지점 거리 (L)
static const float ENCODER_CPR       = 11.0f;  // 엔코더 1회전 펄스 (모터축 기준)
static const float GEAR_RATIO        = 30.0f;  // 감속비
// 출력축 1회전당 엔코더 tick 수 (A상 하강/상승 모두 세면 x2, 여기선 상승만 세므로 x1)
static const float TICKS_PER_REV     = ENCODER_CPR * GEAR_RATIO;

// 모터가 낼 수 있는 최대 바퀴 선속도 [cm/s]. PWM 정규화 및 포화 판단에 쓴다.
static const float MAX_WHEEL_SPEED_CM_S = 60.0f;

/*
 * --- 3륜 옴니 배치 ---
 * 각 바퀴의 장착각 α_i : 로봇 +X축(정면)에서 반시계로 잰 각도.
 * 바퀴는 자기 위치의 접선 방향으로 구동력을 낸다. 따라서
 *
 *     v_i = -sin(α_i)*vx + cos(α_i)*vy + L*w
 *
 * 기본값 0/120/240도 배치에서 검증:
 *   순수 전진(vx>0) -> v0 = 0, v1 = -0.866vx, v2 = +0.866vx  (0도 바퀴는 정지) OK
 *   순수 회전(w>0)  -> 세 바퀴 모두 +L*w (같은 방향)              OK
 *
 * 실제 로봇 배치가 다르면 이 각도만 바꾸면 된다.
 */
static const float WHEEL_ANGLE_DEG[3] = { 0.0f, 120.0f, 240.0f };

// --- 핀 배치 (TB6612FNG / DRV8833 계열: PWM + IN1 + IN2) ---
static const int PIN_PWM[3] = { 25, 26, 27 };
static const int PIN_IN1[3] = { 14, 12, 13 };
static const int PIN_IN2[3] = { 32, 33, 15 };

// --- 엔코더 핀 (A상은 인터럽트 가능 핀이어야 한다) ---
static const int PIN_ENC_A[3] = { 34, 35, 36 };
static const int PIN_ENC_B[3] = { 39, 16, 17 };

// 바퀴 회전 방향이 반대로 달렸으면 -1로 뒤집는다.
static const int WHEEL_DIR_SIGN[3] = { 1, 1, 1 };

// --- PWM 설정 ---
static const int PWM_FREQ_HZ  = 20000;   // 20kHz - 가청 대역 밖
static const int PWM_RES_BITS = 10;      // 0 ~ 1023
static const int PWM_MAX      = (1 << PWM_RES_BITS) - 1;
static const int PWM_MIN_MOVE = 90;      // 이 이하는 모터가 안 도는 데드존

// --- PID 게인 (바퀴 속도 제어, 단위: cm/s -> PWM) ---
static float KP = 12.0f;
static float KI = 45.0f;
static float KD = 0.35f;

static const uint32_t CONTROL_PERIOD_MS = 10;   // 100Hz 제어 주기

// ============================================================
// 2. 전역 상태
// ============================================================
WiFiUDP udp;
char packetBuf[512];

// PC로부터 받은 최신 명령
volatile float cmd_vx = 0.0f;     // cm/s
volatile float cmd_vy = 0.0f;     // cm/s
volatile float cmd_w  = 0.0f;     // rad/s
String  cmd_status    = "STOP";
uint32_t last_cmd_ms  = 0;
int32_t  last_seq     = -1;

bool     estopped     = true;     // 부팅 직후는 정지 상태에서 시작

// 엔코더 카운터 (ISR에서 갱신)
volatile int32_t encTicks[3] = { 0, 0, 0 };
int32_t prevTicks[3] = { 0, 0, 0 };

// PID 상태
float targetSpeed[3]  = { 0, 0, 0 };   // cm/s
float measSpeed[3]    = { 0, 0, 0 };   // cm/s
float integral[3]     = { 0, 0, 0 };
float prevError[3]    = { 0, 0, 0 };

uint32_t lastControlMs = 0;
uint32_t lastStatusMs  = 0;

// ============================================================
// 3. LEDC 호환 래퍼 (ESP32 Core 2.x / 3.x)
// ============================================================
static void pwmSetup(int idx, int pin) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(pin, PWM_FREQ_HZ, PWM_RES_BITS);
#else
  ledcSetup(idx, PWM_FREQ_HZ, PWM_RES_BITS);
  ledcAttachPin(pin, idx);
#endif
}

static void pwmWrite(int idx, int pin, int duty) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(pin, duty);
#else
  (void)pin;
  ledcWrite(idx, duty);
#endif
}

// ============================================================
// 4. 엔코더 ISR
// ============================================================
// A상 상승엣지에서 B상을 읽어 회전 방향을 판별하는 표준 방식.
void IRAM_ATTR encISR0() { encTicks[0] += digitalRead(PIN_ENC_B[0]) ? 1 : -1; }
void IRAM_ATTR encISR1() { encTicks[1] += digitalRead(PIN_ENC_B[1]) ? 1 : -1; }
void IRAM_ATTR encISR2() { encTicks[2] += digitalRead(PIN_ENC_B[2]) ? 1 : -1; }

// ============================================================
// 5. 모터 구동
// ============================================================
static void motorWrite(int i, float pwmSigned) {
  int duty = (int)fabsf(pwmSigned);
  if (duty > PWM_MAX) duty = PWM_MAX;

  // 데드존 보정: 작지만 0이 아닌 명령은 최소 구동 PWM까지 끌어올린다.
  if (duty > 0 && duty < PWM_MIN_MOVE) duty = PWM_MIN_MOVE;

  bool forward = (pwmSigned >= 0);
  if (WHEEL_DIR_SIGN[i] < 0) forward = !forward;

  if (duty == 0) {
    // 브레이크 (양쪽 HIGH) - 관성 주행보다 정지가 빠르다
    digitalWrite(PIN_IN1[i], HIGH);
    digitalWrite(PIN_IN2[i], HIGH);
  } else if (forward) {
    digitalWrite(PIN_IN1[i], HIGH);
    digitalWrite(PIN_IN2[i], LOW);
  } else {
    digitalWrite(PIN_IN1[i], LOW);
    digitalWrite(PIN_IN2[i], HIGH);
  }
  pwmWrite(i, PIN_PWM[i], duty);
}

static void stopAllMotors() {
  for (int i = 0; i < 3; i++) {
    digitalWrite(PIN_IN1[i], HIGH);
    digitalWrite(PIN_IN2[i], HIGH);   // 브레이크
    pwmWrite(i, PIN_PWM[i], 0);
    targetSpeed[i] = 0.0f;
    integral[i]    = 0.0f;            // 적분 와인드업 제거
    prevError[i]   = 0.0f;
  }
}

// ============================================================
// 6. 역기구학 - 몸체 속도 (vx, vy, w) -> 바퀴 선속도 3개
// ============================================================
static void inverseKinematics(float vx, float vy, float w, float out[3]) {
  for (int i = 0; i < 3; i++) {
    float a = WHEEL_ANGLE_DEG[i] * (float)M_PI / 180.0f;
    out[i] = -sinf(a) * vx + cosf(a) * vy + ROBOT_RADIUS_CM * w;
  }

  // 포화 처리: 한 바퀴라도 최대치를 넘으면 세 바퀴를 같은 비율로 줄인다.
  // 개별로 자르면 합성 속도의 '방향'이 틀어져 로봇이 엉뚱하게 간다.
  float maxAbs = 0.0f;
  for (int i = 0; i < 3; i++) maxAbs = fmaxf(maxAbs, fabsf(out[i]));
  if (maxAbs > MAX_WHEEL_SPEED_CM_S) {
    float k = MAX_WHEEL_SPEED_CM_S / maxAbs;
    for (int i = 0; i < 3; i++) out[i] *= k;
  }
}

// ============================================================
// 7. UDP 수신 + JSON 파싱
// ============================================================
static void receiveCommand() {
  int packetSize = udp.parsePacket();
  while (packetSize > 0) {
    int len = udp.read(packetBuf, sizeof(packetBuf) - 1);
    if (len > 0) {
      packetBuf[len] = '\0';

      JsonDocument doc;
      DeserializationError err = deserializeJson(doc, packetBuf);

      if (err) {
        Serial.printf("[JSON] 파싱 실패: %s\n", err.c_str());
      } else {
        int32_t seq = doc["seq"] | -1;

        // 순서가 뒤바뀐(오래된) 패킷은 버린다. UDP는 순서를 보장하지 않는다.
        // seq가 줄어들었는데 차이가 크면 PC가 재시작한 것으로 보고 받아들인다.
        bool stale = (seq >= 0 && last_seq >= 0 && seq <= last_seq && (last_seq - seq) < 1000);
        if (!stale) {
          last_seq = seq;
          cmd_vx = doc["vx"] | 0.0f;
          cmd_vy = doc["vy"] | 0.0f;
          cmd_w  = doc["w"]  | 0.0f;
          cmd_status = String((const char*)(doc["status"] | "STOP"));
          last_cmd_ms = millis();
        }
      }
    }
    packetSize = udp.parsePacket();   // 밀린 패킷이 있으면 최신 것까지 소비
  }
}

// ============================================================
// 8. 속도 측정 + PID
// ============================================================
static void updateSpeeds(float dt) {
  for (int i = 0; i < 3; i++) {
    noInterrupts();
    int32_t ticks = encTicks[i];
    interrupts();

    int32_t delta = ticks - prevTicks[i];
    prevTicks[i] = ticks;

    // tick -> 회전수 -> 바퀴 원주 -> cm/s
    float revs = (float)delta / TICKS_PER_REV;
    float cm   = revs * 2.0f * (float)M_PI * WHEEL_RADIUS_CM;
    float raw  = cm / dt;

    // 1차 저역통과 - 엔코더 양자화 노이즈 제거
    measSpeed[i] = 0.7f * measSpeed[i] + 0.3f * raw * WHEEL_DIR_SIGN[i];
  }
}

static void runPID(float dt) {
  for (int i = 0; i < 3; i++) {
    float error = targetSpeed[i] - measSpeed[i];

    float p = KP * error;
    float d = KD * (error - prevError[i]) / dt;
    prevError[i] = error;

    // 적분: 출력이 포화됐을 땐 적분을 멈춘다 (anti-windup)
    float candidate = integral[i] + error * dt;
    float iTerm = KI * candidate;
    float total = p + iTerm + d;

    if (total > -PWM_MAX && total < PWM_MAX) {
      integral[i] = candidate;   // 포화 아님 -> 적분 반영
    } else {
      iTerm = KI * integral[i];  // 포화 -> 직전 적분 유지
      total = p + iTerm + d;
    }

    // 목표가 0이면 적분을 빠르게 털어낸다 (정지 후 슬금슬금 움직이는 것 방지)
    if (fabsf(targetSpeed[i]) < 0.5f) {
      integral[i] *= 0.85f;
      if (fabsf(measSpeed[i]) < 1.0f) total = 0.0f;
    }

    if (total >  PWM_MAX) total =  PWM_MAX;
    if (total < -PWM_MAX) total = -PWM_MAX;

    motorWrite(i, total);
  }
}

// ============================================================
// 9. setup / loop
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("\n=== D.I.G ESP32 옴니 제어기 ===");

  // --- 핀 초기화 (모터를 먼저 정지 상태로 만든 뒤 WiFi를 붙인다) ---
  for (int i = 0; i < 3; i++) {
    pinMode(PIN_IN1[i], OUTPUT);
    pinMode(PIN_IN2[i], OUTPUT);
    pwmSetup(i, PIN_PWM[i]);

    pinMode(PIN_ENC_A[i], INPUT_PULLUP);
    pinMode(PIN_ENC_B[i], INPUT_PULLUP);
  }
  stopAllMotors();

  attachInterrupt(digitalPinToInterrupt(PIN_ENC_A[0]), encISR0, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_A[1]), encISR1, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_A[2]), encISR2, RISING);

  // --- WiFi ---
  WiFi.mode(WIFI_STA);
  if (!WiFi.config(LOCAL_IP, GATEWAY, SUBNET)) {
    Serial.println("[WiFi] 고정 IP 설정 실패");
  }
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[WiFi] 연결 중");

  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) {
    delay(300);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] 연결됨. IP=%s\n", WiFi.localIP().toString().c_str());
    udp.begin(UDP_PORT);
    Serial.printf("[UDP] %d 포트 수신 대기\n", UDP_PORT);
  } else {
    Serial.println("\n[WiFi] 연결 실패 - 모터는 정지 상태로 유지됩니다.");
  }

  // WiFi가 붙어도 PC 패킷이 오기 전까지는 정지 상태를 유지한다.
  last_cmd_ms = 0;
  lastControlMs = millis();
}

void loop() {
  // --- 1) 수신 ---
  receiveCommand();

  // --- 2) 안전 판정 (★ 필수 안전장치 ★) ---
  uint32_t now = millis();
  bool timedOut = (last_cmd_ms == 0) || ((now - last_cmd_ms) > CMD_TIMEOUT_MS);
  bool wifiDown = (WiFi.status() != WL_CONNECTED);
  bool stopCmd  = (cmd_status == "STOP");

  bool shouldStop = timedOut || wifiDown || stopCmd;

  if (shouldStop != estopped) {
    estopped = shouldStop;
    Serial.printf("[안전] %s  (timeout=%d wifi_down=%d stop_cmd=%d)\n",
                  estopped ? "정지" : "주행 재개",
                  timedOut, wifiDown, stopCmd);
  }

  // --- 3) 제어 주기 ---
  if (now - lastControlMs < CONTROL_PERIOD_MS) return;
  float dt = (now - lastControlMs) / 1000.0f;
  lastControlMs = now;

  updateSpeeds(dt);

  if (estopped) {
    stopAllMotors();
  } else {
    // 역기구학: 몸체 속도 -> 바퀴별 목표 선속도
    float wheels[3];
    inverseKinematics(cmd_vx, cmd_vy, cmd_w, wheels);
    for (int i = 0; i < 3; i++) targetSpeed[i] = wheels[i];

    runPID(dt);
  }

  // --- 4) 상태 로그 (1초에 한 번) ---
  if (now - lastStatusMs > 1000) {
    lastStatusMs = now;
    Serial.printf(
      "[%s] cmd(%.1f, %.1f, %.2f) tgt(%.1f %.1f %.1f) meas(%.1f %.1f %.1f) seq=%ld\n",
      estopped ? "STOP" : cmd_status.c_str(),
      cmd_vx, cmd_vy, cmd_w,
      targetSpeed[0], targetSpeed[1], targetSpeed[2],
      measSpeed[0], measSpeed[1], measSpeed[2],
      (long)last_seq);
  }
}
