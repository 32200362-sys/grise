"""
D.I.G 실습 - 빠름/느림 2클래스 분류 모델 학습 스크립트

목적:
  - data_collector.py가 생성한 motion_dataset.csv를 읽어
  - 손 속도 피처로 '빠름(safe)' / '느림(danger)'을 분류하는 모델을 학습
  - 학습된 모델을 model.pkl로 저장 (realtime_predict.py에서 그대로 사용)

설계 원칙 (실습이라도 지키는 게 좋음):
  - timestamp는 피처로 쓰지 않음 (시간 누수 방지)
  - 연속 프레임이 거의 동일하므로 랜덤 분할 대신 '시간순 분할' 사용
  - 추론 코드와 동일한 피처(hand_speed_px_per_s 중심)만 사용

사용법:
  1) 먼저 data_collector.py를 실행해 데이터를 수집한다
     (s = 빠름, d = 느림 / 각 클래스 수백 행 이상 권장)
  2) motion_dataset.csv와 이 스크립트를 같은 폴더에 둔다
  3) python train.py 실행
  4) model.pkl 이 생성되면 성공
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import sklearn.linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# 설정값
# ---------------------------------------------------------
INPUT_CSV = "motion_dataset.csv"
OUTPUT_MODEL = "model.pkl"

# 학습에 사용할 피처 (추론 코드와 반드시 동일하게 유지할 것)
# 빠름/느림은 본질적으로 속도 문제이므로 속도 피처만 사용한다.
# 손만 움직이는 실습이므로 마커 기반 approach_speed는 제외하고
# 손 속도 하나만 사용한다. (마커가 없으면 approach_speed는 항상 0이라 무의미)
FEATURE_COLUMNS = ["hand_speed_px_per_s"]

# 라벨 컬럼 (data_collector.py가 기록하는 값: "safe" 또는 "danger")
LABEL_COLUMN = "label"

# 시간순 분할 비율 (앞 70%는 학습, 뒤 30%는 평가)
TRAIN_RATIO = 0.7


def main():
    # -----------------------------------------------------
    # 1. CSV 로드
    # -----------------------------------------------------
    if not os.path.isfile(INPUT_CSV):
        print(f"[오류] {INPUT_CSV} 파일이 없습니다. 먼저 data_collector.py로 데이터를 수집하세요.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"[로드] 전체 {len(df)}행 읽음")

    # -----------------------------------------------------
    # 2. 결측값 처리
    #    손/마커가 안 잡힌 프레임은 속도 피처가 비어 있을 수 있다.
    #    실습에서는 결측 행을 제거하는 게 가장 깔끔하다.
    # -----------------------------------------------------
    needed = FEATURE_COLUMNS + [LABEL_COLUMN]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        print(f"[오류] CSV에 필요한 컬럼이 없습니다: {missing_cols}")
        print(f"       현재 컬럼: {list(df.columns)}")
        sys.exit(1)

    before = len(df)
    df = df.dropna(subset=needed).reset_index(drop=True)
    print(f"[정리] 결측 행 제거: {before} -> {len(df)}행")

    # 라벨 분포 확인
    label_counts = df[LABEL_COLUMN].value_counts()
    print(f"[분포] 라벨별 행 수:\n{label_counts.to_string()}")

    if df[LABEL_COLUMN].nunique() < 2:
        print("[오류] 라벨이 한 종류뿐입니다. s(빠름)와 d(느림) 둘 다 수집했는지 확인하세요.")
        sys.exit(1)

    # 클래스 불균형이 심하면 경고만 (학습은 class_weight로 보정)
    ratio = label_counts.max() / label_counts.min()
    if ratio > 3:
        print(f"[경고] 클래스 불균형이 큽니다 (비율 {ratio:.1f}:1). "
              f"가능하면 적은 쪽 라벨을 더 수집하는 것을 권장합니다.")

    # -----------------------------------------------------
    # 3. 피처 / 라벨 분리
    # -----------------------------------------------------
    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    # 문자열 라벨을 그대로 사용 (sklearn이 내부적으로 처리)
    y = df[LABEL_COLUMN].values

    # -----------------------------------------------------
    # 4. 시간순 분할 (랜덤 분할 금지)
    #    연속 프레임이 거의 동일하므로 랜덤 분할 시 누수가 발생한다.
    #    CSV는 수집 시간 순서대로 저장되므로 앞/뒤로 자른다.
    #    주의: 이 방식은 s를 먼저, d를 나중에 길게 수집하면 한쪽으로 쏠릴 수 있다.
    #          수집 시 s/d를 번갈아 여러 번 토글하면 더 균형 잡힌 분할이 된다.
    # -----------------------------------------------------
    split_idx = int(len(df) * TRAIN_RATIO)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"[분할] 학습 {len(X_train)}행 / 평가 {len(X_test)}행 (시간순)")

    # 분할된 양쪽에 두 라벨이 모두 있는지 확인
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        print("[경고] 학습 또는 평가 세트에 한 라벨만 있습니다.")
        print("       수집할 때 s/d를 번갈아 여러 번 토글해서 다시 수집하길 권장합니다.")
        print("       (이대로도 학습은 진행하지만 평가 결과가 부정확할 수 있습니다.)")

    # -----------------------------------------------------
    # 5. 모델 학습
    #    StandardScaler + LogisticRegression 파이프라인.
    #    - Scaler: 두 피처(px/s, cm/s)의 단위가 달라 스케일을 맞춰줌
    #    - LogisticRegression: 선형 분리에 충분, 결정 경계 해석이 명확
    #    파이프라인으로 묶으면 추론 시에도 동일한 스케일링이 자동 적용된다.
    # -----------------------------------------------------
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", sklearn.linear_model.LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    model.fit(X_train, y_train)

    # -----------------------------------------------------
    # 6. 평가
    # -----------------------------------------------------
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 50)
    print(f"[평가] 테스트 정확도: {acc:.3f}")
    print("[평가] 상세 리포트:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("[평가] 혼동 행렬 (행=실제, 열=예측):")
    labels_sorted = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    print(f"       라벨 순서: {labels_sorted}")
    print(cm)
    print("=" * 50)

    if acc > 0.98:
        print("[참고] 정확도가 매우 높습니다. 빠름/느림은 속도로 거의 완벽히 갈리므로 "
              "정상일 수 있으나, 데이터 누수가 없는지(특히 분할 방식) 한 번 더 확인하세요.")

    # -----------------------------------------------------
    # 7. 모델 저장
    #    피처 순서도 함께 저장한다. 추론 코드가 같은 순서로 입력을 만들어야 하므로.
    # -----------------------------------------------------
    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "labels": labels_sorted,
    }
    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[저장] 모델 저장 완료: {OUTPUT_MODEL}")
    print(f"       사용 피처: {FEATURE_COLUMNS}")


if __name__ == "__main__":
    main()
