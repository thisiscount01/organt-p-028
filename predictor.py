"""
predictor.py — 임시 통계 기반 스텁 (Prophet/LSTM 교체 전 서버 검증용)
────────────────────────────────────────────────────────────────────────────
인터페이스:
    from predictor import predict
    result = predict(item_code="112", days=7)
    # → {"predicted_prices": [float,...], "grade": str, "trend_pct": float, "anomaly_flag": bool}

구현 방식:
    - 최근 60일 소매가격 데이터(data/{code}.csv) 로드
    - 지수평활(ETS) + 선형 추세로 N일 예측
    - 이상탐지: IQR 기반 (최근 7일 예측값이 과거 30일 분포의 IQR 범위를 벗어나면 이상)

※ AI 엔지니어가 Prophet/LSTM 구현으로 이 파일을 교체해야 합니다.
   train-serve skew 방지: 학습 feature = data/{code}.csv 의 '날짜','가격' 컬럼 고정.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("predictor")

DATA_DIR = Path(os.getenv("KAMIS_DATA_DIR", "data"))

# 5단계 등급 경계 (trend_pct 기준)
_GRADE_THRESHOLDS = [
    ( 10.0, "급등경보"),
    (  3.0, "상승"),
    ( -3.0, "보합"),
    (-10.0, "하락"),
]


def _grade(trend_pct: float) -> str:
    for threshold, label in _GRADE_THRESHOLDS:
        if trend_pct >= threshold:
            return label
    return "급락경보"


def _load_series(item_code: str, window: int = 120) -> pd.Series:
    """data/{code}.csv → 날짜 인덱스 가격 Series (최근 window 일)."""
    path = DATA_DIR / f"{item_code}.csv"
    if not path.exists():
        raise FileNotFoundError(f"데이터 없음: {path}")

    for enc in ("utf-8-sig", "euc-kr", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f"인코딩 감지 실패: {path}")

    # 컬럼 자동 탐지
    date_col = next((c for c in df.columns if any(h in c for h in ("날짜", "일자", "date"))), df.columns[0])
    price_col = next((c for c in df.columns if any(h in c for h in ("가격", "price", "소매", "도매"))), df.columns[-1])

    df = df[[date_col, price_col]].copy()
    df.columns = ["date", "price"]
    df["price"] = pd.to_numeric(df["price"].astype(str).str.replace(",", ""), errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna().sort_values("date").drop_duplicates("date")
    df = df[df["price"] > 0].tail(window)

    if len(df) < 7:
        raise ValueError(f"학습 데이터 부족: {len(df)}행 (최소 7행 필요)")

    return df.set_index("date")["price"]


def _ets_forecast(series: pd.Series, days: int) -> np.ndarray:
    """
    이중 지수평활 (Holt's Linear) + 장기 평균 회귀 댐핑.
    - α=0.15 (평활) / β=0.05 (추세) — 농산물 가격의 급등락 충격 흡수
    - 예측이 장기 이동평균(60일)에서 ±30% 이탈 시 10% 댐핑 적용
    Prophet/LSTM 교체 전 서버 동작 검증용.
    """
    vals = series.values.astype(float)
    n = len(vals)

    # Holt's Linear Exponential Smoothing
    alpha = 0.15   # 수준 평활 (낮을수록 단기 충격에 둔감)
    beta  = 0.05   # 추세 평활 (낮을수록 추세 변화가 완만)

    level  = np.zeros(n)
    trend_ = np.zeros(n)
    level[0]  = vals[0]
    trend_[0] = vals[1] - vals[0] if n > 1 else 0.0

    for i in range(1, n):
        prev_l = level[i - 1]
        prev_t = trend_[i - 1]
        level[i]  = alpha * vals[i] + (1 - alpha) * (prev_l + prev_t)
        trend_[i] = beta * (level[i] - prev_l) + (1 - beta) * prev_t

    last_level = level[-1]
    last_trend = trend_[-1]

    # 장기 평균 (60일) — 회귀 앵커
    long_mean = float(np.mean(vals[-min(60, n):]))

    # 추세 감쇠 인자 φ (damped trend, φ=0.88)
    phi = 0.88

    # 예측
    forecast = np.zeros(days)
    phi_sum = 0.0
    for d in range(days):
        phi_sum += phi ** (d + 1)
        raw = last_level + last_trend * phi_sum
        # 장기 평균 ±30% 벗어나면 10% 회귀 댐핑
        if raw > long_mean * 1.30:
            raw = raw * 0.90 + long_mean * 0.10
        elif raw < long_mean * 0.70:
            raw = raw * 0.90 + long_mean * 0.10
        forecast[d] = max(raw, 1.0)

    return forecast


def _detect_anomaly(series: pd.Series, forecast: np.ndarray) -> bool:
    """IQR 기반 이상탐지: 예측값 평균이 과거 30일 [Q1-1.5IQR, Q3+1.5IQR] 벗어나면 True."""
    hist = series.values[-30:].astype(float)
    if len(hist) < 4:
        return False
    q1, q3 = np.percentile(hist, 25), np.percentile(hist, 75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mean_fc = float(np.mean(forecast))
    return mean_fc < lo or mean_fc > hi


# ─────────────────────────────────────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────────────────────────────────────

def predict(item_code: str, days: int = 7) -> dict[str, Any]:
    """
    Parameters
    ----------
    item_code : str   서버 내부 품목 코드 (e.g. "112")
    days      : int   예측 일수 (1~30)

    Returns
    -------
    dict with keys:
        predicted_prices : list[float]  — 예측 가격 배열 (len == days)
        grade            : str          — 5단계 구매 타이밍 등급
        trend_pct        : float        — 예측 기간 가격 변화율 (%)
        anomaly_flag     : bool         — 이상가격 여부
    """
    series = _load_series(item_code)
    forecast = _ets_forecast(series, days)

    last_actual = float(series.iloc[-1])
    mean_fc     = float(np.mean(forecast))
    trend_pct   = (mean_fc - last_actual) / last_actual * 100 if last_actual > 0 else 0.0

    grade        = _grade(trend_pct)
    anomaly_flag = _detect_anomaly(series, forecast)

    # 이상탐지 시 등급 강제 조정
    if anomaly_flag and grade == "보합":
        grade = "상승" if trend_pct >= 0 else "하락"

    logger.debug(
        "predict [%s] days=%d last=%.0f mean_fc=%.0f trend=%.2f%% grade=%s anomaly=%s",
        item_code, days, last_actual, mean_fc, trend_pct, grade, anomaly_flag,
    )

    return {
        "predicted_prices": [round(float(p), 0) for p in forecast],
        "grade":            grade,
        "trend_pct":        round(trend_pct, 2),
        "anomaly_flag":     anomaly_flag,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI 검증
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    codes = sys.argv[1:] or ["112", "214", "215", "218", "111", "131", "117"]
    for code in codes:
        try:
            r = predict(code, days=7)
            print(f"[{code}] grade={r['grade']} trend={r['trend_pct']:+.1f}% "
                  f"anomaly={r['anomaly_flag']} prices={[int(p) for p in r['predicted_prices']]}")
        except Exception as e:
            print(f"[{code}] 오류: {e}")
