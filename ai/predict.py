"""
추론 모듈 — Prophet/LightGBM 통합 predict
품목명 입력 → {item, current_price, prices_7d, mape, anomaly_flag, buy_timing, ...}
"""
import os, sys, pickle, glob
from datetime import datetime
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Prophet은 선택적 의존성 — Render 무료 티어에서 pystan 컴파일 타임아웃 방지
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

MODEL_DIR = 'model'
_cache = {}   # 품목 → pkl 패키지 (메모리)

# train_lgbm.py 의 FEAT_COLS 와 동일 — lgbm pkl에 feat_cols 없을 경우 폴백
_LGBM_FEAT_COLS = (
    ['day_of_week', 'day_of_month', 'month', 'week_of_year', 'quarter']
    + [f'lag_{l}' for l in [1, 3, 7, 14, 21]]
    + [f'pct_change_{l}d' for l in [1, 3, 7, 14, 21]]
    + [f'rolling_mean_{w}' for w in [3, 7, 14, 30]]
    + [f'rolling_std_{w}' for w in [3, 7, 14, 30]]
    + [f'rolling_min_{w}' for w in [3, 7, 14, 30]]
    + [f'rolling_max_{w}' for w in [3, 7, 14, 30]]
    + ['mom_7d', 'mom_14d']
)


def _load_pkg(item: str):
    if item not in _cache:
        prophet_path = os.path.join(MODEL_DIR, f'{item}_prophet.pkl')
        lgbm_path    = os.path.join(MODEL_DIR, f'{item}_lgbm.pkl')
        pkg = None

        # Prophet 사용 가능하면 prophet pkl 우선 시도
        if PROPHET_AVAILABLE and os.path.exists(prophet_path):
            try:
                with open(prophet_path, 'rb') as f:
                    pkg = pickle.load(f)
            except Exception:
                pkg = None

        # prophet 불가 또는 실패 시 lgbm pkl 폴백
        if pkg is None and os.path.exists(lgbm_path):
            with open(lgbm_path, 'rb') as f:
                pkg = pickle.load(f)
            if 'feat_cols' not in pkg:
                pkg['feat_cols'] = _LGBM_FEAT_COLS

        if pkg is None:
            return None
        _cache[item] = pkg
    return _cache.get(item)


def _predict_prophet(pkg: dict) -> list:
    """Prophet 모델로 7일 예측 → [{date, price, lower, upper}]"""
    m = pkg['model']
    future = m.make_future_dataframe(periods=7, freq='D')
    fc = m.predict(future)
    fc7 = fc.tail(7)
    return [
        {
            'date': str(row['ds'].date()),
            'price': round(float(max(row['yhat'], 0)), 0),
            'lower': round(float(max(row.get('yhat_lower', row['yhat'] * 0.9), 0)), 0),
            'upper': round(float(max(row.get('yhat_upper', row['yhat'] * 1.1), 0)), 0),
        }
        for _, row in fc7.iterrows()
    ]


def _predict_lgbm(pkg: dict) -> list:
    """LightGBM horizon별 모델로 7일 예측"""
    models = pkg['models_horizon']    # {1..7: LGBMRegressor}
    df_feat = pkg['df_feat']
    feat_cols = pkg.get('feat_cols', [])

    if df_feat is None or len(df_feat) == 0:
        return []

    last_row = df_feat.iloc[[-1]][feat_cols].values
    last_date = pd.to_datetime(pkg['last_date'])

    results = []
    for h in range(1, 8):
        m = models.get(h)
        if m is None:
            continue
        price = float(max(m.predict(last_row)[0], 0))
        # 간단 신뢰구간 (±10%)
        results.append({
            'date': str((last_date + pd.Timedelta(days=h)).date()),
            'price': round(price, 0),
            'lower': round(price * 0.90, 0),
            'upper': round(price * 1.10, 0),
        })

    return results


def _detect_anomaly(current_price: float, pkg: dict) -> tuple:
    """Z-score 기반 이상가격 탐지 (최근 30일 기준)"""
    try:
        if pkg.get('model_type') == 'prophet':
            hist = pkg['model'].history[['ds', 'y']].copy()
            prices = hist['y'].values[-30:]
        else:
            df_feat = pkg.get('df_feat')
            if df_feat is None:
                return False, None
            prices = df_feat['price'].values[-30:]

        if len(prices) < 10:
            return False, None

        mean, std = np.mean(prices), np.std(prices)
        if std < 1e-6:
            return False, None

        z = (current_price - mean) / std
        if z > 2.5:
            pct = (current_price - mean) / mean * 100
            return True, f'현재가가 최근 30일 평균 대비 +{pct:.1f}% (Z={z:.1f})'
        elif z < -2.5:
            pct = (mean - current_price) / mean * 100
            return True, f'현재가가 최근 30일 평균 대비 -{pct:.1f}% (Z={z:.1f})'
    except Exception:
        pass

    return False, None


def _buy_timing(current_price: float, prices_7d: list) -> tuple:
    """
    구매 타이밍 5단계
    7일 후 예측가 기준 등락률로 판정
    Returns: (label, score)
    """
    if not prices_7d:
        return '보합', 0.0

    last_price = prices_7d[-1]['price']
    pct = (last_price - current_price) / (current_price + 1e-6) * 100
    score = float(np.clip(pct * 8, -100, 100))

    if pct >= 5.0:
        label = '급등경보'
    elif pct >= 2.0:
        label = '상승'
    elif pct > -2.0:
        label = '보합'
    elif pct > -5.0:
        label = '하락'
    else:
        label = '급락경보'

    return label, round(score, 1)


def predict(item: str) -> dict:
    """단일 품목 7일 예측"""
    pkg = _load_pkg(item)
    if pkg is None:
        raise ValueError(f"'{item}' 모델 없음. list_available_items()로 목록 확인")

    current_price = float(pkg['current_price'])
    mape_val      = float(pkg['mape'])
    unit          = pkg.get('item_unit', '원')
    last_date     = pkg.get('last_date', '')
    model_type    = pkg.get('model_type', 'prophet')

    # 예측
    if model_type == 'prophet':
        prices_7d = _predict_prophet(pkg)
    else:
        prices_7d = _predict_lgbm(pkg)

    # 이상가격 탐지
    anomaly_flag, anomaly_reason = _detect_anomaly(current_price, pkg)

    # 구매 타이밍
    buy_timing, timing_score = _buy_timing(current_price, prices_7d)

    # 신뢰구간 평균
    conf_low  = float(np.mean([p['lower'] for p in prices_7d])) if prices_7d else 0.0
    conf_high = float(np.mean([p['upper'] for p in prices_7d])) if prices_7d else 0.0

    return {
        'item':             item,
        'current_price':    round(current_price, 0),
        'prices_7d':        prices_7d,
        'mape':             mape_val,
        'anomaly_flag':     anomaly_flag,
        'anomaly_reason':   anomaly_reason,
        'buy_timing':       buy_timing,
        'buy_timing_score': timing_score,
        'confidence_low':   round(conf_low, 0),
        'confidence_high':  round(conf_high, 0),
        'unit':             unit,
        'model_type':       model_type,
        'last_date':        last_date,
        'updated_at':       datetime.now().strftime('%Y-%m-%d %H:%M'),
    }


def list_available_items() -> list:
    prophet_pkls = glob.glob(os.path.join(MODEL_DIR, '*_prophet.pkl'))
    lgbm_pkls    = glob.glob(os.path.join(MODEL_DIR, '*_lgbm.pkl'))
    all_items = set(
        [os.path.basename(p).replace('_prophet.pkl', '') for p in prophet_pkls] +
        [os.path.basename(p).replace('_lgbm.pkl', '') for p in lgbm_pkls]
    )
    return sorted(all_items)


if __name__ == '__main__':
    items = list_available_items()
    print(f'모델 목록 ({len(items)}개): {items}')

    if not items:
        print('ai/train_best.py 먼저 실행하세요.')
        exit(1)

    import json
    for test_item in ['배추', '양파', '마늘']:
        if test_item in items:
            print(f'\n=== [{test_item}] 예측 ===')
            r = predict(test_item)
            print(f'  현재가: {r["current_price"]:,.0f} {r["unit"]}')
            print(f'  MAPE: {r["mape"]}%  모델: {r["model_type"]}')
            print(f'  buy_timing: {r["buy_timing"]} (score={r["buy_timing_score"]})')
            print(f'  anomaly: {r["anomaly_flag"]} {r["anomaly_reason"] or ""}')
            print(f'  7일 예측:')
            for d in r['prices_7d']:
                print(f'    {d["date"]}: {d["price"]:,.0f}원 [{d["lower"]:,.0f}~{d["upper"]:,.0f}]')
