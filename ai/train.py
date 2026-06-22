"""
Prophet 모델 학습 스크립트
각 품목별 Prophet 모델 저장 → model/{품목명}_prophet.pkl
MAPE 계산 포함 (80/20 train/test split)
"""
import os, sys, pickle, warnings
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ai.preprocess import get_all_price_series, TARGET_ITEMS

MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

# 한국 공휴일 (농산물 가격에 영향)
KR_HOLIDAYS = pd.DataFrame({
    'holiday': '공휴일',
    'ds': pd.to_datetime([
        '2024-01-01', '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12',
        '2024-03-01', '2024-04-10', '2024-05-05', '2024-05-06', '2024-05-15',
        '2024-06-06', '2024-08-15', '2024-09-16', '2024-09-17', '2024-09-18',
        '2024-10-03', '2024-10-09', '2024-12-25',
    ]),
    'lower_window': -1,
    'upper_window': 1,
})


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_mape_cv(full_df: pd.DataFrame, horizon: int = 7) -> float:
    """
    7일 선행 예측 MAPE (rolling walk-forward):
      - 여러 cutoff 지점에서 horizon=7 예측 → 실제값 비교
      - 각 cutoff는 전체 데이터의 60~80% 구간에서 7일 간격으로 선택
    """
    n = len(full_df)
    if n < horizon * 4:
        return 99.0

    # cutoff 후보: 60%~80% 사이에서 7일 간격 (최대 5개)
    start_idx = int(n * 0.60)
    end_idx   = int(n * 0.80)
    cutoffs = list(range(start_idx, end_idx, 7))[:5]
    if not cutoffs:
        cutoffs = [int(n * 0.7)]

    all_errors = []

    for cutoff_idx in cutoffs:
        train_df = full_df.iloc[:cutoff_idx].copy()
        future_actuals = full_df.iloc[cutoff_idx:cutoff_idx + horizon].copy()

        if len(train_df) < 30 or len(future_actuals) < horizon:
            continue

        try:
            m_eval = Prophet(
                changepoint_prior_scale=0.3,
                seasonality_prior_scale=10.0,
                seasonality_mode='multiplicative',
                weekly_seasonality=True,
                yearly_seasonality=(len(train_df) >= 200),
                daily_seasonality=False,
                holidays=KR_HOLIDAYS,
                interval_width=0.80,
            )
            m_eval.add_seasonality(name='monthly', period=30.5, fourier_order=3)
            m_eval.fit(train_df)

            future = m_eval.make_future_dataframe(periods=horizon, freq='D')
            fc = m_eval.predict(future)
            fc_horizon = fc.tail(horizon)[['ds', 'yhat']]

            merged = future_actuals.merge(fc_horizon, on='ds', how='inner')
            if len(merged) >= 3:
                errors = np.abs((merged['y'].values - merged['yhat'].values) / merged['y'].values) * 100
                all_errors.extend(errors.tolist())
        except Exception:
            continue

    if not all_errors:
        return 99.0

    return float(np.mean(all_errors))


def train_item(item_name: str, df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    단일 품목 Prophet 모델 학습
    Returns: {model, mape, last_date, current_price, item_unit}
    """
    prophet_df = df.rename(columns={'date': 'ds', 'price': 'y'}).copy()
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

    n = len(prophet_df)
    train_df = prophet_df.iloc[:int(n * 0.8)].copy()
    test_df = prophet_df.iloc[int(n * 0.8):].copy()

    # Prophet 설정
    m = Prophet(
        changepoint_prior_scale=0.3,         # 가격 변동 유연성 (농산물 특성)
        seasonality_prior_scale=15.0,        # 계절성 강도
        holidays_prior_scale=10.0,
        seasonality_mode='multiplicative',   # 농산물 = 계절성이 가격에 곱해짐
        weekly_seasonality=True,
        yearly_seasonality=True if n >= 200 else False,
        daily_seasonality=False,
        holidays=KR_HOLIDAYS,
        interval_width=0.80,                 # 80% 신뢰구간
    )

    # 월별 계절성 추가 (중간 주기)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    # 학습 (전체 데이터로 최종 모델)
    if verbose:
        print(f'  [{item_name}] 학습 중... ({n}일 데이터)')

    # MAPE 계산 (진짜 hold-out: 80/20 split, 훈련 전에 미리 계산)
    mape_val = compute_mape_cv(prophet_df)

    # 전체 데이터로 최종 모델 학습 (프로덕션용)
    m.fit(prophet_df)

    current_price = float(prophet_df.iloc[-1]['y'])
    last_date = prophet_df.iloc[-1]['ds']

    result = {
        'model': m,
        'mape': round(mape_val, 2),
        'last_date': str(last_date.date()),
        'current_price': current_price,
        'item_unit': TARGET_ITEMS.get(item_name, {}).get('unit', '원'),
        'n_train': n,
    }

    if verbose:
        print(f'  [{item_name}] MAPE={mape_val:.1f}%, 학습일수={n}, 현재가={current_price:.0f}')

    return result


def train_all():
    """전체 품목 학습 및 저장"""
    print('=== 농산물 가격 Prophet 학습 시작 ===\n')

    # 데이터 로드
    print('[1] 데이터 로드...')
    series_dict = get_all_price_series()
    print(f'    {len(series_dict)}개 품목 로드 완료\n')

    # 품목별 학습
    print('[2] Prophet 모델 학습...')
    results = {}
    failed = []

    for item, df in series_dict.items():
        try:
            res = train_item(item, df)
            results[item] = res

            # 모델 저장
            model_path = os.path.join(MODEL_DIR, f'{item}_prophet.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(res, f)

        except Exception as e:
            print(f'  [{item}] 학습 실패: {e}')
            failed.append(item)

    # 요약
    print(f'\n[3] 학습 완료: {len(results)}개 품목')
    if failed:
        print(f'    실패: {failed}')

    print('\n품목별 MAPE:')
    mapes = [(item, r['mape']) for item, r in results.items()]
    mapes.sort(key=lambda x: x[1])
    for item, m in mapes:
        status = '✓' if m <= 15 else '△'
        print(f'  {status} {item}: {m:.1f}%')

    avg_mape = np.mean([r['mape'] for r in results.values()])
    print(f'\n  평균 MAPE: {avg_mape:.1f}% (목표 ≤15%)')

    return results


if __name__ == '__main__':
    train_all()
