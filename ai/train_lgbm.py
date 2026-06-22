"""
LightGBM 기반 7일 가격 예측 모델 (Prophet 보완용)
lag 피처 + 캘린더 피처로 단기 예측 특화
"""
import os, sys, pickle, warnings
import pandas as pd
import numpy as np
import lightgbm as lgb

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ai.preprocess import get_all_price_series

MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

HORIZON = 7  # 7일 예측


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """lag + rolling + calendar 피처 생성"""
    df = df.copy().sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])

    # 캘린더 피처
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter

    # 가격 변화율
    for lag in [1, 3, 7, 14, 21]:
        df[f'lag_{lag}'] = df['price'].shift(lag)
        df[f'pct_change_{lag}d'] = df['price'].pct_change(lag)

    # 롤링 통계
    for window in [3, 7, 14, 30]:
        df[f'rolling_mean_{window}'] = df['price'].shift(1).rolling(window).mean()
        df[f'rolling_std_{window}']  = df['price'].shift(1).rolling(window).std()
        df[f'rolling_min_{window}']  = df['price'].shift(1).rolling(window).min()
        df[f'rolling_max_{window}']  = df['price'].shift(1).rolling(window).max()

    # 단기 모멘텀
    df['mom_7d'] = df['price'].shift(1) / (df['price'].shift(8) + 1e-6)
    df['mom_14d'] = df['price'].shift(1) / (df['price'].shift(15) + 1e-6)

    return df


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true > 10  # 너무 낮은 가격 제외
    if mask.sum() == 0:
        return 99.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


FEAT_COLS = (
    ['day_of_week', 'day_of_month', 'month', 'week_of_year', 'quarter']
    + [f'lag_{l}' for l in [1, 3, 7, 14, 21]]
    + [f'pct_change_{l}d' for l in [1, 3, 7, 14, 21]]
    + [f'rolling_mean_{w}' for w in [3, 7, 14, 30]]
    + [f'rolling_std_{w}' for w in [3, 7, 14, 30]]
    + [f'rolling_min_{w}' for w in [3, 7, 14, 30]]
    + [f'rolling_max_{w}' for w in [3, 7, 14, 30]]
    + ['mom_7d', 'mom_14d']
)


def train_lgbm_item(item: str, df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    7-day direct multi-step LightGBM 학습
    각 horizon(1~7)에 대해 별도 모델 학습
    """
    df_feat = make_features(df)
    df_feat = df_feat.dropna()

    n = len(df_feat)
    if n < 50:
        raise ValueError(f'피처 생성 후 데이터 부족: {n}행')

    # walk-forward MAPE 평가 (cutoffs: 60%~80%)
    start_idx = int(n * 0.60)
    end_idx   = int(n * 0.80)
    cutoffs = list(range(start_idx, end_idx, 7))[:5]
    if not cutoffs:
        cutoffs = [int(n * 0.7)]

    all_errors = []
    models_horizon = {}  # h → lgb model

    # 각 horizon별 모델 학습
    for h in range(1, HORIZON + 1):
        df_h = df_feat.copy()
        df_h['target'] = df_h['price'].shift(-h)
        df_h = df_h.dropna(subset=['target'])

        X = df_h[FEAT_COLS].values
        y = df_h['target'].values

        cutoff = int(len(X) * 0.80)
        X_train, X_test = X[:cutoff], X[cutoff:]
        y_train, y_test = y[:cutoff], y[cutoff:]

        if len(X_train) < 20:
            continue

        params = dict(
            objective='regression',
            metric='mae',
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbose=-1,
        )
        model_h = lgb.LGBMRegressor(**params)
        model_h.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
        )
        models_horizon[h] = model_h

        if h == HORIZON and len(X_test) > 0:
            pred = model_h.predict(X_test)
            all_errors.extend(
                (np.abs((y_test - pred) / (y_test + 1e-6)) * 100).tolist()
            )

    mape_val = float(np.mean(all_errors)) if all_errors else 99.0

    current_price = float(df.iloc[-1]['price'])
    last_date = str(pd.to_datetime(df.iloc[-1]['date']).date())

    result = {
        'model_type': 'lgbm',
        'models_horizon': models_horizon,  # h=1..7 모델
        'df_feat': df_feat,               # 마지막 행이 predict에 필요
        'mape': round(mape_val, 2),
        'last_date': last_date,
        'current_price': current_price,
        'n_train': n,
    }

    if verbose:
        status = '✓' if mape_val <= 15 else '△'
        print(f'  {status} [{item}] LGB MAPE={mape_val:.1f}%, N={n}, 현재가={current_price:.0f}')

    return result


def train_lgbm_all(items_to_train=None) -> dict:
    """지정 품목(또는 전체) LightGBM 학습"""
    series_dict = get_all_price_series()
    if items_to_train:
        series_dict = {k: v for k, v in series_dict.items() if k in items_to_train}

    results = {}
    for item, df in series_dict.items():
        try:
            res = train_lgbm_item(item, df)
            results[item] = res
            # 저장
            save_path = os.path.join(MODEL_DIR, f'{item}_lgbm.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(res, f)
        except Exception as e:
            print(f'  [{item}] LGB 실패: {e}')

    return results


if __name__ == '__main__':
    print('=== LightGBM 학습 ===')
    results = train_lgbm_all()
    print('\nLGB MAPE 요약:')
    for item, r in sorted(results.items(), key=lambda x: x[1]['mape']):
        s = '✓' if r['mape'] <= 15 else '△'
        print(f'  {s} {item}: {r["mape"]:.1f}%')
