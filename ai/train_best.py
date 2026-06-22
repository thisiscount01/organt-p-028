"""
Prophet/LightGBM 중 품목별 최적 모델 선택 → model/ 저장
+ 배추 등 고변동성 품목: additive 모드 재시도
"""
import os, sys, pickle, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ai.preprocess import get_all_price_series, TARGET_ITEMS

MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true > 10
    if mask.sum() == 0:
        return 99.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ─── Prophet ──────────────────────────────────────────────────────────────────
from prophet import Prophet

KR_HOLIDAYS = pd.DataFrame({
    'holiday': '공휴일',
    'ds': pd.to_datetime([
        '2024-01-01', '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12',
        '2024-03-01', '2024-04-10', '2024-05-05', '2024-05-06', '2024-05-15',
        '2024-06-06', '2024-08-15', '2024-09-16', '2024-09-17', '2024-09-18',
        '2024-10-03', '2024-10-09', '2024-12-25',
    ]),
    'lower_window': -1, 'upper_window': 1,
})


def cv_mape_prophet(full_df: pd.DataFrame, mode='multiplicative') -> float:
    n = len(full_df)
    if n < 60:
        return 99.0

    start_idx = int(n * 0.60)
    end_idx   = int(n * 0.80)
    cutoffs = list(range(start_idx, end_idx, 7))[:5]
    if not cutoffs:
        cutoffs = [int(n * 0.7)]

    all_errors = []
    for ci in cutoffs:
        train_df = full_df.iloc[:ci].copy()
        act7 = full_df.iloc[ci:ci + 7].copy()
        if len(act7) < 5:
            continue
        try:
            m = Prophet(
                changepoint_prior_scale=0.2,
                seasonality_prior_scale=10.0,
                seasonality_mode=mode,
                weekly_seasonality=True,
                yearly_seasonality=(len(train_df) >= 200),
                daily_seasonality=False,
                holidays=KR_HOLIDAYS,
                interval_width=0.80,
            )
            m.add_seasonality(name='monthly', period=30.5, fourier_order=3)
            m.fit(train_df)
            fut = m.make_future_dataframe(periods=7, freq='D')
            fc = m.predict(fut)
            fc7 = fc.tail(7)[['ds', 'yhat']]
            merged = act7.merge(fc7, on='ds', how='inner')
            if len(merged) >= 3:
                all_errors.extend(
                    (np.abs((merged['y'].values - merged['yhat'].values)
                             / np.maximum(merged['y'].values, 1)) * 100).tolist()
                )
        except Exception:
            continue

    return float(np.mean(all_errors)) if all_errors else 99.0


def fit_prophet_final(full_df: pd.DataFrame, mode='multiplicative') -> Prophet:
    m = Prophet(
        changepoint_prior_scale=0.2,
        seasonality_prior_scale=10.0,
        seasonality_mode=mode,
        weekly_seasonality=True,
        yearly_seasonality=(len(full_df) >= 200),
        daily_seasonality=False,
        holidays=KR_HOLIDAYS,
        interval_width=0.80,
    )
    m.add_seasonality(name='monthly', period=30.5, fourier_order=3)
    m.fit(full_df)
    return m


# ─── LightGBM ─────────────────────────────────────────────────────────────────
import lightgbm as lgb

FEAT_COLS = (
    ['day_of_week', 'day_of_month', 'month', 'week_of_year', 'quarter']
    + [f'lag_{l}' for l in [1, 3, 7, 14, 21]]
    + [f'pct_change_{l}d' for l in [1, 3, 7]]
    + [f'rolling_mean_{w}' for w in [3, 7, 14, 30]]
    + [f'rolling_std_{w}' for w in [7, 14]]
    + [f'rolling_min_{w}' for w in [7, 14]]
    + [f'rolling_max_{w}' for w in [7, 14]]
)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    for lag in [1, 3, 7, 14, 21]:
        df[f'lag_{lag}'] = df['price'].shift(lag)
        df[f'pct_change_{lag}d'] = df['price'].pct_change(lag)
    for w in [3, 7, 14, 30]:
        df[f'rolling_mean_{w}'] = df['price'].shift(1).rolling(w).mean()
    for w in [7, 14]:
        df[f'rolling_std_{w}']  = df['price'].shift(1).rolling(w).std()
        df[f'rolling_min_{w}']  = df['price'].shift(1).rolling(w).min()
        df[f'rolling_max_{w}']  = df['price'].shift(1).rolling(w).max()
    return df


def cv_mape_lgbm(df: pd.DataFrame) -> float:
    df_feat = make_features(df).dropna()
    n = len(df_feat)
    if n < 50:
        return 99.0

    cutoff = int(n * 0.80)
    X = df_feat[FEAT_COLS].values
    y_7 = df_feat['price'].shift(-7).values

    df_feat_h7 = df_feat.copy()
    df_feat_h7['target'] = df_feat_h7['price'].shift(-7)
    df_feat_h7 = df_feat_h7.dropna(subset=['target'])

    n2 = len(df_feat_h7)
    cutoff2 = int(n2 * 0.80)
    X2 = df_feat_h7[FEAT_COLS].values
    y2 = df_feat_h7['target'].values

    if cutoff2 < 20:
        return 99.0

    X_tr, X_te = X2[:cutoff2], X2[cutoff2:]
    y_tr, y_te = y2[:cutoff2], y2[cutoff2:]

    if len(X_te) < 3:
        return 99.0

    params = dict(
        objective='regression', metric='mae', n_estimators=300,
        learning_rate=0.05, num_leaves=15, min_child_samples=5,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, verbose=-1,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
    )
    pred = model.predict(X_te)
    return mape(y_te, pred)


def fit_lgbm_final(df: pd.DataFrame) -> dict:
    """각 horizon 1~7에 대해 별도 모델 학습"""
    df_feat = make_features(df).dropna()
    models = {}
    for h in range(1, 8):
        df_h = df_feat.copy()
        df_h['target'] = df_h['price'].shift(-h)
        df_h = df_h.dropna(subset=['target'])
        X, y = df_h[FEAT_COLS].values, df_h['target'].values
        cutoff = int(len(X) * 0.85)
        X_tr, X_te = X[:cutoff], X[cutoff:]
        y_tr, y_te = y[:cutoff], y[cutoff:]
        params = dict(
            objective='regression', metric='mae', n_estimators=500,
            learning_rate=0.03, num_leaves=15, min_child_samples=5,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, verbose=-1,
        )
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)] if len(X_te) > 5 else None,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )
        models[h] = m
    return models


# ─── 통합 학습 ────────────────────────────────────────────────────────────────
def train_best_all():
    print('=== 최적 모델 선택 학습 ===\n')
    series_dict = get_all_price_series()

    final_mapes = {}
    all_success = 0

    for item, df in series_dict.items():
        print(f'\n[{item}] Prophet/LGB 비교...')
        prophet_df = df.rename(columns={'date': 'ds', 'price': 'y'})
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

        # Prophet: multiplicative + additive 둘 다 시도
        mape_p_mul = cv_mape_prophet(prophet_df, 'multiplicative')
        mape_p_add = cv_mape_prophet(prophet_df, 'additive')
        mape_prophet = min(mape_p_mul, mape_p_add)
        best_mode = 'multiplicative' if mape_p_mul <= mape_p_add else 'additive'
        print(f'  Prophet (mul={mape_p_mul:.1f}%, add={mape_p_add:.1f}%)')

        mape_lgb = cv_mape_lgbm(df)
        print(f'  LightGBM {mape_lgb:.1f}%')

        if mape_prophet <= mape_lgb:
            chosen = 'prophet'
            best_mape = mape_prophet
        else:
            chosen = 'lgbm'
            best_mape = mape_lgb

        print(f'  => 선택: {chosen.upper()} MAPE={best_mape:.1f}%')

        # 최종 모델 학습
        try:
            if chosen == 'prophet':
                final_model = fit_prophet_final(prophet_df, best_mode)
                pkg = {
                    'model_type': 'prophet',
                    'model': final_model,
                    'seasonality_mode': best_mode,
                    'mape': round(best_mape, 2),
                    'last_date': str(df.iloc[-1]['date']),
                    'current_price': float(df.iloc[-1]['price']),
                    'item_unit': TARGET_ITEMS.get(item, {}).get('unit', '원'),
                    'n_train': len(df),
                }
            else:  # lgbm
                lgbm_models = fit_lgbm_final(df)
                df_feat = make_features(df).dropna()
                pkg = {
                    'model_type': 'lgbm',
                    'models_horizon': lgbm_models,
                    'df_feat': df_feat,
                    'feat_cols': FEAT_COLS,
                    'mape': round(best_mape, 2),
                    'last_date': str(df.iloc[-1]['date']),
                    'current_price': float(df.iloc[-1]['price']),
                    'item_unit': TARGET_ITEMS.get(item, {}).get('unit', '원'),
                    'n_train': len(df),
                }

            # 저장 (기존 덮어쓰기)
            save_path = os.path.join(MODEL_DIR, f'{item}_prophet.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(pkg, f)
            print(f'  저장: {save_path}')

            final_mapes[item] = best_mape
            if best_mape <= 15:
                all_success += 1

        except Exception as e:
            print(f'  최종학습 실패: {e}')

    print('\n=== 최종 결과 ===')
    print(f'총 {len(final_mapes)}개 품목, {all_success}개 ≤15%\n')
    for item, m in sorted(final_mapes.items(), key=lambda x: x[1]):
        s = '✓' if m <= 15 else '△'
        print(f'  {s} {item}: {m:.1f}%')

    avg = np.mean(list(final_mapes.values()))
    print(f'\n  평균 MAPE: {avg:.1f}% (목표 ≤15%)')
    return final_mapes


if __name__ == '__main__':
    train_best_all()
