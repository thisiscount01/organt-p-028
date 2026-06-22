"""
농산물 가격 데이터 전처리 모듈
소스:
  1. kamis_vegetables_daily.csv  — 도매일일가격 6품목, 2024 전체
  2. kamis_wholesale_market.csv  — 공판장 50품목, 2024-01~08
"""
import os, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# 분석 대상 품목 (10종+)
TARGET_ITEMS = {
    '배추':   {'source': 'daily',  'col': '도매일일가격', 'unit': '원/10kg'},
    '무':     {'source': 'daily',  'col': '도매일일가격', 'unit': '원/20kg'},
    '양파':   {'source': 'daily',  'col': '도매일일가격', 'unit': '원/20kg'},
    '마늘':   {'source': 'daily',  'col': '도매일일가격', 'unit': '원/10kg'},
    '대파':   {'source': 'daily',  'col': '도매일일가격', 'unit': '원/1kg'},
    '건고추': {'source': 'daily',  'col': '도매일일가격', 'unit': '원/600g'},
    '감자':   {'source': 'market', 'col': '평균가격',     'unit': '원/20kg'},
    '고구마': {'source': 'market', 'col': '평균가격',     'unit': '원/10kg'},
    '토마토': {'source': 'market', 'col': '평균가격',     'unit': '원/5kg'},
    '사과':   {'source': 'market', 'col': '평균가격',     'unit': '원/10kg'},
    '당근':   {'source': 'market', 'col': '평균가격',     'unit': '원/20kg'},
    '양배추': {'source': 'market', 'col': '평균가격',     'unit': '원/8kg'},
    '상추':   {'source': 'market', 'col': '평균가격',     'unit': '원/4kg'},
    '시금치': {'source': 'market', 'col': '평균가격',     'unit': '원/4kg'},
}

DATA_DIR = 'data'


def load_daily_kamis() -> pd.DataFrame:
    """kamis_vegetables_daily.csv 로드 → long 형식 (date, item, price)"""
    path = os.path.join(DATA_DIR, 'kamis_vegetables_daily.csv')
    df = pd.read_csv(path, encoding='cp949', low_memory=False)

    # 컬럼 확인
    df['조회일자'] = pd.to_datetime(df['조회일자'], errors='coerce')
    df = df.dropna(subset=['조회일자'])

    # 도매일일가격 숫자 변환
    df['도매일일가격'] = pd.to_numeric(df['도매일일가격'], errors='coerce')

    records = []
    for item in ['배추', '무', '양파', '마늘', '대파', '건고추']:
        sub = df[df['품목명'] == item][['조회일자', '도매일일가격']].copy()
        sub = sub.dropna(subset=['도매일일가격'])
        sub = sub[sub['도매일일가격'] > 0]
        sub = sub.rename(columns={'조회일자': 'date', '도매일일가격': 'price'})
        sub['item'] = item
        # 하루 중복 시 평균
        sub = sub.groupby('date')['price'].mean().reset_index()
        sub['item'] = item
        records.append(sub)

    daily_df = pd.concat(records, ignore_index=True)
    daily_df = daily_df.sort_values(['item', 'date']).reset_index(drop=True)
    return daily_df


def load_wholesale_kamis() -> pd.DataFrame:
    """kamis_wholesale_market.csv 로드 → long 형식"""
    path = os.path.join(DATA_DIR, 'kamis_wholesale_market.csv')
    df = pd.read_csv(path, encoding='utf-8', low_memory=False)

    df['가격날짜'] = pd.to_datetime(df['가격날짜'], errors='coerce')
    df['평균가격'] = pd.to_numeric(df['평균가격'], errors='coerce')
    df = df.dropna(subset=['가격날짜', '평균가격'])
    df = df[df['평균가격'] > 0]

    # 상 등급 기준 (없으면 전체 평균)
    market_items = [k for k, v in TARGET_ITEMS.items() if v['source'] == 'market']

    records = []
    for item in market_items:
        sub = df[df['품목명'] == item].copy()
        if len(sub) == 0:
            continue
        # 상 등급 우선
        sub_sang = sub[sub['등급'] == '상']
        if len(sub_sang) > 10:
            sub = sub_sang

        daily = sub.groupby('가격날짜')['평균가격'].mean().reset_index()
        daily.columns = ['date', 'price']
        daily['item'] = item
        records.append(daily)

    if not records:
        return pd.DataFrame(columns=['date', 'price', 'item'])

    result = pd.concat(records, ignore_index=True)
    result = result.sort_values(['item', 'date']).reset_index(drop=True)
    return result


def get_all_price_series() -> dict:
    """
    품목명 → pd.DataFrame(date, price) 딕셔너리 반환
    결측일은 forward-fill (최대 3일)
    """
    df_daily = load_daily_kamis()
    df_market = load_wholesale_kamis()

    combined = pd.concat([df_daily, df_market], ignore_index=True)

    result = {}
    for item in TARGET_ITEMS:
        sub = combined[combined['item'] == item][['date', 'price']].copy()
        if len(sub) < 30:
            print(f'  [{item}] 데이터 부족 ({len(sub)}행) — 건너뜀')
            continue

        sub = sub.sort_values('date').set_index('date')
        # 날짜 연속화 (빈 날짜 채우기)
        idx = pd.date_range(sub.index.min(), sub.index.max(), freq='D')
        sub = sub.reindex(idx)
        # forward fill 3일, backward fill 1일
        sub['price'] = sub['price'].ffill(limit=3)
        sub['price'] = sub['price'].bfill(limit=1)
        sub = sub.dropna()
        sub = sub.reset_index().rename(columns={'index': 'date'})

        # 이상치 제거 (IQR 3× 기준)
        q1, q3 = sub['price'].quantile([0.25, 0.75])
        iqr = q3 - q1
        sub = sub[(sub['price'] >= q1 - 3 * iqr) & (sub['price'] <= q3 + 3 * iqr)]
        sub = sub.reset_index(drop=True)

        result[item] = sub
        print(f'  [{item}] {len(sub)}일 데이터 ({sub["date"].min().date()} ~ {sub["date"].max().date()})')

    return result


if __name__ == '__main__':
    print('=== 전처리 테스트 ===')
    series = get_all_price_series()
    print(f'\n총 {len(series)}개 품목 로드 완료')
    for item, df in series.items():
        print(f'  {item}: {len(df)}행, 가격범위 {df["price"].min():.0f}~{df["price"].max():.0f}')
