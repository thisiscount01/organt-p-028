#!/usr/bin/env python3
"""
KAMIS 데이터 전처리 + 추가 다운로드 스크립트
1) kamis_vegetables_daily.csv → 품목별 {task_code}.csv (name 매핑)
2) KAMIS API로 누락 품목 다운로드 (2022-2024)
3) 모든 파일은 data/{code}.csv 형태로 저장 (컬럼: 날짜,가격)
"""

import csv
import json
import os
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ─── Task 코드 ↔ 품목명 ↔ KAMIS API 파라미터 매핑 ─────────────────────────────
# (task_code, item_name, api_category, api_item_code, api_kind)
ITEM_MAP = [
    ("112", "배추",  "02", "211", "01"),
    ("111", "무",    "02", "212", "01"),
    ("214", "양파",  "02", "214", "01"),
    ("215", "마늘",  "02", "215", "01"),
    ("218", "대파",  "02", "216", "01"),
    ("117", "감자",  "02", "232", "01"),
    ("216", "고구마","02", "235", "01"),
    ("411", "사과",  "04", "411", "01"),
    ("412", "배",    "04", "412", "01"),
    ("131", "쌀",    "01", "111", "01"),
    ("114", "상추",  "02", "227", "01"),
    ("118", "오이",  "02", "221", "01"),
    ("121", "호박",  "02", "220", "01"),
    ("213", "당근",  "02", "231", "01"),
    ("217", "생강",  "02", "252", "01"),
]

COMBINED_CSV = DATA_DIR / "kamis_vegetables_daily.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FarmPriceBot/1.0)"}

NAME_TO_CODE = {name: code for code, name, *_ in ITEM_MAP}


# ─── 1. 통합 CSV → 품목별 파일 분리 ───────────────────────────────────────────
def split_combined_csv():
    if not COMBINED_CSV.exists():
        print("[준비] 통합 CSV 없음 → 건너뜀")
        return set()

    saved = {}
    try:
        import pandas as pd
        df = pd.read_csv(COMBINED_CSV, encoding="euc-kr")
    except Exception:
        try:
            import pandas as pd
            df = pd.read_csv(COMBINED_CSV, encoding="utf-8-sig")
        except Exception as e:
            print(f"[오류] CSV 읽기 실패: {e}")
            return set()

    name_col = None
    for c in df.columns:
        if "품목명" in c or "name" in c.lower():
            name_col = c
            break

    date_col = None
    for c in df.columns:
        if "조회일자" in c or "날짜" in c or "일자" in c:
            date_col = c
            break

    price_col = None
    for c in df.columns:
        if "소매일일가격" in c:
            price_col = c
            break
    if price_col is None:
        for c in df.columns:
            if "도매일일가격" in c:
                price_col = c
                break

    if not all([name_col, date_col, price_col]):
        print(f"[오류] 필수 컬럼 못 찾음 — name={name_col}, date={date_col}, price={price_col}")
        print(f"  columns: {df.columns.tolist()}")
        return set()

    print(f"[분리] 통합CSV: name={name_col}, date={date_col}, price={price_col}")

    for name, task_code in NAME_TO_CODE.items():
        sub = df[df[name_col] == name].copy()
        if sub.empty:
            continue
        sub = sub[[date_col, price_col]].copy()
        sub.columns = ["날짜", "가격"]
        import numpy as np
        sub["가격"] = pd.to_numeric(
            sub["가격"].astype(str).str.replace(",", "").str.strip(),
            errors="coerce"
        )
        sub = sub.dropna()
        sub = sub[sub["가격"] > 0]
        sub = sub.sort_values("날짜").drop_duplicates("날짜")

        out_path = DATA_DIR / f"{task_code}.csv"
        sub.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  {task_code}({name}): {len(sub)}건 → {out_path}")
        saved[task_code] = name

    return set(saved.keys())


# ─── 2. KAMIS API 다운로드 (누락 품목) ────────────────────────────────────────
def fetch_kamis_api(task_code: str, name: str,
                    cat: str, item: str, kind: str,
                    start: str = "2020-01-01", end: str = "2025-12-31") -> list:
    url = (
        "https://www.kamis.or.kr/service/price/xml.do"
        f"?action=periodProductList"
        f"&p_startday={start}"
        f"&p_endday={end}"
        f"&p_itemcategorycode={cat}"
        f"&p_itemcode={item}"
        f"&p_kindcode={kind}"
        f"&p_productrankcode=04"
        f"&p_countrycode=1101"
        "&p_convert_kg_yn=N"
        "&p_cert_key=sample"
        "&p_cert_id=sample"
        "&p_returntype=json"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
        items = data.get("data", {}).get("item", [])
        err = data.get("data", {}).get("error_code", "?")
        if err != "000":
            return []
        records = []
        for row in items:
            if row.get("countyname") != "평균":
                continue
            yyyy = str(row.get("yyyy", "")).strip()
            regday = str(row.get("regday", "")).strip()
            price_s = str(row.get("price", "0")).replace(",", "").strip()
            if not yyyy or "/" not in regday or not price_s.isdigit():
                continue
            m, d = regday.split("/")
            date = f"{yyyy}-{m.zfill(2)}-{d.zfill(2)}"
            price = float(price_s)
            if price > 0:
                records.append((date, price))
        return sorted(set(records), key=lambda x: x[0])
    except Exception as e:
        print(f"  API 오류: {e}")
        return []


def download_missing(already_saved: set):
    for task_code, name, cat, item, kind in ITEM_MAP:
        out_path = DATA_DIR / f"{task_code}.csv"
        if task_code in already_saved and out_path.exists():
            size = out_path.stat().st_size
            if size > 500:
                print(f"[스킵] {task_code}({name}): 이미 존재 ({size}B)")
                continue

        print(f"[API] {task_code}({name}) cat={cat} item={item} kind={kind}")
        records = fetch_kamis_api(task_code, name, cat, item, kind)
        if len(records) >= 30:
            with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                w.writerow(["날짜", "가격"])
                for date, price in records:
                    w.writerow([date, f"{price:.0f}"])
            print(f"  → 저장 {out_path} ({len(records)}건)")
        else:
            print(f"  → 데이터 부족 ({len(records)}건) — 건너뜀")
        time.sleep(0.5)


# ─── 3. 결과 요약 ─────────────────────────────────────────────────────────────
def summary():
    print("\n=== 데이터 준비 완료 ===")
    total_ok = 0
    for task_code, name, *_ in ITEM_MAP:
        path = DATA_DIR / f"{task_code}.csv"
        if path.exists():
            size = path.stat().st_size
            try:
                with open(path, encoding="utf-8-sig") as f:
                    rows = sum(1 for _ in f) - 1
                status = "✓" if rows >= 100 else "⚠"
                if rows >= 100:
                    total_ok += 1
                print(f"  {status} {task_code}({name}): {rows}행")
            except Exception:
                print(f"  ? {task_code}({name}): 읽기 오류")
        else:
            print(f"  ✗ {task_code}({name}): 파일 없음")
    print(f"\n유효 품목(100행+): {total_ok}/{len(ITEM_MAP)}")


if __name__ == "__main__":
    print("=== KAMIS 데이터 준비 ===\n")
    print("[1단계] 통합 CSV 분리")
    saved = split_combined_csv()
    print(f"\n[2단계] 누락 품목 API 다운로드")
    download_missing(saved)
    summary()
