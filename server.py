"""
KAMIS 농산물 가격 AI 예측 서비스 — FastAPI 백엔드
포트 : 8000
실행 : uvicorn server:app --port 8000 --reload

API 계약 (프론트엔드 호환):
  GET /api/items      → [{code, name, category, has_data}, ...]
  GET /api/predict    → {item, current_price, prices_7d, buy_timing, ...}
  GET /api/predict/{item_code}?days=7  → 원시 예측 (하위호환)
  GET /api/latest/{item_code}          → 최근 N일 실제가
  GET /api/anomaly                     → 이상탐지 목록
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ─────────────────────────────────────────────────────────────────────────────
# 품목 코드 상수
# ─────────────────────────────────────────────────────────────────────────────
ITEMS: dict[str, str] = {
    "112": "배추",
    "214": "양파",
    "215": "마늘",
    "218": "대파",
    "111": "무",
    "216": "고구마",
    "117": "감자",
    "411": "사과",
    "412": "배",
    "131": "쌀",
    "114": "상추",
    "118": "오이",
    "121": "호박",
    "213": "당근",
    "217": "생강",
    # AI 모델 보유 품목 추가
    "211": "건고추",
    "115": "시금치",
    "113": "양배추",
    "225": "토마토",
}

CATEGORY: dict[str, str] = {
    "112": "채소류", "214": "채소류", "215": "채소류",
    "218": "채소류", "111": "채소류", "216": "채소류",
    "117": "채소류", "114": "채소류", "118": "채소류",
    "121": "채소류", "213": "채소류", "217": "채소류",
    "411": "과일류", "412": "과일류",
    "131": "곡류",
    # AI 모델 보유 품목
    "211": "채소류", "115": "채소류", "113": "채소류", "225": "채소류",
}

# grade → 구매 타이밍 점수 (높을수록 지금 사기 좋음)
GRADE_SCORE: dict[str, int] = {
    "급등경보": 10,
    "상승":     30,
    "보합":     50,
    "하락":     70,
    "급락경보": 90,
}

VALID_GRADES = set(GRADE_SCORE.keys())

DATA_DIR   = Path("data")
PUBLIC_DIR = Path("public")
DATA_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 로깅
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kamis_api")

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI 앱
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="KAMIS 농산물 가격 AI 예측 API",
    description=(
        "aT KAMIS 공공데이터 기반 농산물 가격 예측·이상탐지·구매 타이밍 추천.\n\n"
        "**5단계 등급** : 급등경보 / 상승 / 보합 / 하락 / 급락경보"
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# predictor.py 동적 로드 + 결과 캐시 (TTL = PREDICT_CACHE_TTL 초)
# ─────────────────────────────────────────────────────────────────────────────
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_TTL: float = float(os.getenv("PREDICT_CACHE_TTL", "300"))


def _get_predict_fn():
    try:
        mod = importlib.import_module("predictor")
        importlib.reload(mod)
        return mod.predict
    except (ImportError, AttributeError) as exc:
        logger.warning("predictor.py 로드 실패: %s", exc)
        return None


def _call_predict(item_code: str, days: int) -> dict[str, Any]:
    """캐시 적용 예측 호출. ai/predict.py Prophet/LGB 우선, ValueError → ETS 스텁 폴백."""
    key = f"{item_code}:{days}"
    now = time.monotonic()
    if key in _CACHE:
        ts, result = _CACHE[key]
        if now - ts < _CACHE_TTL:
            return result

    item_name = ITEMS.get(item_code, "")
    result: dict[str, Any] | None = None

    # ── 1) AI 모델(Prophet/LightGBM) 우선 시도 ──────────────────────────────
    try:
        from ai.predict import predict as ai_predict  # noqa: PLC0415
        ai_res = ai_predict(item_name)

        # ai/predict.py buy_timing 이 VALID_GRADES 에 없으면 보합
        grade = ai_res.get("buy_timing", "보합")
        if grade not in VALID_GRADES:
            grade = "보합"

        # trend_pct 를 current_price → 7일 마지막 예측가 기준으로 직접 산출
        cur = float(ai_res.get("current_price") or 0)
        p7d = ai_res.get("prices_7d") or []
        if cur > 0 and p7d:
            last_fc = float(p7d[-1]["price"])
            trend_pct = round((last_fc - cur) / cur * 100, 2)
        else:
            trend_pct = 0.0

        result = {
            "source":           "ai",
            "predicted_prices": [float(p["price"]) for p in p7d],
            "grade":            grade,
            "trend_pct":        trend_pct,
            "anomaly_flag":     bool(ai_res.get("anomaly_flag", False)),
            # AI 전용 추가 필드 — predict_by_name 에서 바로 사용
            "current_price":    ai_res.get("current_price"),
            "prices_7d":        p7d,          # {date, price, lower, upper} 포함
            "mape":             ai_res.get("mape"),
            "buy_timing_score": GRADE_SCORE.get(grade, 50),
            "confidence_low":   ai_res.get("confidence_low"),
            "confidence_high":  ai_res.get("confidence_high"),
        }
        logger.info("AI 모델 사용 [%s/%s] grade=%s trend=%.2f%%", item_code, item_name, grade, trend_pct)

    except ValueError:
        # 모델 파일 없음 → ETS 스텁으로 폴백
        logger.info("AI 모델 없음 [%s/%s] → ETS 스텁 폴백", item_code, item_name)
    except Exception as exc:
        logger.warning("AI 모델 실패 [%s/%s]: %s — ETS 스텁 폴백", item_code, item_name, exc)

    # ── 2) ETS 스텁 폴백 ─────────────────────────────────────────────────────
    if result is None:
        fn = _get_predict_fn()
        if fn is None:
            raise RuntimeError("predictor.py를 찾을 수 없습니다. AI 모델을 배포하세요.")

        raw   = fn(item_code=item_code, days=days)
        grade = str(raw.get("grade", "보합"))
        if grade not in VALID_GRADES:
            grade = "보합"

        result = {
            "source":           "ets",
            "predicted_prices": [float(p) for p in raw.get("predicted_prices", [])],
            "grade":            grade,
            "trend_pct":        float(raw.get("trend_pct", 0.0)),
            "anomaly_flag":     bool(raw.get("anomaly_flag", False)),
            # ETS 는 아래 필드 없음 → None (predict_by_name 이 별도 계산)
            "current_price":    None,
            "prices_7d":        None,
            "mape":             None,
            "buy_timing_score": GRADE_SCORE.get(grade, 50),
            "confidence_low":   None,
            "confidence_high":  None,
        }

    _CACHE[key] = (now, result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CSV 유틸리티
# ─────────────────────────────────────────────────────────────────────────────
_DATE_HINTS  = ("날짜", "일자", "date", "조사일", "기준일", "조회일", "연도")
_PRICE_HINTS = ("가격", "소매가", "도매가", "price", "평균가", "소매일일가격", "도매일일가격", "평균가격")


def _detect_cols(df: pd.DataFrame) -> tuple[str, str]:
    date_col = price_col = None
    for col in df.columns:
        cl = col.strip().lower()
        if date_col is None and any(h in cl for h in _DATE_HINTS):
            date_col = col
        if price_col is None and any(h in cl for h in _PRICE_HINTS):
            price_col = col
    if date_col is None:
        date_col = df.columns[0]
    if price_col is None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        price_col = num_cols[-1] if len(num_cols) else df.columns[-1]
    return date_col, price_col


def _read_item_csv(item_code: str, window: int = 30) -> pd.DataFrame:
    """품목 전용 CSV(data/{code}.csv) → 최근 window 일 DataFrame."""
    item_name = ITEMS.get(item_code, "")

    # 1) 품목 전용 파일 우선
    for candidate in [DATA_DIR / f"{item_code}.csv", DATA_DIR / f"kamis_{item_code}.csv"]:
        if candidate.exists():
            for enc in ("utf-8-sig", "euc-kr", "utf-8"):
                try:
                    return _clean(pd.read_csv(candidate, encoding=enc), window)
                except UnicodeDecodeError:
                    continue
            break

    # 2) 통합 CSV 에서 품목명 매칭
    for csv_file in sorted(DATA_DIR.glob("*.csv")):
        for enc in ("utf-8-sig", "euc-kr", "utf-8"):
            try:
                df = pd.read_csv(csv_file, encoding=enc)
                name_cols = [c for c in df.columns if "품목명" in c or c.lower() == "name"]
                if not name_cols:
                    continue
                sub = df[df[name_cols[0]] == item_name]
                if len(sub) > 0:
                    return _clean(sub.reset_index(drop=True), window)
            except UnicodeDecodeError:
                continue
            except Exception:
                break
    return pd.DataFrame()


def _clean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    date_col, price_col = _detect_cols(df)
    out = df[[date_col, price_col]].copy()
    out.columns = ["date", "price"]
    out["price"] = pd.to_numeric(
        out["price"].astype(str).str.replace(",", "").str.strip(), errors="coerce"
    )
    out = out.dropna()
    out = out[out["price"] > 0]
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.dropna()
    return (
        out.sort_values("date")
        .drop_duplicates("date")
        .tail(window)
        .reset_index(drop=True)
    )


def _get_last_price(item_code: str) -> float | None:
    df = _read_item_csv(item_code, window=3)
    if df.empty:
        return None
    return float(df.iloc[-1]["price"])


def _get_last_date(item_code: str) -> date | None:
    df = _read_item_csv(item_code, window=3)
    if df.empty:
        return None
    try:
        return date.fromisoformat(str(df.iloc[-1]["date"]))
    except Exception:
        return None


def _make_future_dates(item_code: str, days: int) -> list[str]:
    """마지막 데이터 날짜 다음날부터 N일치 날짜 문자열 생성."""
    last = _get_last_date(item_code)
    if last is None:
        last = date.today()
    return [(last + timedelta(days=i + 1)).isoformat() for i in range(days)]


def _confidence_band(prices: list[float], trend_pct: float) -> tuple[float, float]:
    """예측 가격의 신뢰구간 하한·상한 (불확실성 ≈ |trend_pct|/2 + 기본 5%)."""
    if not prices:
        return 0.0, 0.0
    mean_p = float(np.mean(prices))
    uncertainty_pct = min(max(abs(trend_pct) / 2 + 5, 5), 25) / 100
    return round(mean_p * (1 - uncertainty_pct), 0), round(mean_p * (1 + uncertainty_pct), 0)


def _compute_mape(item_code: str) -> float | None:
    """Walk-forward MAPE 계산 (일별 데이터 21일 후보). 캐시됨."""
    if item_code in _mape_cache:
        return _mape_cache[item_code]

    try:
        from predictor import _load_series, _ets_forecast
        series = _load_series(item_code, window=9999)
        if len(series) < 21:
            _mape_cache[item_code] = None
            return None

        test_n = 14
        mapes = []
        for start in range(0, test_n - 7 + 1, 7):
            cut = len(series) - (test_n - start)
            if cut < 14:
                continue
            train = series.iloc[:cut]
            actual = series.iloc[cut: cut + 7].values
            fc = _ets_forecast(train, 7)[: len(actual)]
            if len(fc) == len(actual) and len(actual) > 0:
                m = float(np.mean(np.abs((actual - fc) / np.maximum(actual, 1)))) * 100
                mapes.append(m)

        result = round(float(np.mean(mapes)), 1) if mapes else None
        _mape_cache[item_code] = result
        return result
    except Exception:
        _mape_cache[item_code] = None
        return None


_mape_cache: dict[str, float | None] = {}

# ─────────────────────────────────────────────────────────────────────────────
# 이름 → 코드 역방향 색인
# ─────────────────────────────────────────────────────────────────────────────
_NAME_TO_CODE: dict[str, str] = {v: k for k, v in ITEMS.items()}


def _code_from_name(name: str) -> str | None:
    return _NAME_TO_CODE.get(name.strip())


# ─────────────────────────────────────────────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["운영"])
def health():
    predictor_ok = _get_predict_fn() is not None
    csv_counts = {code: len(_read_item_csv(code, window=9999)) for code in ITEMS}
    return {
        "status": "ok",
        "predictor_loaded": predictor_ok,
        "supported_items": len(ITEMS),
        "items_with_data": sum(1 for v in csv_counts.values() if v >= 7),
        "cache_ttl_sec": _CACHE_TTL,
        "data_row_counts": csv_counts,
    }


# ── /api/items ────────────────────────────────────────────────────────────────
@app.get("/api/items", tags=["품목"])
def get_items():
    """지원 품목 전체 목록 (배열 직반환 — 프론트엔드 호환)."""
    result = []
    for code, name in ITEMS.items():
        df = _read_item_csv(code, window=1)
        result.append({
            "code":     code,
            "name":     name,
            "category": CATEGORY.get(code, "농산물"),
            "has_data": len(df) > 0,
        })
    return result   # JSONResponse 아닌 list → FastAPI가 배열로 직렬화


# ── /api/predict (프론트엔드 주 엔드포인트: ?item=NAME) ──────────────────────
@app.get("/api/predict", tags=["예측"])
def predict_by_name(
    item: str = Query(..., description="품목명 (예: 배추, 양파)"),
    days: int = Query(default=7, ge=1, le=30),
):
    """
    품목명으로 7일 예측 + 구매 타이밍 등급 반환 (프론트엔드 호환).

    **buy_timing** : 급등경보(10점) / 상승(30점) / 보합(50점) / 하락(70점) / 급락경보(90점)
    """
    code = _code_from_name(item)
    if code is None:
        raise HTTPException(404, f"지원하지 않는 품목명: {item}")

    try:
        pred = _call_predict(code, days)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        logger.error("predict 오류 [%s]: %s", code, exc, exc_info=True)
        raise HTTPException(500, f"예측 중 오류: {exc}") from exc

    grade     = pred["grade"]
    trend_pct = pred["trend_pct"]
    anomaly   = pred["anomaly_flag"]

    if pred.get("source") == "ai" and pred.get("prices_7d"):
        # ── AI 모델 경로: prices_7d에 date/price/lower/upper 이미 포함 ─────
        prices_7d     = pred["prices_7d"]
        current_price = pred["current_price"]
        mape          = pred["mape"]
        conf_low      = pred["confidence_low"]
        conf_high     = pred["confidence_high"]
    else:
        # ── ETS 스텁 경로: 별도 날짜 생성 + 신뢰구간 계산 ──────────────────
        prices       = pred["predicted_prices"]
        future_dates = _make_future_dates(code, days)
        prices_7d    = [{"date": d, "price": p} for d, p in zip(future_dates, prices)]
        current_price = _get_last_price(code)
        mape          = _compute_mape(code)
        conf_low, conf_high = _confidence_band(prices, trend_pct)

    return {
        "item":             ITEMS[code],
        "item_code":        code,
        "category":         CATEGORY.get(code, "농산물"),
        "current_price":    current_price,
        "prices_7d":        prices_7d,
        "mape":             mape,
        "anomaly_flag":     anomaly,
        "buy_timing":       grade,
        "buy_timing_score": GRADE_SCORE.get(grade, 50),
        "trend_pct":        trend_pct,
        "confidence_low":   conf_low,
        "confidence_high":  conf_high,
        "model_source":     pred.get("source", "ets"),
        "updated_at":       pd.Timestamp.now().isoformat(),
    }


# ── /api/predict/{item_code} (하위호환 — 명세 원형) ──────────────────────────
@app.get("/api/predict/{item_code}", tags=["예측"])
def predict_by_code(
    item_code: str,
    days: int = Query(default=7, ge=1, le=30, description="예측 일수"),
):
    """품목코드로 예측 (원형 명세 — 하위호환)."""
    if item_code not in ITEMS:
        raise HTTPException(404, f"지원하지 않는 품목코드: {item_code}")
    try:
        pred = _call_predict(item_code, days)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"예측 오류: {exc}") from exc

    return {
        "item_code":       item_code,
        "item_name":       ITEMS[item_code],
        "days":            days,
        "predicted_prices": pred["predicted_prices"],
        "grade":           pred["grade"],
        "trend_pct":       pred["trend_pct"],
        "anomaly_flag":    pred["anomaly_flag"],
        # 프론트 호환 필드도 함께 포함
        "buy_timing":      pred["grade"],
        "buy_timing_score": GRADE_SCORE.get(pred["grade"], 50),
    }


# ── /api/latest/{item_code} ───────────────────────────────────────────────────
@app.get("/api/latest/{item_code}", tags=["실가격"])
def latest_prices(
    item_code: str,
    days: int = Query(default=30, ge=1, le=365),
):
    """data/ 디렉터리 CSV에서 최근 N일 실제 가격 이력."""
    if item_code not in ITEMS:
        raise HTTPException(404, f"지원하지 않는 품목코드: {item_code}")
    df = _read_item_csv(item_code, window=days)
    if df.empty:
        raise HTTPException(
            404,
            f"데이터 없음 — data/{item_code}.csv 를 배치하거나 "
            "scripts/prepare_data.py 를 실행하세요.",
        )
    return {
        "item_code":      item_code,
        "item_name":      ITEMS[item_code],
        "requested_days": days,
        "actual_rows":    len(df),
        "data":           df.to_dict(orient="records"),
    }


# ── /api/anomaly ──────────────────────────────────────────────────────────────
@app.get("/api/anomaly", tags=["이상탐지"])
def detect_anomalies():
    """이상가격(급등/급락경보 또는 anomaly_flag=True) 품목 목록."""
    if _get_predict_fn() is None:
        return {"anomalies": [], "count": 0, "message": "predictor.py 미로드"}

    anomalies: list[dict] = []
    for code, name in ITEMS.items():
        try:
            pred = _call_predict(code, days=7)
            if pred["anomaly_flag"] or pred["grade"] in ("급등경보", "급락경보"):
                anomalies.append({
                    "item_code":    code,
                    "item_name":    name,
                    "buy_timing":   pred["grade"],
                    "trend_pct":    pred["trend_pct"],
                    "anomaly_flag": pred["anomaly_flag"],
                })
        except Exception as exc:
            logger.warning("이상탐지 스킵 [%s]: %s", code, exc)

    return {"anomalies": anomalies, "count": len(anomalies)}


# ── 운영 도구 ─────────────────────────────────────────────────────────────────
@app.delete("/api/cache", tags=["운영"], include_in_schema=False)
def clear_cache(item_code: str | None = None):
    if item_code:
        _CACHE.pop(f"{item_code}:7", None)
        _mape_cache.pop(item_code, None)
    else:
        _CACHE.clear()
        _mape_cache.clear()
    return {"invalidated": item_code or "ALL"}


# ─────────────────────────────────────────────────────────────────────────────
# 정적 파일 서빙 (public/) — 반드시 API 라우트 등록 이후 마운트
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def serve_root():
    idx = PUBLIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({
        "service": "KAMIS 농산물 가격 AI 예측 API",
        "version": "1.0.0",
        "docs":    "/api/docs",
        "items":   "/api/items",
    })


if PUBLIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="public")
