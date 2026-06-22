"""
농산물 AI 가격예측 Flask 서버
포트 3000
"""
import os, sys, time, json
from functools import lru_cache
from flask import Flask, jsonify, request, send_from_directory, abort

sys.path.insert(0, os.path.dirname(__file__))
from ai.predict import predict, list_available_items

app = Flask(__name__, static_folder='public', static_url_path='')

# 예측 캐시 (1시간 TTL)
_pred_cache = {}
_cache_time = {}
CACHE_TTL = 3600  # 1시간


def get_cached_prediction(item: str) -> dict:
    """캐시된 예측 반환 (없거나 만료 시 재계산)"""
    now = time.time()
    if item in _pred_cache and (now - _cache_time.get(item, 0)) < CACHE_TTL:
        return _pred_cache[item]

    result = predict(item)
    _pred_cache[item] = result
    _cache_time[item] = now
    return result


# ─── 정적 파일 ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('public', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('public', filename)


# ─── API ──────────────────────────────────────────────────────────────────────

@app.route('/api/items')
def api_items():
    """이용 가능한 품목 목록"""
    items = list_available_items()
    item_meta = {
        '배추':   {'category': '채소', 'emoji': '🥬'},
        '무':     {'category': '채소', 'emoji': '🫛'},
        '양파':   {'category': '채소', 'emoji': '🧅'},
        '마늘':   {'category': '채소', 'emoji': '🧄'},
        '대파':   {'category': '채소', 'emoji': '🌿'},
        '건고추': {'category': '채소', 'emoji': '🌶️'},
        '감자':   {'category': '채소', 'emoji': '🥔'},
        '고구마': {'category': '채소', 'emoji': '🍠'},
        '토마토': {'category': '채소', 'emoji': '🍅'},
        '사과':   {'category': '과일', 'emoji': '🍎'},
        '당근':   {'category': '채소', 'emoji': '🥕'},
        '양배추': {'category': '채소', 'emoji': '🥬'},
        '상추':   {'category': '채소', 'emoji': '🥗'},
        '시금치': {'category': '채소', 'emoji': '🌱'},
    }
    result = []
    for item in items:
        meta = item_meta.get(item, {'category': '농산물', 'emoji': '🌾'})
        result.append({'name': item, **meta})
    return jsonify(result)


@app.route('/api/predict')
def api_predict():
    """품목 7일 가격 예측"""
    item = request.args.get('item', '배추').strip()
    if not item:
        return jsonify({'error': '품목명 필요 (item=배추)'}), 400

    available = list_available_items()
    if item not in available:
        return jsonify({'error': f"'{item}' 모델 없음", 'available': available}), 404

    try:
        start = time.time()
        result = get_cached_prediction(item)
        elapsed_ms = (time.time() - start) * 1000
        result['response_ms'] = round(elapsed_ms, 1)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/summary')
def api_summary():
    """전체 품목 현재 시세 + 타이밍 요약"""
    items = list_available_items()
    results = []
    for item in items:
        try:
            r = get_cached_prediction(item)
            results.append({
                'item': r['item'],
                'current_price': r['current_price'],
                'buy_timing': r['buy_timing'],
                'buy_timing_score': r['buy_timing_score'],
                'anomaly_flag': r['anomaly_flag'],
                'mape': r['mape'],
                'unit': r['unit'],
            })
        except Exception as e:
            results.append({'item': item, 'error': str(e)})

    return jsonify({'items': results, 'count': len(results)})


@app.route('/api/health')
def api_health():
    """헬스 체크 + 시스템 상태"""
    items = list_available_items()
    return jsonify({
        'status': 'ok',
        'models_loaded': len(items),
        'items': items,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    })


# ─── 시작 ─────────────────────────────────────────────────────────────────────

def warmup():
    """서버 시작 시 예측 캐시 선열 (P95 응답 보장)"""
    items = list_available_items()
    if not items:
        print('[warmup] 모델 없음 — ai/train_best.py 실행 필요')
        return
    print(f'[warmup] {len(items)}개 품목 예측 캐시 초기화...')
    for item in items:
        try:
            get_cached_prediction(item)
        except Exception as e:
            print(f'  [{item}] FAIL: {e}')
    print(f'[warmup] {len(items)}개 완료')


# 모듈 로드 시 즉시 warmup (gunicorn --preload 에서도 동작)
warmup()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    print(f'\n서버 시작 → http://localhost:{port}')
    app.run(host='0.0.0.0', port=port, debug=False)
