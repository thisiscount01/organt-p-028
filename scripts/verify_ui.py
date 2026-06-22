import http.server
import threading
import json
import time
from urllib.parse import unquote, urlparse, parse_qs
from playwright.sync_api import sync_playwright

class MockHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory='public', **kw)
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == '/api/items':
            self.send_json([
                {"code": "1001", "name": "배추",   "category": "채소류"},
                {"code": "1101", "name": "무",     "category": "채소류"},
                {"code": "1201", "name": "양파",   "category": "채소류"},
                {"code": "1202", "name": "대파",   "category": "채소류"},
                {"code": "1207", "name": "건고추", "category": "채소류"},
                {"code": "1209", "name": "마늘",   "category": "채소류"},
            ])
        elif self.path.startswith('/api/predict'):
            qs   = parse_qs(urlparse(self.path).query)
            item = unquote(qs.get('item', ['배추'])[0])
            base = 1850
            self.send_json({
                "item": item,
                "current_price": base,
                "prices_7d": [
                    {"date": f"2024-06-{21+i:02d}", "price": base + i * 30 + (i % 2) * 20}
                    for i in range(7)
                ],
                "mape": 8.4,
                "anomaly_flag": False,
                "buy_timing": "하락",
                "buy_timing_score": 72,
                "confidence_low": 1720,
                "confidence_high": 1990,
                "updated_at": "2024-06-21T14:30:00",
                "category": "채소류",
            })
        else:
            super().do_GET()

    def send_json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


def main():
    server = http.server.HTTPServer(("0.0.0.0", 13002), MockHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.4)

    with sync_playwright() as p:
        browser = p.chromium.launch()

        # ── 데스크탑 검증 ──
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        page.goto("http://localhost:13002/", wait_until="networkidle")
        time.sleep(1.6)

        title     = page.title()
        tabs      = page.query_selector_all(".item-tab")
        badge     = page.query_selector(".timing-badge")
        badge_txt = badge.inner_text().strip() if badge else ""
        price_el  = page.query_selector(".cprice-value")
        price_txt = price_el.inner_text() if price_el else ""
        chart_el  = page.query_selector("#price-chart")
        rows      = page.query_selector_all("#table-body tr")
        sk_hidden = not page.query_selector("#pred-skeleton:not([hidden])")
        ct_shown  = bool(page.query_selector("#pred-content:not([hidden])"))
        timing    = page.get_attribute("#price-card", "data-timing")
        anomaly   = bool(page.query_selector("#anomaly-banner:not([hidden])"))
        score_num = page.query_selector("#score-num")
        score_txt = score_num.inner_text() if score_num else ""
        stat_max  = page.query_selector("#stat-max")
        stat_txt  = stat_max.inner_text() if stat_max else ""

        print("=== 데스크탑 검증 ===")
        print(f"타이틀:         {title}")
        print(f"품목탭 수:      {len(tabs)}개  {'OK' if len(tabs)>=6 else 'NG'}")
        print(f"타이밍배지:     {badge_txt}")
        print(f"현재가:         {price_txt}  {'OK' if price_txt else 'NG'}")
        print(f"data-timing:    {timing}  {'OK' if timing else 'NG'}")
        print(f"차트캔버스:     {'OK' if chart_el else 'NG'}")
        print(f"테이블행:       {len(rows)}행  {'OK' if len(rows)==7 else 'NG'}")
        print(f"스켈레톤 숨김:  {'OK' if sk_hidden else 'NG'}")
        print(f"콘텐츠 표시:    {'OK' if ct_shown else 'NG'}")
        print(f"이상가격배너:   {'표시' if anomaly else '숨김(정상)'}")
        print(f"점수 지수:      {score_txt}")
        print(f"7일 최고가:     {stat_txt}")
        print(f"JS 에러:        {len(errors)}건  {errors[:2] if errors else ''}")

        # 탭 전환 테스트
        onion = page.query_selector('[data-item="양파"]')
        if onion:
            onion.click()
            page.wait_for_selector("#pred-content:not([hidden])", timeout=5000)
            time.sleep(0.5)
            item_name = page.query_selector("#item-name").inner_text()
            print(f"탭전환(양파):   item-name={item_name}  {'OK' if item_name=='양파' else 'NG'}")
        else:
            print("탭전환:         양파 탭 없음 NG")

        page.screenshot(path="scripts/screenshot_desktop.png")
        print("스크린샷:       scripts/screenshot_desktop.png")

        # ── 모바일 검증 ──
        mob = browser.new_page(viewport={"width": 390, "height": 844})
        mob.goto("http://localhost:13002/", wait_until="networkidle")
        time.sleep(1.4)
        mob_ct = bool(mob.query_selector("#pred-content:not([hidden])"))
        mob_tabs = mob.query_selector_all(".item-tab")
        mob.screenshot(path="scripts/screenshot_mobile.png")
        print(f"\n=== 모바일(390px) 검증 ===")
        print(f"콘텐츠 표시:    {'OK' if mob_ct else 'NG'}")
        print(f"탭 수:          {len(mob_tabs)}개")
        print("스크린샷:       scripts/screenshot_mobile.png")

        browser.close()

    server.shutdown()
    print("\n모든 검증 완료")


main()
