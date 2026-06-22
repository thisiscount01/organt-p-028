#!/usr/bin/env python3
"""
KAMIS/공공데이터 농산물 가격 실데이터 다운로드
순서: 1) data.go.kr Playwright  2) KAMIS 웹 스크래핑  3) 농넷 AJAX  4) data.mafra.go.kr
"""
import os, sys, time, re, json, urllib.request, urllib.parse
os.makedirs('data', exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120',
    'Accept': 'application/json, text/html, */*',
    'Accept-Language': 'ko-KR,ko;q=0.9',
}

# ─── 1. data.go.kr Playwright ─────────────────────────────────────────────────
def try_playwright_datagokr():
    from playwright.sync_api import sync_playwright

    datasets = [
        # (페이지ID, UDDI, 저장파일명, 설명)
        ('15087352', 'uddi:2cf914d1-5fd1-4135-8986-f1d899743cc7', 'kamis_vegetables_daily.csv', '채소류 일일가격(2196행)'),
        ('15087482', None, 'kamis_monthly_retail.csv', '월별 소매가격'),
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(accept_downloads=True, ignore_https_errors=True)
        page = ctx.new_page()

        for ds_id, uddi, save_name, desc in datasets:
            print(f'[PW] {desc} ({ds_id}) 시도...')
            try:
                page.goto(f'https://www.data.go.kr/data/{ds_id}/fileData.do', timeout=30000)
                page.wait_for_load_state('domcontentloaded', timeout=20000)

                if uddi is None:
                    content = page.content()
                    match = re.search(r"fn_fileDataDown\('[^']+',\s*'(uddi:[^']+)'", content)
                    if match:
                        uddi = match.group(1)
                    else:
                        print(f'  UDDI 찾기 실패')
                        continue

                with page.expect_download(timeout=30000) as dl_info:
                    page.evaluate(f"fileDetailObj.fn_fileDataDown('{ds_id}', '{uddi}', '','1', '4')")

                dl = dl_info.value
                dl.save_as(f'data/{save_name}')
                print(f'  OK: data/{save_name} ({dl.suggested_filename})')
                size = os.path.getsize(f'data/{save_name}')
                print(f'  크기: {size:,} bytes')
                if size > 1000:
                    browser.close()
                    return save_name

            except Exception as e:
                print(f'  FAIL: {e}')

        browser.close()
    return None

# ─── 2. KAMIS 웹사이트 직접 스크래핑 ──────────────────────────────────────────
def try_kamis_scrape():
    """KAMIS 가격통계 페이지에서 CSV 엑셀 직접 다운로드 시도"""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(accept_downloads=True, ignore_https_errors=True)
        page = ctx.new_page()

        items = [
            ('배추', '211'),
            ('무', '212'),
            ('양파', '214'),
            ('마늘', '215'),
            ('대파', '216'),
            ('건고추', '218'),
            ('감자', '232'),
            ('고추', '219'),
        ]

        all_rows = []
        collected = set()

        # 기간별 가격조회 페이지
        url = 'https://www.kamis.or.kr/customer/price/product/period.do'
        print(f'[KAMIS] 가격통계 접근: {url}')

        try:
            page.goto(url, timeout=20000)
            page.wait_for_load_state('domcontentloaded', timeout=15000)
            print(f'  title: {page.title()}')

            # 페이지 내 CSV/엑셀 다운로드 버튼 찾기
            content = page.content()
            dl_patterns = re.findall(r'(excel|csv|download|Export)[^\n]{0,100}', content, re.I)
            for pat in dl_patterns[:5]:
                print(f'  패턴: {pat[:100]}')

            # 엑셀 다운로드 버튼 클릭 시도
            btns = page.locator('button, a, input[type=button]').all()
            for b in btns:
                txt = (b.text_content() or '').strip()
                if any(k in txt for k in ['엑셀', 'Excel', 'CSV', '다운']):
                    print(f'  버튼 발견: {txt}')

        except Exception as e:
            print(f'  FAIL: {e}')

        # 통계 데이터 직접 API 탐색
        print('\n[KAMIS] 통계 API 탐색...')
        ajax_urls = []
        page.on('request', lambda req: ajax_urls.append(req.url) if 'price' in req.url.lower() or 'stat' in req.url.lower() else None)

        try:
            page.goto('https://www.kamis.or.kr/customer/price/statistics.do', timeout=20000)
            page.wait_for_load_state('networkidle', timeout=15000)
            time.sleep(2)
            print(f'  캡처된 AJAX URL 수: {len(ajax_urls)}')
            for u in ajax_urls[:10]:
                print(f'    {u}')
        except Exception as e:
            print(f'  FAIL: {e}')

        browser.close()
        return None

# ─── 3. KAMIS Open API (공개 샘플 키) ─────────────────────────────────────────
def try_kamis_api_scrape_all():
    """KAMIS API로 다년간 데이터를 품목별로 수집"""
    from playwright.sync_api import sync_playwright

    # 품목코드 (KAMIS 표준)
    items = {
        '배추': ('01', '211', '11'),   # (분류코드, 품목코드, 품종코드)
        '무':   ('01', '212', '11'),
        '양파': ('01', '214', '11'),
        '마늘': ('01', '215', '11'),
        '대파': ('01', '216', '11'),
        '건고추': ('01', '218', '11'),
        '감자': ('01', '232', '11'),
        '사과': ('04', '411', '11'),
        '배':   ('04', '412', '11'),
        '쌀':   ('01', '111', '11'),
        '토마토': ('02', '214', '11'),
        '오이': ('02', '215', '11'),
    }

    # 연도범위
    years = range(2019, 2025)

    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(ignore_https_errors=True)
        page = ctx.new_page()

        for item_name, (cat_code, item_code, kind_code) in items.items():
            for year in years:
                url = (
                    f'https://www.kamis.or.kr/service/price/xml.do'
                    f'?action=periodProductList'
                    f'&p_startday={year}-01-01'
                    f'&p_endday={year}-12-31'
                    f'&p_productclscode=01'
                    f'&p_itemcategorycode={cat_code}'
                    f'&p_itemcode={item_code}'
                    f'&p_kindcode={kind_code}'
                    f'&p_graderank=1'
                    f'&p_convert_kg_yn=N'
                    f'&p_cert_key=111'
                    f'&p_cert_id=222'
                    f'&p_returntype=json'
                )
                try:
                    resp = page.evaluate(f'''
                        fetch("{url}", {{headers:{{"User-Agent":"Mozilla/5.0"}}}})
                            .then(r => r.text())
                            .catch(e => "ERROR:" + e.message)
                    ''')
                    if resp and not resp.startswith('ERROR'):
                        data = json.loads(resp)
                        print(f'  {item_name} {year}: {str(data)[:100]}')
                except Exception as e:
                    print(f'  {item_name} {year}: {e}')

        browser.close()
    return rows

# ─── 4. nongnet.or.kr 도매가격 ─────────────────────────────────────────────────
def try_nongnet_wholesale():
    """농넷 도매시장 가격 데이터 수집"""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(ignore_https_errors=True)
        page = ctx.new_page()

        ajax_calls = []
        def capture(req):
            if any(x in req.url for x in ['price', 'market', 'json', 'ajax']):
                ajax_calls.append({'url': req.url, 'method': req.method})

        page.on('request', capture)

        try:
            print('[nongnet] 페이지 로드...')
            page.goto('https://www.nongnet.or.kr/', timeout=20000)
            page.wait_for_load_state('networkidle', timeout=15000)
            print(f'  캡처된 AJAX: {len(ajax_calls)}')
            for c in ajax_calls[:10]:
                print(f'    {c["method"]} {c["url"]}')
        except Exception as e:
            print(f'  nongnet FAIL: {e}')

        browser.close()
    return None

# ─── 5. data.mafra.go.kr ──────────────────────────────────────────────────────
def try_mafra():
    """농림축산식품부 공공데이터 포털"""
    import urllib.request
    urls = [
        'https://data.mafra.go.kr/opendata/data/indexOpenDataDetail.do?data_id=20151114000000000541',
        'https://data.mafra.go.kr/opendata/data/selectDataProfPage.do?data_id=20151114000000000541',
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=10) as r:
                data = r.read().decode('utf-8', errors='replace')
                print(f'[mafra] {url[:60]}: {len(data)}자, 미리보기: {data[:200]}')
                return data
        except Exception as e:
            print(f'[mafra] {url[:60]}: {e}')
    return None


if __name__ == '__main__':
    print('=== 농산물 가격 실데이터 다운로드 ===\n')

    # 1순위: data.go.kr Playwright
    result = try_playwright_datagokr()
    if result:
        print(f'\n성공: {result}')
        sys.exit(0)

    # 2순위: KAMIS 스크래핑 (AJAX URL 캡처)
    print('\n--- KAMIS 웹 스크래핑 ---')
    try_kamis_scrape()

    # 3순위: nongnet
    print('\n--- nongnet 도매가격 ---')
    try_nongnet_wholesale()

    # 4순위: mafra
    print('\n--- data.mafra.go.kr ---')
    try_mafra()

    files = [f for f in os.listdir('data') if os.path.getsize(f'data/{f}') > 1000]
    print(f'\n=== 결과: data/ 폴더 유효 파일: {files} ===')
