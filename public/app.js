/**
 * 장보기 어드바이저 — Frontend App
 * API 계약: GET /api/predict?item=NAME, GET /api/items
 * 모든 색상·상태는 CSS data-timing 속성으로 제어 (JS 색상값 0건)
 */

/* ═══════════════════════════════════════════════════════
   1. 상수 & 설정
   ══════════════════════════════════════════════════════ */

/** buy_timing 5단계 UI 설정 — 색상 없음, CSS가 처리 */
const TIMING_CONFIG = {
  '급등경보': {
    icon: '🔴',
    advice: '⚠️ 현재 가격이 비정상적으로 높습니다. 구매를 최대한 미루고 가격 안정을 기다리세요.',
  },
  '상승': {
    icon: '🟠',
    advice: '📈 가격이 오를 전망입니다. 꼭 필요한 양만 구매하고 추가 구매는 잠시 보류하세요.',
  },
  '보합': {
    icon: '🟡',
    advice: '✅ 가격이 안정적입니다. 평소 필요한 양만큼 구매하셔도 좋습니다.',
  },
  '하락': {
    icon: '🟢',
    advice: '💡 가격이 내릴 전망입니다. 소량씩 분할 구매를 권장합니다.',
  },
  '급락경보': {
    icon: '💜',
    advice: '🎉 가격이 크게 낮아질 전망입니다! 지금이 대량 구매의 최적 타이밍입니다.',
  },
};

/** 품목 이모지 매핑 (API 응답 보완용) */
const ITEM_EMOJI = {
  '배추':   '🥬',
  '양파':   '🧅',
  '마늘':   '🧄',
  '대파':   '🌿',
  '무':     '⬜',
  '사과':   '🍎',
  '토마토': '🍅',
  '오이':   '🥒',
  '쌀':     '🌾',
  '감자':   '🥔',
  '건고추': '🌶️',
  '당근':   '🥕',
  '상추':   '🥗',
  '고구마': '🍠',
  '배':     '🍐',
  '포도':   '🍇',
};

/** 클라이언트 측 기본 품목 목록 (API 실패 대비 fallback) */
const DEFAULT_ITEMS = [
  { code: '1001', name: '배추',   category: '채소류' },
  { code: '1101', name: '무',     category: '채소류' },
  { code: '1201', name: '양파',   category: '채소류' },
  { code: '1202', name: '대파',   category: '채소류' },
  { code: '1207', name: '건고추', category: '채소류' },
  { code: '1209', name: '마늘',   category: '채소류' },
];

/* ═══════════════════════════════════════════════════════
   2. 상태
   ══════════════════════════════════════════════════════ */

let currentItem   = '배추';
let chartInstance = null;
let items         = [];

/* ═══════════════════════════════════════════════════════
   3. DOM 참조
   ══════════════════════════════════════════════════════ */

const $ = id => document.getElementById(id);

const DOM = {
  itemTabs:       $('item-tabs'),
  itemTabsSkel:   $('item-tabs-skeleton'),
  predSkeleton:   $('pred-skeleton'),
  predContent:    $('pred-content'),
  errorState:     $('error-state'),
  errorDesc:      $('error-desc'),
  retryBtn:       $('retry-btn'),
  updatedAt:      $('updated-at'),
  anomalyBanner:  $('anomaly-banner'),
  anomalyText:    $('anomaly-text'),
  anomalyClose:   $('anomaly-close'),
  priceCard:      $('price-card'),
  itemEmoji:      $('item-emoji'),
  itemName:       $('item-name'),
  itemCategory:   $('item-category'),
  timingBadge:    $('timing-badge'),
  timingIcon:     $('timing-icon'),
  timingLabel:    $('timing-label'),
  currentPrice:   $('current-price'),
  scoreNum:       $('score-num'),
  scoreFill:      $('score-fill'),
  timingAdvice:   $('timing-advice'),
  mapeValue:      $('mape-value'),
  anomalyChip:    $('anomaly-chip'),
  statMax:        $('stat-max'),
  statMin:        $('stat-min'),
  statAvg:        $('stat-avg'),
  statRange:      $('stat-range'),
  priceChart:     $('price-chart'),
  tableBody:      $('table-body'),
};

/* ═══════════════════════════════════════════════════════
   4. 유틸리티
   ══════════════════════════════════════════════════════ */

/** 숫자를 한국식 천단위 포맷 */
function fmtPrice(n) {
  if (n == null || isNaN(n)) return '—';
  return Math.round(n).toLocaleString('ko-KR');
}

/** 날짜 문자열(YYYY-MM-DD)을 "6/21(토)" 형식으로 */
function fmtDate(dateStr) {
  const d = new Date(dateStr + 'T00:00:00');
  const days = ['일', '월', '화', '수', '목', '금', '토'];
  return `${d.getMonth() + 1}/${d.getDate()}(${days[d.getDay()]})`;
}

/** 날짜 문자열을 "2024.06.21 14:30" 형식으로 */
function fmtDatetime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  const pad = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}.${pad(d.getMonth()+1)}.${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())} 기준`;
}

/** 이모지 반환 (없으면 기본값) */
function getEmoji(name, fallback = '🛒') {
  return ITEM_EMOJI[name] || fallback;
}

/** 신뢰구간 밴드: 7일치 각 일별 상하한 계산 (선형 확장) */
function buildBands(prices7d, confLow, confHigh) {
  const halfRange = (confHigh - confLow) / 2;
  const n = prices7d.length;
  return prices7d.map((pt, i) => {
    const factor = (i + 1) / n;   // 1일차 좁음 → 7일차 넓음
    const unc = halfRange * factor;
    return {
      high: Math.round(pt.price + unc),
      low:  Math.max(0, Math.round(pt.price - unc)),
    };
  });
}

/* ═══════════════════════════════════════════════════════
   5. UI 상태 전환
   ══════════════════════════════════════════════════════ */

/**
 * @param {'loading'|'content'|'error'} state
 */
function setViewState(state) {
  const show = el => { el.removeAttribute('hidden'); el.setAttribute('aria-hidden', 'false'); };
  const hide = el => { el.setAttribute('hidden', '');   el.setAttribute('aria-hidden', 'true'); };

  hide(DOM.predSkeleton);
  hide(DOM.predContent);
  hide(DOM.errorState);

  if (state === 'loading') show(DOM.predSkeleton);
  if (state === 'content') show(DOM.predContent);
  if (state === 'error')   show(DOM.errorState);
}

/* ═══════════════════════════════════════════════════════
   6. API 호출
   ══════════════════════════════════════════════════════ */

async function fetchItems() {
  try {
    const res = await fetch('/api/items');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch {
    return null;   // fallback 처리는 호출부에서
  }
}

async function fetchPrediction(itemName) {
  const res = await fetch(`/api/predict?item=${encodeURIComponent(itemName)}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `서버 오류 (${res.status})`);
  }
  return res.json();
}

/* ═══════════════════════════════════════════════════════
   7. 렌더: 품목 탭
   ══════════════════════════════════════════════════════ */

function renderItemTabs(itemList) {
  // 스켈레톤 제거
  DOM.itemTabsSkel.remove();

  // 기존 탭 버튼 제거
  DOM.itemTabs.querySelectorAll('.item-tab').forEach(el => el.remove());

  itemList.forEach(item => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.role = 'tab';
    btn.className = 'item-tab';
    btn.dataset.item = item.name;
    btn.setAttribute('aria-selected', item.name === currentItem ? 'true' : 'false');
    btn.setAttribute('aria-label', `${item.name} 가격 예측 보기`);

    const emoji = document.createElement('span');
    emoji.className = 'item-tab-emoji';
    emoji.setAttribute('aria-hidden', 'true');
    emoji.textContent = getEmoji(item.name);

    const label = document.createElement('span');
    label.textContent = item.name;

    btn.appendChild(emoji);
    btn.appendChild(label);

    btn.addEventListener('click', () => {
      if (currentItem === item.name) return;
      selectItem(item.name);
    });

    DOM.itemTabs.appendChild(btn);
  });
}

/** 선택된 탭 시각 업데이트 */
function updateActiveTab(itemName) {
  DOM.itemTabs.querySelectorAll('.item-tab').forEach(btn => {
    const active = btn.dataset.item === itemName;
    btn.setAttribute('aria-selected', active ? 'true' : 'false');
  });
}

/* ═══════════════════════════════════════════════════════
   8. 렌더: 예측 결과
   ══════════════════════════════════════════════════════ */

function renderPrediction(data) {
  const {
    item, current_price, prices_7d,
    mape, anomaly_flag,
    buy_timing, buy_timing_score,
    confidence_low, confidence_high,
    updated_at,
  } = data;

  const cfg  = TIMING_CONFIG[buy_timing] || TIMING_CONFIG['보합'];
  const bands = buildBands(prices_7d, confidence_low, confidence_high);
  const prices = prices_7d.map(p => p.price);

  /* 메타 */
  DOM.updatedAt.textContent = fmtDatetime(updated_at);

  /* 이상가격 배너 */
  if (anomaly_flag) {
    DOM.anomalyText.textContent = `⚠️ ${item} 가격에 이상 급변이 감지되었습니다. 최근 수급 상황을 확인하세요.`;
    DOM.anomalyBanner.removeAttribute('hidden');
  } else {
    DOM.anomalyBanner.setAttribute('hidden', '');
  }

  /* 품목 정체성 */
  DOM.itemEmoji.textContent     = getEmoji(item);
  DOM.itemName.textContent      = item;
  DOM.itemCategory.textContent  = data.category || '농산물';

  /* buy_timing — data-timing 세팅으로 CSS 전환 */
  DOM.priceCard.setAttribute('data-timing', buy_timing);
  DOM.timingBadge.setAttribute('data-timing', buy_timing);  // 배지도 독립 스위치
  DOM.timingIcon.textContent  = cfg.icon;
  DOM.timingLabel.textContent = buy_timing;

  /* 현재가 */
  DOM.currentPrice.textContent = fmtPrice(current_price);

  /* 점수 미터 */
  const score = Math.round(buy_timing_score ?? 50);
  DOM.scoreNum.textContent      = `${score}점`;
  DOM.scoreFill.style.width     = `${score}%`;
  DOM.scoreFill.setAttribute('aria-valuenow', score);

  /* 조언 */
  DOM.timingAdvice.textContent = cfg.advice;

  /* MAPE */
  DOM.mapeValue.textContent = mape != null ? `${mape.toFixed(1)}%` : '—';

  /* 이상가격 칩 */
  if (anomaly_flag) DOM.anomalyChip.removeAttribute('hidden');
  else              DOM.anomalyChip.setAttribute('hidden', '');

  /* 통계 카드 */
  const maxP = Math.max(...prices);
  const minP = Math.min(...prices);
  const avgP = prices.reduce((a, b) => a + b, 0) / prices.length;

  DOM.statMax.textContent   = fmtPrice(maxP);
  DOM.statMin.textContent   = fmtPrice(minP);
  DOM.statAvg.textContent   = fmtPrice(avgP);
  DOM.statRange.textContent = `${fmtPrice(confidence_low)}~${fmtPrice(confidence_high)}원`;

  /* 차트 */
  renderChart(prices_7d, bands);

  /* 테이블 */
  renderTable(prices_7d, bands, current_price);
}

/* ═══════════════════════════════════════════════════════
   9. 차트 (Chart.js)
   ══════════════════════════════════════════════════════ */

function renderChart(prices7d, bands) {
  const labels    = prices7d.map(p => fmtDate(p.date));
  const priceData = prices7d.map(p => p.price);
  const upperData = bands.map(b => b.high);
  const lowerData = bands.map(b => b.low);

  /* 이전 차트 파괴 */
  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }

  const ctx = DOM.priceChart.getContext('2d');

  /* 차트 색상 — CSS Custom Property에서 읽어 단일 소스 유지
   * (Canvas는 CSS 상속 불가, getComputedStyle로 동기화)             */
  const cssRoot    = getComputedStyle(document.documentElement);
  const cssVar     = name => cssRoot.getPropertyValue(name).trim();
  const brandColor = cssVar('--clr-accent');   /* oklch(52% 0.18 145) */
  const gridColor  = cssVar('--clr-border');   /* oklch(89% 0.010 220) */
  const tickColor  = cssVar('--clr-text-3');   /* oklch(60% 0.016 250) */
  /* 신뢰구간 반투명 밴드 — CSS --clr-accent-a14 토큰에서 읽음 */
  const bandFill      = cssVar('--clr-accent-a14');
  /* 툴팁 색상 — CSS --chart-tooltip-* 토큰에서 읽음 (JS 색상 0건 유지) */
  const ttBg          = cssVar('--chart-tooltip-bg');
  const ttTitle       = cssVar('--chart-tooltip-title');
  const ttBody        = cssVar('--chart-tooltip-body');
  const ttBorder      = cssVar('--chart-tooltip-border');

  chartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        /* [0] 신뢰구간 상단 — fill to dataset[1] */
        {
          label: '신뢰구간 상단',
          data: upperData,
          borderColor: 'transparent',
          backgroundColor: bandFill,
          fill: 1,             // fill to dataset index 1
          pointRadius: 0,
          tension: 0.4,
          order: 2,
        },
        /* [1] 신뢰구간 하단 */
        {
          label: '신뢰구간 하단',
          data: lowerData,
          borderColor: 'transparent',
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
          tension: 0.4,
          order: 2,
        },
        /* [2] 예측 가격 메인 라인 */
        {
          label: '예측 가격 (원/kg)',
          data: priceData,
          borderColor: brandColor,
          backgroundColor: brandColor,
          fill: false,
          tension: 0.4,
          pointRadius: 5,
          pointHoverRadius: 8,
          pointBorderColor: 'white',
          pointBorderWidth: 2,
          borderWidth: 2.5,
          order: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      plugins: {
        legend: {
          display: false,   // 커스텀 범례(HTML)를 씀
        },
        tooltip: {
          backgroundColor: ttBg,
          titleColor:   ttTitle,
          bodyColor:    ttBody,
          borderColor:  ttBorder,
          borderWidth: 1,
          padding: 12,
          cornerRadius: 8,
          callbacks: {
            title(ctx) {
              return ctx[0]?.label || '';
            },
            label(ctx) {
              if (ctx.dataset.label.startsWith('신뢰')) return null;  // CI 라인은 툴팁 숨김
              const v = ctx.parsed.y;
              return ` 예측가: ${Math.round(v).toLocaleString('ko-KR')}원/kg`;
            },
            afterBody(ctx) {
              const idx = ctx[0]?.dataIndex;
              if (idx == null) return [];
              // bands는 클로저로 접근
              const b = _currentBands?.[idx];
              if (!b) return [];
              return [
                ` 신뢰 하한: ${b.low.toLocaleString('ko-KR')}원`,
                ` 신뢰 상한: ${b.high.toLocaleString('ko-KR')}원`,
              ];
            },
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          border: { color: gridColor },
          ticks: {
            color: tickColor,
            font: { size: 11 },
          },
        },
        y: {
          grid: {
            color: gridColor,
            drawBorder: false,
          },
          border: { dash: [4, 4], color: 'transparent' },
          ticks: {
            color: tickColor,
            font: { size: 11 },
            callback: v => `${Math.round(v).toLocaleString('ko-KR')}원`,
          },
        },
      },
    },
  });
}

/* bands를 클로저로 툴팁 afterBody에서 접근 */
let _currentBands = null;

/* ═══════════════════════════════════════════════════════
   10. 테이블 렌더
   ══════════════════════════════════════════════════════ */

function renderTable(prices7d, bands, currentPrice) {
  _currentBands = bands;  // 차트 툴팁 클로저에도 공유
  DOM.tableBody.innerHTML = '';

  let prevPrice = currentPrice;

  prices7d.forEach((pt, i) => {
    const b    = bands[i];
    const diff = pt.price - prevPrice;
    const pct  = prevPrice ? (diff / prevPrice * 100) : 0;

    const tr = document.createElement('tr');

    /* 날짜 */
    const tdDate = document.createElement('td');
    tdDate.textContent = fmtDate(pt.date);

    /* 예측가 */
    const tdPrice = document.createElement('td');
    tdPrice.textContent = `${fmtPrice(pt.price)}원`;
    tdPrice.style.fontWeight = '600';

    /* 신뢰 하한 */
    const tdLow = document.createElement('td');
    tdLow.textContent = `${fmtPrice(b.low)}원`;
    tdLow.style.color = 'var(--clr-text-3)';

    /* 신뢰 상한 */
    const tdHigh = document.createElement('td');
    tdHigh.textContent = `${fmtPrice(b.high)}원`;
    tdHigh.style.color = 'var(--clr-text-3)';

    /* 전일比 */
    const tdDiff = document.createElement('td');
    if (Math.abs(pct) < 0.05) {
      tdDiff.textContent = '—';
      tdDiff.className = 'td-flat';
    } else if (diff > 0) {
      tdDiff.textContent = `▲ ${pct.toFixed(1)}%`;
      tdDiff.className = 'td-rise';
    } else {
      tdDiff.textContent = `▼ ${Math.abs(pct).toFixed(1)}%`;
      tdDiff.className = 'td-drop';
    }

    tr.appendChild(tdDate);
    tr.appendChild(tdPrice);
    tr.appendChild(tdLow);
    tr.appendChild(tdHigh);
    tr.appendChild(tdDiff);
    DOM.tableBody.appendChild(tr);

    prevPrice = pt.price;
  });
}

/* ═══════════════════════════════════════════════════════
   11. 품목 선택
   ══════════════════════════════════════════════════════ */

async function selectItem(itemName) {
  currentItem = itemName;
  updateActiveTab(itemName);
  setViewState('loading');

  try {
    const data = await fetchPrediction(itemName);
    renderPrediction(data);
    setViewState('content');
  } catch (err) {
    console.error('[selectItem]', err);
    DOM.errorDesc.textContent = err.message || '알 수 없는 오류가 발생했습니다.';
    setViewState('error');
  }
}

/* ═══════════════════════════════════════════════════════
   12. 초기화
   ══════════════════════════════════════════════════════ */

async function init() {
  /* 탭 로딩 */
  let fetchedItems = await fetchItems();
  items = fetchedItems && fetchedItems.length > 0 ? fetchedItems : DEFAULT_ITEMS;
  renderItemTabs(items);

  /* 기본 품목 즉시 로드 */
  await selectItem(currentItem);

  /* 이상가격 배너 닫기 */
  DOM.anomalyClose.addEventListener('click', () => {
    DOM.anomalyBanner.setAttribute('hidden', '');
  });

  /* 재시도 버튼 */
  DOM.retryBtn.addEventListener('click', () => {
    selectItem(currentItem);
  });
}

/* DOM 준비 후 시작 */
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
