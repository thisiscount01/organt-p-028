/**
 * Node.js reverse-proxy for KAMIS AI service.
 * PORT를 즉시 listen — Render startup timeout 방지.
 * pip install / uvicorn은 비동기 spawn으로 기동.
 */
const http   = require('http');
const { spawn } = require('child_process');
const PORT   = parseInt(process.env.PORT || '3000', 10);
const PY_PORT = 8001;

let pythonReady = false;

/* ── 1. PORT 즉시 오픈 ── */
const proxyServer = http.createServer((req, res) => {
  if (!pythonReady) {
    const isHtmlReq = (req.headers['accept'] || '').includes('text/html');
    res.writeHead(200, { 'Content-Type': isHtmlReq ? 'text/html; charset=utf-8' : 'application/json' });
    if (isHtmlReq) {
      res.end(`<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>장보기 어드바이저 — 준비 중</title>
<style>body{font-family:sans-serif;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;background:#f0faf4;}
.box{text-align:center;padding:2rem;}.logo{font-size:3rem;}.title{font-size:1.4rem;font-weight:700;color:#1a9e5c;margin:.5rem 0;}
.sub{color:#666;margin-bottom:1.5rem;}.bar{width:200px;height:6px;background:#e0e0e0;border-radius:3px;margin:0 auto;overflow:hidden;}
.bar-fill{height:100%;background:#1a9e5c;border-radius:3px;animation:load 2s ease-in-out infinite;}
@keyframes load{0%{width:0}60%{width:80%}100%{width:100%}}</style>
<script>setTimeout(()=>location.reload(),6000)</script>
</head><body><div class="box">
<div class="logo">🌾</div>
<div class="title">장보기 어드바이저</div>
<div class="sub">AI 모델 로딩 중입니다… 잠시만 기다려 주세요.<br>(약 1~2분 소요)</div>
<div class="bar"><div class="bar-fill"></div></div>
</div></body></html>`);
    } else {
      res.end(JSON.stringify({ status: 'loading', message: '서버 준비 중' }));
    }
    return;
  }

  const options = {
    host: '127.0.0.1',
    port: PY_PORT,
    path: req.url,
    method: req.method,
    headers: req.headers,
  };
  const proxy = http.request(options, pr => {
    res.writeHead(pr.statusCode, pr.headers);
    pr.pipe(res);
  });
  proxy.on('error', e => {
    res.writeHead(502);
    res.end('Gateway error: ' + e.message);
  });
  req.pipe(proxy);
});

proxyServer.listen(PORT, () => console.log('Proxy listening on port', PORT));

/* ── 2. pip install (비동기 spawn) ── */
function installDeps(cb) {
  console.log('pip install 시작…');
  const pip = spawn('pip', [
    'install',
    'fastapi>=0.110',
    'uvicorn[standard]>=0.29',
    'python-multipart>=0.0.9',
    'pandas>=2.0',
    'numpy>=1.24',
    'prophet>=1.1',
    'lightgbm>=4.0',
    'scikit-learn>=1.3',
    '--quiet',
  ], { stdio: 'inherit' });

  pip.on('error', e => { console.warn('pip error:', e.message); cb(); });
  pip.on('exit', code => {
    if (code !== 0) console.warn('pip exit code:', code);
    else console.log('pip install 완료');
    cb();
  });
}

/* ── 3. uvicorn 기동 ── */
function startUvicorn() {
  console.log('uvicorn 기동 중…');
  const py = spawn('uvicorn', [
    'server:app', '--host', '127.0.0.1', '--port', String(PY_PORT),
  ], { stdio: 'inherit' });

  py.on('error', e => console.error('uvicorn error:', e));
  py.on('exit', (code, signal) => {
    pythonReady = false;
    console.error('uvicorn 종료: code=' + code + ' signal=' + signal);
  });

  waitPy(90, () => {
    pythonReady = true;
    console.log('Python ready — 요청 전달 시작');
  });
}

/* ── 4. /health 폴링 (3초×90회=4.5분) ── */
function waitPy(n, cb) {
  http.get('http://127.0.0.1:' + PY_PORT + '/health', r => {
    if (r.statusCode === 200) { console.log('Python 준비 완료'); cb(); }
    else if (n > 0) { setTimeout(() => waitPy(n - 1, cb), 3000); }
    else { console.warn('Python 미준비 — 강제 시작'); cb(); }
    r.resume();
  }).on('error', () => {
    if (n > 0) { setTimeout(() => waitPy(n - 1, cb), 3000); }
    else { console.warn('Python 미응답 — 강제 시작'); cb(); }
  });
}

installDeps(startUvicorn);
