// CDP client for the CANNJudge CLI.
//
// 浏览器里已经登录了 cannjudge.cn，所以 CLI 不存任何 token/密码：
// 它通过本机 CDP daemon（默认 127.0.0.1:9400）在**页面上下文**里跑 fetch，
// 复用浏览器已有的 HttpOnly 会话。
//
// 注意：Chrome 会冻结后台标签页（timer / 网络回调全停），此时 Runtime.evaluate
// 会挂住。这里用「截一帧图」唤醒渲染进程后重试。
import { spawn } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';

const DAEMON = process.env.CANNJUDGE_DAEMON || 'http://127.0.0.1:9400';
const MATCH = process.env.CANNJUDGE_PAGE_MATCH || 'cannjudge';
const WAKE_FILE = path.join(tmpdir(), 'cannjudge-wake.png');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function post(pathname, body) {
  try {
    const res = await fetch(DAEMON + pathname, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body)
    });
    return await res.json();
  } catch (e) {
    return { error: String(e && e.message ? e.message : e) };
  }
}

export async function ping() {
  try {
    const res = await fetch(DAEMON + '/ping', { signal: AbortSignal.timeout(1500) });
    const j = await res.json();
    return { up: true, browser: !!j.connected };
  } catch {
    return { up: false, browser: false };
  }
}

export async function ensureDaemon() {
  if ((await ping()).up) return true;
  const script = path.join(import.meta.dirname, 'cdp-daemon.mjs');
  const child = spawn(process.execPath, [script], {
    detached: true,
    stdio: 'ignore',
    windowsHide: true,
    env: { ...process.env, CDP_DAEMON_PORT: new URL(DAEMON).port || '9400' }
  });
  child.unref();
  for (let i = 0; i < 40; i++) {
    await sleep(500);
    if ((await ping()).up) return true;
  }
  throw new Error('CDP daemon 启动失败：' + script);
}

export async function wakePage() {
  await post('/shot', { match: MATCH, file: WAKE_FILE });
}

export async function pages() {
  await ensureDaemon();
  const r = await post('/pages', {});
  return r.pages || [];
}

// 在 cannjudge 页面里执行表达式，返回解析后的 JSON
export async function evalPage(expression) {
  await ensureDaemon();
  let r = await post('/eval', { match: MATCH, expression });
  if (typeof r.result !== 'string') {
    await wakePage();
    await sleep(1200);
    r = await post('/eval', { match: MATCH, expression });
  }
  if (r.error) {
    if (r.error === 'no matching page') {
      throw new Error('没找到 cannjudge 页面：请先在 Edge 里打开 https://cannjudge.cn');
    }
    throw new Error(r.error);
  }
  if (r.exception) {
    throw new Error('页面脚本异常：' + JSON.stringify(r.exception).slice(0, 400));
  }
  if (typeof r.result !== 'string') {
    throw new Error('页面无响应（后台标签页被冻结），请把 cannjudge 标签页切到前台后重试');
  }
  return JSON.parse(r.result);
}

// 打开一个新的 cannjudge 标签页（会切到前台）
export async function openPage(url = 'https://cannjudge.cn') {
  await ensureDaemon();
  const r = await post('/open', { url });
  if (r.error) throw new Error(r.error);
  return r.targetId;
}
