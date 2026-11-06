#!/usr/bin/env node
// cannjudge — CANN Judge 命令行（复用浏览器已登录会话，不存 token）
//
//   node tools/cannjudge.mjs whoami
//   node tools/cannjudge.mjs contests
//   node tools/cannjudge.mjs problems vector
//   node tools/cannjudge.mjs fetch vector/topkpqdistance problems/vector/topkpqdistance
//   node tools/cannjudge.mjs submit problems/vector/topkpqdistance --watch
//   node tools/cannjudge.mjs status <submissionId>
//   node tools/cannjudge.mjs open
import { readFileSync, writeFileSync, mkdirSync, readdirSync, existsSync, statSync } from 'node:fs';
import path from 'node:path';
import { evalPage, openPage, ping } from './cdp.mjs';

// ---------------------------------------------------------------------------
// 页面内工具函数（每个命令拼在表达式前面）
// ---------------------------------------------------------------------------
const PRELUDE = `
const J = async (u, o) => { const r = await fetch(u, Object.assign({ credentials: 'include' }, o || {})); const t = await r.text(); try { return JSON.parse(t); } catch (e) { return { __raw: t }; } };
const D = (x) => (x && typeof x === 'object' && 'data' in x) ? x.data : x;
const L = (x) => Array.isArray(x) ? x : ((x && (x.items || x.list || x.rows)) || []);
const me = () => { try { return JSON.parse(localStorage.getItem('cannjudge_user') || 'null'); } catch (e) { return null; } };
const publicGroup = async () => D(await J('/api/groups/public'));
const allContests = async () => { const g = await publicGroup(); return L(D(await J('/api/contests/group/' + g._id))); };
const findContest = async (tok) => (await allContests()).find(function (c) { return (c.token || c.slug || c.name) === tok; }) || null;
const problemsOf = async (cid) => L(D(await J('/api/problems/contest/' + cid)));
const findProblem = async (ctok, ptok) => {
  const c = await findContest(ctok); if (!c) return { error: '没找到 contest: ' + ctok };
  const p = (await problemsOf(c._id)).find(function (x) { return (x.token || x.slug || x.name) === ptok; });
  if (!p) return { error: '没找到题目: ' + ptok };
  return { contest: c, problem: p };
};
`;

const run = (body) => evalPage(`(async () => { ${PRELUDE}\n${body} })()`);

const fmt = (ms) => (ms == null ? '-' : ms >= 1000 ? (ms / 1000).toFixed(1) + ' ms' : ms.toFixed(1) + ' µs');

// ---------------------------------------------------------------------------
// 命令
// ---------------------------------------------------------------------------
async function whoami() {
  const out = await run(`
    const u = me();
    if (!u) return JSON.stringify({ error: 'localStorage 里没有登录信息，请先在浏览器登录 cannjudge.cn' });
    return JSON.stringify({ id: u._id, ID: u.ID, nickname: u.nickname, admin: !!u.admin });
  `);
  if (out.error) throw new Error(out.error);
  console.log(`已登录：${out.nickname}（ID ${out.ID}${out.admin ? '，admin' : ''}）`);
  console.log(`user._id = ${out.id}`);
}

async function contests() {
  const list = await run(`
    const cs = await allContests();
    return JSON.stringify(cs.map(c => ({ token: c.token || c.slug || c.name, title: c.title || c.name, id: c._id })));
  `);
  console.log('token'.padEnd(38) + 'title');
  console.log('-'.repeat(78));
  for (const c of list) console.log(String(c.token).padEnd(38) + (c.title || ''));
}

async function problems(contestToken) {
  if (!contestToken) throw new Error('用法：cannjudge problems <contestToken>');
  const out = await run(`
    const c = await findContest(${JSON.stringify(contestToken)});
    if (!c) return JSON.stringify({ error: '没找到 contest: ' + ${JSON.stringify(contestToken)} });
    const ps = await problemsOf(c._id);
    return JSON.stringify({ title: c.title, items: ps.map(p => ({ token: p.token || p.slug || p.name, title: p.title, id: p._id, template: p.code_template })) });
  `);
  if (out.error) throw new Error(out.error);
  console.log(`${out.title}\n`);
  console.log('token'.padEnd(34) + 'title'.padEnd(32) + 'problemId');
  console.log('-'.repeat(100));
  for (const p of out.items) console.log(String(p.token).padEnd(34) + String(p.title).padEnd(32) + p.id);
}

async function fetchProblem(spec, dir) {
  if (!spec || !spec.includes('/')) throw new Error('用法：cannjudge fetch <contest>/<problem> [dir]');
  const [ctok, ptok] = spec.split('/');
  const out = await run(`
    const f = await findProblem(${JSON.stringify(ctok)}, ${JSON.stringify(ptok)});
    if (f.error) return JSON.stringify(f);
    const u = me();
    const tpl = D(await J('/api/problems/' + f.problem._id + '/template?userId=' + (u ? u.ID : '')));
    return JSON.stringify({
      contest: { token: ${JSON.stringify(ctok)}, title: f.contest.title, id: f.contest._id },
      problem: { token: ${JSON.stringify(ptok)}, title: f.problem.title, id: f.problem._id, template: f.problem.code_template },
      files: (tpl && tpl.files) || []
    });
  `);
  if (out.error) throw new Error(out.error);

  const target = dir || path.join('problems', ctok, ptok);
  mkdirSync(target, { recursive: true });
  for (const f of out.files) {
    const p = path.join(target, 'ref', f.path);
    mkdirSync(path.dirname(p), { recursive: true });
    writeFileSync(p, f.content ?? '', 'utf8');
    const editable = f.editable !== false;
    // 可编辑文件同时放一份到题目根目录，形成"可直接提交"的骨架
    if (editable) writeFileSync(path.join(target, f.path), f.content ?? '', 'utf8');
    console.log(`${editable ? '  *' : '   '} ${f.path}${editable ? '  （可编辑，已放到题目根目录）' : ''}`);
  }
  const meta = {
    contest: out.contest.token,
    contestId: out.contest.id,
    contestTitle: out.contest.title,
    problem: out.problem.token,
    problemId: out.problem.id,
    problemTitle: out.problem.title,
    codeTemplate: out.problem.template,
    fetchedAt: new Date().toISOString()
  };
  writeFileSync(path.join(target, 'meta.json'), JSON.stringify(meta, null, 2) + '\n', 'utf8');
  console.log(`\n已生成 ${target}/（meta.json 记录 problemId，供 submit 使用）`);
}

function collectSources(dir) {
  const protectedNames = new Set(['judge.asc', 'data_utils.h', 'main.asc']);
  const files = [];
  for (const name of readdirSync(dir)) {
    const full = path.join(dir, name);
    if (!statSync(full).isFile()) continue;
    if (!/\.(asc|h)$/i.test(name)) continue;
    if (protectedNames.has(name)) continue;
    files.push({ path: name, content: readFileSync(full, 'utf8') });
  }
  files.sort((a, b) => (a.path === 'kernel.asc' ? -1 : b.path === 'kernel.asc' ? 1 : a.path.localeCompare(b.path)));
  return files;
}

async function submit(dir, watch) {
  const metaPath = path.join(dir, 'meta.json');
  if (!existsSync(metaPath)) throw new Error(`${dir}/meta.json 不存在，先跑 cannjudge fetch`);
  const meta = JSON.parse(readFileSync(metaPath, 'utf8'));
  const files = collectSources(dir);
  if (!files.length) throw new Error(`${dir} 里没有 .asc/.h 源文件`);
  console.log(`提交 ${meta.contest}/${meta.problem}（${files.length} 个文件：${files.map((f) => f.path).join(', ')}）`);

  const out = await run(`
    const u = me();
    if (!u) return JSON.stringify({ error: '未登录' });
    const r = await fetch('/api/submissions/submit', {
      method: 'POST',
      credentials: 'include',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        problemId: ${JSON.stringify(meta.problemId)},
        files: ${JSON.stringify(files)},
        tiling_h: '', tiling_key_h: '', host_cpp: '', kernel_cpp: '',
        userId: u._id
      })
    });
    const t = await r.text();
    return JSON.stringify({ status: r.status, body: t });
  `);
  if (out.status !== 200) {
    let msg = out.body;
    try { const j = JSON.parse(out.body); msg = j.msg || j.message || out.body; } catch {}
    throw new Error(`提交失败（HTTP ${out.status}）：${msg}`);
  }
  const submissionId = JSON.parse(out.body).data.submissionId;
  console.log(`submissionId = ${submissionId}`);
  if (watch) await watchSubmission(submissionId);
  return submissionId;
}

async function watchSubmission(id) {
  process.stdout.write('评测中');
  for (let i = 0; i < 200; i++) {
    const d = await run(`
      const u = me();
      const r = await J('/api/submissions/${id}?userId=' + (u ? u._id : '') + '&_=' + Date.now());
      return JSON.stringify(D(r));
    `);
    const st = d && d.status;
    if (st && !/wait|pending|running|queue|judg/i.test(st)) {
      process.stdout.write('\n');
      printResult(d);
      return d;
    }
    process.stdout.write('.');
    await new Promise((r) => setTimeout(r, 6000));
  }
  process.stdout.write('\n');
  throw new Error('等待超时');
}

function printResult(d) {
  console.log(`\n状态：${d.status}`);
  const rows = d.result || [];
  if (rows.length) {
    console.log('用例  状态            耗时        最好成绩');
    console.log('-'.repeat(52));
    let total = 0;
    rows.forEach((x, i) => {
      total += x.time || 0;
      console.log(
        String(i + 1).padStart(3) + '  ' +
        String(x.testcase_status).padEnd(15) +
        fmt(x.time).padStart(10) + '  ' +
        fmt(x.best_time).padStart(10)
      );
      if (x.msg) console.log('     ↳ ' + String(x.msg).split('\n').slice(0, 8).join('\n       '));
    });
    console.log('-'.repeat(52));
    console.log(`合计 ${fmt(total)}`);
  }
}

async function status(id) {
  const d = await run(`
    const u = me();
    const r = await J('/api/submissions/${id}?userId=' + (u ? u._id : '') + '&_=' + Date.now());
    return JSON.stringify(D(r));
  `);
  printResult(d);
}

// ---------------------------------------------------------------------------
const [, , cmd, ...args] = process.argv;

const help = `cannjudge — CANN Judge 命令行

  whoami                          看当前登录用户
  contests                        列出公共题库的比赛/题目集
  problems <contest>              列出某比赛下的题目（含 problemId）
  fetch <contest>/<problem> [dir] 拉模板到目录（ref/ 存原始模板，可编辑文件放根目录）
  submit <dir> [--watch]          提交该目录下的 .asc/.h 并可选等待评测
  status <submissionId>           查一次评测结果
  open                            在 Edge 里打开 cannjudge.cn

环境变量：
  CANNJUDGE_DAEMON   默认 http://127.0.0.1:9400
`;

try {
  switch (cmd) {
    case 'whoami': await whoami(); break;
    case 'contests': await contests(); break;
    case 'problems': await problems(args[0]); break;
    case 'fetch': await fetchProblem(args[0], args[1]); break;
    case 'submit': await submit(args[0] || '.', args.includes('--watch')); break;
    case 'status': await status(args[0]); break;
    case 'watch': await watchSubmission(args[0]); break;
    case 'open': { const id = await openPage(); console.log('已打开 targetId=' + id); break; }
    case 'ping': { const p = await ping(); console.log(JSON.stringify(p)); break; }
    default: console.log(help);
  }
} catch (e) {
  console.error('✗ ' + (e && e.message ? e.message : e));
  process.exitCode = 1;
}
