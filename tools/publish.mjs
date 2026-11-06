#!/usr/bin/env node
// 把当前仓库的改动提交并推到 GitHub。
//
//   node tools/publish.mjs -m "feat(topkpqdistance): 单核 6 cycle/element"
//   node tools/publish.mjs -m "..." --no-push
import { execFileSync } from 'node:child_process';

const args = process.argv.slice(2);
const mi = args.indexOf('-m');
const message = mi >= 0 ? args[mi + 1] : null;
const noPush = args.includes('--no-push');

const git = (...a) => execFileSync('git', a, { stdio: 'inherit' });
const gitOut = (...a) => execFileSync('git', a, { encoding: 'utf8' }).trim();

if (!message) {
  console.error('用法：node tools/publish.mjs -m "<commit message>" [--no-push]');
  process.exit(2);
}

try {
  git('add', '-A');
  try {
    git('commit', '-m', message);
  } catch {
    console.log('没有需要提交的改动');
  }
  const branch = gitOut('rev-parse', '--abbrev-ref', 'HEAD');
  const remotes = gitOut('remote');
  if (!remotes.includes('origin')) {
    console.log('\n还没配远端。先在 GitHub 建一个空仓库，然后执行：');
    console.log('  git remote add origin git@github.com:<user>/<repo>.git');
    console.log(`  git push -u origin ${branch}`);
    process.exit(0);
  }
  if (!noPush) git('push', 'origin', branch);
  console.log(`\n✓ ${branch} 已同步（${gitOut('rev-parse', '--short', 'HEAD')}）`);
} catch (e) {
  console.error('✗ ' + (e && e.message ? e.message : e));
  process.exit(1);
}
