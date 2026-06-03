#!/usr/bin/env node
/**
 * memo CLI shim — 确保 bun 存在后转发给 scripts/memo.ts
 *
 * 安装方式：npm install -g .  或  从 npm registry 安装
 * 使用方式：memo init ./docs / memo server start / 等等
 */
const { execFileSync } = require('child_process');
const { join, dirname } = require('path');

function findBun() {
  try {
    const which = process.platform === 'win32' ? 'where' : 'which';
    return execFileSync(which, ['bun'], { encoding: 'utf-8' }).trim();
  } catch {
    return null;
  }
}

function main() {
  const bun = findBun();
  if (!bun) {
    console.error('');
    console.error('  ❌ 需要 Bun 运行时才能执行 memo 命令');
    console.error('');
    console.error('  安装 Bun:');
    console.error('    curl -fsSL https://bun.sh/install | bash');
    console.error('');
    console.error('  或使用 npm:');
    console.error('    npm install -g bun');
    console.error('');
    process.exit(1);
  }

  // memo-shim.js 在 scripts/ 下，memo.ts 在同目录
  const scriptPath = join(__dirname, 'memo.ts');
  const args = process.argv.slice(2);

  try {
    execFileSync(bun, ['run', scriptPath, ...args], {
      stdio: 'inherit',
    });
  } catch (err) {
    // execFileSync 在子进程非零退出时会抛异常，我们直接透传退出码
    process.exit(err.status ?? 1);
  }
}

main();
