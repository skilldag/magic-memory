#!/usr/bin/env bun
/**
 * magic-memory CLI — 知识图谱项目管理工具
 *
 * Usage:
 *   magic-memory init <path>     扫描目录构建图谱并注册到全局服务
 *   magic-memory list            列出已注册项目
 *   magic-memory remove <id>     删除项目
 *   magic-memory server start    启动全局服务
 *   magic-memory server stop     停止全局服务
 *   magic-memory server status   查看全局服务状态
 */

import { buildGraphFromDir, registerProject, removeProject, listProjects, saveGraph } from '../server/graphBuilder';
import { join, dirname } from 'path';
import { homedir } from 'os';
import { existsSync, readFileSync, writeFileSync, unlinkSync, openSync, readdirSync } from 'fs';

const GLOBAL_SERVICE_PORT = 4321;
const GLOBAL_SERVICE_URL = `http://localhost:${GLOBAL_SERVICE_PORT}`;
const STATE_FILE = join(homedir(), '.magic-memory', 'state.json');
const MEMORY_DIR = dirname(STATE_FILE);
const FRONTEND_PORT = 3000;

function loadState(): { gs?: number; frontend?: number } {
  try { return JSON.parse(readFileSync(STATE_FILE, 'utf-8')); } catch { return {}; }
}

function saveState(state: { gs?: number; frontend?: number }) {
  writeFileSync(STATE_FILE, JSON.stringify(state));
}

function isRunning(pid: number): boolean {
  try { process.kill(pid, 0); return true; } catch { return false; }
}

function log(msg: string) {
  console.log(`  ${msg}`);
}

function error(msg: string) {
  console.error(`  ERROR: ${msg}`);
}

async function cmdInit(sourceDir: string) {
  const resolvedDir = sourceDir.startsWith('/') ? sourceDir : join(process.cwd(), sourceDir);
  log(`构建知识图谱: ${resolvedDir}`);

  const result = await buildGraphFromDir(resolvedDir, (msg) => log(msg));

  const name = resolvedDir.split('/').pop() || 'untitled';
  const projectId = `proj_${Date.now()}`;

  log(`\n注册项目到全局服务...`);

  const resp = await fetch(`${GLOBAL_SERVICE_URL}/api/projects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id: projectId,
      name,
      sourceDir: resolvedDir,
      concepts: result.concepts,
      edges: result.edges,
    }),
  });

  if (!resp.ok) {
    const text = await resp.text();
    error(`注册失败 (${resp.status}): ${text}`);
    process.exit(1);
  }

  const data = await resp.json();
  log(`✓ 项目已注册: ${data.id}`);
  log(`  名称: ${name}`);
  log(`  概念: ${result.concepts.length}`);
  log(`  边: ${result.edges.length}`);
  log(`\n启动 Web UI 后可在界面上看到此项目`);
}

async function cmdInitRepo(sourceDir: string) {
  const resolvedDir = sourceDir.startsWith('/') ? sourceDir : join(process.cwd(), sourceDir);

  if (!existsSync(join(resolvedDir, '.git'))) {
    error(`不是 Git 仓库: ${resolvedDir}`);
    error('请确保路径是一个 Git 仓库（包含 .git 目录）');
    process.exit(1);
  }

  const name = resolvedDir.split('/').pop() || 'untitled';
  const projectId = `proj_${Date.now()}`;

  log(`注册仓库项目: ${name}`);
  log(`  路径: ${resolvedDir}`);

  const { getProjectDir, registerProject } = await import('../server/graphBuilder');
  const { join: joinPath } = await import('path');
  const { mkdirSync } = await import('fs');
  const projectDir = getProjectDir(projectId);
  if (!existsSync(projectDir)) {
    mkdirSync(projectDir, { recursive: true });
  }
  const docsDir = joinPath(projectDir, 'docs');
  if (!existsSync(docsDir)) {
    mkdirSync(docsDir, { recursive: true });
  }

  await registerProject({
    id: projectId,
    name,
    sourceDir: resolvedDir,
    sourceType: 'repo',
    createdAt: new Date().toISOString(),
    conceptCount: 0,
    edgeCount: 0,
  });

  log(`✓ 项目已注册: ${projectId}`);
  log(`  名称: ${name}`);
  log(`  类型: repo`);
  log(`\n启动 Web UI 后可在界面上看到此项目`);
  log(`使用 memo add-concept ${projectId} --name "概念名" --file "path/to/file" --lines 42-85 添加概念`);
}

interface AddConceptArgs {
  name: string;
  file: string;
  lines?: string;
  desc?: string;
  doc?: string;
  dependsOn?: string[];
}

function parseAddConceptArgs(): AddConceptArgs {
  const args = process.argv.slice(2);
  const result: AddConceptArgs = { name: '', file: '' };
  for (let i = 2; i < args.length; i++) {
    switch (args[i]) {
      case '--name': result.name = args[++i] || ''; break;
      case '--file': result.file = args[++i] || ''; break;
      case '--lines': result.lines = args[++i] || ''; break;
      case '--desc': result.desc = args[++i] || ''; break;
      case '--doc': result.doc = args[++i] || ''; break;
      case '--depends-on': {
        if (!result.dependsOn) result.dependsOn = [];
        result.dependsOn.push(args[++i] || '');
        break;
      }
    }
  }
  return result;
}

async function cmdAddConcept(projectId: string) {
  const opts = parseAddConceptArgs();
  if (!opts.name) { error('--name 是必填参数'); process.exit(1); }
  if (!opts.file) { error('--file 是必填参数'); process.exit(1); }

  const { loadGraph, saveGraph, registerProject, listProjects, getProjectDir } = await import('../server/graphBuilder');
  const { join: joinPath } = await import('path');
  const { existsSync, readFileSync, writeFileSync } = await import('fs');

  const graph = await loadGraph(projectId);
  if (!graph) {
    error(`项目 ${projectId} 不存在`);
    process.exit(1);
  }

  const projects = await listProjects();
  const project = projects.find(p => p.id === projectId);
  if (!project) { error(`项目 ${projectId} 未找到`); process.exit(1); }

  const codeFilePath = joinPath(project.sourceDir, opts.file);
  if (!existsSync(codeFilePath)) {
    error(`文件不存在: ${codeFilePath}`);
    process.exit(1);
  }

  let lineStart: number | undefined;
  let lineEnd: number | undefined;
  if (opts.lines) {
    const parts = opts.lines.split('-');
    lineStart = parseInt(parts[0], 10);
    if (parts.length > 1) lineEnd = parseInt(parts[1], 10);
  }

  const allLines = readFileSync(codeFilePath, 'utf-8').split('\n');
  const snippetStart = lineStart ? Math.max(0, lineStart - 1) : 0;
  const snippetEnd = lineEnd ? Math.min(lineEnd, snippetStart + 20) : Math.min(snippetStart + 20, allLines.length);
  const snippet = allLines.slice(snippetStart, snippetEnd).join('\n');

  const conceptId = opts.name.toLowerCase().replace(/[\s\/]+/g, '-');

  if (graph.concepts.some(c => c.id === conceptId)) {
    error(`概念 "${opts.name}" 已存在 (ID: ${conceptId})`);
    process.exit(1);
  }

  const newConcept = {
    id: conceptId,
    title: opts.name,
    level: 1,
    category: '',
    problem: '',
    gap_anticipate: '',
    depends_on: opts.dependsOn || [],
    leads_to: [],
    related: [],
    codeRefs: [{
      file: opts.file,
      lineStart,
      lineEnd,
      snippet,
      description: opts.desc || '',
    }],
    tags: [],
    path: joinPath('docs', conceptId + '.md'),
    lastModified: new Date().toISOString(),
  };

  graph.concepts.push(newConcept);

  await saveGraph(projectId, graph.concepts, graph.edges);

  await registerProject({ ...project, conceptCount: graph.concepts.length, edgeCount: graph.edges.length });


  if (opts.doc) {
    const projectDir = getProjectDir(projectId);
    const docPath = joinPath(projectDir, 'docs', conceptId + '.md');
    writeFileSync(docPath, opts.doc, 'utf-8');
    log(`文档已创建: ${docPath}`);
  }

  log(`✓ 概念已添加: ${opts.name} (${conceptId})`);
  log(`  代码引用: ${opts.file}${opts.lines ? ` L${opts.lines}` : ''}`);
  log(`  概念总数: ${graph.concepts.length}`);
}

async function cmdList() {
  const resp = await fetch(`${GLOBAL_SERVICE_URL}/api/projects`);
  if (!resp.ok) {
    error(`无法连接全局服务 (${resp.status})`);
    error('请先运行: magic-memory server start');
    process.exit(1);
  }
  const data = await resp.json();
  const projects = data.projects || [];

  console.log(`\n  已注册项目 (${projects.length}):\n`);
  for (const p of projects) {
    console.log(`  ${p.id.padEnd(30)} ${p.name.padEnd(20)} ${p.conceptCount} 概念, ${p.edgeCount} 边`);
  }
  if (projects.length === 0) {
    console.log('  (无)');
  }
  console.log('');
}

async function cmdRemove(projectId: string) {
  const resp = await fetch(`${GLOBAL_SERVICE_URL}/api/projects/${projectId}`, {
    method: 'DELETE',
  });

  if (!resp.ok) {
    const text = await resp.text();
    error(`删除失败 (${resp.status}): ${text}`);
    process.exit(1);
  }
  log(`✓ 项目已删除: ${projectId}`);
}

function killProjectVites() {
  const projectDir = join(import.meta.dir, '..');
  const result = Bun.spawnSync(['ps', 'aux'], { stdio: ['ignore', 'pipe', 'pipe'] });
  const stdout = result.stdout.toString();
  const lines = stdout.split('\n');
  for (const line of lines) {
    if (!line.includes('vite')) continue;
    if (!line.includes(projectDir)) continue;
    if (line.includes('grep')) continue;
    const parts = line.trim().split(/\s+/);
    const pid = parseInt(parts[1]);
    if (isNaN(pid)) continue;
    try { process.kill(pid, 'SIGTERM'); log(`停止旧 Vite (PID: ${pid})`); } catch {}
  }
  for (let i = 0; i < 15; i++) {
    const check = Bun.spawnSync(['lsof', '-i', `:${FRONTEND_PORT}`, '-P', '-n'], { stdio: ['ignore', 'pipe', 'pipe'] });
    if (!check.stdout.toString().includes('LISTEN')) break;
    Bun.sleepSync(200);
  }
}

function cmdServerStart() {
  const state = loadState();
  if (state.gs && isRunning(state.gs)) {
    error(`服务已在运行中 (GS PID: ${state.gs})`);
    process.exit(1);
  }

  const serverPath = join(import.meta.dir, '..', 'server', 'explore.ts');
  const projectDir = join(import.meta.dir, '..');

  log(`启动全局服务...`);
  const gs = Bun.spawn(['bun', 'run', serverPath], {
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env as any },
    detached: true,
  });
  gs.unref();

  log(`启动前端页面 (端口 ${FRONTEND_PORT})...`);
  killProjectVites();
  const devNull = openSync('/dev/null', 'w');
  const frontend = Bun.spawn(['node', join(projectDir, 'node_modules', 'vite', 'bin', 'vite.js'), '--port', String(FRONTEND_PORT)], {
    cwd: projectDir,
    detached: true,
    stdio: ['ignore', devNull, devNull],
  });
  frontend.unref();

  const newState = { gs: gs.pid, frontend: frontend.pid };
  saveState(newState);
  log(`✓ 全局服务已启动 (PID: ${gs.pid})`);
  log(`✓ 前端已启动 (PID: ${frontend.pid})`);
  log(`  API:  http://localhost:${GLOBAL_SERVICE_PORT}`);
  log(`  Web:  http://localhost:${FRONTEND_PORT}`);
}

function cmdServerStop() {
  const state = loadState();
  if (!state.gs && !state.frontend) {
    error('服务未在运行');
    process.exit(1);
  }

  let stopped = false;
  if (state.frontend && isRunning(state.frontend)) {
    try { process.kill(state.frontend, 'SIGTERM'); log(`✓ 前端已停止 (PID: ${state.frontend})`); stopped = true; } catch {}
  }
  if (state.gs && isRunning(state.gs)) {
    try { process.kill(state.gs, 'SIGTERM'); log(`✓ 全局服务已停止 (PID: ${state.gs})`); stopped = true; } catch {}
  }
  try { unlinkSync(STATE_FILE); } catch {}

  if (!stopped) error('没有运行中的服务 (已清理状态文件)');
}

function cmdServerRestart() {
  cmdServerStop();
  cmdServerStart();
}

function cmdServerStatus() {
  const state = loadState();
  const gsRunning = state.gs ? isRunning(state.gs) : false;
  const feRunning = state.frontend ? isRunning(state.frontend) : false;

  if (!gsRunning && !feRunning) {
    console.log('  状态: 未运行');
    return;
  }
  if (gsRunning) console.log(`  ● 全局服务: 运行中 (PID: ${state.gs})`);
  else console.log(`  ○ 全局服务: 已停止`);
  if (feRunning) console.log(`  ● 前端页面: 运行中 (PID: ${state.frontend})`);
  else console.log(`  ○ 前端页面: 已停止`);
  if (gsRunning) console.log(`  API: http://localhost:${GLOBAL_SERVICE_PORT}`);
  if (feRunning) console.log(`  Web: http://localhost:${FRONTEND_PORT}`);
}

function printHelp() {
  console.log(`
  memo — 知识图谱项目管理

  用法:
    init <path>             扫描目录构建图谱并注册到全局服务
    init-repo <path>        注册 Git 仓库项目
    add-concept <id>        从代码位置添加概念
    list                    列出已注册项目
    remove <id>             删除项目
    server start     启动全局服务 (端口 ${GLOBAL_SERVICE_PORT})
    server stop      停止全局服务
    server restart   重启全局服务
    server status    查看全局服务状态
`);
}

async function main() {
  const args = process.argv.slice(2);
  const cmd = args[0];

  if (args.length === 0 || cmd === '-h' || cmd === '--help' || cmd === 'help') {
    printHelp();
    return;
  }

  switch (cmd) {
    case 'init':
      if (!args[1]) { error('请指定文档目录路径'); process.exit(1); }
      await cmdInit(args[1]);
      break;

    case 'init-repo':
      if (!args[1]) { error('请指定仓库路径'); process.exit(1); }
      await cmdInitRepo(args[1]);
      break;

    case 'add-concept':
      if (!args[1]) { error('请指定项目 ID'); process.exit(1); }
      await cmdAddConcept(args[1]);
      break;

    case 'list':
      await cmdList();
      break;

    case 'remove':
      if (!args[1]) { error('请指定项目 ID'); process.exit(1); }
      await cmdRemove(args[1]);
      break;

    case 'server':
      if (!args[1] || args[1] === '-h' || args[1] === '--help' || args[1] === 'help') {
        console.log(`
  memo server — 管理全局服务

  用法:
    start      启动全局服务 (端口 ${GLOBAL_SERVICE_PORT})
    stop       停止全局服务
    restart    重启全局服务
    status     查看服务状态
`);
        return;
      }
      switch (args[1]) {
        case 'start': cmdServerStart(); break;
        case 'stop': cmdServerStop(); break;
        case 'restart': cmdServerRestart(); break;
        case 'status': cmdServerStatus(); break;
        default: error('未知命令: server ' + (args[1] || ''));
      }
      break;

    default:
      error(`未知命令: ${cmd}`);
      process.exit(1);
  }
}

main().catch((e) => {
  error(e.message);
  process.exit(1);
});
