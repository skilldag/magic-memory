import { readdir, readFile, writeFile, rm } from 'fs/promises';
import { join } from 'path';
import { existsSync, mkdirSync } from 'fs';
import { homedir } from 'os';

interface ParsedFrontmatter {
  id?: string;
  title?: string;
  alias?: string[];
  level?: number;
  category?: string;
  problem?: string;
  gap_anticipate?: string;
  depends_on?: string[];
  leads_to?: string[];
  related?: string[];
  elements?: { name: string; description: string; type: string; order: number }[];
  tags?: string[];
}

export interface Concept {
  id: string;
  title: string;
  alias?: string[];
  level: number;
  category: string;
  problem?: string;
  gap_anticipate?: string;
  depends_on: string[];
  leads_to: string[];
  related: string[];
  path: string;
  tags: string[];
  lastModified: string;
  metadata?: { status?: string };
}

export interface ConceptEdge {
  id: string;
  source: string;
  target: string;
  type: 'depends_on' | 'leads_to' | 'related';
}

export interface BuildResult {
  concepts: Concept[];
  edges: ConceptEdge[];
}

export interface ProjectMeta {
  id: string;
  name: string;
  sourceDir: string;
  createdAt: string;
  conceptCount: number;
  edgeCount: number;
}

const MAGIC_MEMORY_DIR = join(homedir(), '.magic-memory');
const PROJECTS_DIR = join(MAGIC_MEMORY_DIR, 'projects');
const PROJECT_LIST_FILE = join(PROJECTS_DIR, 'project-list.json');

function ensureDir(dir: string) {
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
}

async function scanMdFiles(dirPath: string): Promise<string[]> {
  const results: string[] = [];
  const entries = await readdir(dirPath, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    const fullPath = join(dirPath, entry.name);
    if (entry.isDirectory()) {
      results.push(...(await scanMdFiles(fullPath)));
    } else if (entry.isFile() && entry.name.endsWith('.md')) {
      results.push(fullPath);
    }
  }
  return results;
}

function parseFrontmatter(content: string): { meta: ParsedFrontmatter; body: string } {
  const fmRegex = /^---\n([\s\S]*?)\n---/;
  const match = content.match(fmRegex);
  if (!match) return { meta: {}, body: content };

  const fmRaw = match[1];
  const body = content.slice(match[0].length).trim();
  const meta: ParsedFrontmatter = {};

  fmRaw.split('\n').forEach((line) => {
    const colonIdx = line.indexOf(':');
    if (colonIdx === -1) return;
    const key = line.slice(0, colonIdx).trim();
    let value = line.slice(colonIdx + 1).trim();
    if (!value) return;

    if (value.startsWith('[') && value.endsWith(']')) {
      try {
        (meta as any)[key] = JSON.parse(value);
      } catch {
        (meta as any)[key] = value
          .slice(1, -1)
          .split(',')
          .map((s) => s.trim().replace(/^['"]|['"]$/g, ''));
      }
    } else if (!isNaN(Number(value))) {
      (meta as any)[key] = Number(value);
    } else {
      (meta as any)[key] = value;
    }
  });

  return { meta, body };
}

const DEEPSEEK_API_KEY = () => process.env.DEEPSEEK_API_KEY || '';
const DEEPSEEK_BASE_URL = () => process.env.DEEPSEEK_BASE_URL || 'https://api.deepseek.com/v1';

const INFER_SYSTEM_PROMPT = `你是一个技术文档分析助手。分析 Markdown 文档内容，提取结构化元数据。

输出必须是 JSON 对象，包含以下字段：
- title: 文档标题（精简，10字以内）
- level: 难度级别（1=基础概念，2=核心概念，3=高级概念）
- category: 分类（Foundation / Model / Attention / Performance / Scheduling / Serving / Infrastructure / Advanced 之一）
- problem: 这个概念要解决的核心问题（一句话，20字以内）
- gap_anticipate: 学习此概念时可能产生的疑问（一句话，20字以内）
- depends_on_titles: 前置概念名称列表（从文档中提取，没有则为空数组）
- leads_to_titles: 引出的概念名称列表（没有则为空数组）
- related_titles: 相关概念名称列表（没有则为空数组）
- tags: 标签数组（从内容中提取 2-5 个关键词）

只输出 JSON，不要包含其他文字。`;

async function inferSingleFrontmatter(
  filename: string,
  content: string,
): Promise<ParsedFrontmatter | null> {
  const apiKey = DEEPSEEK_API_KEY();
  const baseUrl = DEEPSEEK_BASE_URL();
  if (!apiKey) return null;

  try {
    const resp = await fetch(`${baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: 'deepseek-chat',
        messages: [
          { role: 'system', content: INFER_SYSTEM_PROMPT },
          {
            role: 'user',
            content: `文档文件名: ${filename}\n文档内容:\n${content.slice(0, 3000)}\n\n请分析并返回 JSON。`,
          },
        ],
        temperature: 0.3,
        max_tokens: 1000,
      }),
    });

    if (!resp.ok) return null;
    const data = (await resp.json()) as any;
    const text = data?.choices?.[0]?.message?.content;
    if (!text) return null;

    try {
      return JSON.parse(text);
    } catch {
      const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
      if (jsonMatch) return JSON.parse(jsonMatch[1].trim());
      return null;
    }
  } catch {
    return null;
  }
}

function matchTitlesToIds(
  titles: string[],
  concepts: { id: string; title: string; alias?: string[] }[],
): string[] {
  const ids: string[] = [];
  for (const raw of titles) {
    const t = raw.trim();
    if (!t) continue;

    let found = concepts.find((c) => c.title === t || c.alias?.includes(t));
    if (found) { ids.push(found.id); continue; }

    found = concepts.find(
      (c) => c.title.includes(t) || t.includes(c.title) || c.alias?.some((a) => a.includes(t) || t.includes(a)),
    );
    if (found) { ids.push(found.id); continue; }

    const norm = t.replace(/[\s\-_]/g, '').toLowerCase();
    found = concepts.find(
      (c) =>
        c.title.replace(/[\s\-_]/g, '').toLowerCase() === norm ||
        c.alias?.some((a) => a.replace(/[\s\-_]/g, '').toLowerCase() === norm),
    );
    if (found) { ids.push(found.id); continue; }
  }
  return ids;
}

function buildEdges(concepts: Concept[]): ConceptEdge[] {
  const edges: ConceptEdge[] = [];
  const ids = new Set(concepts.map((c) => c.id));
  const edgeSet = new Set<string>();

  const relTypes: [string, 'leads_to' | 'depends_on' | 'related'][] = [
    ['leads_to', 'leads_to'],
    ['depends_on', 'depends_on'],
    ['related', 'related'],
  ];
  for (const c of concepts) {
    for (const [rel, type] of relTypes) {
      const targets = (c as any)[rel] as string[];
      for (const t of targets) {
        if (ids.has(t)) {
          const eid = `${c.id}-${rel}-${t}`;
          if (!edgeSet.has(eid)) {
            edgeSet.add(eid);
            edges.push({ id: eid, source: c.id, target: t, type });
          }
        }
      }
    }
  }

  return edges;
}

export async function buildGraphFromDir(
  dirPath: string,
  onProgress?: (msg: string) => void,
): Promise<BuildResult> {
  onProgress?.(`扫描目录: ${dirPath}`);
  const mdFiles = await scanMdFiles(dirPath);
  onProgress?.(`发现 ${mdFiles.length} 个 Markdown 文件`);

  const concepts: Concept[] = [];
  const filesNeedingLLM: { path: string; content: string }[] = [];

  for (const filePath of mdFiles) {
    const content = await readFile(filePath, 'utf-8');
    const { meta } = parseFrontmatter(content);

    const relPath = filePath.replace(dirPath, '').replace(/^\//, '');
    const id = relPath.replace(/\.md$/, '').replace(/\//g, '-');

    if (meta.title) {
      concepts.push({
        id: meta.id || id,
        title: meta.title,
        alias: meta.alias,
        level: meta.level ?? 1,
        category: meta.category || '',
        problem: meta.problem || '',
        gap_anticipate: meta.gap_anticipate || '',
        depends_on: meta.depends_on || [],
        leads_to: meta.leads_to || [],
        related: meta.related || [],
        tags: meta.tags || [],
        path: filePath,
        lastModified: new Date().toISOString(),
      });
    } else {
      filesNeedingLLM.push({ path: filePath, content });
    }
  }

  if (filesNeedingLLM.length > 0) {
    onProgress?.(`→ AI 推断 ${filesNeedingLLM.length} 个文件的元数据...`);
    const CONCURRENCY = 5;

    for (let i = 0; i < filesNeedingLLM.length; i += CONCURRENCY) {
      const batch = filesNeedingLLM.slice(i, i + CONCURRENCY);
      onProgress?.(`  批次 ${Math.floor(i / CONCURRENCY) + 1}/${Math.ceil(filesNeedingLLM.length / CONCURRENCY)}`);

      const results = await Promise.allSettled(
        batch.map((f) => inferSingleFrontmatter(f.path, f.content)),
      );

      for (let j = 0; j < batch.length; j++) {
        const file = batch[j];
        const result = results[j];
        if (result.status === 'fulfilled' && result.value) {
          const meta = result.value;
          const rel = file.path.replace(dirPath, '').replace(/^\//, '');
          const id = rel.replace(/\.md$/, '').replace(/\//g, '-');

          concepts.push({
            id: meta.id || id,
            title: meta.title || file.path.split('/').pop()?.replace('.md', '') || 'unknown',
            level: meta.level ?? 1,
            category: meta.category || '',
            problem: meta.problem || '',
            gap_anticipate: meta.gap_anticipate || '',
            depends_on: (meta as any).depends_on_titles || meta.depends_on || [],
            leads_to: (meta as any).leads_to_titles || meta.leads_to || [],
            related: (meta as any).related_titles || meta.related || [],
            tags: meta.tags || [],
            path: file.path,
            lastModified: new Date().toISOString(),
          });
        }
      }
    }
  }

  const resolved = concepts.map((c) => ({
    ...c,
    depends_on: matchTitlesToIds(c.depends_on as string[], concepts),
    leads_to: matchTitlesToIds(c.leads_to as string[], concepts),
    related: matchTitlesToIds(c.related as string[], concepts),
  }));

  const edges = buildEdges(resolved);

  onProgress?.(`完成: ${resolved.length} 个概念, ${edges.length} 条边`);

  return { concepts: resolved, edges };
}

export async function saveGraph(
  projectId: string,
  concepts: Concept[],
  edges: ConceptEdge[],
): Promise<void> {
  const projectDir = join(PROJECTS_DIR, projectId);
  ensureDir(projectDir);
  await writeFile(join(projectDir, 'concepts.json'), JSON.stringify(concepts, null, 2));
  await writeFile(join(projectDir, 'edges.json'), JSON.stringify(edges, null, 2));
}

export async function loadGraph(projectId: string): Promise<BuildResult | null> {
  const projectDir = join(PROJECTS_DIR, projectId);
  const conceptsPath = join(projectDir, 'concepts.json');
  const edgesPath = join(projectDir, 'edges.json');
  if (!existsSync(conceptsPath) || !existsSync(edgesPath)) return null;

  const concepts: Concept[] = JSON.parse(await readFile(conceptsPath, 'utf-8'));
  const edges: ConceptEdge[] = JSON.parse(await readFile(edgesPath, 'utf-8'));
  return { concepts, edges };
}

export async function listProjects(): Promise<ProjectMeta[]> {
  ensureDir(PROJECTS_DIR);
  if (!existsSync(PROJECT_LIST_FILE)) return [];
  return JSON.parse(await readFile(PROJECT_LIST_FILE, 'utf-8'));
}

export async function registerProject(meta: ProjectMeta): Promise<void> {
  ensureDir(PROJECTS_DIR);
  const list = await listProjects();
  const idx = list.findIndex((p) => p.id === meta.id);
  if (idx >= 0) {
    list[idx] = meta;
  } else {
    list.push(meta);
  }
  await writeFile(PROJECT_LIST_FILE, JSON.stringify(list, null, 2));
}

export async function removeProject(projectId: string): Promise<void> {
  const list = (await listProjects()).filter((p) => p.id !== projectId);
  await writeFile(PROJECT_LIST_FILE, JSON.stringify(list, null, 2));

  const projectDir = join(PROJECTS_DIR, projectId);
  if (existsSync(projectDir)) {
    await rm(projectDir, { recursive: true, force: true });
  }
}

export { MAGIC_MEMORY_DIR, PROJECTS_DIR, PROJECT_LIST_FILE };
