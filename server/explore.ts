const envFile = Bun.file(import.meta.dir + "/../../.env");
if (await envFile.exists()) {
  const text = await envFile.text();
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eqIndex = trimmed.indexOf("=");
    if (eqIndex === -1) continue;
    const key = trimmed.slice(0, eqIndex).trim();
    const value = trimmed.slice(eqIndex + 1).trim();
    if (!process.env[key]) {
      process.env[key] = value;
    }
  }
}

const PORT = 4321;

// ── graphBuilder (GS) imports ──
import {
  listProjects,
  registerProject,
  removeProject,
  saveGraph,
  loadGraph,
  getProjectDir,
} from "./graphBuilder";

import type { ProjectMeta, Concept, ConceptEdge } from "./graphBuilder";

// ── documents (server.ts) imports ──
import { readdir, readFile, stat, writeFile, rm } from 'fs/promises'
import { join, dirname } from 'path'
import { homedir } from 'os'
import { existsSync, mkdirSync } from 'fs'

const DOCS_DIR = join(import.meta.dir, '..', 'docs')

interface Document {
  id: string
  title: string
  path: string
  content: string
  level: number
  category: string
  tags: string[]
  lastModified: Date
  metadata?: { author?: string; version?: string; status?: string }
}

interface Annotation {
  id: string
  documentId: string
  type: 'comment' | 'question' | 'suggestion' | 'correction'
  content: string
  position: { start: number; end: number; line?: number }
  author: string
  createdAt: Date
  updatedAt: Date
  status: 'open' | 'resolved' | 'closed'
}

const documents: Document[] = []

async function loadDocuments() {
  try {
    const entries = await readdir(DOCS_DIR, { withFileTypes: true })
    for (const entry of entries) {
      const fullPath = join(DOCS_DIR, entry.name)
      if (entry.isDirectory()) {
        await loadDocumentsFromDirectory(fullPath, entry.name)
      } else if (entry.name.endsWith('.md')) {
        try {
          const content = await readFile(fullPath, 'utf-8')
          const stats = await stat(fullPath)
          const title = entry.name.replace('.md', '')
          documents.push({
            id: entry.name, title, path: fullPath, content,
            level: 1, category: '', tags: [],
            lastModified: stats.mtime, metadata: { status: 'draft' },
          })
        } catch (error) {
          console.error(`Failed to load document: ${fullPath}`, error)
        }
      }
    }
    console.log(`Loaded ${documents.length} documents`)
  } catch (error) {
    console.error('Failed to load documents:', error)
  }
}

async function loadDocumentsFromDirectory(dirPath: string, relativePath: string) {
  try {
    const entries = await readdir(dirPath, { withFileTypes: true })
    for (const entry of entries) {
      const fullPath = join(dirPath, entry.name)
      const entryRelativePath = relativePath ? join(relativePath, entry.name) : entry.name
      if (entry.isDirectory()) {
        await loadDocumentsFromDirectory(fullPath, entryRelativePath)
      } else if (entry.name.endsWith('.md')) {
        try {
          const content = await readFile(fullPath, 'utf-8')
          const stats = await stat(fullPath)
          documents.push({
            id: entryRelativePath, title: entry.name.replace('.md', ''),
            path: fullPath, content, level: 1, category: '', tags: [],
            lastModified: stats.mtime, metadata: { status: 'draft' },
          })
        } catch (error) {
          console.error(`Failed to load document: ${fullPath}`, error)
        }
      }
    }
  } catch (error) {
    console.error(`Failed to read directory: ${dirPath}`, error)
  }
}

// ── LLM helpers ──
const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY || "";
const DEEPSEEK_BASE_URL = process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com/v1";

async function llmChat(system: string, user: string, maxTokens = 2000) {
  const resp = await fetch(`${DEEPSEEK_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${DEEPSEEK_API_KEY}` },
    body: JSON.stringify({ model: 'deepseek-chat', messages: [{ role: 'system', content: system }, { role: 'user', content: user }], max_tokens: maxTokens }),
  })
  if (!resp.ok) {
    const errText = await resp.text().catch(() => "unknown error");
    throw new Error(`LLM API error (${resp.status}): ${errText}`);
  }
  const data = await resp.json() as any
  return data?.choices?.[0]?.message?.content || ''
}

// ── Route handlers ──

function json(data: any, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
  })
}

function jsonError(msg: string, status = 500): Response {
  return json({ error: msg }, status)
}

// ── Documents ──

function handleListDocuments(): Response {
  return json(documents)
}

function handleGetDocument(id: string): Response {
  const doc = documents.find(d => d.id === id)
  return doc ? json(doc) : jsonError('Document not found', 404)
}

function handleDocumentAnnotations(id: string): Response {
  return json([])
}

function handleStats(): Response {
  return json({
    total: documents.length,
    byLevel: {
      'Level 1': documents.filter(d => d.level === 1).length,
      'Level 2': documents.filter(d => d.level === 2).length,
      'Level 3': documents.filter(d => d.level === 3).length,
    },
    byCategory: documents.reduce((acc, doc) => {
      acc[doc.category] = (acc[doc.category] || 0) + 1
      return acc
    }, {} as Record<string, number>),
  })
}

// ── Write / Read / Delete doc ──

async function handleWriteDoc(req: Request): Promise<Response> {
  let body: any = {};
  try {
    body = await req.json()
    let { path: filePath, content, baseDir } = body
    if (!filePath || content === undefined) return jsonError('path and content required', 400)
    if (baseDir && !filePath.startsWith('/')) {
      filePath = join(baseDir, filePath.replace(/^\.\//, ''))
    }
    const dir = dirname(filePath)
    if (!existsSync(dir)) { mkdirSync(dir, { recursive: true }) }
    await Bun.write(filePath, content)
    const doc = documents.find(d => d.path === filePath)
    if (doc) {
      const fmMatch = doc.content.match(/^---[\s\S]*?---\n/)
      doc.content = fmMatch ? fmMatch[0] + content : content
    }
    return json({ success: true, filePath })
  } catch (error) {
    console.error('/api/write-doc failed:', error, 'body:', JSON.stringify(body).slice(0, 200))
    return jsonError(String(error))
  }
}

async function handleReadDoc(req: Request): Promise<Response> {
  try {
    const url = new URL(req.url)
    const pathParam = url.searchParams.get('path') || ''
    const baseDir = url.searchParams.get('baseDir') || ''
    let resolvedPath = pathParam
    if (baseDir && !pathParam.startsWith('/')) {
      resolvedPath = join(baseDir, pathParam.replace(/^\.\//, ''))
    }
    if (!existsSync(resolvedPath)) {
      return jsonError('File not found', 404)
    }
    const content = await readFile(resolvedPath, 'utf-8')
    return json({ content })
  } catch (error) {
    return jsonError(String(error))
  }
}

async function handleDeleteDoc(req: Request): Promise<Response> {
  try {
    const body = await req.json()
    let { path: filePath, baseDir } = body
    if (!filePath) return jsonError('path required', 400)
    if (baseDir && !filePath.startsWith('/')) {
      filePath = join(baseDir, filePath.replace(/^\.\//, ''))
    }
    try { await rm(filePath) } catch {}
    const docIdx = documents.findIndex(d => d.path === filePath)
    if (docIdx !== -1) documents.splice(docIdx, 1)
    return json({ success: true })
  } catch (error) {
    return jsonError(String(error))
  }
}

// ── LLM features ──

async function handleGenerateDoc(req: Request): Promise<Response> {
  try {
    const body = await req.json() as any
    const { concept } = body
    if (!concept) return jsonError('concept required', 400)
    const content = await llmChat(
      '你是一个技术文档撰写助手。',
      `你是一个技术文档撰写助手。请根据以下概念信息生成一份中文 Markdown 文档。

概念名称: ${concept.title || ''}
核心问题: ${concept.problem || ''}
认知缺口: ${concept.gap_anticipate || ''}

文档要求：
1. 以"# 标题"开头
2. 包含"## 问题"章节，解释这个概念要解决什么问题
3. 包含"## 核心设计"章节，描述核心设计思路
4. 包含"## 使用示例"章节，给出代码或伪代码示例

只输出 Markdown 内容，不要额外说明。`
    )
    return json({ content })
  } catch (error) {
    return jsonError(String(error))
  }
}

async function handleAskQuestion(req: Request): Promise<Response> {
  try {
    const body = await req.json() as any
    const { selectedText, question } = body
    if (!selectedText || !question) {
      return jsonError('selectedText and question required', 400)
    }
    const answer = await llmChat(
      '你是一个技术学习助手，帮助用户理解技术概念和回答问题。',
      `你是一个技术学习助手。用户在学习 vLLM 相关知识时，针对一段文本提出了一个问题。

选中文本:
"""
${selectedText}
"""

用户问题:
${question}

请用中文回答用户的问题。回答要简明扼要、切中要点。如果选中文本不足以回答问题，请基于你的知识补充说明。`,
      1000
    )
    return json({ answer })
  } catch (error) {
    return jsonError(String(error))
  }
}

async function handleInferRelations(req: Request): Promise<Response> {
  try {
    const body = await req.json() as any
    const { conceptId, content, concepts: allConcepts } = body
    if (!conceptId || !content || !allConcepts) {
      return jsonError('conceptId, content, and concepts required', 400)
    }
    const sourceConcept = allConcepts.find((c: any) => c.id === conceptId)
    const otherConcepts = allConcepts.filter((c: any) => c.id !== conceptId)
    const conceptListStr = otherConcepts
      .map((c: any) => `- ID: ${c.id}, 名称: ${c.title}${c.problem ? `, 核心问题: ${c.problem}` : ''}`)
      .join('\n')

    const rawContent = await llmChat(
      '你是一个知识图谱分析助手，只输出 JSON，不输出其他内容。',
      `你是一个知识图谱分析助手。请根据以下概念文档内容，从候选概念列表中找出与之直接相关的概念。

当前概念名称: ${sourceConcept?.title || conceptId}
当前概念文档内容:
"""
${content.slice(0, 3000)}
"""

候选概念列表:
${conceptListStr}

请分析当前概念的文档内容，找出候选概念中与它最直接相关的概念（通常 2-5 个）。
判断依据：概念之间的因果依赖、功能关联、或共同解决一个问题。

只返回 JSON 格式的数组，不要额外说明：
["concept_id_1", "concept_id_2", ...]`,
      1000
    )

    let relatedIds: string[] = []
    try {
      relatedIds = JSON.parse(rawContent)
    } catch {
      const jsonMatch = rawContent.match(/```(?:json)?\s*([\s\S]*?)```/)
      if (jsonMatch) {
        try { relatedIds = JSON.parse(jsonMatch[1]) } catch {}
      }
    }
    if (!Array.isArray(relatedIds)) relatedIds = []

    const relations = relatedIds
      .filter((id: string) => allConcepts.some((c: any) => c.id === id))
      .map((targetId: string) => ({ targetId }))

    return json({ relations })
  } catch (error) {
    console.error('关系推断失败:', error)
    return jsonError(String(error))
  }
}

// ── Explore (GS original) ──

const EXPLORE_SYSTEM_PROMPT = `你是一个知识图谱概念生成助手。你的任务是基于用户提供的源概念和探索问题，生成一个新的概念节点。
输出必须是一个 JSON 对象，包含以下字段：
- title: 概念名称（简短、精确，10个字以内）
- problem: 这个概念解决的核心问题（一句话）
- gap_anticipate: 学习这个概念时可能产生的疑问（一句话，30字以内）
- content: 概念的详细内容，用 Markdown 格式（至少 200 字，包含概述、核心原理、关键要点）
只输出 JSON，不要包含其他文字。`;

async function handleExplore(req: Request): Promise<Response> {
  try {
    const body = await req.json() as any;
    const userPrompt = `源概念: ${body.sourceConcept.title}
${body.sourceConcept.problem ? `源概念问题: ${body.sourceConcept.problem}` : ""}
探索问题: ${body.userQuestion}
请基于源概念「${body.sourceConcept.title}」和探索问题「${body.userQuestion}」，生成一个新的概念。输出 JSON。`;

    const text = await llmChat(EXPLORE_SYSTEM_PROMPT, userPrompt, 2000);

    let result: any;
    try {
      result = JSON.parse(text);
    } catch {
      const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
      if (jsonMatch) result = JSON.parse(jsonMatch[1].trim());
      else throw new Error("Failed to parse LLM response as JSON");
    }

    return json(result);
  } catch (error: any) {
    console.error("Explore error:", error);
    return jsonError(error.message);
  }
}

// ── Project / Graph handlers (from explore.ts) ──

async function handleListProjects(): Promise<Response> {
  try {
    const projects = await listProjects();
    return json({ projects });
  } catch (error: any) {
    return jsonError(error.message);
  }
}

async function handleCreateProject(req: Request): Promise<Response> {
  try {
    const body = await req.json() as {
      id?: string; name: string; sourceDir: string;
      concepts?: Concept[]; edges?: ConceptEdge[];
    };
    if (!body.name || body.sourceDir === undefined) {
      return jsonError("name and sourceDir required", 400);
    }
    const projectId = body.id || `proj_${Date.now()}`;
    const meta: ProjectMeta = {
      id: projectId, name: body.name, sourceDir: body.sourceDir,
      sourceType: 'doc',
      createdAt: new Date().toISOString(),
      conceptCount: body.concepts?.length || 0,
      edgeCount: body.edges?.length || 0,
    };
    await registerProject(meta);
    if (body.concepts && body.edges) {
      await saveGraph(projectId, body.concepts, body.edges);
    }
    return json(meta, 201);
  } catch (error: any) {
    return jsonError(error.message);
  }
}

async function handleDeleteProject(projectId: string): Promise<Response> {
  try {
    await removeProject(projectId);
    return json({ success: true });
  } catch (error: any) {
    return jsonError(error.message);
  }
}

async function handleGetGraph(projectId: string): Promise<Response> {
  try {
    const graph = await loadGraph(projectId);
    if (!graph) return jsonError("Graph not found", 404);
    return json(graph);
  } catch (error: any) {
    return jsonError(error.message);
  }
}

async function handleUpdateGraph(projectId: string, req: Request): Promise<Response> {
  try {
    const body = await req.json() as { concepts: Concept[]; edges: ConceptEdge[] };
    if (!body.concepts || !body.edges) {
      return jsonError("concepts and edges required", 400);
    }
    await saveGraph(projectId, body.concepts, body.edges);
    return json({ success: true, conceptCount: body.concepts.length, edgeCount: body.edges.length });
  } catch (error: any) {
    return jsonError(error.message);
  }
}

// ── Repo project docs ──

async function handleListProjectDocs(projectId: string): Promise<Response> {
  try {
    const docsDir = join(getProjectDir(projectId), 'docs');
    if (!existsSync(docsDir)) return json([]);
    const files = await readdir(docsDir);
    const docs = files.filter(f => f.endsWith('.md')).map(f => ({
      id: f.replace('.md', ''),
      title: f.replace('.md', ''),
    }));
    return json(docs);
  } catch (error: any) {
    return jsonError(error.message);
  }
}

async function handleGetProjectDoc(projectId: string, docId: string): Promise<Response> {
  try {
    const docPath = join(getProjectDir(projectId), 'docs', docId + '.md');
    if (!existsSync(docPath)) return jsonError('Document not found', 404);
    const content = await readFile(docPath, 'utf-8');
    return json({ id: docId, content });
  } catch (error: any) {
    return jsonError(error.message);
  }
}

async function handleWriteProjectDoc(projectId: string, req: Request): Promise<Response> {
  try {
    const body = await req.json() as { docId: string; content: string };
    if (!body.docId || body.content === undefined) {
      return jsonError('docId and content required', 400);
    }
    const docsDir = join(getProjectDir(projectId), 'docs');
    if (!existsSync(docsDir)) mkdirSync(docsDir, { recursive: true });
    const docPath = join(docsDir, body.docId + '.md');
    await writeFile(docPath, body.content, 'utf-8');
    return json({ success: true });
  } catch (error: any) {
    return jsonError(error.message);
  }
}

// ── Server ──

Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);
    const method = req.method;

    if (method === "OPTIONS") {
      return json(null, 200);
    }

    // Documents
    if (url.pathname === '/api/documents' && method === 'GET') return handleListDocuments()
    if (url.pathname.match(/\/api\/documents\/[^/]+\/annotations/)) {
      const id = url.pathname.split('/')[3]
      return handleDocumentAnnotations(id)
    }
    if (url.pathname.startsWith('/api/documents/') && method === 'GET') {
      const id = url.pathname.split('/')[3]
      return handleGetDocument(id)
    }
    if (url.pathname === '/api/stats' && method === 'GET') return handleStats()

    // Doc CRUD
    if (url.pathname === '/api/write-doc' && method === 'POST') return handleWriteDoc(req)
    if (url.pathname === '/api/read-doc' && method === 'GET') return handleReadDoc(req)
    if (url.pathname === '/api/delete-doc' && method === 'DELETE') return handleDeleteDoc(req)

    // LLM routes
    if (url.pathname === '/api/generate-doc' && method === 'POST') return handleGenerateDoc(req)
    if (url.pathname === '/api/ask-question' && method === 'POST') return handleAskQuestion(req)
    if (url.pathname === '/api/infer-relations-from-content' && method === 'POST') return handleInferRelations(req)

    // Explore (GS)
    if (url.pathname === "/api/explore" && method === "POST") return handleExplore(req)

    // Projects
    if (url.pathname === "/api/projects" && method === "GET") return handleListProjects()
    if (url.pathname === "/api/projects" && method === "POST") return handleCreateProject(req)

    const graphMatch = url.pathname.match(/^\/api\/projects\/([^\/]+)\/graph$/);
    if (graphMatch && method === "GET") return handleGetGraph(graphMatch[1])
    if (graphMatch && method === "PUT") return handleUpdateGraph(graphMatch[1], req)

    const projectMatch = url.pathname.match(/^\/api\/projects\/([^\/]+)$/);
    if (projectMatch && method === "DELETE") return handleDeleteProject(projectMatch[1])

    // Project docs
    const docListMatch = url.pathname.match(/^\/api\/projects\/([^\/]+)\/docs$/);
    if (docListMatch && method === 'GET') return handleListProjectDocs(docListMatch[1]);
    if (docListMatch && method === 'POST') return handleWriteProjectDoc(docListMatch[1], req);

    const docGetMatch = url.pathname.match(/^\/api\/projects\/([^\/]+)\/docs\/([^\/]+)$/);
    if (docGetMatch && method === 'GET') return handleGetProjectDoc(docGetMatch[1], docGetMatch[2]);

    return jsonError("Not Found", 404);
  },
});

console.log(`🚀 Server running on http://localhost:${PORT}`);
console.log(`   Documents: GET /api/documents, /api/stats`);
console.log(`   Doc CRUD:  POST/GET/DELETE /api/write-doc, /api/read-doc, /api/delete-doc`);
console.log(`   LLM:       POST /api/generate-doc, /api/ask-question, /api/infer-relations-from-content`);
console.log(`   Explore:   POST /api/explore`);
console.log(`   Projects:  GET/POST /api/projects`);
console.log(`   Graph:     GET/PUT /api/projects/:id/graph`);

await loadDocuments()
