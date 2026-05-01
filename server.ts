import { serve } from 'bun'
import { readdir, readFile, stat, writeFile, rm } from 'fs/promises'
import { join, relative, dirname } from 'path'
import { homedir } from 'os'
import { existsSync, mkdirSync } from 'fs'
import { clusterPipeline } from './scripts/cluster'
import type { Document, Annotation, Concept, ConceptEdge, Project } from './src/types'
import type { Edge as ClusterEdge } from './scripts/cluster'
import { analyzeGraph, formatAnalysisToString } from './src/utils/graphAnalysis'

const PORT = 3001
const DOCS_DIR = join(process.cwd(), '../docs')
const ANALYSIS_CACHE_PATH = join(process.cwd(), 'data', 'graph-analysis.json')

// ===== Project storage constants =====
const MAGIC_MEMORY_DIR = join(homedir(), '.magic-memory');
const PROJECTS_DIR = join(MAGIC_MEMORY_DIR, 'projects');
const PROJECT_LIST_FILE = join(PROJECTS_DIR, 'project-list.json');

// Ensure directories exist
function ensureProjectDirs() {
  if (!existsSync(PROJECTS_DIR)) {
    mkdirSync(PROJECTS_DIR, { recursive: true });
  }
}
ensureProjectDirs();

const documents: Document[] = []
const concepts: Concept[] = []
const conceptEdges: ConceptEdge[] = []
let graphAnalysisCache: ReturnType<typeof analyzeGraph> | null = null

async function loadDocuments() {
  // 优先从缓存文件加载分析结果
  const cacheLoaded = await tryLoadAnalysisCache()
  if (cacheLoaded) {
    console.log(`Loaded graph analysis from cache (${ANALYSIS_CACHE_PATH})`)
    return
  }

  try {
    await loadDocumentsFromDirectory(DOCS_DIR, '')
    console.log(`Loaded ${documents.length} documents`)
    
    documents.forEach(doc => {
      concepts.push(buildConceptFromDocument(doc))
    })

    // 用 cluster 管道自动构建图（交叉引用 + 目录 + 编号邻近）
    const clusterResult = clusterPipeline(DOCS_DIR, null, 0.5)
    if (clusterResult.edges.length > 0) {
      conceptEdges.push(...assignDirections(clusterResult.edges))
    }
    console.log(`Built knowledge graph: ${concepts.length} concepts, ${conceptEdges.length} edges from cluster analysis`)

    const edgeConceptIds = new Set<string>()
    conceptEdges.forEach(e => { edgeConceptIds.add(e.source); edgeConceptIds.add(e.target) })
    const graphConcepts = concepts.filter(c => edgeConceptIds.has(c.id))
    graphAnalysisCache = analyzeGraph(graphConcepts, conceptEdges)
    console.log(`Computed graph analysis: ${graphAnalysisCache.stats.rootsCount} roots, ${graphAnalysisCache.stats.totalConcepts} concepts`)

    // 写入缓存文件
    await saveAnalysisCache(graphAnalysisCache)
    console.log(`Saved graph analysis cache (${ANALYSIS_CACHE_PATH})`)
  } catch (error) {
    console.error('Failed to load documents:', error)
  }
}

// ===== Project Management Helpers =====
async function loadProjectList(): Promise<Project[]> {
  try {
    // If no list file yet, return empty list
    if (!existsSync(PROJECT_LIST_FILE)) return []
    const content = await readFile(PROJECT_LIST_FILE, 'utf-8')
    return JSON.parse(content) as Project[]
  } catch {
    return []
  }
}

async function saveProjectList(projects: Project[]): Promise<void> {
  await writeFile(PROJECT_LIST_FILE, JSON.stringify(projects, null, 2))
}

async function createProjectDir(projectId: string): Promise<string> {
  const projectDir = join(PROJECTS_DIR, projectId)
  mkdirSync(projectDir, { recursive: true })
  return projectDir
}

function generateProjectId(): string {
  return `proj_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
}
// ===== End Project Management Helpers =====

async function tryLoadAnalysisCache(): Promise<boolean> {
  try {
    if (!existsSync(ANALYSIS_CACHE_PATH)) return false
    const content = await readFile(ANALYSIS_CACHE_PATH, 'utf-8')
    const parsed = JSON.parse(content)
    // 重建 analyzeGraph 返回结构（JSON 反序列化后类型一致）
    graphAnalysisCache = parsed
    console.log(`Found ${parsed.stats.totalConcepts} concepts, ${parsed.stats.totalEdges} edges in cache`)
    return true
  } catch {
    return false
  }
}

async function saveAnalysisCache(data: any): Promise<void> {
  const dir = dirname(ANALYSIS_CACHE_PATH)
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  await writeFile(ANALYSIS_CACHE_PATH, JSON.stringify(data, null, 2))
}

/** 给 cluster 产生的无向边赋予方向：低 level → 高 level，同 level 按编号小→大 */
function assignDirections(clusterEdges: ClusterEdge[]): ConceptEdge[] {
  // cluster 的 concept ID 是文件名去前缀（vllm-config），而 server 的 concept ID 是路径（Foundation/00-vllm-config.md）
  // 构建短名 → 全路径 ID 映射
  const shortToFull = new Map<string, string>()
  for (const c of concepts) {
    const filename = c.path.split('/').pop() || ''
    const shortId = filename.replace(/\.md$/i, '').replace(/^\d+[-_\s]+/, '')
    if (shortId) shortToFull.set(shortId, c.id)
  }

  const result: ConceptEdge[] = []
  const seen = new Set<string>()

  for (const e of clusterEdges) {
    // 编号邻近边 (number_proximity) 产生自然的学习路径 0→1→2→...→50
    // references 边 (wiki 链接) 产生强关联
    if (e.type === 'co_directory') continue

    const srcFull = shortToFull.get(e.source)
    const tgtFull = shortToFull.get(e.target)
    if (!srcFull || !tgtFull) continue

    const src = concepts.find(c => c.id === srcFull)
    const tgt = concepts.find(c => c.id === tgtFull)
    if (!src || !tgt) continue

    let source: string, target: string
    let type: ConceptEdge['type']

    if (src.level !== tgt.level) {
      if (src.level < tgt.level) { source = src.id; target = tgt.id }
      else { source = tgt.id; target = src.id }
      type = 'leads_to'
    } else {
      const srcNum = extractConceptNumber(src.path.split('/').pop() || '')
      const tgtNum = extractConceptNumber(tgt.path.split('/').pop() || '')
      if (srcNum !== null && tgtNum !== null && srcNum !== tgtNum) {
        if (srcNum < tgtNum) { source = src.id; target = tgt.id }
        else { source = tgt.id; target = src.id }
        type = 'leads_to'
      } else {
        source = src.id; target = tgt.id
        type = 'related'
      }
    }

    const key = `${source}→${target}`
    if (seen.has(key)) continue
    seen.add(key)
    result.push({ id: `${source}-${target}`, source, target, type })
  }

  console.log(`Assigned directions to ${result.length} cluster edges (from ${clusterEdges.length} undirected)`)
  return result
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

          const title = entry.name.replace('.md', '')
          const level = extractLevel(title)
          const category = extractCategory(entryRelativePath)

          documents.push({
            id: entryRelativePath,
            title,
            path: fullPath,
            content,
            level,
            category,
            tags: extractTags(content),
            lastModified: stats.mtime,
            metadata: {
              status: 'draft',
            },
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

function extractLevel(title: string): number {
  const levelMatch = title.match(/level-(\d+)/i)
  if (levelMatch) {
    return parseInt(levelMatch[1], 10)
  }

  const numberMatch = title.match(/^(\d+)-/)
  if (numberMatch) {
    const num = parseInt(numberMatch[1], 10)
    if (num >= 0 && num <= 9) return 1
    if (num >= 10 && num <= 29) return 2
    if (num >= 30 && num <= 50) return 3
  }

  return 1
}

function extractCategory(path: string): string {
  const pathLower = path.toLowerCase()

  if (pathLower.includes('vllm')) return 'vLLM'
  if (pathLower.includes('methodology')) return '方法论'
  if (pathLower.includes('framework')) return '框架'
  if (pathLower.includes('level-1')) return 'Level 1'
  if (pathLower.includes('level-2')) return 'Level 2'
  if (pathLower.includes('level-3')) return 'Level 3'

  const parts = path.split('/')
  if (parts.length > 1) {
    return parts[0]
  }

  return '其他'
}

function extractTags(content: string): string[] {
  const tags: string[] = []

  const tagMatch = content.match(/^tags:\s*(.+)$/m)
  if (tagMatch) {
    const tagString = tagMatch[1]
    tags.push(...tagString.split(',').map(tag => tag.trim()))
  }

  const keywords = ['vllm', 'transformer', 'attention', 'quantization', 'batching', 'scheduling']
  keywords.forEach(keyword => {
    if (content.toLowerCase().includes(keyword.toLowerCase())) {
      tags.push(keyword)
    }
  })

  return [...new Set(tags)]
}

function parseFrontmatter(content: string): Record<string, any> {
  const fmRegex = /^---\n([\s\S]*?)\n---/
  const match = content.match(fmRegex)
  if (!match) return {}
  
  const fmRaw = match[1]
  const meta: Record<string, any> = {}
  
  fmRaw.split('\n').forEach(line => {
    const colonIdx = line.indexOf(':')
    if (colonIdx === -1) return
    
    const key = line.slice(0, colonIdx).trim()
    let value: any = line.slice(colonIdx + 1).trim()
    
    if (value.startsWith('[') && value.endsWith(']')) {
      try { 
        value = JSON.parse(value) 
      } catch { 
        value = value.slice(1, -1).split(',').map(s => s.trim())
      }
    } else if (!isNaN(Number(value))) {
      value = Number(value)
    }
    
    meta[key] = value
  })
  
  return meta
}

function buildConceptFromDocument(doc: Document): Concept {
  const meta = parseFrontmatter(doc.content)
  const body = doc.content.replace(/^---[\s\S]*?---\n/, '')
  
  return {
    id: doc.id,
    title: meta.title || doc.title,
    alias: meta.alias,
    level: doc.level,
    category: doc.category,
    problem: meta.problem || '',
    gap_anticipate: meta.gap_anticipate || '',
    depends_on: meta.depends_on || [],
    leads_to: meta.leads_to || [],
    related: meta.related || [],
    content: body,
    path: doc.path,
    tags: doc.tags,
    lastModified: doc.lastModified,
    metadata: doc.metadata
  }
}

/** 从文件名提取概念编号（如 "00-vllm-config.md" → 0, "40-vllm-engine.md" → 40） */
function extractConceptNumber(filename: string): number | null {
  const m = filename.match(/^(\d+)/)
  return m ? parseInt(m[1], 10) : null
}

/** 解析 docs/概念关联图.md，提取所有概念关系边 */
async function buildEdgesFromConceptGraph(): Promise<ConceptEdge[]> {
  const graphFilePath = join(DOCS_DIR, '概念关联图.md')
  let content: string
  try {
    content = await readFile(graphFilePath, 'utf-8')
  } catch { return [] }

  // 构建 编号 → 文档 ID 映射
  const numToId = new Map<number, string>()
  for (const doc of documents) {
    const filename = doc.path.split('/').pop() || ''
    const num = extractConceptNumber(filename)
    if (num !== null) numToId.set(num, doc.id)
  }

  const edges: ConceptEdge[] = []
  const edgeSet = new Set<string>()

  // 按 ## 章节拆分
  const sections = content.split(/^## /m)

  for (const section of sections) {
    // 判断章节类型
    const isDataFlow = /一、数据流关联|数据流/.test(section)
    const isDependency = /二、依赖关系|依赖/.test(section)
    if (!isDataFlow && !isDependency) continue

    const edgeType: ConceptEdge['type'] = isDataFlow ? 'leads_to' : 'depends_on'

    // 按顺序提取所有 数字(Concept) 引用，相邻引用之间创建边
    // 这样能正确处理多行链（如 35(Scheduler)\n    → 34(...)）
    const refRegex = /(\d+)\([^)]+\)/g
    let refMatch: RegExpExecArray | null
    const numbers: number[] = []
    while ((refMatch = refRegex.exec(section)) !== null) {
      numbers.push(parseInt(refMatch[1], 10))
    }
    for (let i = 0; i < numbers.length - 1; i++) {
      const srcNum = numbers[i]
      const tgtNum = numbers[i + 1]
      const sourceId = numToId.get(srcNum)
      const targetId = numToId.get(tgtNum)
      if (!sourceId || !targetId) continue

      const eid = `${sourceId}-graph-${targetId}`
      if (!edgeSet.has(eid)) {
        edgeSet.add(eid)
        edges.push({ id: eid, source: sourceId, target: targetId, type: edgeType })
      }
    }
  }

  return edges
}

function buildEdges(): ConceptEdge[] {
  const edges: ConceptEdge[] = []
  const conceptMap = new Map(concepts.map(c => [c.id, c]))
  
  concepts.forEach(source => {
    source.depends_on.forEach(targetId => {
      if (conceptMap.has(targetId)) {
        edges.push({
          id: `${source.id}-depends-${targetId}`,
          source: source.id,
          target: targetId,
          type: 'depends_on'
        })
      }
    })
    source.leads_to.forEach(targetId => {
      if (conceptMap.has(targetId)) {
        edges.push({
          id: `${source.id}-leads-${targetId}`,
          source: source.id,
          target: targetId,
          type: 'leads_to'
        })
      }
    })
    source.related.forEach(targetId => {
      if (conceptMap.has(targetId)) {
        edges.push({
          id: `${source.id}-related-${targetId}`,
          source: source.id,
          target: targetId,
          type: 'related'
        })
      }
    })
  })
  
  return edges
}

// 扫描目录构建索引（不含文档正文，只提取 frontmatter 元数据）
async function scanDirectoryForIndex(dirPath: string, result: any[]) {
  const entries = await readdir(dirPath, { withFileTypes: true })
  for (const entry of entries) {
    const fullPath = join(dirPath, entry.name)
    if (entry.isDirectory() && !entry.name.startsWith('.')) {
      await scanDirectoryForIndex(fullPath, result)
    } else if (entry.name.endsWith('.md')) {
      try {
        const content = await readFile(fullPath, 'utf-8')
        const meta = parseFrontmatter(content)
        const title = meta.title || entry.name.replace('.md', '').replace(/^\d+-/, '')
        const id = meta.id || entry.name.replace('.md', '')
        const levelNum = meta.level ?? 1
        const category = meta.category || ''
        result.push({
          id,
          title,
          path: fullPath,
          level: levelNum,
          category,
          depends_on: meta.depends_on || [],
          leads_to: meta.leads_to || [],
          related: meta.related || [],
          problem: meta.problem || '',
          gap_anticipate: meta.gap_anticipate || '',
          alias: meta.alias,
          tags: meta.tags || [],
        })
      } catch (e) {
        console.error('scan skip:', fullPath, e)
      }
    }
  }
}

const server = serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url)

    // POST /api/scan-docs — 扫描目录建索引
    if (url.pathname === '/api/scan-docs' && req.method === 'POST') {
      try {
        const body = await req.json()
        const scanPath = body.path
        if (!scanPath) {
          return new Response(JSON.stringify({ error: 'path required' }), { status: 400, headers: { 'Content-Type': 'application/json' } })
        }
        const result: { id: string; title: string; path: string; level: number; category: string; depends_on: string[]; leads_to: string[]; related: string[]; problem: string; gap_anticipate: string; alias?: string[]; tags: string[] }[] = []
        await scanDirectoryForIndex(scanPath, result)
        // 从 depends_on/leads_to/related 推导边
        const ids = new Map(result.map(c => [c.id, c]))
        const edges: { id: string; source: string; target: string; type: string }[] = []
        const edgeSet = new Set<string>()
        for (const c of result) {
          for (const t of c.leads_to) {
            if (ids.has(t)) {
              const eid = `${c.id}-leads-${t}`
              if (!edgeSet.has(eid)) { edgeSet.add(eid); edges.push({ id: eid, source: c.id, target: t, type: 'leads_to' }) }
            }
          }
          for (const t of c.depends_on) {
            if (ids.has(t)) {
              const eid = `${c.id}-depends-${t}`
              if (!edgeSet.has(eid)) { edgeSet.add(eid); edges.push({ id: eid, source: c.id, target: t, type: 'depends_on' }) }
            }
          }
          for (const t of c.related) {
            if (ids.has(t)) {
              const eid = `${c.id}-related-${t}`
              if (!edgeSet.has(eid)) { edgeSet.add(eid); edges.push({ id: eid, source: c.id, target: t, type: 'related' }) }
            }
          }
        }
        return new Response(JSON.stringify({ concepts: result, edges }), {
          headers: { 'Content-Type': 'application/json' },
        })
      } catch (error) {
        return new Response(JSON.stringify({ error: String(error) }), { status: 500, headers: { 'Content-Type': 'application/json' } })
      }
    }

    if (url.pathname === '/api/documents') {
      return new Response(JSON.stringify(documents), {
        headers: { 'Content-Type': 'application/json' },
      })
    }

    if (url.pathname.startsWith('/api/documents/')) {
      const id = url.pathname.split('/')[3]
      const document = documents.find((doc) => doc.id === id)

      if (document) {
        return new Response(JSON.stringify(document), {
          headers: { 'Content-Type': 'application/json' },
        })
      }

      return new Response('Document not found', { status: 404 })
    }

    if (url.pathname.match(/\/api\/documents\/[^/]+\/annotations/)) {
      const id = url.pathname.split('/')[3]
      const documentAnnotations: Annotation[] = []

      return new Response(JSON.stringify(documentAnnotations), {
        headers: { 'Content-Type': 'application/json' },
      })
    }

    if (url.pathname === '/api/stats') {
      const stats = {
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
      }

      return new Response(JSON.stringify(stats), {
        headers: { 'Content-Type': 'application/json' },
      })
    }

    if (url.pathname === '/api/graph') {
      // 只返回有边连接的概念，过滤掉孤立的方法论/自测/vllm-tree 重复文档
      const edgeConceptIds = new Set<string>()
      conceptEdges.forEach(e => { edgeConceptIds.add(e.source); edgeConceptIds.add(e.target) })
      const graphConcepts = concepts.filter(c => edgeConceptIds.has(c.id))
      return new Response(JSON.stringify({ concepts: graphConcepts, edges: conceptEdges }), {
        headers: { 'Content-Type': 'application/json' },
      })
    }

    if (url.pathname === '/api/graph/analysis') {
      const analysis = graphAnalysisCache ?? await (async () => {
        const edgeConceptIds = new Set<string>()
        conceptEdges.forEach(e => { edgeConceptIds.add(e.source); edgeConceptIds.add(e.target) })
        const graphConcepts = concepts.filter(c => edgeConceptIds.has(c.id))
        return analyzeGraph(graphConcepts, conceptEdges)
      })()

      const describe = url.searchParams.get('describe') === 'true'
      if (describe) {
        await enrichFlowsWithDescription(analysis)
      }

      const format = url.searchParams.get('format')
      if (format === 'text') {
        return new Response(formatAnalysisToString(analysis), {
          headers: { 'Content-Type': 'text/plain; charset=utf-8' },
        })
      }

      return new Response(JSON.stringify(analysis), {
        headers: { 'Content-Type': 'application/json' },
      })
    }

    if (url.pathname.startsWith('/api/graph/')) {
      const id = url.pathname.split('/')[3]
      const concept = concepts.find(c => c.id === id)
      if (concept) {
        const relatedIds = [
          ...concept.depends_on,
          ...concept.leads_to,
          ...concept.related
        ]
        const related = concepts.filter(c => relatedIds.includes(c.id))
        return new Response(JSON.stringify({ concept, related }), {
          headers: { 'Content-Type': 'application/json' },
        })
      }
      return new Response('Concept not found', { status: 404 })
    }

    if (url.pathname === '/api/review/due') {
      const now = new Date()
      const due = concepts.filter(c => true)
      return new Response(JSON.stringify(due.slice(0, 10)), {
        headers: { 'Content-Type': 'application/json' },
      })
    }

    // POST /api/infer-frontmatter — LLM 从 .md 内容推断概念元数据
    if (url.pathname === '/api/infer-frontmatter' && req.method === 'POST') {
      try {
        const body = await req.json()
        const { filename, content } = body
        if (!content) return new Response(JSON.stringify({ error: 'content required' }), { status: 400, headers: { 'Content-Type': 'application/json' } })

        const prompt = `你是一个知识图谱概念分析助手。请阅读以下 Markdown 文档，提取出该文档描述的核心概念信息。

文档文件名: ${filename || 'unknown'}

文档内容:
${content.slice(0, 3000)}

请返回 JSON（不要其他文字）：
{
  "id": "唯一英文ID",
  "title": "概念完整名称",
  "alias": ["别名1", "别名2"],
  "level": 1|2|3,
  "category": "所属分类",
  "problem": "该概念要解决的核心问题",
  "gap_anticipate": "学习时常见的认知缺口",
  "depends_on_titles": ["前置概念标题"],
  "leads_to_titles": ["引出概念标题"],
  "related_titles": [],
  "elements": [
    { "name": "要素名", "description": "简要描述", "type": "core_field|design_pattern|key_insight|boundary", "order": 1 }
  ]
}

注意：
- depends_on_titles / leads_to_titles 是用概念标题而非ID
- elements 列出该概念的核心组成部分（3-5个）
- category 从以下选: Foundation, Model, Performance, Scheduling, Serving, Advanced, Optimization, Infrastructure`
        
        const apiKey = process.env.DEEPSEEK_API_KEY || ''
        const baseUrl = process.env.DEEPSEEK_BASE_URL || 'https://api.deepseek.com/v1'
        const resp = await fetch(`${baseUrl}/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
          body: JSON.stringify({
            model: 'deepseek-chat',
            messages: [
              { role: 'system', content: '你是一个知识图谱概念分析助手。只输出 JSON，不要额外文字。' },
              { role: 'user', content: prompt }
            ],
            max_tokens: 2000,
            response_format: { type: 'json_object' },
          }),
        })
        const data = await resp.json() as any
        const raw = data?.choices?.[0]?.message?.content || '{}'
        const result = JSON.parse(raw)
        return new Response(JSON.stringify(result), {
          headers: { 'Content-Type': 'application/json' },
        })
      } catch (error) {
        return new Response(JSON.stringify({ error: String(error) }), { status: 500, headers: { 'Content-Type': 'application/json' } })
      }
    }

    // POST /api/generate-doc — LLM 生成文档正文
    if (url.pathname === '/api/generate-doc' && req.method === 'POST') {
      try {
        const body = await req.json()
        const { concept } = body
        if (!concept) return new Response(JSON.stringify({ error: 'concept required' }), { status: 400, headers: { 'Content-Type': 'application/json' } })
        
        const prompt = `你是一个技术文档撰写助手。请根据以下概念信息生成一份中文 Markdown 文档。

概念名称: ${concept.title || ''}
核心问题: ${concept.problem || ''}
认知缺口: ${concept.gap_anticipate || ''}

文档要求：
1. 以"# 标题"开头
2. 包含"## 问题"章节，解释这个概念要解决什么问题
3. 包含"## 核心设计"章节，描述核心设计思路
4. 包含"## 使用示例"章节，给出代码或伪代码示例

只输出 Markdown 内容，不要额外说明。`

        const apiKey = process.env.DEEPSEEK_API_KEY || ''
        const baseUrl = process.env.DEEPSEEK_BASE_URL || 'https://api.deepseek.com/v1'
        const resp = await fetch(`${baseUrl}/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
          body: JSON.stringify({
            model: 'deepseek-chat',
            messages: [{ role: 'system', content: '你是一个技术文档撰写助手。' }, { role: 'user', content: prompt }],
            max_tokens: 2000,
          }),
        })
        const data = await resp.json() as any
        const content = data?.choices?.[0]?.message?.content || ''
        return new Response(JSON.stringify({ content }), {
          headers: { 'Content-Type': 'application/json' },
        })
      } catch (error) {
        return new Response(JSON.stringify({ error: String(error) }), { status: 500, headers: { 'Content-Type': 'application/json' } })
      }
    }

    // GET /api/cluster — 对选中文档运行聚类
    if (url.pathname === '/api/cluster' && req.method === 'GET') {
      try {
        const docsPath = url.searchParams.get('path') || DOCS_DIR
        const resParam = url.searchParams.get('resolution')
        const resolution = resParam ? parseFloat(resParam) : 0.5

        const result = clusterPipeline(docsPath, null, resolution)
        return new Response(JSON.stringify(result), {
          headers: { 'Content-Type': 'application/json' },
        })
      } catch (error) {
        return new Response(JSON.stringify({ error: String(error) }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' },
        })
      }
    }

    // POST /api/ask-question — AI 回答用户关于选中文本的问题
    if (url.pathname === '/api/ask-question' && req.method === 'POST') {
      try {
        const body = await req.json()
        const { selectedText, question } = body
        if (!selectedText || !question) {
          return new Response(JSON.stringify({ error: 'selectedText and question required' }), { status: 400, headers: { 'Content-Type': 'application/json' } })
        }

        const prompt = `你是一个技术学习助手。用户在学习 vLLM 相关知识时，针对一段文本提出了一个问题。

选中文本:
"""
${selectedText}
"""

用户问题:
${question}

请用中文回答用户的问题。回答要简明扼要、切中要点。如果选中文本不足以回答问题，请基于你的知识补充说明。`

        const apiKey = process.env.DEEPSEEK_API_KEY || ''
        const baseUrl = process.env.DEEPSEEK_BASE_URL || 'https://api.deepseek.com/v1'
        const resp = await fetch(`${baseUrl}/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
          body: JSON.stringify({
            model: 'deepseek-chat',
            messages: [
              { role: 'system', content: '你是一个技术学习助手，帮助用户理解技术概念和回答问题。' },
              { role: 'user', content: prompt },
            ],
            max_tokens: 1000,
          }),
        })
        const data = await resp.json() as any
        const answer = data?.choices?.[0]?.message?.content || ''
        return new Response(JSON.stringify({ answer }), {
          headers: { 'Content-Type': 'application/json' },
        })
      } catch (error) {
        return new Response(JSON.stringify({ error: String(error) }), { status: 500, headers: { 'Content-Type': 'application/json' } })
      }
    }

    // POST /api/infer-relations-from-content — 基于文本相似度推断概念关系（纯算法，无 LLM）
    if (url.pathname === '/api/infer-relations-from-content' && req.method === 'POST') {
      try {
        const body = await req.json()
        const { conceptId, content } = body
        if (!conceptId || !content) {
          return new Response(JSON.stringify({ error: 'conceptId and content required' }), { status: 400, headers: { 'Content-Type': 'application/json' } })
        }

        // 从服务端内存中获取所有其他概念的内容
        const candidates = concepts.filter(c => c.id !== conceptId && c.content && c.content.trim())
        if (candidates.length === 0) {
          return new Response(JSON.stringify({ relations: [] }), { headers: { 'Content-Type': 'application/json' } })
        }

        // Tokenize 函数：去 Markdown 语法，提取中文 + 英文关键词
        function tokenize(text: string): Set<string> {
          const plain = text
            .replace(/^---[\s\S]*?---\n/, '')  // 移除 frontmatter
            .replace(/[#*`~\[\]()>|\\]/g, ' ')  // 移除 markdown 符号
            .replace(/\s+/g, ' ')
            .toLowerCase()
          // 提取中文词组（2-4 字）和英文单词（>=3 字母）
          const tokens = new Set<string>()
          // 中文：2-4 字滑动窗口
          for (let i = 0; i < plain.length - 1; i++) {
            if (/[\u4e00-\u9fff]/.test(plain[i])) {
              for (let len = 2; len <= 4 && i + len <= plain.length; len++) {
                const phrase = plain.slice(i, i + len)
                if (/^[\u4e00-\u9fff]+$/.test(phrase)) tokens.add(phrase)
              }
            }
          }
          // 英文：>=3 字母且非停用词
          const stopWords = new Set(['the', 'and', 'for', 'this', 'that', 'with', 'from', 'which', 'when', 'what', 'into', 'over', 'such', 'each', 'also', 'will', 'can', 'has', 'had', 'but', 'not', 'are', 'was', 'its', 'than', 'then', 'they', 'been', 'more', 'very', 'just', 'should', 'about', 'their', 'there', 'these', 'those', 'have', 'does', 'done', 'being', 'some', 'would', 'could', 'other', 'after', 'before', 'between', 'through', 'during', 'without', 'within', 'across', 'along', 'among', 'around', 'above', 'below', 'under'])
          for (const m of plain.matchAll(/[a-z]{3,}/g)) {
            if (!stopWords.has(m[0])) tokens.add(m[0])
          }
          return tokens
        }

        const targetTokens = tokenize(content)
        if (targetTokens.size === 0) {
          return new Response(JSON.stringify({ relations: [] }), { headers: { 'Content-Type': 'application/json' } })
        }

        // Jaccard 相似度计算
        interface RelationScore {
          targetId: string
          targetTitle: string
          score: number
        }
        const scores: RelationScore[] = []

        for (const c of candidates) {
          const candidateTokens = tokenize(c.content || '')
          if (candidateTokens.size === 0) continue

          // Jaccard 相似度
          let intersectionSize = 0
          for (const t of targetTokens) {
            if (candidateTokens.has(t)) intersectionSize++
          }
          const unionSize = targetTokens.size + candidateTokens.size - intersectionSize
          const similarity = unionSize > 0 ? intersectionSize / unionSize : 0

          // 标题匹配加分（概念标题在内容中出现）
          let titleBoost = 0
          const titleWords = c.title.toLowerCase().split(/[\s_-]+/).filter(w => w.length >= 2)
          for (const w of titleWords) {
            if (content.toLowerCase().includes(w)) { titleBoost += 0.05 }
          }

          const finalScore = Math.min(1, similarity + titleBoost)
          if (finalScore > 0.01) {
            scores.push({ targetId: c.id, targetTitle: c.title, score: finalScore })
          }
        }

        // 按分数排序，取 top 8
        scores.sort((a, b) => b.score - a.score)
        const topRelations = scores.slice(0, 8)

        return new Response(JSON.stringify({ relations: topRelations }), {
          headers: { 'Content-Type': 'application/json' },
        })
      } catch (error) {
        return new Response(JSON.stringify({ error: String(error) }), { status: 500, headers: { 'Content-Type': 'application/json' } })
      }
    }

    // POST /api/write-doc — 写入 .md 文件
    if (url.pathname === '/api/write-doc' && req.method === 'POST') {
      try {
        const body = await req.json()
        const { path: filePath, content } = body
        if (!filePath || content === undefined) return new Response(JSON.stringify({ error: 'path and content required' }), { status: 400, headers: { 'Content-Type': 'application/json' } })
        await Bun.write(filePath, content)
        // 同步更新服务端内存中的 documents 和 concepts，确保刷新后 /api/graph 返回最新内容
        const doc = documents.find(d => d.path === filePath)
        if (doc) {
          const fmMatch = doc.content.match(/^---[\s\S]*?---\n/)
          doc.content = fmMatch ? fmMatch[0] + content : content
          const c = concepts.find(c => c.id === doc.id)
          if (c) c.content = content
        }
        return new Response(JSON.stringify({ success: true }), {
          headers: { 'Content-Type': 'application/json' },
        })
      } catch (error) {
        return new Response(JSON.stringify({ error: String(error) }), { status: 500, headers: { 'Content-Type': 'application/json' } })
      }
    }

    // === PROJECT API ROUTES ===
    if (url.pathname === '/api/projects' && req.method === 'GET') {
      try {
        const projects = await loadProjectList()
        return new Response(JSON.stringify({ projects }), {
          headers: { 'Content-Type': 'application/json' },
        })
      } catch (error) {
        return new Response(JSON.stringify({ error: String(error) }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' },
        })
      }
    }

    if (url.pathname === '/api/projects' && req.method === 'POST') {
      try {
        const body = await req.json()
        const { name, folderPath } = body
        if (!name || !folderPath) {
          return new Response(JSON.stringify({ error: 'name and folderPath required' }), {
            status: 400, headers: { 'Content-Type': 'application/json' },
          })
        }
        if (!existsSync(folderPath)) {
          return new Response(JSON.stringify({ error: 'Folder does not exist' }), {
            status: 400, headers: { 'Content-Type': 'application/json' },
          })
        }
        const projectId = generateProjectId()
        const projectDir = await createProjectDir(projectId)
        const concepts: any[] = []
        await scanDirectoryForIndex(folderPath, concepts)
        const ids = new Map(concepts.map(c => [c.id, c]))
        const edges: any[] = []
        const edgeSet = new Set<string>()
        for (const c of concepts) {
          for (const t of c.leads_to || []) {
            if (ids.has(t)) {
              const eid = c.id + '-leads-' + t
              if (!edgeSet.has(eid)) { edgeSet.add(eid); edges.push({ id: eid, source: c.id, target: t, type: 'leads_to' }) }
            }
          }
          for (const t of c.depends_on || []) {
            if (ids.has(t)) {
              const eid = c.id + '-depends-' + t
              if (!edgeSet.has(eid)) { edgeSet.add(eid); edges.push({ id: eid, source: c.id, target: t, type: 'depends_on' }) }
            }
          }
          for (const t of c.related || []) {
            if (ids.has(t)) {
              const eid = c.id + '-related-' + t
              if (!edgeSet.has(eid)) { edgeSet.add(eid); edges.push({ id: eid, source: c.id, target: t, type: 'related' }) }
            }
          }
        }
        const now = new Date().toISOString()
        const project: Project = {
          id: projectId, name, folderPath, createdAt: now, lastOpenedAt: now,
        }
        await writeFile(join(projectDir, 'config.json'), JSON.stringify(project, null, 2))
        await writeFile(join(projectDir, 'concepts.json'), JSON.stringify(concepts, null, 2))
        await writeFile(join(projectDir, 'edges.json'), JSON.stringify(edges, null, 2))
        const projects = await loadProjectList()
        projects.push(project)
        await saveProjectList(projects)
        return new Response(JSON.stringify({ project, concepts, edges }), {
          headers: { 'Content-Type': 'application/json' },
        })
      } catch (error) {
        return new Response(JSON.stringify({ error: String(error) }), {
          status: 500, headers: { 'Content-Type': 'application/json' },
        })
      }
    }

    const projectsGetMatch = url.pathname.match(/^\/api\/projects\/([^\/]+)$/)
    if (projectsGetMatch && req.method === 'GET') {
      const projectId = projectsGetMatch[1]
      const projectDir = join(PROJECTS_DIR, projectId)
      if (!existsSync(projectDir)) {
        return new Response(JSON.stringify({ error: 'Project not found' }), {
          status: 404, headers: { 'Content-Type': 'application/json' },
        })
      }
      try {
        const configContent = await readFile(join(projectDir, 'config.json'), 'utf-8')
        const project = JSON.parse(configContent)
        let concepts: any[] = []
        let edges: any[] = []
        if (existsSync(join(projectDir, 'concepts.json'))) {
          concepts = JSON.parse(await readFile(join(projectDir, 'concepts.json'), 'utf-8'))
        }
        if (existsSync(join(projectDir, 'edges.json'))) {
          edges = JSON.parse(await readFile(join(projectDir, 'edges.json'), 'utf-8'))
        }
        project.lastOpenedAt = new Date().toISOString()
        await writeFile(join(projectDir, 'config.json'), JSON.stringify(project, null, 2))
        const projects = await loadProjectList()
        const idx = projects.findIndex(p => p.id === projectId)
        if (idx >= 0) {
          projects[idx].lastOpenedAt = project.lastOpenedAt
          await saveProjectList(projects)
        }
        return new Response(JSON.stringify({ project, concepts, edges }), {
          headers: { 'Content-Type': 'application/json' },
        })
      } catch (error) {
        return new Response(JSON.stringify({ error: String(error) }), {
          status: 500, headers: { 'Content-Type': 'application/json' },
        })
      }
    }

    const projectsDelMatch = url.pathname.match(/^\/api\/projects\/([^\/]+)$/)
    if (projectsDelMatch && req.method === 'DELETE') {
      const projectId = projectsDelMatch[1]
      const projectDir = join(PROJECTS_DIR, projectId)
      if (!existsSync(projectDir)) {
        return new Response(JSON.stringify({ error: 'Project not found' }), {
          status: 404, headers: { 'Content-Type': 'application/json' },
        })
      }
      try {
        await rm(projectDir, { recursive: true, force: true })
        const projects = await loadProjectList()
        const filtered = projects.filter(p => p.id !== projectId)
        await saveProjectList(filtered)
        return new Response(JSON.stringify({ success: true }), {
          headers: { 'Content-Type': 'application/json' },
        })
      } catch (error) {
        return new Response(JSON.stringify({ error: String(error) }), {
          status: 500, headers: { 'Content-Type': 'application/json' },
        })
      }
    }

    // END PROJECT API ROUTES
    // POST /api/write-doc — 写入 .md 文件
    if (url.pathname === '/api/write-doc' && req.method === 'POST') {
      try {
        const body = await req.json()
        const { path: filePath, content } = body
        if (!filePath || content === undefined) return new Response(JSON.stringify({ error: 'path and content required' }), { status: 400, headers: { 'Content-Type': 'application/json' } })
        await Bun.write(filePath, content)
        // 同步更新服务端内存中的 documents 和 concepts，确保刷新后 /api/graph 返回最新内容
        const doc = documents.find(d => d.path === filePath)
        if (doc) {
          const fmMatch = doc.content.match(/^---[\s\S]*?---\n/)
          doc.content = fmMatch ? fmMatch[0] + content : content
          const c = concepts.find(c => c.id === doc.id)
          if (c) c.content = content
        }
        return new Response(JSON.stringify({ success: true }), {
          headers: { 'Content-Type': 'application/json' }
        })
      } catch (error) {
        return new Response(JSON.stringify({ error: String(error) }), { status: 500, headers: { 'Content-Type': 'application/json' } })
      }
    }

    // DELETE /api/delete-doc
    if (url.pathname === '/api/delete-doc' && req.method === 'DELETE') {
      try {
        const body = await req.json()
        const { path: filePath } = body
        if (!filePath) return new Response(JSON.stringify({ error: 'path required' }), { status: 400, headers: { 'Content-Type': 'application/json' } })

        try { await rm(filePath) } catch {}

        const docIdx = documents.findIndex(d => d.path === filePath)
        const docId = docIdx !== -1 ? documents[docIdx].id : null
        if (docIdx !== -1) documents.splice(docIdx, 1)

        const conceptIdx = concepts.findIndex(c => c.path === filePath)
        const conceptId = conceptIdx !== -1 ? concepts[conceptIdx].id : (docId || filePath)
        if (conceptIdx !== -1) concepts.splice(conceptIdx, 1)

        for (let i = conceptEdges.length - 1; i >= 0; i--) {
          if (conceptEdges[i].source === conceptId || conceptEdges[i].target === conceptId) {
            conceptEdges.splice(i, 1)
          }
        }

        return new Response(JSON.stringify({ success: true, conceptId }), {
          headers: { 'Content-Type': 'application/json' },
        })
      } catch (error) {
        return new Response(JSON.stringify({ error: String(error) }), { status: 500, headers: { 'Content-Type': 'application/json' } })
      }
    }

    return new Response('Not found', { status: 404 })
  },
})

console.log(`Server running on http://localhost:${PORT}`)
await loadDocuments()
