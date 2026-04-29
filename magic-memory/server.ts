import { serve } from 'bun'
import { readdir, readFile, stat } from 'fs/promises'
import { join, relative } from 'path'
import type { Document, Annotation, Concept, ConceptEdge } from './src/types'

const PORT = 3001
const DOCS_DIR = join(process.cwd(), '../docs')

const documents: Document[] = []
const concepts: Concept[] = []
const conceptEdges: ConceptEdge[] = []

async function loadDocuments() {
  try {
    await loadDocumentsFromDirectory(DOCS_DIR, '')
    console.log(`Loaded ${documents.length} documents`)
    
    documents.forEach(doc => {
      concepts.push(buildConceptFromDocument(doc))
    })
    conceptEdges.push(...buildEdges())
    console.log(`Built knowledge graph: ${concepts.length} concepts, ${conceptEdges.length} edges`)
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
      return new Response(JSON.stringify({ concepts, edges: conceptEdges }), {
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

    // POST /api/write-doc — 写入 .md 文件
    if (url.pathname === '/api/write-doc' && req.method === 'POST') {
      try {
        const body = await req.json()
        const { path: filePath, content } = body
        if (!filePath || content === undefined) return new Response(JSON.stringify({ error: 'path and content required' }), { status: 400, headers: { 'Content-Type': 'application/json' } })
        await Bun.write(filePath, content)
        return new Response(JSON.stringify({ success: true }), {
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