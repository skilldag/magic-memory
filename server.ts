import { serve } from 'bun'
import { readdir, readFile, stat, writeFile, rm } from 'fs/promises'
import { join, dirname } from 'path'
import { homedir } from 'os'
import { existsSync, mkdirSync } from 'fs'
import type { Document, Project, Annotation } from './src/types'

const PORT = 3001
const DOCS_DIR = join(import.meta.dir, 'docs')

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

// ===== Project Management Helpers =====
async function loadProjectList(): Promise<Project[]> {
  try {
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
            id: entry.name,
            title,
            path: fullPath,
            content,
            level: 1,
            category: '',
            tags: [],
            lastModified: stats.mtime,
            metadata: { status: 'draft' },
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
            id: entryRelativePath,
            title: entry.name.replace('.md', ''),
            path: fullPath,
            content,
            level: 1,
            category: '',
            tags: [],
            lastModified: stats.mtime,
            metadata: { status: 'draft' },
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

const server = serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url)

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

    // POST /api/write-doc — 写入 .md 文件
    if (url.pathname === '/api/write-doc' && req.method === 'POST') {
      try {
        const body = await req.json()
        const { path: filePath, content } = body
        if (!filePath || content === undefined) return new Response(JSON.stringify({ error: 'path and content required' }), { status: 400, headers: { 'Content-Type': 'application/json' } })
        await Bun.write(filePath, content)
        // 同步更新服务端内存中的文档缓存
        const doc = documents.find(d => d.path === filePath)
        if (doc) {
          const fmMatch = doc.content.match(/^---[\s\S]*?---\n/)
          doc.content = fmMatch ? fmMatch[0] + content : content
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
        const now = new Date().toISOString()
        const project: Project = {
          id: projectId, name, folderPath, createdAt: now, lastOpenedAt: now,
        }
        await writeFile(join(projectDir, 'config.json'), JSON.stringify(project, null, 2))
        const projects = await loadProjectList()
        projects.push(project)
        await saveProjectList(projects)
        return new Response(JSON.stringify({ project }), {
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

    // DELETE /api/delete-doc
    if (url.pathname === '/api/delete-doc' && req.method === 'DELETE') {
      try {
        const body = await req.json()
        const { path: filePath } = body
        if (!filePath) return new Response(JSON.stringify({ error: 'path required' }), { status: 400, headers: { 'Content-Type': 'application/json' } })

        try { await rm(filePath) } catch {}

        const docIdx = documents.findIndex(d => d.path === filePath)
        if (docIdx !== -1) documents.splice(docIdx, 1)

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
