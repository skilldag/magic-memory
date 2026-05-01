import React, { useState } from 'react'
import { useAnnotationStore } from '../store/annotationStore'
import type { Document } from '../types'

interface ExportModalProps {
  document: Document
  isOpen: boolean
  onClose: () => void
}

export function ExportModal({ document, isOpen, onClose }: ExportModalProps) {
  const [exportFormat, setExportFormat] = useState<'json' | 'markdown' | 'html'>('json')
  const [includeAnnotations, setIncludeAnnotations] = useState(true)
  const [includeMetadata, setIncludeMetadata] = useState(true)
  const [isExporting, setIsExporting] = useState(false)

  const { getAnnotationsByDocument } = useAnnotationStore()
  const annotations = getAnnotationsByDocument(document.id)

  const handleExport = async () => {
    setIsExporting(true)
    try {
      let content = ''
      let filename = ''
      let mimeType = ''

      switch (exportFormat) {
        case 'json':
          content = JSON.stringify(
            {
              document: includeMetadata
                ? document
                : { id: document.id, title: document.title, content: document.content },
              annotations: includeAnnotations ? annotations : [],
            },
            null,
            2
          )
          filename = `${document.title}.json`
          mimeType = 'application/json'
          break

        case 'markdown':
          content = generateMarkdownExport(document, includeAnnotations ? annotations : [])
          filename = `${document.title}.md`
          mimeType = 'text/markdown'
          break

        case 'html':
          content = generateHtmlExport(document, includeAnnotations ? annotations : [])
          filename = `${document.title}.html`
          mimeType = 'text/html'
          break
      }

      const blob = new Blob([content], { type: mimeType })
      const url = URL.createObjectURL(blob)
      const a = window.document.createElement('a')
      a.href = url
      a.download = filename
      window.document.body.appendChild(a)
      a.click()
      window.document.body.removeChild(a)
      URL.revokeObjectURL(url)

      onClose()
    } catch (error) {
      console.error('Export failed:', error)
    } finally {
      setIsExporting(false)
    }
  }

  const generateMarkdownExport = (doc: Document, anns: any[]) => {
    let md = `# ${doc.title}\n\n`
    md += `${doc.content}\n\n`

    if (anns.length > 0) {
      md += `## 注释 (${anns.length})\n\n`
      anns.forEach((ann, index) => {
        md += `### ${index + 1}. ${ann.type}\n\n`
        md += `**位置**: ${ann.position.start}-${ann.position.end}\n\n`
        md += `**内容**: ${ann.content}\n\n`
        md += `**状态**: ${ann.status}\n\n`
        md += `**作者**: ${ann.author}\n\n`
        md += `**时间**: ${new Date(ann.createdAt).toLocaleString()}\n\n`

        if (ann.replies && ann.replies.length > 0) {
          md += `**回复**:\n\n`
          ann.replies.forEach((reply: any) => {
            md += `- ${reply.author}: ${reply.content}\n`
          })
          md += '\n'
        }
        md += '---\n\n'
      })
    }

    return md
  }

  const generateHtmlExport = (doc: Document, anns: any[]) => {
    let html = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${doc.title}</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 2rem; line-height: 1.6; }
    h1 { border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }
    .annotation { background: #f8fafc; border-left: 4px solid #3b82f6; padding: 1rem; margin: 1rem 0; }
    .annotation-header { font-weight: 600; margin-bottom: 0.5rem; }
    .annotation-content { margin-bottom: 0.5rem; }
    .annotation-meta { font-size: 0.875rem; color: #64748b; }
    .reply { background: #f1f5f9; padding: 0.5rem; margin-top: 0.5rem; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>${doc.title}</h1>
  <div class="content">${doc.content}</div>
`

    if (anns.length > 0) {
      html += `  <h2>注释 (${anns.length})</h2>\n`
      anns.forEach((ann, index) => {
        html += `  <div class="annotation">
    <div class="annotation-header">${index + 1}. ${ann.type} - ${ann.status}</div>
    <div class="annotation-content">${ann.content}</div>
    <div class="annotation-meta">
      位置: ${ann.position.start}-${ann.position.end} | 
      作者: ${ann.author} | 
      时间: ${new Date(ann.createdAt).toLocaleString()}
    </div>`

        if (ann.replies && ann.replies.length > 0) {
          html += `    <div class="replies">\n`
          ann.replies.forEach((reply: any) => {
            html += `      <div class="reply">
        <strong>${reply.author}</strong>: ${reply.content}
      </div>\n`
          })
          html += `    </div>\n`
        }

        html += `  </div>\n`
      })
    }

    html += `</body>
</html>`

    return html
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">导出文档</h3>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-200 rounded"
            aria-label="关闭"
          >
            <svg width={20} height={20} className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">导出格式</label>
            <select
              value={exportFormat}
              onChange={(e) => setExportFormat(e.target.value as 'json' | 'markdown' | 'html')}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="json">JSON</option>
              <option value="markdown">Markdown</option>
              <option value="html">HTML</option>
            </select>
          </div>

          <div className="space-y-2">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={includeAnnotations}
                onChange={(e) => setIncludeAnnotations(e.target.checked)}
                className="mr-2"
              />
              <span className="text-sm text-gray-700">包含注释 ({annotations.length})</span>
            </label>

            <label className="flex items-center">
              <input
                type="checkbox"
                checked={includeMetadata}
                onChange={(e) => setIncludeMetadata(e.target.checked)}
                className="mr-2"
              />
              <span className="text-sm text-gray-700">包含元数据</span>
            </label>
          </div>

          <div className="flex gap-2 pt-4">
            <button
              onClick={handleExport}
              disabled={isExporting}
              className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isExporting ? '导出中...' : '导出'}
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300"
            >
              取消
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
