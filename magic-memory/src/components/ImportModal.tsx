import React, { useState } from 'react'
import { useAnnotationStore } from '../store/annotationStore'
import type { Document } from '../types'

interface ImportModalProps {
  isOpen: boolean
  onClose: () => void
  onImport: (data: { document: Document; annotations: any[] }) => void
}

export function ImportModal({ isOpen, onClose, onImport }: ImportModalProps) {
  const [file, setFile] = useState<File | null>(null)
  const [importFormat, setImportFormat] = useState<'json' | 'markdown'>('json')
  const [isImporting, setIsImporting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      setError(null)

      const extension = selectedFile.name.split('.').pop()?.toLowerCase()
      if (extension === 'json') {
        setImportFormat('json')
      } else if (extension === 'md') {
        setImportFormat('markdown')
      }
    }
  }

  const handleImport = async () => {
    if (!file) {
      setError('请选择文件')
      return
    }

    setIsImporting(true)
    setError(null)

    try {
      const content = await file.text()

      if (importFormat === 'json') {
        const data = JSON.parse(content)

        if (!data.document || !data.document.title || !data.document.content) {
          throw new Error('无效的文档格式')
        }

        onImport({
          document: {
            id: data.document.id || `imported-${Date.now()}`,
            title: data.document.title,
            content: data.document.content,
            path: data.document.path || '',
            level: data.document.level || 1,
            category: data.document.category || '导入',
            tags: data.document.tags || [],
            lastModified: new Date(data.document.lastModified || Date.now()),
            metadata: data.document.metadata,
          },
          annotations: data.annotations || [],
        })
      } else if (importFormat === 'markdown') {
        const lines = content.split('\n')
        const titleMatch = lines[0].match(/^#\s+(.+)$/)
        const title = titleMatch ? titleMatch[1] : file.name.replace('.md', '')
        const content = lines.slice(1).join('\n')

        onImport({
          document: {
            id: `imported-${Date.now()}`,
            title,
            content,
            path: '',
            level: 1,
            category: '导入',
            tags: [],
            lastModified: new Date(),
          },
          annotations: [],
        })
      }

      onClose()
      setFile(null)
    } catch (error) {
      setError(error instanceof Error ? error.message : '导入失败')
    } finally {
      setIsImporting(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">导入文档</h3>
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
            <label className="block text-sm font-medium text-gray-700 mb-2">选择文件</label>
            <input
              type="file"
              accept=".json,.md"
              onChange={handleFileChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            {file && (
              <p className="mt-2 text-sm text-gray-600">已选择: {file.name}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">导入格式</label>
            <select
              value={importFormat}
              onChange={(e) => setImportFormat(e.target.value as 'json' | 'markdown')}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="json">JSON (包含注释)</option>
              <option value="markdown">Markdown (仅文档)</option>
            </select>
          </div>

          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-md">
              <p className="text-sm text-red-600">{error}</p>
            </div>
          )}

          <div className="flex gap-2 pt-4">
            <button
              onClick={handleImport}
              disabled={!file || isImporting}
              className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isImporting ? '导入中...' : '导入'}
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
