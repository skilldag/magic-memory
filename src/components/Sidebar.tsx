import { useState } from 'react'
import { ProjectList } from './ProjectList'
import { useProjectStore } from '../store/projectStore'
import type { Document } from '../types'

interface SidebarProps {
  documents: Document[]
  selectedDoc: Document | null
  onDocumentSelect: (doc: Document) => void
  onClose: () => void
}

export function Sidebar({ documents, selectedDoc, onDocumentSelect, onClose }: SidebarProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedLevel, setSelectedLevel] = useState<number | null>(null)
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)

  const handleAddProject = async () => {
    if ('showDirectoryPicker' in window) {
      try {
        const handle = await (window as any).showDirectoryPicker()
        const name = handle.name
        const demoPaths = ['/Users/meetai/source/', '/home/user/', './']
        const suggested = demoPaths[0] + name
        const folderPath = prompt(
          `已选择文件夹「${name}」。\n请确认完整路径，否则文件读写可能失败:`,
          suggested
        )
        if (folderPath) {
          await useProjectStore.getState().createProject(name, folderPath)
        }
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          console.error('Failed to select folder:', err)
        }
      }
    } else {
      alert('Your browser does not support folder picker. Please enter the path manually.')
      const folderPath = prompt('请输入文件夹完整路径:')
      if (folderPath) {
        const name = folderPath.split('/').pop() || 'New Project'
        await useProjectStore.getState().createProject(name, folderPath)
      }
    }
  }

  const filteredDocuments = documents.filter((doc) => {
    const matchesSearch =
      searchQuery === '' ||
      doc.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      doc.content.toLowerCase().includes(searchQuery.toLowerCase())

    const matchesLevel = selectedLevel === null || doc.level === selectedLevel
    const matchesCategory = selectedCategory === null || doc.category === selectedCategory

    return matchesSearch && matchesLevel && matchesCategory
  })

  const levels = Array.from(new Set(documents.map((doc) => doc.level))).sort((a, b) => a - b)
  const categories = Array.from(new Set(documents.map((doc) => doc.category)))

  return (
    <div className="w-72 lg:w-80 border-r border-gray-200 flex flex-col bg-gray-50 shrink-0">
      <div className="border-b border-gray-200">
        <ProjectList onAddProject={handleAddProject} />
      </div>
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">文档列表</h2>
          <div className="flex items-center gap-2">
            {/* Removed: 导入文档按钮; replaced by ProjectList-based import flow */}
            <button
              onClick={onClose}
              className="p-1 hover:bg-gray-200 rounded"
              aria-label="关闭侧边栏"
            >
              <svg width={20} height={20} className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <input
          type="text"
          placeholder="搜索文档..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />

        <div className="mt-4 space-y-2">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">级别</label>
            <select
              value={selectedLevel ?? ''}
              onChange={(e) => setSelectedLevel(e.target.value ? Number(e.target.value) : null)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">全部级别</option>
              {levels.map((level) => (
                <option key={level} value={level}>
                  Level {level}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">分类</label>
            <select
              value={selectedCategory ?? ''}
              onChange={(e) => setSelectedCategory(e.target.value || null)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">全部分类</option>
              {categories.map((category) => (
                <option key={category} value={category}>
                  {category}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {filteredDocuments.length === 0 ? (
          <div className="p-4 text-center text-gray-500">
            <p>没有找到匹配的文档</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {filteredDocuments.map((doc) => (
              <button
                key={doc.path || doc.id}
                onClick={() => onDocumentSelect(doc)}
                className={`w-full text-left px-4 py-3 hover:bg-gray-100 transition-colors ${
                  selectedDoc?.id === doc.id ? 'bg-blue-50 border-l-4 border-blue-500' : ''
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm font-medium text-gray-900 truncate">{doc.title}</h3>
                    <div className="mt-1 flex items-center gap-2">
                      <span className="text-xs px-2 py-0.5 bg-blue-100 text-blue-800 rounded">
                        L{doc.level}
                      </span>
                      <span className="text-xs text-gray-500">{doc.category}</span>
                    </div>
                  </div>
                  {doc.metadata?.status && (
                    <span
                      className={`text-xs px-2 py-0.5 rounded ${
                        doc.metadata.status === 'approved'
                          ? 'bg-green-100 text-green-800'
                          : doc.metadata.status === 'review'
                          ? 'bg-yellow-100 text-yellow-800'
                          : 'bg-gray-100 text-gray-800'
                      }`}
                    >
                      {doc.metadata.status}
                    </span>
                  )}
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="p-4 border-t border-gray-200 bg-gray-50">
        <div className="text-xs text-gray-500">
          共 {filteredDocuments.length} 个文档
        </div>
      </div>
    </div>
  )
}
