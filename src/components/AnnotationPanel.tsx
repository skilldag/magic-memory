import React from 'react'
import { useAnnotationStore } from '../store/annotationStore'
import type { Document } from '../types'

interface AnnotationPanelProps {
  document: Document
  onClose: () => void
  onNavigateToAnnotation?: (annotationId: string) => void
}

export function AnnotationPanel({ document, onClose, onNavigateToAnnotation }: AnnotationPanelProps) {
  const {
    annotations,
    selectedAnnotation,
    selectAnnotation,
    updateAnnotation,
    deleteAnnotation,
    addReply,
    getStats,
  } = useAnnotationStore()

  const docAnnotations = annotations.filter((ann) => ann.documentId === document.id)
  const stats = getStats(document.id)

  const handleStatusChange = (annotationId: string, status: 'open' | 'resolved' | 'closed') => {
    updateAnnotation(annotationId, { status })
  }

  const handleReplySubmit = (annotationId: string, content: string) => {
    if (content.trim()) {
      addReply(annotationId, {
        content: content.trim(),
        author: 'User',
      })
    }
  }

  return (
    <div className="w-80 lg:w-96 border-l border-gray-200 flex flex-col bg-white shrink-0">
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">注释</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-200 rounded"
            aria-label="关闭注释面板"
          >
            <svg width={20} height={20} className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="bg-blue-50 rounded-lg p-2">
            <div className="text-2xl font-bold text-blue-600">{stats.total}</div>
            <div className="text-xs text-blue-600">总计</div>
          </div>
          <div className="bg-green-50 rounded-lg p-2">
            <div className="text-2xl font-bold text-green-600">{stats.recent}</div>
            <div className="text-xs text-green-600">最近</div>
          </div>
          <div className="bg-purple-50 rounded-lg p-2">
            <div className="text-2xl font-bold text-purple-600">
              {Object.keys(stats.byType).length}
            </div>
            <div className="text-xs text-purple-600">类型</div>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {docAnnotations.length === 0 ? (
          <div className="p-4 text-center text-gray-500">
            <div className="mb-4 text-4xl">💬</div>
            <p>还没有注释</p>
            <p className="text-sm mt-2">选择文本后点击工具栏添加注释</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {docAnnotations.map((annotation) => (
              <div
                key={annotation.id}
                className={`p-4 hover:bg-gray-50 cursor-pointer transition-colors ${
                  selectedAnnotation?.id === annotation.id ? 'bg-blue-50' : ''
                }`}
                onClick={() => selectAnnotation(annotation)}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-xs px-2 py-0.5 rounded ${
                        annotation.type === 'comment'
                          ? 'bg-blue-100 text-blue-800'
                          : annotation.type === 'question'
                          ? 'bg-purple-100 text-purple-800'
                          : annotation.type === 'suggestion'
                          ? 'bg-green-100 text-green-800'
                          : 'bg-red-100 text-red-800'
                      }`}
                    >
                      {annotation.type}
                    </span>
                    <span className="text-xs text-gray-500">
                      {new Date(annotation.createdAt).toLocaleDateString()}
                    </span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        selectAnnotation(annotation)
                        onNavigateToAnnotation?.(annotation.id)
                      }}
                      className="text-xs text-blue-500 hover:text-blue-700 ml-1"
                      title="在文档中定位"
                    >
                      📍
                    </button>
                  </div>
                  <select
                    value={annotation.status}
                    onChange={(e) =>
                      handleStatusChange(annotation.id, e.target.value as 'open' | 'resolved' | 'closed')
                    }
                    className="text-xs border border-gray-300 rounded px-2 py-1"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <option value="open">开放</option>
                    <option value="resolved">已解决</option>
                    <option value="closed">已关闭</option>
                  </select>
                </div>

                <p className="text-sm text-gray-700 mb-2">{annotation.content}</p>

                {annotation.replies && annotation.replies.length > 0 && (
                  <div className="mt-3 space-y-2">
                    {annotation.replies.map((reply) => (
                      <div key={reply.id} className="bg-gray-50 rounded p-2">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs font-medium text-gray-700">{reply.author}</span>
                          <span className="text-xs text-gray-500">
                            {new Date(reply.createdAt).toLocaleDateString()}
                          </span>
                        </div>
                        <p className="text-xs text-gray-600">{reply.content}</p>
                      </div>
                    ))}
                  </div>
                )}

                <div className="mt-3 flex items-center gap-2">
                  <input
                    type="text"
                    placeholder="添加回复..."
                    className="flex-1 text-xs border border-gray-300 rounded px-2 py-1"
                    onClick={(e) => e.stopPropagation()}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && e.currentTarget.value.trim()) {
                        handleReplySubmit(annotation.id, e.currentTarget.value)
                        e.currentTarget.value = ''
                      }
                    }}
                  />
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      deleteAnnotation(annotation.id)
                    }}
                    className="text-xs text-red-600 hover:text-red-800"
                  >
                    删除
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
