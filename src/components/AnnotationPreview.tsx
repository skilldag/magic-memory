import React, { useEffect, useRef } from 'react'
import type { Annotation } from '../types'

interface AnnotationPreviewProps {
  annotation: Annotation
  position: { x: number; y: number }
  onViewDetails: (annotation: Annotation) => void
  onClose: () => void
}

const typeIcons: Record<string, string> = {
  comment: '💬',
  question: '❓',
  suggestion: '💡',
  correction: '✏️',
}

const typeLabels: Record<string, string> = {
  comment: '评论',
  question: '提问',
  suggestion: '建议',
  correction: '纠正',
}

const statusLabels: Record<string, string> = {
  open: '开放',
  resolved: '已解决',
  closed: '已关闭',
}

function timeAgo(date: Date): string {
  const now = new Date()
  const diff = now.getTime() - new Date(date).getTime()
  const days = Math.floor(diff / (1000 * 60 * 60 * 24))
  if (days === 0) return '今天'
  if (days === 1) return '昨天'
  if (days < 7) return `${days}天前`
  return new Date(date).toLocaleDateString()
}

export function AnnotationPreview({
  annotation,
  position,
  onViewDetails,
  onClose,
}: AnnotationPreviewProps) {
  const previewRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (previewRef.current && !previewRef.current.contains(e.target as Node)) {
        onClose()
      }
    }
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    const timer = setTimeout(() => {
      document.addEventListener('mousedown', handleClickOutside)
      document.addEventListener('keydown', handleEscape)
    }, 0)
    return () => {
      clearTimeout(timer)
      document.removeEventListener('mousedown', handleClickOutside)
      document.removeEventListener('keydown', handleEscape)
    }
  }, [onClose])

  const previewStyle: React.CSSProperties = {
    position: 'fixed',
    left: position.x,
    top: position.y - 10,
    transform: 'translate(-50%, -100%)',
    zIndex: 1000,
  }

  if (position.y < 150) {
    previewStyle.top = position.y + 10
    previewStyle.transform = 'translate(-50%, 0)'
  }

  return (
    <div
      ref={previewRef}
      style={previewStyle}
      className="w-72 bg-white rounded-lg shadow-lg border border-gray-200 overflow-hidden"
    >
      <div className="px-3 py-2 border-b border-gray-100 flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <span className="text-sm">{typeIcons[annotation.type] || '💬'}</span>
          <span className="text-xs font-medium text-gray-700">
            {typeLabels[annotation.type] || annotation.type}
          </span>
          <span className="text-xs text-gray-400">·</span>
          <span className="text-xs text-gray-400">{annotation.author}</span>
          <span className="text-xs text-gray-400">·</span>
          <span className="text-xs text-gray-400">{timeAgo(annotation.createdAt)}</span>
        </div>
        <span className={`text-xs px-1.5 py-0.5 rounded ${
          annotation.status === 'open' ? 'bg-green-100 text-green-700' :
          annotation.status === 'resolved' ? 'bg-blue-100 text-blue-700' :
          'bg-gray-100 text-gray-500'
        }`}>
          {statusLabels[annotation.status] || annotation.status}
        </span>
      </div>
      <div className="px-3 py-2">
        <p className="text-sm text-gray-700 line-clamp-3 leading-relaxed">
          {annotation.content}
        </p>
        {annotation.replies && annotation.replies.length > 0 && (
          <div className="mt-1.5 text-xs text-gray-400">
            ↩ {annotation.replies.length} 条回复
          </div>
        )}
      </div>
      <div className="px-3 py-1.5 bg-gray-50 border-t border-gray-100">
        <button
          onClick={() => onViewDetails(annotation)}
          className="text-xs text-blue-600 hover:text-blue-800 font-medium"
        >
          查看详情 →
        </button>
      </div>
    </div>
  )
}
