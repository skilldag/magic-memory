import React, { useCallback, useEffect, useRef } from 'react'

interface SelectionPopoverProps {
  position: { x: number; y: number }
  selectedText: string
  onAddAnnotation: (type: 'comment' | 'question' | 'suggestion' | 'correction') => void
  onConceptElevation: () => void
  onClose: () => void
}

export function SelectionPopover({
  position,
  selectedText,
  onAddAnnotation,
  onConceptElevation,
  onClose,
}: SelectionPopoverProps) {
  const popoverRef = useRef<HTMLDivElement>(null)

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(selectedText)
    onClose()
  }, [selectedText, onClose])

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        onClose()
      }
    }
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    // Delay to avoid the same click that triggered the popover
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

  // Determine popover position: above selection by default, flip below if near top
  const popoverStyle: React.CSSProperties = {
    position: 'fixed',
    left: position.x,
    top: position.y - 10,
    transform: 'translate(-50%, -100%)',
    zIndex: 1000,
  }

  // If near top of viewport (< 120px from top), flip below
  if (position.y < 120) {
    popoverStyle.top = position.y + 10
    popoverStyle.transform = 'translate(-50%, 0)'
  }

  return (
    <div
      ref={popoverRef}
      style={popoverStyle}
      className="flex items-center gap-1 bg-white rounded-lg shadow-lg border border-gray-200 px-2 py-1.5"
    >
      <button
        onClick={() => onAddAnnotation('comment')}
        className="px-2 py-1 text-xs font-medium text-blue-700 bg-blue-50 rounded hover:bg-blue-100 transition-colors whitespace-nowrap"
        title="添加评论"
      >
        💬 评论
      </button>
      <button
        onClick={handleCopy}
        className="px-2 py-1 text-xs font-medium text-purple-700 bg-purple-50 rounded hover:bg-purple-100 transition-colors whitespace-nowrap"
        title="复制选中文本"
      >
        📋 复制
      </button>
      <button
        onClick={() => onAddAnnotation('correction')}
        className="px-2 py-1 text-xs font-medium text-red-700 bg-red-50 rounded hover:bg-red-100 transition-colors whitespace-nowrap"
        title="纠正错误"
      >
        ✏️ 纠正
      </button>
      <div className="w-px h-5 bg-gray-200 mx-1" />
      <button
        onClick={onConceptElevation}
        className="px-2 py-1 text-xs font-medium text-green-700 bg-green-50 rounded hover:bg-green-100 transition-colors whitespace-nowrap"
        title="提升为概念"
      >
        🔗 概念
      </button>
      <button
        onClick={onClose}
        className="p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded transition-colors"
        title="关闭"
      >
        <svg width={14} height={14} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  )
}
