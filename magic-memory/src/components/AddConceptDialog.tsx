import { useState, useRef, useEffect } from 'react'

interface AddConceptDialogProps {
  onClose: () => void
  onConfirm: (title: string) => void
  initialTitle?: string
}

export function AddConceptDialog({ onClose, onConfirm, initialTitle = '' }: AddConceptDialogProps) {
  const [title, setTitle] = useState(initialTitle)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const handleSubmit = () => {
    const trimmed = title.trim()
    if (!trimmed) return
    onConfirm(trimmed)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div
      className="fixed inset-0 z-[9999] flex justify-center bg-black/40"
      style={{ alignItems: 'flex-start', paddingTop: '96px' }}
      onClick={onClose}
    >
      <div className="w-[420px] bg-white rounded-xl shadow border border-gray-200 p-5" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-gray-900">添加新概念</h3>
          <button type="button" onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <svg width={16} height={16} className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <p className="text-xs text-gray-400 mb-3">双击空白处创建一个新概念节点。可以稍后在右侧面板编辑文档和关系。</p>

        <input
          ref={inputRef}
          type="text"
          value={title}
          onChange={e => setTitle(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入概念名称..."
          className="w-full px-3 py-2 text-sm border border-gray-200 rounded-lg outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-colors"
        />

        <div className="mt-4 flex justify-end gap-2 border-t border-gray-100 pt-3">
          <button type="button" className="px-3 py-1.5 text-xs text-gray-500 hover:text-gray-700" onClick={onClose}>
            取消
          </button>
          <button
            type="button"
            className="px-4 py-1.5 text-xs rounded-lg bg-gray-900 text-white hover:bg-gray-800 disabled:opacity-40 disabled:cursor-not-allowed"
            disabled={!title.trim()}
            onClick={handleSubmit}
          >
            确认添加
          </button>
        </div>
      </div>
    </div>
  )
}
