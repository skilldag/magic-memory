import type { Concept, SuggestionItem } from '../types'

interface BatchLinkDialogProps {
  sourceConcept: Concept
  suggestions: SuggestionItem[]
  loading: boolean
  onClose: () => void
  onToggle: (index: number) => void
  onConfirm: () => void
}

export function BatchLinkDialog({ sourceConcept, suggestions, loading, onClose, onToggle, onConfirm }: BatchLinkDialogProps) {
  return (
    <div
      className="fixed inset-0 z-[9999] flex justify-center bg-black/40"
      style={{ alignItems: 'flex-start', paddingTop: '96px' }}
      onClick={onClose}
    >
      <div className="w-[480px] bg-white rounded-xl shadow border border-gray-200 p-5" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-gray-900">AI 生成概念</h3>
          <button type="button" onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <p className="text-xs text-gray-400 mt-1">当前概念：{sourceConcept.title}。勾选后确认添加。</p>

        {loading ? (
          <div className="mt-4 text-sm text-gray-500">正在生成建议...</div>
        ) : suggestions.length === 0 ? (
          <div className="mt-4 text-sm text-gray-400">没有新的概念建议。</div>
        ) : (
          <div className="mt-3 space-y-1.5 max-h-[320px] overflow-y-auto">
            {suggestions.map((item, idx) => (
              <label
                key={`${item.title}-${idx}`}
                className="flex items-center gap-2 p-2 rounded border border-gray-200 hover:bg-gray-50 cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={item.checked}
                  onChange={() => onToggle(idx)}
                  className="rounded"
                />
                <div className="min-w-0">
                  <div className="text-sm text-gray-800">{item.title}</div>
                  <div className="text-xs text-gray-400">{item.problem}</div>
                </div>
              </label>
            ))}
          </div>
        )}

        <div className="mt-3 flex justify-end gap-2 border-t border-gray-100 pt-3">
          <button type="button" className="px-3 py-1.5 text-xs text-gray-500 hover:text-gray-700" onClick={onClose}>
            取消
          </button>
          <button
            type="button"
            className="px-4 py-1.5 text-xs rounded-lg bg-gray-900 text-white hover:bg-gray-800"
            onClick={onConfirm}
          >
            确认添加
          </button>
        </div>
      </div>
    </div>
  )
}
