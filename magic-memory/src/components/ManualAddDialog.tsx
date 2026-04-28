import { useState } from 'react'
import type { Concept } from '../types'

type RelationType = 'leads_to' | 'depends_on' | 'related'

interface ManualAddDialogProps {
  sourceConcept: Concept
  onClose: () => void
  onAdd: (source: Concept, titles: string[], relationType: RelationType) => void
}

export function ManualAddDialog({ sourceConcept, onClose, onAdd }: ManualAddDialogProps) {
  const [inputText, setInputText] = useState('')
  const [titles, setTitles] = useState<string[]>([])
  const [relationType, setRelationType] = useState<RelationType>('related')

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && e.currentTarget.value.trim()) {
      e.preventDefault()
      const val = e.currentTarget.value.trim()
      setTitles(prev => [...prev, val])
      setInputText('')
    }
  }

  const removeTitle = (index: number) => {
    setTitles(prev => prev.filter((_, i) => i !== index))
  }

  const handleSubmit = () => {
    if (titles.length === 0) return
    onAdd(sourceConcept, titles, relationType)
    setTitles([])
    setInputText('')
  }

  return (
    <div
      className="fixed inset-0 z-[9999] flex justify-center bg-black/40"
      style={{ alignItems: 'flex-start', paddingTop: '96px' }}
      onClick={onClose}
    >
      <div className="w-[480px] bg-white rounded-xl shadow border border-gray-200 p-5" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-gray-900">添加概念</h3>
          <button type="button" onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <p className="text-xs text-gray-400 mt-1">当前概念：{sourceConcept.title}。输入概念名后回车添加。</p>

        <div className="mt-3 flex flex-wrap gap-1.5 p-2 bg-gray-50 rounded-lg min-h-[36px]">
          {titles.map((tag, i) => (
            <span
              key={i}
              className="inline-flex items-center gap-1 px-2 py-0.5 bg-white rounded-md border border-gray-200 text-xs text-gray-700 shadow-sm"
            >
              {tag}
              <button type="button" onClick={() => removeTitle(i)} className="text-gray-400 hover:text-gray-600 leading-none">×</button>
            </span>
          ))}
          <input
            type="text"
            value={inputText}
            onChange={e => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入概念名，回车添加..."
            className="flex-1 min-w-[120px] text-xs outline-none border-none bg-transparent"
          />
        </div>

        <div className="mt-2.5">
          <label className="text-xs text-gray-400">关系类型</label>
          <div className="mt-1 flex gap-2">
            {(['related', 'depends_on', 'leads_to'] as const).map(t => (
              <button
                key={t}
                type="button"
                onClick={() => setRelationType(t)}
                className={`px-2.5 py-1 text-xs rounded border ${
                  relationType === t
                    ? 'bg-blue-50 border-blue-300 text-blue-700 font-medium'
                    : 'border-gray-200 text-gray-500'
                }`}
              >
                {t === 'related' ? '相关' : t === 'depends_on' ? '依赖' : '引出'}
              </button>
            ))}
          </div>
        </div>

        <div className="mt-3 flex justify-end gap-2 border-t border-gray-100 pt-3">
          <button type="button" className="px-3 py-1.5 text-xs text-gray-500 hover:text-gray-700" onClick={onClose}>
            取消
          </button>
          <button
            type="button"
            className="px-4 py-1.5 text-xs rounded-lg bg-gray-900 text-white hover:bg-gray-800 disabled:opacity-40"
            disabled={titles.length === 0}
            onClick={handleSubmit}
          >
            提交 {titles.length > 0 ? `(${titles.length}个)` : ''}
          </button>
        </div>
      </div>
    </div>
  )
}
