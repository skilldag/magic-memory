import React, { useState } from 'react'

interface QuestionDialogProps {
  selectedText: string
  onClose: () => void
  onSubmit: (data: {
    question: string
    selectedText: string
    enableAI: boolean
    enableConcept: boolean
  }) => void
}

export function QuestionDialog({
  selectedText,
  onClose,
  onSubmit,
}: QuestionDialogProps) {
  const [question, setQuestion] = useState('')
  const [enableAI, setEnableAI] = useState(true)
  const [enableConcept, setEnableConcept] = useState(false)

  const handleSubmit = () => {
    if (!question.trim()) return
    onSubmit({
      question: question.trim(),
      selectedText,
      enableAI,
      enableConcept,
    })
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-xl shadow-2xl w-[480px] max-w-[90vw] max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-200">
          <h2 className="text-base font-semibold text-gray-900">提出问题</h2>
          <button onClick={onClose} className="p-1 rounded hover:bg-gray-100 text-gray-400">
            <svg width={20} height={20} className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1.5">选中文本</label>
            <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg text-sm text-gray-700 leading-relaxed select-auto">
              {selectedText}
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1.5">你的问题</label>
            <textarea
              className="w-full h-24 p-3 text-sm border border-gray-200 rounded-lg resize-none outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-400"
              placeholder="针对选中文本提出你的问题..."
              value={question}
              onChange={e => setQuestion(e.target.value)}
              autoFocus
            />
          </div>

          <div className="space-y-3 pt-2">
            <label className="flex items-center gap-3 cursor-pointer">
              <div className="relative">
                <input
                  type="checkbox"
                  className="sr-only"
                  checked={enableAI}
                  onChange={e => setEnableAI(e.target.checked)}
                />
                <div className={`w-10 h-5 rounded-full transition-colors ${enableAI ? 'bg-blue-500' : 'bg-gray-300'}`}>
                  <div className={`w-4 h-4 bg-white rounded-full shadow-sm transition-transform mt-0.5 ${enableAI ? 'translate-x-5' : 'translate-x-0.5'}`} />
                </div>
              </div>
              <div>
                <span className="text-sm font-medium text-gray-800">AI 自动回答</span>
                <p className="text-xs text-gray-500">提交后用 AI 生成问题的回答</p>
              </div>
            </label>

            <label className="flex items-center gap-3 cursor-pointer">
              <div className="relative">
                <input
                  type="checkbox"
                  className="sr-only"
                  checked={enableConcept}
                  onChange={e => setEnableConcept(e.target.checked)}
                />
                <div className={`w-10 h-5 rounded-full transition-colors ${enableConcept ? 'bg-blue-500' : 'bg-gray-300'}`}>
                  <div className={`w-4 h-4 bg-white rounded-full shadow-sm transition-transform mt-0.5 ${enableConcept ? 'translate-x-5' : 'translate-x-0.5'}`} />
                </div>
              </div>
              <div>
                <span className="text-sm font-medium text-gray-800">转为新概念</span>
                <p className="text-xs text-gray-500">将这个问题创建为一个新的知识图概念</p>
              </div>
            </label>
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 px-5 py-4 border-t border-gray-200">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
          >
            取消
          </button>
          <button
            onClick={handleSubmit}
            disabled={!question.trim()}
            className="px-4 py-2 text-sm font-medium text-white bg-blue-500 rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            提交
          </button>
        </div>
      </div>
    </div>
  )
}
