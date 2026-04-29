import { useState } from 'react'
import { useKnowledgeGraphStore } from '../store/knowledgeGraphStore'
import type { Concept } from '../types'

interface QuickExploreDialogProps {
  sourceConcept: Concept
  onClose: () => void
}

export function QuickExploreDialog({ sourceConcept, onClose }: QuickExploreDialogProps) {
  const createConceptWithEdges = useKnowledgeGraphStore(state => state.createConceptWithEdges)

  const [question, setQuestion] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [aiError, setAiError] = useState<string | null>(null)

  const EXPLORE_SERVER = 'http://localhost:4321'
  const canSubmit = question.trim().length > 0

  const handleSubmit = async () => {
    if (!canSubmit) return
    setSubmitting(true)
    setAiError(null)

    try {
      const resp = await fetch(`${EXPLORE_SERVER}/api/explore`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(30000),
        body: JSON.stringify({
          sourceConcept: {
            id: sourceConcept.id,
            title: sourceConcept.title,
            problem: sourceConcept.problem,
          },
          userQuestion: question.trim(),
          relationType: 'leads_to',
        }),
      })

      if (resp.ok) {
        const data = await resp.json()
        createConceptWithEdges(sourceConcept, {
          title: data.title,
          problem: data.problem,
          gap_anticipate: data.gap_anticipate,
          relationType: 'leads_to',
          metadataStatus: 'ai-generated',
        })
      } else {
        throw new Error(resp.status === 0 ? 'server not running' : `server error: ${resp.status}`)
      }
    } catch (err: any) {
      if (err.name === 'TimeoutError' || err.name === 'AbortError') {
        setAiError('AI 生成超时，请重试')
      } else if (err.message?.includes('fetch') || err.name === 'TypeError') {
        setAiError('AI 生成服务未启动')
      } else {
        setAiError(err.message || '请求失败')
      }
      setSubmitting(false)
      return
    }

    setSubmitting(false)
    onClose()
  }

  return (
    <div className="fixed inset-0 z-[9999] flex bg-black/40" style={{ alignItems: 'flex-start', paddingTop: '96px', paddingLeft: '24px' }} onClick={onClose}>
      <div
        className="w-[380px] bg-white rounded-xl shadow-2xl border border-gray-200 p-5"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="text-sm font-semibold text-gray-900">基于问题探索</h3>
            <p className="text-xs text-gray-400 mt-0.5">
              当前概念：<span className="text-gray-600">{sourceConcept.title}</span>
            </p>
          </div>
          <button type="button" onClick={onClose} className="text-gray-400 hover:text-gray-600 p-1">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="mb-4">
          <textarea
            value={question}
            onChange={e => setQuestion(e.target.value)}
            placeholder="输入你想探索的问题，例如：都需要支持哪些不同的 LLM Provider？"
            rows={3}
            className="w-full px-3 py-2 border-2 border-blue-400 rounded-lg text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white resize-none"
            autoFocus
          />
          <p className="text-xs text-gray-400 mt-1">输入问题后提交，AI 会自动生成新概念</p>
        </div>

        {aiError && (
          <div className="mb-3 px-3 py-2 rounded-lg bg-red-50 border border-red-200 text-xs text-red-600">
            {aiError}
          </div>
        )}

        <div className="flex items-center justify-end gap-2 border-t border-gray-100 pt-3">
          <button
            type="button"
            onClick={onClose}
            className="px-3 py-1.5 text-xs text-gray-500 hover:text-gray-700 transition-colors"
          >
            取消
          </button>
          <button
            type="button"
            onClick={handleSubmit}
            disabled={!canSubmit || submitting}
            className="px-4 py-1.5 text-xs rounded-lg bg-blue-500 text-white hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            {submitting ? '生成中...' : '生成概念'}
          </button>
        </div>
      </div>
    </div>
  )
}
