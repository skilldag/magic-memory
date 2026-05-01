import { useMemo, useState } from 'react'
import { useKnowledgeGraphStore } from '../store/knowledgeGraphStore'
import type { Concept } from '../types'

interface ExploreDialogProps {
  sourceConcept: Concept
  onClose: () => void
}

type RelationType = 'leads_to' | 'depends_on' | 'related'

const RELATION_OPTIONS: { value: RelationType; label: string; desc: string }[] = [
  { value: 'leads_to', label: '引出', desc: '当前概念引出的新概念' },
  { value: 'depends_on', label: '依赖于', desc: '新概念是当前概念的前置依赖' },
  { value: 'related', label: '相关', desc: '新概念与当前概念相关' },
]

const QUESTION_HISTORY_KEY = 'mm_explore_question_history_v1'
const MAX_QUESTION_HISTORY = 20

interface ExploreQuestionHistoryItem {
  id: string
  sourceConceptId: string
  sourceConceptTitle: string
  question: string
  relationType: RelationType
  createdAt: string
}

const loadQuestionHistory = (): ExploreQuestionHistoryItem[] => {
  try {
    const raw = localStorage.getItem(QUESTION_HISTORY_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.filter(item => item?.question && item?.sourceConceptId)
  } catch {
    return []
  }
}

const saveQuestionHistory = (items: ExploreQuestionHistoryItem[]) => {
  try {
    localStorage.setItem(QUESTION_HISTORY_KEY, JSON.stringify(items))
  } catch {
    // ignore storage failures
  }
}

export function ExploreDialog({ sourceConcept, onClose }: ExploreDialogProps) {
  const createConceptWithEdges = useKnowledgeGraphStore(state => state.createConceptWithEdges)

  const [title, setTitle] = useState('')
  const [problem, setProblem] = useState('')
  const [gapAnticipate, setGapAnticipate] = useState('')
  const [content, setContent] = useState('')
  const [sourceText, setSourceText] = useState('')
  const [relationType, setRelationType] = useState<RelationType>('leads_to')
  const [submitting, setSubmitting] = useState(false)
  const [aiError, setAiError] = useState<string | null>(null)
  const [questionHistory, setQuestionHistory] = useState<ExploreQuestionHistoryItem[]>(() => loadQuestionHistory())
  const [derivedQuestions, setDerivedQuestions] = useState<string[]>([])

  const EXPLORE_SERVER = 'http://localhost:4321'

  const canSubmit = title.trim() || problem.trim()
  const conceptQuestionHistory = useMemo(
    () => questionHistory.filter(item => item.sourceConceptId === sourceConcept.id).slice(0, 6),
    [questionHistory, sourceConcept.id]
  )

  const upsertQuestionHistory = (questionText: string, relation: RelationType) => {
    const normalizedQuestion = questionText.trim()
    if (!normalizedQuestion) return

    const nextItem: ExploreQuestionHistoryItem = {
      id: `${sourceConcept.id}_${Date.now()}`,
      sourceConceptId: sourceConcept.id,
      sourceConceptTitle: sourceConcept.title,
      question: normalizedQuestion,
      relationType: relation,
      createdAt: new Date().toISOString(),
    }

    const deduped = questionHistory.filter(
      item => !(item.sourceConceptId === sourceConcept.id && item.question === normalizedQuestion)
    )
    const nextHistory = [nextItem, ...deduped].slice(0, MAX_QUESTION_HISTORY)
    setQuestionHistory(nextHistory)
    saveQuestionHistory(nextHistory)
  }

  const deriveQuestionsFromSourceText = () => {
    const raw = sourceText.trim()
    if (!raw) {
      setDerivedQuestions([])
      return
    }

    const normalized = raw.replace(/\s+/g, ' ').trim()
    const short = normalized.length > 80 ? `${normalized.slice(0, 80)}...` : normalized
    const candidates = [
      `这段内容围绕「${sourceConcept.title}」的核心问题是什么？`,
      `要理解这段内容，关于「${sourceConcept.title}」需要先掌握哪些前置概念？`,
      `这段内容可以沉淀出哪些新的概念节点？`,
      `这段信息与「${sourceConcept.title}」的关系应该如何建立？`,
      `如果只保留一个关键问题，最值得继续探索的是：${short}`,
    ]
    setDerivedQuestions(candidates.slice(0, 4))
  }

  const handleSubmit = async () => {
    if (!canSubmit) return
    setSubmitting(true)
    setAiError(null)

    const userQuestion = problem.trim() || title.trim()
    let aiData: any = null

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
          userQuestion,
          relationType,
          sourceText: sourceText.trim() || undefined,
        }),
      })

      if (resp.ok) {
        aiData = await resp.json()
      } else {
        throw new Error(resp.status === 0 ? 'server not running' : `server error: ${resp.status}`)
      }
    } catch (err: any) {
      if (err.name === 'TimeoutError' || err.name === 'AbortError') {
        setAiError('AI 生成超时，使用本地生成')
      } else {
        setAiError(err.message?.includes('server') ? 'AI 生成服务未启动，使用本地生成' : `AI 生成失败: ${err.message}`)
      }
    }

    upsertQuestionHistory(userQuestion, relationType)

    if (aiData) {
      createConceptWithEdges(sourceConcept, {
        title: aiData.title,
        problem: aiData.problem,
        gap_anticipate: aiData.gap_anticipate,
        relationType,
        metadataStatus: 'ai-generated',
      })
    } else {
      const finalTitle = title.trim() || userQuestion.slice(0, 40)
      const finalProblem = problem.trim() || `从「${sourceConcept.title}」引出的问题`

      createConceptWithEdges(sourceConcept, {
        title: finalTitle,
        problem: finalProblem,
        gap_anticipate: gapAnticipate.trim() || undefined,
        relationType,
        metadataStatus: 'draft',
      })
    }

    setSubmitting(false)
    onClose()
  }

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 9999, display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(0,0,0,0.4)' }} onClick={onClose}>
      <div style={{ backgroundColor: 'white', borderRadius: '12px', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.25)' }} className="w-[520px] max-h-[85vh] flex flex-col pointer-events-auto" onClick={e => e.stopPropagation()}>
        <div className="shrink-0 flex items-center justify-between px-5 py-4 border-b border-gray-100">
          <div>
            <h2 className="text-base font-semibold text-gray-900">从概念探索</h2>
            <p className="text-xs text-gray-500 mt-0.5">基于「{sourceConcept.title}」扩展新概念</p>
          </div>
          <button onClick={onClose} className="p-1 rounded hover:bg-gray-100 text-gray-400">
            <svg width={20} height={20} className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          <div>
            <label className="block text-sm font-bold text-gray-900 mb-1.5">关系类型*</label>
            <div className="flex gap-2">
              {RELATION_OPTIONS.map(opt => (
                <button
                  key={opt.value}
                  onClick={() => setRelationType(opt.value)}
                  className={`flex-1 px-3 py-2.5 rounded-lg text-sm border-2 text-left transition-colors ${
                    relationType === opt.value
                      ? 'border-blue-500 bg-blue-50 text-blue-800'
                      : 'border-gray-300 text-gray-800 hover:border-gray-400 bg-gray-50'
                  }`}
                >
                  <div className="font-semibold">{opt.label}</div>
                  <div className="text-xs mt-0.5 text-gray-600">{opt.desc}</div>
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-bold text-gray-900 mb-1.5">探索问题（必填）</label>
            <textarea
              value={problem}
              onChange={e => setProblem(e.target.value)}
              placeholder="输入你想探索的问题，例如：都需要支持哪些不同的 LLM Provider？"
              rows={3}
              className="w-full px-3.5 py-2.5 border-2 border-blue-400 rounded-lg text-base text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white resize-none"
            />
            <p className="text-xs text-gray-500 mt-1">只需填写问题即可提交，系统会自动生成概念</p>
          </div>

          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="block text-sm font-bold text-gray-900">文本抽离（可选）</label>
              <button
                type="button"
                onClick={deriveQuestionsFromSourceText}
                disabled={!sourceText.trim()}
                className="px-2.5 py-1 text-xs rounded-md border border-gray-300 text-gray-700 hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                提炼问题建议
              </button>
            </div>
            <textarea
              value={sourceText}
              onChange={e => setSourceText(e.target.value)}
              placeholder="粘贴原始资料、笔记或对话内容，系统会帮你提炼可探索问题..."
              rows={4}
              className="w-full px-3.5 py-2.5 border-2 border-gray-300 rounded-lg text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white resize-none"
            />
            <p className="text-xs text-gray-500 mt-1">点击“提炼问题建议”后，可将建议一键带入上方探索问题。</p>
            {derivedQuestions.length > 0 && (
              <div className="mt-2 space-y-1.5">
                {derivedQuestions.map((q, index) => (
                  <button
                    key={`${index}-${q}`}
                    type="button"
                    onClick={() => setProblem(q)}
                    className="w-full text-left px-3 py-2 rounded-lg border border-blue-200 bg-blue-50 text-xs text-blue-800 hover:border-blue-300 hover:bg-blue-100 transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="block text-sm font-bold text-gray-900">历史提问</label>
              {conceptQuestionHistory.length > 0 && (
                <span className="text-xs text-gray-400">点击可回填</span>
              )}
            </div>
            {conceptQuestionHistory.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {conceptQuestionHistory.map(item => (
                  <button
                    key={item.id}
                    onClick={() => {
                      setProblem(item.question)
                      setRelationType(item.relationType)
                    }}
                    className="max-w-full px-2.5 py-1.5 rounded-full border border-gray-200 bg-gray-50 text-xs text-gray-700 hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700 transition-colors"
                    title={item.question}
                  >
                    <span className="truncate inline-block max-w-[300px] align-bottom">{item.question}</span>
                  </button>
                ))}
              </div>
            ) : (
              <div className="px-3 py-2 rounded-lg border border-gray-200 bg-gray-50 text-xs text-gray-500">
                当前概念还没有历史提问，提交后会自动记录在这里。
              </div>
            )}
          </div>

          <div>
            <label className="block text-sm font-bold text-gray-900 mb-1.5">概念名称（可选）</label>
            <input
              type="text"
              value={title}
              onChange={e => setTitle(e.target.value)}
              placeholder="留空则自动从问题生成"
              className="w-full px-3.5 py-2.5 border-2 border-gray-300 rounded-lg text-base text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white"
              autoFocus
            />
          </div>

          <div>
            <label className="block text-sm font-bold text-gray-900 mb-1.5">可能产生的疑问 (认知缺口)</label>
            <input
              type="text"
              value={gapAnticipate}
              onChange={e => setGapAnticipate(e.target.value)}
              placeholder="学习这个概念时可能有什么疑问？"
              className="w-full px-3.5 py-2.5 border-2 border-gray-300 rounded-lg text-base text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white"
            />
          </div>

          <div>
            <label className="block text-sm font-bold text-gray-900 mb-1.5">详细内容 (Markdown)</label>
            <textarea
              value={content}
              onChange={e => setContent(e.target.value)}
              placeholder="用 Markdown 写详细说明... (可选)"
              rows={5}
              className="w-full px-3.5 py-2.5 border-2 border-gray-300 rounded-lg text-base text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none bg-white font-mono"
            />
          </div>
        </div>

        <div className="shrink-0 flex items-center justify-end gap-2 px-5 py-4 border-t border-gray-100">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
          >
            取消
          </button>
          <button
            onClick={handleSubmit}
            disabled={!canSubmit || submitting}
            className="px-5 py-2.5 text-sm font-medium text-white bg-blue-500 rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            {submitting ? '添加中...' : canSubmit ? '添加到图谱并探索' : '请填写问题'}
          </button>
        </div>
      </div>
      </div>
  )
}
