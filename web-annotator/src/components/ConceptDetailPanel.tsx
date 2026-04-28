import { useState, useMemo } from 'react'
import { DocumentViewer } from './DocumentViewer'
import { DependencyChainSVG } from './DependencyChainSVG'
import { getRelatedConcepts, getDependencyChain, getWhatIsSummary, getRelationReason, getReviewRecordFor } from '../utils/knowledgeGraph'
import type { Concept, ConceptEdge, ReviewRecord } from '../types'

interface ConceptDetailPanelProps {
  concept: Concept
  viewMode: 'explore' | 'review'
  concepts: Concept[]
  edges: ConceptEdge[]
  reviewRecords: Map<string, ReviewRecord>
  lastReviewFeedback: string | null
  onNavigate: (conceptId: string) => void
  onDeselect: () => void
  onReviewScore: (quality: number) => void
  onOpenExplore: () => void
}

type ExploreTab = 'snapshot' | 'ask' | 'extract' | 'reasoning'
type LearnTab = 'path' | 'quiz' | 'memory' | 'mistakes'

type ReviewEntry = { concept: Concept; record: NonNullable<ReturnType<typeof getReviewRecordFor>> }

export function ConceptDetailPanel({
  concept,
  viewMode,
  concepts,
  edges,
  reviewRecords,
  lastReviewFeedback,
  onNavigate,
  onDeselect,
  onReviewScore,
  onOpenExplore,
}: ConceptDetailPanelProps) {
  const [exploreTab, setExploreTab] = useState<ExploreTab>('snapshot')
  const [learnTab, setLearnTab] = useState<LearnTab>('path')

  const dueConcepts = useMemo(() => {
    const now = new Date()
    const list = concepts
      .map(c => {
        const record = getReviewRecordFor(c.id, reviewRecords)
        if (!record) return null
        return { concept: c, record }
      })
      .filter(Boolean)
      .filter(item => item!.record.next_review <= now)
    return (list as ReviewEntry[]).slice(0, 6)
  }, [concepts, reviewRecords])

  const weakConcepts = useMemo(() => {
    const list = concepts
      .map(c => {
        const record = getReviewRecordFor(c.id, reviewRecords)
        if (!record) return null
        return { concept: c, record }
      })
      .filter(Boolean)
      .filter(item => item!.record.status === 'learning' || item!.record.interval <= 1)
    return (list as ReviewEntry[]).slice(0, 6)
  }, [concepts, reviewRecords])

  const exploreTabs = [
    { key: 'snapshot' as const, label: '概念快照' },
    { key: 'ask' as const, label: '提问生成' },
    { key: 'extract' as const, label: '文本抽离' },
    { key: 'reasoning' as const, label: '推导预览' },
  ]

  const learnTabs = [
    { key: 'path' as const, label: '学习路径' },
    { key: 'quiz' as const, label: '自测练习' },
    { key: 'memory' as const, label: '记忆状态' },
    { key: 'mistakes' as const, label: '错题回顾' },
  ]

  return (
    <>
      {/* 头部 */}
      <div className="shrink-0 px-5 pt-4 pb-3 border-b border-gray-100">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <h2 className="text-base font-semibold text-gray-900 truncate">{concept.title}</h2>
            {concept.alias && concept.alias.length > 0 && (
              <p className="text-xs text-gray-500 mt-0.5 truncate">别名: {concept.alias.join(' / ')}</p>
            )}
          </div>
          <button onClick={onDeselect} className="shrink-0 ml-2 p-1 rounded hover:bg-gray-100 text-gray-400 hover:text-gray-600" title="取消选择">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div className="flex items-center gap-2 mt-2">
          <span className={`px-2 py-0.5 rounded text-xs font-medium ${
            concept.level === 1 ? 'bg-green-100 text-green-700' :
            concept.level === 2 ? 'bg-blue-100 text-blue-700' :
            'bg-purple-100 text-purple-700'
          }`}>
            L{concept.level}
          </span>
          <span className="px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-600">
            {concept.category}
          </span>
        </div>
      </div>

      {/* Tab 切换 */}
      <div className="shrink-0 flex border-b border-gray-100">
        {(viewMode === 'explore' ? exploreTabs : learnTabs).map(t => (
          <button
            key={t.key}
            onClick={() => {
              if (viewMode === 'explore') setExploreTab(t.key)
              else setLearnTab(t.key)
            }}
            className={`flex-1 px-3 py-2.5 text-sm font-medium text-center transition-colors ${
              (viewMode === 'explore' ? exploreTab : learnTab) === t.key
                ? 'text-blue-600 border-b-2 border-blue-500'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* 内容区域 */}
      <div className="flex-1 overflow-y-auto">
        {/* 探索模式 - 概念快照 */}
        {viewMode === 'explore' && exploreTab === 'snapshot' && (
          <>
            <div className="px-5 py-4 space-y-4">
              <div>
                <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">这个概念是什么</h3>
                <div className="p-3 bg-indigo-50 border-l-4 border-indigo-400 rounded-r-lg">
                  <p className="text-sm leading-relaxed text-indigo-900">
                    {getWhatIsSummary(concept) || '定义尚未补充，可先查看"深入理解"获取完整说明。'}
                  </p>
                </div>
              </div>
              <div>
                <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">它所解决的问题</h3>
                <div className="p-3 bg-amber-50 border-l-4 border-amber-400 rounded-r-lg">
                  <p className="text-sm leading-relaxed text-amber-900 font-medium">
                    {concept.problem || '背景问题尚未记录'}
                  </p>
                </div>
              </div>
            </div>
            <div className="px-5 py-4">
              <div className="prose prose-sm max-w-none">
                <DocumentViewer document={{
                  id: concept.id, title: concept.title, path: concept.path,
                  content: concept.content, level: concept.level, category: concept.category,
                  tags: concept.tags, lastModified: concept.lastModified, metadata: concept.metadata,
                }} />
              </div>
            </div>
          </>
        )}

        {/* 探索模式 - 提问生成 */}
        {viewMode === 'explore' && exploreTab === 'ask' && (
          <div className="px-5 py-4 space-y-4">
            <div className="p-4 rounded-lg border border-blue-200 bg-blue-50">
              <h3 className="text-sm font-semibold text-blue-900">问题驱动探索</h3>
              <p className="text-xs text-blue-700 mt-1 leading-relaxed">
                围绕「{concept.title}」提出新问题，系统会生成候选概念并建立关联关系。
              </p>
            </div>
            <button onClick={onOpenExplore} className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors">
              <span className="text-sm font-medium">开始提问生成</span>
            </button>
            <div className="p-3 rounded-lg border border-gray-200 bg-gray-50 text-xs text-gray-600 leading-relaxed">
              你可以输入问题，也可以直接使用历史提问回填，快速展开下一轮探索。
            </div>
          </div>
        )}

        {/* 探索模式 - 文本抽离 */}
        {viewMode === 'explore' && exploreTab === 'extract' && (
          <div className="px-5 py-4 space-y-4">
            <div className="p-4 rounded-lg border border-indigo-200 bg-indigo-50">
              <h3 className="text-sm font-semibold text-indigo-900">从原始资料抽离知识</h3>
              <p className="text-xs text-indigo-700 mt-1 leading-relaxed">
                粘贴笔记、对话或文章内容，提炼可探索问题并建立与「{concept.title}」的关系。
              </p>
            </div>
            <button onClick={onOpenExplore} className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 transition-colors">
              <span className="text-sm font-medium">打开文本抽离面板</span>
            </button>
            <div className="text-xs text-gray-500 leading-relaxed">
              提示：进入弹窗后在"文本抽离（可选）"区域粘贴内容，再点击"提炼问题建议"。
            </div>
          </div>
        )}

        {/* 探索模式 - 推导预览 */}
        {viewMode === 'explore' && exploreTab === 'reasoning' && (
          <div className="px-5 py-4 space-y-5">
            <div>
              <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">推导路径</h3>
              <div className="p-4 bg-white rounded-lg border border-gray-200">
                <DependencyChainSVG concept={concept} concepts={concepts} />
              </div>
            </div>
            {(() => {
              const related = getRelatedConcepts(concept.id, edges, concepts)
              const byType: Record<string, {concept: Concept; edgeType: string}[]> = {
                leads_to: related.filter(r => r.edgeType === 'leads_to'),
                depends_on: related.filter(r => r.edgeType === 'depends_on'),
                related: related.filter(r => r.edgeType === 'related'),
              }
              return (
                <>
                  {byType.leads_to.length > 0 && (
                    <div>
                      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">由此推出</h3>
                      <div className="space-y-1.5">
                        {byType.leads_to.map(({ concept: c }) => (
                          <button key={c.id} onClick={() => onNavigate(c.id)} className="w-full text-left px-3 py-2 rounded-lg border border-green-200 hover:border-green-300 hover:bg-green-50 transition-colors group">
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium text-gray-800 group-hover:text-green-700">{c.title}</span>
                              <span className="text-xs text-gray-400">→</span>
                            </div>
                            <p className="text-xs text-gray-500 mt-0.5">{getRelationReason(concept, c, 'leads_to')}</p>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  {byType.depends_on.length > 0 && (
                    <div>
                      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">前置基础</h3>
                      <div className="space-y-1.5">
                        {byType.depends_on.map(({ concept: c }) => (
                          <button key={c.id} onClick={() => onNavigate(c.id)} className="w-full text-left px-3 py-2 rounded-lg border border-red-200 hover:border-red-300 hover:bg-red-50 transition-colors group">
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium text-gray-800 group-hover:text-red-700">{c.title}</span>
                              <span className="text-xs text-gray-400">←</span>
                            </div>
                            <p className="text-xs text-gray-500 mt-0.5">{getRelationReason(concept, c, 'depends_on')}</p>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  {byType.related.length > 0 && (
                    <div>
                      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">平行关联</h3>
                      <div className="space-y-1.5">
                        {byType.related.map(({ concept: c }) => (
                          <button key={c.id} onClick={() => onNavigate(c.id)} className="w-full text-left px-3 py-2 rounded-lg border border-gray-200 hover:border-gray-300 hover:bg-gray-50 transition-colors group">
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium text-gray-800 group-hover:text-gray-900">{c.title}</span>
                              <span className="text-xs text-gray-400">↔</span>
                            </div>
                            <p className="text-xs text-gray-500 mt-0.5">{getRelationReason(concept, c, 'related')}</p>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )
            })()}
          </div>
        )}

        {/* 学习模式 - 学习路径 */}
        {viewMode === 'review' && learnTab === 'path' && (
          <div className="px-5 py-4 space-y-4">
            <div className="p-3 bg-emerald-50 border-l-4 border-emerald-400 rounded-r-lg">
              <p className="text-sm leading-relaxed text-emerald-900">
                先掌握前置基础，再回到当前概念，最后延伸到由此推出的主题。
              </p>
            </div>
            <div className="p-3 rounded-lg border border-gray-200 bg-white">
              <h3 className="text-sm font-semibold text-gray-800 mb-2">当前学习顺序</h3>
              <p className="text-xs text-gray-600 leading-relaxed">
                {getDependencyChain(concept.id, concepts).map(c => c.title).join(' → ') || '（无前置）'}{getDependencyChain(concept.id, concepts).length > 0 ? ' → ' : ''}{concept.title}
              </p>
            </div>
            <div className="p-3 rounded-lg border border-gray-200 bg-white">
              <h3 className="text-sm font-semibold text-gray-800 mb-2">建议下一步</h3>
              <div className="space-y-1.5">
                {getRelatedConcepts(concept.id, edges, concepts)
                  .filter(item => item.edgeType === 'leads_to')
                  .slice(0, 3)
                  .map(({ concept: c }) => (
                    <button key={c.id} onClick={() => onNavigate(c.id)} className="w-full text-left px-2.5 py-2 rounded border border-gray-200 hover:border-emerald-300 hover:bg-emerald-50 text-xs text-gray-700 transition-colors">
                      {c.title}
                    </button>
                  ))}
                {getRelatedConcepts(concept.id, edges, concepts).filter(item => item.edgeType === 'leads_to').length === 0 && (
                  <p className="text-xs text-gray-500">暂无明确后继概念，可先完成自测练习。</p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* 学习模式 - 自测练习 */}
        {viewMode === 'review' && learnTab === 'quiz' && (
          <div className="px-5 py-4 space-y-4">
            <div className="p-3 rounded-lg border border-violet-200 bg-violet-50">
              <h3 className="text-sm font-semibold text-violet-900">自测题（概念理解）</h3>
              <p className="text-xs text-violet-800 mt-1">请用一句话回答：{concept.title} 解决了什么问题？</p>
            </div>
            <div className="p-3 rounded-lg border border-gray-200 bg-gray-50 text-xs text-gray-600">
              建议先完成"是什么 + 解决什么问题 + 与易混概念差异"三步再进入下一概念。
            </div>
            <div className="space-y-2">
              <button onClick={() => onReviewScore(2)} className="w-full px-3 py-2 rounded-lg border border-red-200 bg-red-50 text-red-700 text-sm hover:bg-red-100 transition-colors">
                没想起来（标记薄弱）
              </button>
              <button onClick={() => onReviewScore(3)} className="w-full px-3 py-2 rounded-lg border border-amber-200 bg-amber-50 text-amber-700 text-sm hover:bg-amber-100 transition-colors">
                模糊记得（继续巩固）
              </button>
              <button onClick={() => onReviewScore(5)} className="w-full px-3 py-2 rounded-lg border border-emerald-200 bg-emerald-50 text-emerald-700 text-sm hover:bg-emerald-100 transition-colors">
                清晰掌握（可延后复习）
              </button>
            </div>
            {lastReviewFeedback && (
              <div className="p-3 rounded-lg border border-blue-200 bg-blue-50 text-xs text-blue-800">{lastReviewFeedback}</div>
            )}
          </div>
        )}

        {/* 学习模式 - 记忆状态 */}
        {viewMode === 'review' && learnTab === 'memory' && (
          <div className="px-5 py-4 space-y-4">
            <div className="grid grid-cols-2 gap-2">
              <div className="p-3 rounded-lg border border-gray-200 bg-white text-center">
                <div className="text-xs text-gray-500">掌握度</div>
                <div className="text-sm font-semibold text-gray-800 mt-1">{getReviewRecordFor(concept.id, reviewRecords)?.status || 'new'}</div>
              </div>
              <div className="p-3 rounded-lg border border-gray-200 bg-white text-center">
                <div className="text-xs text-gray-500">复习间隔</div>
                <div className="text-sm font-semibold text-gray-800 mt-1">{getReviewRecordFor(concept.id, reviewRecords)?.interval ?? 0} 天</div>
              </div>
            </div>
            <div className="p-3 rounded-lg border border-gray-200 bg-white text-xs text-gray-700 leading-relaxed">
              <div>复习次数：{getReviewRecordFor(concept.id, reviewRecords)?.review_count ?? 0}</div>
              <div>下次复习：{getReviewRecordFor(concept.id, reviewRecords)?.next_review?.toLocaleString() ?? '尚未安排'}</div>
              <div>难度因子：{getReviewRecordFor(concept.id, reviewRecords)?.ease_factor?.toFixed(2) ?? '2.50'}</div>
            </div>
          </div>
        )}

        {/* 学习模式 - 错题回顾 */}
        {viewMode === 'review' && learnTab === 'mistakes' && (
          <div className="px-5 py-4 space-y-4">
            <div className="p-3 rounded-lg border border-amber-200 bg-amber-50 text-sm text-amber-900">
              优先处理"到期待复习"和"薄弱概念"，再回到当前概念继续推进。
            </div>
            <div className="p-3 rounded-lg border border-gray-200 bg-white">
              <h3 className="text-sm font-semibold text-gray-800 mb-2">到期待复习</h3>
              <div className="space-y-1.5">
                {dueConcepts.map(({ concept: c }) => (
                  <button key={c.id} onClick={() => onNavigate(c.id)} className="w-full text-left px-2.5 py-2 rounded border border-gray-200 hover:border-amber-300 hover:bg-amber-50 text-xs text-gray-700 transition-colors">
                    {c.title}
                  </button>
                ))}
                {dueConcepts.length === 0 && <p className="text-xs text-gray-500">暂无到期项</p>}
              </div>
            </div>
            <div className="p-3 rounded-lg border border-gray-200 bg-white">
              <h3 className="text-sm font-semibold text-gray-800 mb-2">薄弱概念</h3>
              <div className="space-y-1.5">
                {weakConcepts.map(({ concept: c }) => (
                  <button key={c.id} onClick={() => onNavigate(c.id)} className="w-full text-left px-2.5 py-2 rounded border border-gray-200 hover:border-red-300 hover:bg-red-50 text-xs text-gray-700 transition-colors">
                    {c.title}
                  </button>
                ))}
                {weakConcepts.length === 0 && <p className="text-xs text-gray-500">暂无薄弱项</p>}
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  )
}
