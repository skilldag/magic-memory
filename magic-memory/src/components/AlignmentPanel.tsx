import { useState, useCallback, useEffect } from 'react'
import type { Concept } from '../types'
import {
  compareTexts,
  type GraphAlignmentResult,
  type AlignedNodePair,
  type AlignedEdgePair,
} from '../utils/alignment'
import { loadDocContent } from '../utils/docLoader'

interface AlignmentPanelProps {
  concept: Concept
  allConcepts: Concept[]
  onNavigate: (conceptId: string) => void
}

function NodeRow({ node }: { node: AlignedNodePair }) {
  const dot = { matched: 'bg-emerald-500', missing: 'bg-amber-500', extra: 'bg-gray-400' }
  const bg  = { matched: 'border-emerald-200 bg-emerald-50/50', missing: 'border-amber-200 bg-amber-50', extra: 'border-gray-200 bg-gray-50' }
  const lb  = { matched: '已理解', missing: '未提及', extra: '多余' }
  return (
    <div className={`rounded-lg border p-2.5 text-xs ${bg[node.status]}`}>
      <div className="flex items-center gap-1.5">
        <span className={`inline-block w-1.5 h-1.5 rounded-full shrink-0 ${dot[node.status]}`} />
        <span className="font-medium truncate">{node.label}</span>
        {node.isKnownConcept && <span className="text-[9px] text-blue-500 font-medium">KG</span>}
        <span className={`ml-auto px-1 py-0.5 rounded text-[9px] font-medium ${
          node.status === 'matched' ? 'bg-emerald-100 text-emerald-700' :
          node.status === 'missing' ? 'bg-amber-100 text-amber-700' :
          'bg-gray-100 text-gray-500'
        }`}>{lb[node.status]}</span>
      </div>
    </div>
  )
}

function EdgeRow({ edge }: { edge: AlignedEdgePair }) {
  const bg = { matched: 'border-emerald-200 bg-emerald-50/30', missing: 'border-amber-200 bg-amber-50/30', extra: 'border-gray-200 bg-gray-50' }
  const lb = { matched: '共有关联', missing: '遗漏关联', extra: '多余关联' }
  return (
    <div className={`rounded-lg border p-2 text-xs ${bg[edge.status]}`}>
      <div className="flex items-center gap-1">
        <span className="font-medium text-gray-700 truncate">{edge.sourceLabel}</span>
        <span className="text-gray-400 text-[10px]">──</span>
        <span className="font-medium text-gray-700 truncate">{edge.targetLabel}</span>
        <span className="ml-auto text-[9px] text-gray-500">{lb[edge.status]}</span>
      </div>
    </div>
  )
}

export function AlignmentPanel({ concept, allConcepts, onNavigate }: AlignmentPanelProps) {
  const [userText, setUserText] = useState('')
  const [result, setResult] = useState<GraphAlignmentResult | null>(null)
  const [hasAligned, setHasAligned] = useState(false)
  const [originalContent, setOriginalContent] = useState<string | null>(null)
  const [contentLoading, setContentLoading] = useState(false)
  const [showTab, setShowTab] = useState<'nodes' | 'edges'>('nodes')

  useEffect(() => {
    if (concept.content) { setOriginalContent(concept.content); return }
    setContentLoading(true)
    loadDocContent(concept.path).then(c => {
      if (c) {
        const body = c.replace(/^---[\s\S]*?---\n*/, '').trim()
        setOriginalContent(body || c)
      }
      setContentLoading(false)
    })
  }, [concept.id, concept.path, concept.content])

  const handleAlign = useCallback(() => {
    if (!userText.trim() || !originalContent) return
    const r = compareTexts(userText, originalContent, allConcepts, concept.id)
    setResult(r)
    setHasAligned(true)
  }, [userText, originalContent, allConcepts])

  return (
    <div className="px-5 py-4 space-y-4">
      <div>
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">图结构对齐</h3>
        <p className="text-xs text-gray-500 leading-relaxed">
          把你的理解和原文正文各自建图（n-gram 术语提取 + KG 概念匹配 + 共现关系），然后比对图结构。
        </p>
        <p className="text-[10px] text-gray-400 mt-1">
          {originalContent ? `原文已加载（${originalContent.length} 字符）` : contentLoading ? '加载原文中...' : '⚠ 原文不可用'}
        </p>
      </div>

      <textarea value={userText}
        onChange={e => { setUserText(e.target.value); setHasAligned(false); setResult(null) }}
        placeholder={`用你自己的话描述对「${concept.title}」的理解...`} rows={5}
        className="w-full px-3 py-2.5 border-2 border-gray-200 rounded-lg text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-y bg-white" />
      <div className="flex items-center justify-between">
        <p className="text-[10px] text-gray-400">{userText.trim() ? `${userText.trim().length} 字` : ''}</p>
        <button onClick={handleAlign} disabled={!userText.trim() || !originalContent}
          className="px-4 py-1.5 text-xs font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors">
          执行图对齐</button>
      </div>

      {hasAligned && result && (
        <div className="space-y-4 pt-2 border-t border-gray-100">
          <div>
            <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">图对比概览</h4>
            <div className="mb-3">
              <div className="text-[10px] text-gray-500 mb-1.5 font-medium">节点（术语/概念）</div>
              <div className="grid grid-cols-3 gap-2 mb-2">
                <div className="p-2.5 rounded-lg border border-emerald-200 bg-emerald-50 text-center">
                  <div className="text-lg font-bold text-emerald-700">{result.stats.matchedNodeCount}</div>
                  <div className="text-[10px] text-emerald-600">共现</div></div>
                <div className="p-2.5 rounded-lg border border-amber-200 bg-amber-50 text-center">
                  <div className="text-lg font-bold text-amber-700">{result.stats.missingNodeCount}</div>
                  <div className="text-[10px] text-amber-600">仅原文有</div></div>
                <div className="p-2.5 rounded-lg border border-gray-200 bg-gray-50 text-center">
                  <div className="text-lg font-bold text-gray-500">{result.stats.extraNodeCount}</div>
                  <div className="text-[10px] text-gray-500">仅用户有</div></div>
              </div>
              <div className="flex items-center justify-between text-[10px] text-gray-500 mb-1">
                <span>节点覆盖率</span><span>{result.stats.nodeCoverage}%</span></div>
              <div className="w-full h-1.5 bg-gray-100 rounded-full overflow-hidden">
                <div className={`h-full rounded-full transition-all ${result.stats.nodeCoverage >= 80 ? 'bg-emerald-500' : result.stats.nodeCoverage >= 50 ? 'bg-amber-500' : 'bg-red-400'}`}
                  style={{ width: `${result.stats.nodeCoverage}%` }} /></div>
            </div>
            <div className="mt-2 p-2.5 rounded-lg border border-gray-200 bg-gray-50">
              <p className="text-xs text-gray-600">
                原文提取 {result.originalNodeCount} 个术语/概念，你的输入提取 {result.userNodeCount} 个。
                {result.stats.matchedNodeCount > 0
                  ? <span className="text-emerald-700 font-medium"> 其中 {result.stats.matchedNodeCount} 个共现。</span>
                  : <span className="text-amber-700 font-medium"> 没有术语共现，关键词可能不同。</span>}
              </p>
            </div>
          </div>

          {result.fuzzyMatches.length > 0 && (
            <div className="p-3 rounded-lg border border-blue-200 bg-blue-50/50">
              <h4 className="text-xs font-semibold text-blue-800 mb-2">🔗 模糊匹配（字符级相似）</h4>
              {result.fuzzyMatches.map((f, i) => (
                <p key={i} className="text-[10px] text-blue-700">
                  你的"{f.userLabel}" ≈ 原文"{f.originalLabel}"（{f.similarity}% 相似）
                </p>
              ))}
            </div>
          )}

          {result.stats.missingNodeCount > 0 && (
            <div className="p-3 rounded-lg border border-amber-200 bg-amber-50/50">
              <h4 className="text-xs font-semibold text-amber-800 mb-2">原文有但你的描述中未出现的术语</h4>
              <div className="flex flex-wrap gap-1.5">
                {result.nodes.filter(n => n.status === 'missing').map(n => (
                  <button key={n.nodeId}
                    onClick={() => n.isKnownConcept ? onNavigate(n.nodeId) : null}
                    className={`px-2 py-0.5 text-[10px] font-medium bg-white border rounded-full transition-colors ${
                      n.isKnownConcept
                        ? 'border-amber-300 text-amber-800 hover:bg-amber-100 cursor-pointer'
                        : 'border-gray-200 text-gray-500 cursor-default'
                    }`}>
                    {n.label}{n.isKnownConcept ? '' : ''}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="flex gap-2 border-b border-gray-100 pb-2">
            <button onClick={() => setShowTab('nodes')}
              className={`px-3 py-1 text-xs font-medium rounded transition-colors ${showTab === 'nodes' ? 'bg-blue-100 text-blue-700' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'}`}>
              节点详情（{result.nodes.length}）</button>
            <button onClick={() => setShowTab('edges')}
              className={`px-3 py-1 text-xs font-medium rounded transition-colors ${showTab === 'edges' ? 'bg-blue-100 text-blue-700' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'}`}>
              边详情（{result.edges.length}）</button>
          </div>

          {showTab === 'nodes' && (
            <div className="space-y-1.5">
              {result.nodes.sort((a, b) => ({ matched: 0, missing: 1, extra: 2 }[a.status] - { matched: 0, missing: 1, extra: 2 }[b.status]))
                .map(n => <NodeRow key={n.nodeId} node={n} />)}
            </div>
          )}
          {showTab === 'edges' && (
            <div className="space-y-1.5">
              {result.edges.length === 0 && <p className="text-xs text-gray-400 text-center py-4">无边数据</p>}
              {result.edges.sort((a, b) => ({ matched: 0, missing: 1, extra: 2 }[a.status] - { matched: 0, missing: 1, extra: 2 }[b.status]))
                .map((e, i) => <EdgeRow key={`${e.sourceId}-${e.targetId}-${i}`} edge={e} />)}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
