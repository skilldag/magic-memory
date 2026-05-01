import { useEffect, useState } from 'react'

interface FlowPath {
  name: string
  pathIds: string[]
  pathTitles: string[]
  length: number
}

interface RootConceptInfo {
  id: string
  title: string
  level: number
  category: string
  inDegree: number
  outDegree: number
}

interface HubConceptInfo {
  id: string
  title: string
  level: number
  totalDegree: number
}

interface AnalysisData {
  rootConcepts: RootConceptInfo[]
  dataFlowPaths: FlowPath[]
  dependencyChains: FlowPath[]
  longestPaths: FlowPath[]
  hubConcepts: HubConceptInfo[]
  stats: {
    totalConcepts: number
    totalEdges: number
    rootsCount: number
    crossLayerJumpsCount: number
  }
}

interface AnalysisPanelProps {
  onNavigate?: (conceptId: string) => void
  onPathFocus?: (conceptIds: string[]) => void
}

export function AnalysisPanel({ onNavigate, onPathFocus }: AnalysisPanelProps) {
  const [data, setData] = useState<AnalysisData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expandedSection, setExpandedSection] = useState<string | null>('roots')
  const [expandedPath, setExpandedPath] = useState<number | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fetch('/api/graph/analysis')
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then(d => {
        if (!cancelled) setData(d)
      })
      .catch(e => {
        if (!cancelled) setError(e.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [])

  const toggleSection = (name: string) => {
    setExpandedSection(expandedSection === name ? null : name)
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-32 text-gray-400 text-sm">
        分析中...
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="text-xs text-red-500 px-2 py-1">
        {error || '无数据'}
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full text-sm">
      {/* 标题 */}
      <div className="px-4 py-3 border-b border-gray-100">
        <h2 className="font-bold text-gray-800">图谱分析</h2>
        <p className="text-xs text-gray-400 mt-0.5">
          {data.stats.totalConcepts} 概念 / {data.stats.totalEdges} 边 / {data.stats.rootsCount} 入口
        </p>
      </div>

      {/* 滚动内容 */}
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-2">
        {/* ===== 入口（根节点） ===== */}
        <SectionCard
          title="入口"
          count={data.rootConcepts.length}
          expanded={expandedSection === 'roots'}
          onToggle={() => toggleSection('roots')}
        >
          <div className="space-y-1">
            {data.rootConcepts.map(r => (
              <button
                key={r.id}
                className="flex items-center gap-2 px-2 py-1.5 rounded-md bg-blue-50/60 border border-blue-100 cursor-pointer hover:bg-blue-100"
                onClick={() => onNavigate?.(r.id)}
                title="点击聚焦到图谱"
              >
                <span className="w-2 h-2 rounded-full bg-blue-500 shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-medium text-gray-800 truncate">{r.title}</div>
                  <div className="text-[10px] text-gray-400">L{r.level} · {r.category} · 出度 {r.outDegree}</div>
                </div>
              </button>
            ))}
            {data.rootConcepts.length === 0 && (
              <p className="text-xs text-gray-400 text-center py-2">无根节点</p>
            )}
          </div>
        </SectionCard>

        {/* ===== 枢纽节点 ===== */}
        <SectionCard
          title="枢纽节点"
          count={Math.min(data.hubConcepts.length, 5)}
          expanded={expandedSection === 'hubs'}
          onToggle={() => toggleSection('hubs')}
        >
          <div className="space-y-0.5">
            {data.hubConcepts.slice(0, 5).map((h, i) => (
              <button
                key={h.id}
                className="flex items-center gap-2 px-2 py-1 text-xs hover:bg-gray-50 rounded cursor-pointer"
                onClick={() => onNavigate?.(h.id)}
                title="点击聚焦到图谱"
              >
                <span className="text-gray-300 font-mono w-4 shrink-0 text-right">{i + 1}</span>
                <div className="flex-1 min-w-0 truncate font-medium text-gray-700">{h.title}</div>
                <span className="text-gray-400 shrink-0">{h.totalDegree}</span>
              </button>
            ))}
          </div>
        </SectionCard>

        {/* ===== 最长路径 ===== */}
        <SectionCard
          title="最长路径"
          count={data.longestPaths.length > 0 ? data.longestPaths[0].length : 0}
          expanded={expandedSection === 'flows'}
          onToggle={() => toggleSection('flows')}
        >
          <div className="space-y-1.5">
            {data.longestPaths.slice(0, 10).map((p, i) => (
              <div key={i}>
                <div className="flex items-center w-full">
                  <button
                    className="w-6 h-6 flex items-center justify-center rounded-md hover:bg-gray-50 shrink-0"
                    onClick={() => setExpandedPath(expandedPath === i ? null : i)}
                    aria-label="展开路径"
                  >
                    <svg width={6} height={6} viewBox="0 0 24 24" fill="none" stroke="currentColor" className={`transition-transform ${expandedPath === i ? 'rotate-90' : ''}`} style={{ display: 'block' }}>
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l6 7-6 7" />
                    </svg>
                  </button>
                  <button
                    className="flex-1 px-2 py-1.5 rounded-md hover:bg-gray-50 text-left"
                    onClick={() => onPathFocus?.(p.pathIds)}
                    title="聚焦路径"
                  >
                    <span className="text-xs font-medium text-gray-600">{p.name} ({p.length}步)</span>
                  </button>
                </div>
                {expandedPath === i && (
                  <div className="px-3 pb-1 pt-1">
                    <PathPreview titles={p.pathTitles} />
                  </div>
                )}
              </div>
            ))}
            {data.longestPaths.length === 0 && (
              <p className="text-xs text-gray-400 text-center py-2">未发现路径</p>
            )}
          </div>
        </SectionCard>
      </div>
    </div>
  )
}

function Arrow({ color }: { color: string }) {
  const colors: Record<string, string> = { 'blue-300': '#93c5fd', 'amber-300': '#fcd34d' }
  return (
    <svg width={8} height={8} viewBox="0 0 24 24" fill="none" style={{ margin: '0 1px', display: 'inline', verticalAlign: 'baseline' }}>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} stroke={colors[color] || '#93c5fd'} d="M9 5l7 7-7 7" />
    </svg>
  )
}

function PathPreview({ titles }: { titles: string[] }) {
  const MAX_VISIBLE = 7
  const showFull = titles.length <= MAX_VISIBLE
  const displayed = showFull ? titles : [...titles.slice(0, 3), '...', ...titles.slice(-3)]

  return (
    <div className="text-[10px] leading-none">
      {displayed.map((title, j) => (
        <span key={j} className="inline items-baseline">
          {title === '...'
            ? <span className="text-gray-300 font-mono">···</span>
            : <span className="text-gray-600">{title}</span>}
          {j < displayed.length - 1 && <Arrow color="blue-300" />}
        </span>
      ))}
      {!showFull && <span className="text-[9px] text-gray-400 ml-1">({titles.length}步)</span>}
    </div>
  )
}

function SectionCard({
  title,
  count,
  expanded,
  onToggle,
  children,
}: {
  title: string
  count: number
  expanded: boolean
  onToggle: () => void
  children: React.ReactNode
}) {
  return (
    <div className="border border-gray-200 rounded-md overflow-hidden">
      <button
        className="w-full flex items-center gap-1 px-2.5 py-1.5 hover:bg-gray-50 transition-colors text-left"
        onClick={onToggle}
      >
        <svg
          width={8} height={8}
          className={`text-gray-300 shrink-0 transition-transform ${expanded ? 'rotate-90' : ''}`}
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
          style={{ display: 'block' }}
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5l7 7-7 7" />
        </svg>
        <span className="text-[11px] font-medium text-gray-600">{title}</span>
        <span className="text-[10px] text-gray-300 ml-auto">{count}</span>
      </button>
      {expanded && (
        <div className="px-2.5 pb-1.5 border-t border-gray-100 pt-1.5">
          {children}
        </div>
      )}
    </div>
  )
}
