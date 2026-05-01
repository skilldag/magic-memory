import { useState, useEffect } from 'react'
import { AnalysisPanel } from './AnalysisPanel'

interface Concept {
  id: string
  title: string
  number: number | null
  category: string
  path: string
}

interface Community {
  name: string
  size: number
  numRange: string
  cohesion: number
  members: Concept[]
}

interface ClusterResult {
  totalConcepts: number
  totalEdges: number
  totalCommunities: number
  concepts: Concept[]
  communities: Community[]
}

const COMMUNITY_COLORS = [
  'bg-blue-50 border-blue-300',
  'bg-emerald-50 border-emerald-300',
  'bg-amber-50 border-amber-300',
  'bg-purple-50 border-purple-300',
  'bg-rose-50 border-rose-300',
  'bg-cyan-50 border-cyan-300',
  'bg-orange-50 border-orange-300',
  'bg-teal-50 border-teal-300',
  'bg-pink-50 border-pink-300',
  'bg-indigo-50 border-indigo-300',
]

function getBarColor(cohesion: number): string {
  if (cohesion >= 0.7) return 'bg-green-500'
  if (cohesion >= 0.4) return 'bg-amber-500'
  return 'bg-gray-400'
}

export function ClusterView() {
  const [result, setResult] = useState<ClusterResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [resolution, setResolution] = useState(0.5)

  const fetchCluster = async (res: number) => {
    setLoading(true)
    setError(null)
    try {
      const resp = await fetch(`/api/cluster?resolution=${res}&path=../docs`)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()
      setResult(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : '请求失败')
    }
    setLoading(false)
  }

  useEffect(() => {
    fetchCluster(resolution)
  }, [])

  const handleRerun = () => {
    fetchCluster(resolution)
  }

  return (
    <div className="flex h-full">
      {/* 左侧：聚类内容 */}
      <div className="flex-1 overflow-y-auto p-6 min-w-0">
        <div className="mb-6">
          <h1 className="text-xl font-bold mb-2">文档聚类图谱</h1>
          <p className="text-sm text-gray-500">
            基于 Louvain 算法的社区检测，自动发现概念分组
          </p>
        </div>

        {/* 控制栏 */}
        <div className="flex items-center gap-4 mb-6 p-3 bg-gray-50 rounded-lg">
          <label className="flex items-center gap-2 text-sm">
            <span className="text-gray-600">分辨率:</span>
            <input
              type="range"
              min="0.1"
              max="1.5"
              step="0.1"
              value={resolution}
              onChange={(e) => setResolution(parseFloat(e.target.value))}
              className="w-32"
            />
            <span className="font-mono text-sm w-8">{resolution.toFixed(1)}</span>
          </label>
          <button
            onClick={handleRerun}
            disabled={loading}
            className="px-4 py-1.5 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? '计算中...' : '重新聚类'}
          </button>
          {result && (
            <span className="text-xs text-gray-400 ml-auto">
              {result.totalConcepts} 概念 / {result.totalEdges} 边 /{' '}
              {result.totalCommunities} 社区
            </span>
          )}
        </div>

        {/* 错误提示 */}
        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md text-sm text-red-700">
            {error}
          </div>
        )}

        {/* 加载中 */}
        {loading && (
          <div className="flex items-center justify-center py-20">
            <div className="animate-spin h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full" />
          </div>
        )}

        {/* 社区列表 */}
        {result && !loading && (
          <div className="space-y-4">
            {result.communities.map((comm, i) => (
              <CommunityCard
                key={comm.name}
                community={comm}
                colorIndex={i % COMMUNITY_COLORS.length}
              />
            ))}
          </div>
        )}
      </div>

      {/* 右侧：图谱分析面板 */}
      <div className="w-80 shrink-0 border-l border-gray-200 bg-white overflow-hidden">
        <AnalysisPanel />
      </div>
    </div>
  )
}

function CommunityCard({
  community,
  colorIndex,
}: {
  community: Community
  colorIndex: number
}) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div
      className={`border rounded-lg overflow-hidden ${COMMUNITY_COLORS[colorIndex]}`}
    >
      {/* 头部 */}
      <div
        className="flex items-center gap-3 p-3 cursor-pointer hover:opacity-80"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm">{community.name}</span>
            <span className="text-xs text-gray-400">
              {community.size} 个概念
            </span>
          </div>
          <div className="flex items-center gap-2 mt-1">
            <div className="flex-1 h-1.5 bg-gray-200 rounded-full overflow-hidden max-w-[120px]">
              <div
                className={`h-full rounded-full transition-all ${getBarColor(community.cohesion)}`}
                style={{ width: `${Math.round(community.cohesion * 100)}%` }}
              />
            </div>
            <span className="text-xs text-gray-400">
              内聚度 {community.cohesion.toFixed(2)}
            </span>
          </div>
        </div>
        <svg
          className={`w-4 h-4 text-gray-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </div>

      {/* 成员列表 */}
      {expanded && (
        <div className="border-t border-inherit">
          {community.members.map((m) => (
            <a
              key={m.id}
              href={m.path}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-3 py-1.5 text-xs hover:bg-black/5 transition-colors"
            >
              <span className="text-gray-400 font-mono w-6 text-right shrink-0">
                {m.number ?? '--'}
              </span>
              <span className="flex-1 truncate">{m.title}</span>
              <span className="text-gray-300 shrink-0">{m.category}</span>
            </a>
          ))}
        </div>
      )}
    </div>
  )
}
