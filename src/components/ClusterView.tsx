import { useState, useEffect, useMemo } from 'react'
import { SummaryPanel } from './SummaryPanel'
import { useKnowledgeGraphStore } from '../store/knowledgeGraphStore'
import Graph from 'graphology'
import louvain from 'graphology-communities-louvain'
import type { Concept as StoreConcept, ConceptEdge } from '../types'

interface ClusterConcept {
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
  members: ClusterConcept[]
}

interface ClusterResult {
  totalConcepts: number
  totalEdges: number
  totalCommunities: number
  concepts: ClusterConcept[]
  communities: Community[]
}

function extractNumber(id: string): number | null {
  const m = id.match(/(\d+)/)
  return m ? parseInt(m[1], 10) : null
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

function runLouvain(concepts: StoreConcept[], edges: ConceptEdge[], resolution: number): ClusterResult {
  const graph = new Graph()

  for (const c of concepts) {
    graph.addNode(c.id, { title: c.title, category: c.category, path: c.path })
  }
  for (const e of edges) {
    if (graph.hasNode(e.source) && graph.hasNode(e.target)) {
      if (!graph.hasEdge(e.source, e.target)) {
        graph.addEdge(e.source, e.target)
      }
    }
  }

  const communities = louvain(graph, { resolution })

  const communityMap = new Map<string, StoreConcept[]>()
  for (const c of concepts) {
    const commId = communities[c.id]
    const commName = commId !== undefined ? String(commId) : 'isolated'
    if (!communityMap.has(commName)) communityMap.set(commName, [])
    communityMap.get(commName)!.push(c)
  }

  const commEntries = Array.from(communityMap.entries())
    .map(([name, members]) => {
      const memberConcepts: ClusterConcept[] = members.map(c => ({
        id: c.id,
        title: c.title,
        number: extractNumber(c.id),
        category: c.category,
        path: c.path,
      }))
      memberConcepts.sort((a, b) => (a.number ?? 999) - (b.number ?? 999))

      const numbers = memberConcepts.map(m => m.number).filter((n): n is number => n !== null)
      const numRange = numbers.length > 0
        ? `${Math.min(...numbers)}-${Math.max(...numbers)}`
        : ''

      // 内聚度 = 社区内部边数 / 可能的最大内部边数
      const memberIds = new Set(members.map(m => m.id))
      let internalEdges = 0
      for (const e of edges) {
        if (memberIds.has(e.source) && memberIds.has(e.target)) internalEdges++
      }
      const possibleEdges = members.length * (members.length - 1) / 2
      const cohesion = possibleEdges > 0 ? internalEdges / possibleEdges : 0

      return { name, size: members.length, numRange, cohesion, members: memberConcepts }
    })
    .sort((a, b) => b.size - a.size)

  return {
    totalConcepts: concepts.length,
    totalEdges: edges.length,
    totalCommunities: commEntries.length,
    concepts: commEntries.flatMap(c => c.members),
    communities: commEntries,
  }
}

export function ClusterView() {
  const storeConcepts = useKnowledgeGraphStore(s => s.concepts)
  const storeEdges = useKnowledgeGraphStore(s => s.edges)
  const [resolution, setResolution] = useState(0.5)

  const result = useMemo(() => {
    if (storeConcepts.length === 0) return null
    return runLouvain(storeConcepts, storeEdges, resolution)
  }, [storeConcepts, storeEdges, resolution])

  return (
    <div className="flex h-full">
      <div className="flex-1 overflow-y-auto p-6 min-w-0">
        <div className="mb-6">
          <h1 className="text-xl font-bold mb-2">文档聚类图谱</h1>
          <p className="text-sm text-gray-500">
            基于 Louvain 算法的社区检测，自动发现概念分组
          </p>
        </div>

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
          {result && (
            <span className="text-xs text-gray-400 ml-auto">
              {result.totalConcepts} 概念 / {result.totalEdges} 边 /{' '}
              {result.totalCommunities} 社区
            </span>
          )}
        </div>

        {storeConcepts.length === 0 && (
          <div className="flex items-center justify-center py-20 text-gray-400 text-sm">
            暂无概念数据，请先导入文档
          </div>
        )}

        {result && (
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

      <div className="w-80 shrink-0 border-l border-gray-200 bg-white overflow-hidden">
        <SummaryPanel />
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
    <div className={`border rounded-lg overflow-hidden ${COMMUNITY_COLORS[colorIndex]}`}>
      <div
        className="flex items-center gap-3 p-3 cursor-pointer hover:opacity-80"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm">{community.name}</span>
            <span className="text-xs text-gray-400">{community.size} 个概念</span>
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
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </div>

      {expanded && (
        <div className="border-t border-inherit">
          {community.members.map((m) => (
            <a
              key={m.id}
              href={m.path}
              target="_blank" rel="noopener noreferrer"
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
