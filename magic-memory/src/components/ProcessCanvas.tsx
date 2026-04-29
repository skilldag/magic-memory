import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ReactFlow,
  ReactFlowProvider,
  useNodesState,
  useEdgesState,
  useReactFlow,
  addEdge,
  Background,
  MiniMap,
  Handle,
  Position,
  type Node,
  type Edge,
  type NodeProps,
  type Connection,
  type ReactFlowInstance,
  MarkerType,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type { Concept, ProcessChain } from '../types'
import type { SkeletonNodeDef } from '../utils/processComparison'

// ========== 自定义节点组件 ==========

function ConceptNode({ data, id }: NodeProps) {
  const isCurrent = data.isCurrent
  const isCustom = data.isCustom
  const resizeNode = data.onResize as ((nodeId: string, w: number, h: number, px: number, py: number) => void) | undefined
  const nodeRef = useRef<HTMLDivElement>(null)

  // 用原生 capture 监听器在 ReactFlow 之前拦截手柄的 mousedown
  useEffect(() => {
    const el = nodeRef.current
    if (!el || !resizeNode) return

    const nativeDown = (e: MouseEvent) => {
      const target = e.target as HTMLElement
      const corner = target.getAttribute('data-corner')
      if (!corner) return

      e.stopPropagation()
      e.preventDefault()

      const rect = el.getBoundingClientRect()
      const start = { w: rect.width, h: rect.height, x: e.clientX, y: e.clientY, corner }

      const onMove = (ev: MouseEvent) => {
        let dw = ev.clientX - start.x
        let dh = ev.clientY - start.y
        let newW = start.corner.includes('e') ? start.w + dw : start.w - dw
        let newH = start.corner.includes('s') ? start.h + dh : start.h - dh
        newW = Math.max(80, newW)
        newH = Math.max(40, newH)
        const px = start.corner.includes('w') ? start.w - newW : 0
        const py = start.corner.includes('n') ? start.h - newH : 0
        resizeNode(id, newW, newH, px, py)
      }

      const onUp = () => {
        document.removeEventListener('mousemove', onMove)
        document.removeEventListener('mouseup', onUp)
      }

      document.addEventListener('mousemove', onMove)
      document.addEventListener('mouseup', onUp)
    }

    el.addEventListener('mousedown', nativeDown, true)
    return () => el.removeEventListener('mousedown', nativeDown, true)
  }, [id, resizeNode])

  const hasResize = data.resizeW != null
  const nodeStyle: React.CSSProperties | undefined = hasResize
    ? { width: data.resizeW, height: data.resizeH }
    : undefined

  return (
    <div ref={nodeRef} className={`relative px-3 py-2 rounded-lg border-2 shadow-sm text-xs text-center ${
      isCurrent
        ? 'bg-blue-50 border-blue-400 text-blue-900 font-semibold'
        : isCustom
        ? 'bg-purple-50 border-purple-300 text-purple-800'
        : 'bg-emerald-50 border-emerald-300 text-gray-800'
    }`}
      style={nodeStyle}
    >
      <Handle type="target" position={Position.Top} id="t" className="!w-2 !h-2 !bg-gray-400" />
      <Handle type="target" position={Position.Left} id="l" className="!w-2 !h-2 !bg-gray-400" />
      <div className="truncate">{data.label}</div>
      {data.sub && <div className="text-[10px] text-gray-500 mt-0.5 truncate">{data.sub}</div>}
      <Handle type="source" position={Position.Right} id="r" className="!w-2 !h-2 !bg-gray-400" />
      <Handle type="source" position={Position.Bottom} id="b" className="!w-2 !h-2 !bg-gray-400" />
      <div data-corner="nw" style={{ position: 'absolute', top: -5, left: -5, width: 10, height: 10, background: '#3b82f6', border: '1px solid white', borderRadius: 2, zIndex: 10, cursor: 'nw-resize' }} />
      <div data-corner="ne" style={{ position: 'absolute', top: -5, right: -5, width: 10, height: 10, background: '#3b82f6', border: '1px solid white', borderRadius: 2, zIndex: 10, cursor: 'ne-resize' }} />
      <div data-corner="sw" style={{ position: 'absolute', bottom: -5, left: -5, width: 10, height: 10, background: '#3b82f6', border: '1px solid white', borderRadius: 2, zIndex: 10, cursor: 'sw-resize' }} />
      <div data-corner="se" style={{ position: 'absolute', bottom: -5, right: -5, width: 10, height: 10, background: '#3b82f6', border: '1px solid white', borderRadius: 2, zIndex: 10, cursor: 'se-resize' }} />
    </div>
  )
}

function GapNode({ data }: NodeProps) {
  return (
    <div className="px-3 py-4 rounded-lg border-2 border-dashed border-amber-300 bg-amber-50/50 text-center text-xs min-w-[100px]">
      <Handle type="target" position={Position.Top} id="t" className="!w-2 !h-2 !bg-amber-400" />
      <Handle type="target" position={Position.Left} id="l" className="!w-2 !h-2 !bg-amber-400" />
      <div className="text-amber-500 font-bold text-lg">?</div>
      <div className="text-amber-600 mt-0.5">空缺</div>
      {data.question && <div className="text-[10px] text-gray-500 mt-1 italic">{data.question}</div>}
      <Handle type="source" position={Position.Right} id="r" className="!w-2 !h-2 !bg-amber-400" />
      <Handle type="source" position={Position.Bottom} id="b" className="!w-2 !h-2 !bg-amber-400" />
    </div>
  )
}

// ========== 文字输入节点（T 工具） ==========

function TextInputNode({ data }: NodeProps) {
  const [value, setValue] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (inputRef.current) inputRef.current.focus()
  }, [])

  const handleDone = () => {
    data.onDone?.(value || '未命名')
  }

  return (
    <div className="px-2 py-1.5 rounded-lg border-2 border-dashed border-blue-300 bg-white shadow-sm text-xs min-w-[120px]">
      <Handle type="target" position={Position.Top} id="t" className="!w-2 !h-2 !bg-blue-300" />
      <Handle type="target" position={Position.Left} id="l" className="!w-2 !h-2 !bg-blue-300" />
      <input
        ref={inputRef}
        value={value}
        onChange={e => setValue(e.target.value)}
        onBlur={handleDone}
        onKeyDown={e => {
          if (e.key === 'Enter') handleDone()
          if (e.key === 'Escape') data.onDone?.('')
        }}
        className="w-full outline-none bg-transparent text-gray-700 placeholder-gray-400"
        placeholder="输入文本..."
      />
      <Handle type="source" position={Position.Right} id="r" className="!w-2 !h-2 !bg-blue-300" />
      <Handle type="source" position={Position.Bottom} id="b" className="!w-2 !h-2 !bg-blue-300" />
    </div>
  )
}

type ToolMode = 'select' | 'box' | 'text'

const nodeTypes = { conceptNode: ConceptNode, gapNode: GapNode, textInputNode: TextInputNode }

// ========== 主组件 ==========

interface ProcessCanvasProps {
  concept: Concept
  chain: ProcessChain | null
  allConcepts: Concept[]
  onComplete: (userFlow: string[]) => void
  onNavigate: (conceptId: string) => void
  /** 首次进入时启用骨架填充模式 */
  skeletonMode?: boolean
  /** 骨架模式的引导节点 */
  skeletonNodes?: SkeletonNodeDef[]
  /** 提交骨架填充结果 */
  onSkeletonSubmit?: (results: { gapId: string; filledConceptId: string | null }[]) => void
  /** 打开提问 */
  onOpenQuestion?: () => void
}

const NODE_W = 160
const NODE_H = 70
const GAP = 80

function initialEdges(concept: Concept, nodes: Node[]): Edge[] {
  const edges: Edge[] = []
  const ordered = nodes
  for (let i = 0; i < ordered.length - 1; i++) {
    edges.push({
      id: `e_${ordered[i].id}_${ordered[i + 1].id}`,
      source: ordered[i].id,
      target: ordered[i + 1].id,
      type: 'smoothstep',
      animated: true,
      style: { stroke: '#94a3b8', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
    })
  }
  return edges
}

export function ProcessCanvas({
  concept,
  chain,
  allConcepts,
  onComplete,
  onNavigate,
}: ProcessCanvasProps) {
  const setNodesRef = useRef<(nds: Node[] | ((nds: Node[]) => Node[])) => void>(() => {})

  const handleResize = useCallback((nodeId: string, w: number, h: number, px: number, py: number) => {
    setNodesRef.current(nds => nds.map(n =>
      n.id === nodeId
        ? { ...n, data: { ...n.data, resizeW: w, resizeH: h }, position: { x: n.position.x + px, y: n.position.y + py } }
        : n
    ))
    console.log('[resize] setNodes called')
  }, [])

  const initialNodesWithResize = useMemo((): Node[] => {
    const nodes: Node[] = []
    const knownIds = new Set(concept.depends_on)
    const knownConcepts = allConcepts.filter(c => knownIds.has(c.id))

    knownConcepts.forEach((c, i) => {
      nodes.push({
        id: `known_${c.id}`,
        type: 'conceptNode',
        position: { x: i * (NODE_W + GAP), y: 0 },
        data: { label: c.title, sub: c.problem?.slice(0, 20), isCurrent: false, onResize: handleResize },
      })
    })

    const offsetX = knownConcepts.length * (NODE_W + GAP)

    nodes.push({
      id: `current_${concept.id}`,
      type: 'conceptNode',
      position: { x: offsetX, y: 0 },
      data: { label: concept.title, sub: '← 当前概念', isCurrent: true, onResize: handleResize },
    })

    if (chain) {
      const chainStepIds = new Set(chain.steps.map(s => s.leads_to_id).filter(Boolean))
      const gapConcepts = allConcepts.filter(c =>
        c.id !== concept.id && !knownIds.has(c.id) && chainStepIds.has(c.id)
      )
      gapConcepts.forEach((c, i) => {
        nodes.push({
          id: `gap_${c.id}`,
          type: 'gapNode',
          position: { x: offsetX + (i + 1) * (NODE_W + GAP), y: 0 },
          data: { label: c.title, question: c.problem?.slice(0, 30) || '这里应该是什么？' },
        })
      })
    }

    return nodes
  }, [concept, chain, allConcepts])

  const reactFlowWrapper = useRef<HTMLDivElement>(null)
  const [initialEdgesList] = useState(() => initialEdges(concept, initialNodesWithResize))
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodesWithResize)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdgesList)
  setNodesRef.current = setNodes
  const [rfInstance, setRfInstance] = useState<ReactFlowInstance | null>(null)
  const [submitted, setSubmitted] = useState(false)
  const [toolMode, setToolMode] = useState<ToolMode>('box')
  const lastPaneClickRef = useRef(0)

  // 骨架填充状态
  const [skeletonFills, setSkeletonFills] = useState<Map<string, string | null>>(new Map())
  const [skeletonResults, setSkeletonResults] = useState<{ gapId: string; correct: boolean; filledLabel: string | null }[] | null>(null)

  // 骨架拖拽
  useEffect(() => {
    const handleDrop = (e: DragEvent) => {
      const target = (e.target as HTMLElement).closest('[data-gap-id]')
      if (!target) return
      const gapId = target.getAttribute('data-gap-id')
      const conceptId = e.dataTransfer?.getData('text/plain')
      if (!gapId || !conceptId) return
      e.preventDefault()
      setSkeletonFills(prev => { const m = new Map(prev); m.set(gapId, conceptId); return m })
      setSkeletonResults(null)
    }
    const handleDragOver = (e: DragEvent) => {
      const target = (e.target as HTMLElement).closest('[data-gap-id]')
      if (target) e.preventDefault()
    }
    document.addEventListener('drop', handleDrop)
    document.addEventListener('dragover', handleDragOver)
    return () => {
      document.removeEventListener('drop', handleDrop)
      document.removeEventListener('dragover', handleDragOver)
    }
  }, [])

  function CombinedControls({ toolMode, setToolMode: onToolChange }: { toolMode: ToolMode; setToolMode: (m: ToolMode) => void }) {
    const { zoomIn, zoomOut, fitView } = useReactFlow()

    const toolBtn = (mode: ToolMode, label: string | React.ReactNode, title: string) => (
      <button onClick={() => onToolChange(mode)}
        className={`react-flow__controls-button flex items-center justify-center !text-xs ${toolMode === mode ? '!bg-blue-100 !text-blue-600' : ''}`}
        title={title}
      >{label}</button>
    )

    return (
      <div className="react-flow__controls !shadow !border !border-gray-200 !rounded-lg"
        style={{ position: 'absolute', bottom: 20, left: 20, zIndex: 5 }}>
        <button onClick={() => zoomIn()} className="react-flow__controls-button" title="放大">+</button>
        <button onClick={() => zoomOut()} className="react-flow__controls-button" title="缩小">-</button>
        <button onClick={() => fitView()} className="react-flow__controls-button flex items-center justify-center" title="适应屏幕">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" /></svg>
        </button>
        <div className="w-full h-px bg-gray-200 my-0.5" />
        {toolBtn('box',
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v14a1 1 0 01-1 1H5a1 1 0 01-1-1V5z" /></svg>,
          '流程框')}
        {toolBtn('text', <span className="font-bold">T</span>, '文本')}
      </div>
    )
  }

  const onPaneClick = useCallback((event: React.MouseEvent) => {
    if (!rfInstance) return

    const now = Date.now()
    if (now - lastPaneClickRef.current < 400) {
      lastPaneClickRef.current = 0
      const position = rfInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      })

      if (toolMode === 'box') {
        const customCount = nodes.filter(n => n.id.startsWith('custom_')).length
        setNodes(nds => [...nds, {
          id: `custom_${Date.now()}`,
          type: 'conceptNode',
          position,
          data: { label: `新概念 ${customCount + 1}`, sub: '双击重命名', isCurrent: false, isCustom: true, onResize: handleResize },
        }])
      } else if (toolMode === 'text') {
        const id = `text_${Date.now()}`
        setNodes(nds => [...nds, {
          id,
          type: 'textInputNode',
          position,
          data: {
            onResize: handleResize,
            onDone: (text: string) => {
              setNodes(prev => prev.map(n =>
                n.id === id
                  ? { ...n, type: 'conceptNode', data: { label: text, sub: '', isCustom: true, isCurrent: false, onResize: handleResize } }
                  : n
              ))
            },
          },
        }])
      }
    } else {
      lastPaneClickRef.current = now
    }
  }, [toolMode, rfInstance, setNodes, handleResize])

  const onNodeDoubleClick = useCallback((_event: React.MouseEvent, node: Node) => {
    if (!node.id.startsWith('custom_')) return
    const newLabel = window.prompt('重命名节点', node.data.label)
    if (newLabel && newLabel.trim()) {
      setNodes(nds => nds.map(n =>
        n.id === node.id ? { ...n, data: { ...n.data, label: newLabel.trim(), sub: '' } } : n
      ))
    }
  }, [setNodes])

  const onConnect = useCallback((params: Connection) => {
    setEdges(eds => addEdge({
      ...params,
      type: 'smoothstep',
      animated: true,
      style: { stroke: '#94a3b8', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
    }, eds))
  }, [setEdges])

  const onInit = useCallback((instance: ReactFlowInstance) => {
    setRfInstance(instance)
    setTimeout(() => instance.fitView({ padding: 0.3 }), 100)
  }, [])

  const handleAutoLayout = useCallback(() => {
    setNodes(nds => {
      const ys = new Map<string, number>()
      const xs = new Map<string, number>()
      nds.forEach((n, i) => {
        const col = i
        xs.set(n.id, col * (NODE_W + GAP))
        ys.set(n.id, 0)
      })
      return nds.map(n => ({
        ...n,
        position: { x: xs.get(n.id) ?? 0, y: ys.get(n.id) ?? 0 },
      }))
    })
    setTimeout(() => rfInstance?.fitView({ padding: 0.3 }), 50)
  }, [setNodes, rfInstance])

  const handleSubmit = useCallback(() => {
    const flow = nodes.map(n => n.id)
    onComplete(flow)
    setSubmitted(true)
  }, [nodes, onComplete])

  const handleDeleteNode = useCallback((nodeId: string) => {
    setNodes(nds => nds.filter(n => n.id !== nodeId))
    setEdges(eds => eds.filter(e => e.source !== nodeId && e.target !== nodeId))
  }, [setNodes, setEdges])

  // ========== 骨架填充模式 ==========

  if (skeletonMode && skeletonNodes) {
    const gapNodes = skeletonNodes.filter(n => n.type === 'gap')
    const filledCount = gapNodes.filter(g => skeletonFills.get(g.id) != null).length

    return (
      <div className="flex flex-col h-full">
        <div className="flex-1 overflow-y-auto px-6 py-4">
          <div className="max-w-3xl mx-auto space-y-4">
            {/* 进度条 */}
            <div className="flex items-center gap-2 text-xs text-gray-500 mb-4">
              <span>填充进度</span>
              <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                <div className="h-full bg-blue-500 rounded-full transition-all"
                  style={{ width: `${gapNodes.length > 0 ? (filledCount / gapNodes.length) * 100 : 0}%` }}
                />
              </div>
              <span>{filledCount}/{gapNodes.length}</span>
            </div>

            {/* 步骤卡片 */}
            <div className="space-y-3">
              {skeletonNodes.map((node) => {
                const filledId = skeletonFills.get(node.id)
                const filledConcept = filledId ? allConcepts.find(c => c.id === filledId) : null
                const res = skeletonResults?.find(r => r.gapId === node.id)

                if (node.type === 'current') {
                  return (
                    <div key={node.id} className="p-4 rounded-lg border-2 border-blue-300 bg-blue-50">
                      <div className="flex items-center gap-2">
                        <span className="text-lg">📍</span>
                        <div>
                          <div className="text-sm font-semibold text-blue-900">{node.label}</div>
                          <div className="text-xs text-blue-600">当前概念：{node.question}</div>
                        </div>
                      </div>
                    </div>
                  )
                }

                return (
                  <div key={node.id} data-gap-id={node.id} className={`p-4 rounded-lg border-2 transition-colors ${
                    res
                      ? res.correct ? 'border-emerald-300 bg-emerald-50' : 'border-red-300 bg-red-50'
                      : filledId
                      ? 'border-blue-300 bg-blue-50'
                      : 'border-amber-200 bg-amber-50/50 border-dashed'
                  }`}>
                    <div className="flex items-start gap-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="w-5 h-5 rounded-full bg-gray-200 text-xs flex items-center justify-center text-gray-600 font-medium">
                            {skeletonNodes.indexOf(node) + 1}
                          </span>
                          <span className="text-xs font-medium text-gray-500">{node.label}</span>
                        </div>
                        <div className="text-sm text-gray-700 ml-7">
                          <span className="italic">{node.question}</span>
                        </div>
                        {node.hint && (
                          <div className="text-xs text-gray-400 ml-7 mt-1">💡 {node.hint}</div>
                        )}
                        <div className="ml-7 mt-2">
                          {filledConcept ? (
                            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-white border border-blue-200 text-sm text-blue-700">
                              <span>✓</span>
                              <span>{filledConcept.title}</span>
                              <button onClick={() => {
                                setSkeletonFills(prev => { const m = new Map(prev); m.delete(node.id); return m })
                                setSkeletonResults(null)
                              }} className="text-gray-400 hover:text-red-500 ml-1">✕</button>
                            </div>
                          ) : (
                            <span className="text-xs text-amber-500">从下方拖拽概念到此处</span>
                          )}
                        </div>
                      </div>
                      {res && (
                        <div className={`shrink-0 text-lg ${res.correct ? 'text-emerald-500' : 'text-red-500'}`}>
                          {res.correct ? '✓' : '✗'}
                        </div>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        {/* 候选概念区 */}
        <div className="shrink-0 px-6 py-3 border-t border-gray-200 bg-gray-50">
          <div className="max-w-3xl mx-auto">
            <div className="text-xs text-gray-500 mb-2">拖动概念到空缺节点：</div>
            <div className="flex flex-wrap gap-2">
              {allConcepts
                .filter(c => c.id !== concept.id && !skeletonFills.has(`gap_step-${c.id}`) && !Array.from(skeletonFills.values()).includes(c.id))
                .slice(0, 15)
                .map(c => (
                  <button key={c.id}
                    draggable
                    onDragStart={(e) => e.dataTransfer.setData('text/plain', c.id)}
                    className="px-2.5 py-1.5 text-xs font-medium rounded-md border border-gray-200 bg-white text-gray-700 hover:border-blue-300 hover:bg-blue-50 cursor-grab active:cursor-grabbing transition-colors"
                  >
                    {c.title}
                  </button>
                ))}
              <button onClick={onOpenQuestion}
                className="px-2.5 py-1.5 text-xs font-medium rounded-md border border-dashed border-purple-200 bg-purple-50 text-purple-600 hover:bg-purple-100 transition-colors"
              >
                💬 提问
              </button>
            </div>
          </div>
        </div>

        {/* 底部操作栏 */}
        <div className="shrink-0 flex items-center gap-2 px-6 py-3 border-t border-gray-100 bg-white">
          <div className="flex-1" />
          {!skeletonResults && (
            <button onClick={() => {
              const results = gapNodes.map(g => ({
                gapId: g.id,
                correct: skeletonFills.get(g.id) === g.correctConceptId,
                filledLabel: allConcepts.find(c => c.id === skeletonFills.get(g.id))?.title ?? null,
              }))
              setSkeletonResults(results)
              onSkeletonSubmit?.(gapNodes.map(g => ({
                gapId: g.id,
                filledConceptId: skeletonFills.get(g.id) ?? null,
              })))
            }} disabled={filledCount === 0}
              className="px-4 py-1.5 text-xs font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              提交验证
            </button>
          )}
          {skeletonResults && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">
                正确 {skeletonResults.filter(r => r.correct).length}/{gapNodes.length}
              </span>
              <button onClick={() => { setSkeletonFills(new Map()); setSkeletonResults(null) }}
                className="px-3 py-1.5 text-xs font-medium text-gray-600 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors"
              >
                重新填充
              </button>
            </div>
          )}
        </div>
      </div>
    )
  }

  // ========== 自由画板模式 ==========

  if (!chain) {
    return (
      <div className="px-5 py-4">
        <div className="p-4 rounded-lg border border-gray-200 bg-gray-50 text-center">
          <p className="text-sm text-gray-500">当前概念没有关联的过程链</p>
          <p className="text-xs text-gray-400 mt-1">切换到"查阅文档"直接阅读概念详情</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* 画布 */}
      <div ref={reactFlowWrapper} className="flex-1">
        <ReactFlowProvider>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onInit={onInit}
            onClick={onPaneClick}
            onNodeDoubleClick={onNodeDoubleClick}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.3 }}
            deleteKeyCode={['Backspace', 'Delete']}
            snapToGrid
            snapGrid={[15, 15]}
            minZoom={0.2}
            maxZoom={3}
          >
            <Background color="#f1f5f9" gap={20} />
            <MiniMap
              nodeStrokeColor="#94a3b8"
              nodeColor="#e2e8f0"
              maskColor="rgba(0,0,0,0.08)"
              className="!shadow !border !border-gray-200 !rounded-lg"
            />
          </ReactFlow>
          <CombinedControls toolMode={toolMode} setToolMode={setToolMode} />
        </ReactFlowProvider>
      </div>

      {/* 底部工具栏 */}
      <div className="shrink-0 flex items-center gap-2 px-4 py-2 border-t border-gray-100 bg-white">
        <button
          onClick={handleAutoLayout}
          className="px-3 py-1.5 text-xs font-medium text-gray-600 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors"
        >
          自动排列
        </button>
        <div className="flex-1" />
        {!submitted && (
          <button
            onClick={handleSubmit}
            disabled={nodes.length === 0}
            className="px-4 py-1.5 text-xs font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            提交梳理，生成对照
          </button>
        )}
        {submitted && (
          <span className="text-xs text-emerald-600 font-medium">✓ 已提交</span>
        )}
        <div className="text-[10px] text-gray-400">
          {nodes.length} 节点 · {edges.length} 连线 · Del 删除
        </div>
      </div>
    </div>
  )
}
