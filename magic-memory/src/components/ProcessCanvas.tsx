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
// Skeleton mode types removed - skeleton mode is no longer supported

// ========== 自定义节点组件 ==========

function ConceptNode({ data, id }: NodeProps) {
  const d = data as any
  const isCurrent = d.isCurrent
  const isCustom = d.isCustom
  const resizeNode = d.onResize as ((nodeId: string, w: number, h: number, px: number, py: number) => void) | undefined
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

  const hasResize = d.resizeW != null
  const nodeStyle: React.CSSProperties | undefined = hasResize
    ? { width: d.resizeW as number, height: d.resizeH as number }
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
      <div className="truncate">{d.label}</div>
      {d.sub && <div className="text-[10px] text-gray-500 mt-0.5 truncate">{d.sub}</div>}
      <Handle type="source" position={Position.Right} id="r" className="!w-2 !h-2 !bg-gray-400" />
      <Handle type="source" position={Position.Bottom} id="b" className="!w-2 !h-2 !bg-gray-400" />
      <div data-corner="nw" style={{ position: 'absolute', top: -5, left: -5, width: 10, height: 10, background: '#3b82f6', border: '1px solid white', borderRadius: 2, zIndex: 10, cursor: 'nw-resize' }} />
      <div data-corner="ne" style={{ position: 'absolute', top: -5, right: -5, width: 10, height: 10, background: '#3b82f6', border: '1px solid white', borderRadius: 2, zIndex: 10, cursor: 'ne-resize' }} />
      <div data-corner="sw" style={{ position: 'absolute', bottom: -5, left: -5, width: 10, height: 10, background: '#3b82f6', border: '1px solid white', borderRadius: 2, zIndex: 10, cursor: 'sw-resize' }} />
      <div data-corner="se" style={{ position: 'absolute', bottom: -5, right: -5, width: 10, height: 10, background: '#3b82f6', border: '1px solid white', borderRadius: 2, zIndex: 10, cursor: 'se-resize' }} />
    </div>
  )
}

// GapNode with inline editing capability
function GapNode({ data }: NodeProps) {
  const d = data as any
  const isEditing = Boolean(d.isEditing)
  const editingValue = d.editingValue ?? ''

  // Local draft mirrors editingValue while editing; keep a tiny internal state for UX snappiness
  const [draft, setDraft] = useState<string>(editingValue)

  useEffect(() => {
    // whenever editing state toggles, reset local draft
    if (isEditing) {
      setDraft(editingValue)
    }
  }, [isEditing, editingValue])

  if (isEditing) {
    return (
      <div className="px-3 py-3 rounded-lg border-2 border-dashed border-amber-500 bg-amber-50/60 text-center text-xs min-w-[100px]">
        <Handle type="target" position={Position.Top} id="t" className="!w-2 !h-2 !bg-amber-400" />
        <Handle type="target" position={Position.Left} id="l" className="!w-2 !h-2 !bg-amber-400" />
        <input
          className="w-full border border-amber-300 rounded px-2 py-1 text-sm bg-white"
          value={draft}
          onChange={e => setDraft(e.target.value)}
          onBlur={() => { d.onDone?.(draft) }}
          onKeyDown={e => {
            if (e.key === 'Enter') { d.onDone?.(draft) }
            if (e.key === 'Escape') { d.onCancel?.() }
          }}
          placeholder={"输入你的答案..."}
        />
        <Handle type="source" position={Position.Right} id="r" className="!w-2 !h-2 !bg-amber-400" />
        <Handle type="source" position={Position.Bottom} id="b" className="!w-2 !h-2 !bg-amber-400" />
      </div>
    )
  }

  return (
    <div className="px-3 py-4 rounded-lg border-2 border-dashed border-amber-300 bg-amber-50/50 text-center text-xs min-w-[100px]" onClick={() => d.onStartEditing?.()}>
      <Handle type="target" position={Position.Top} id="t" className="!w-2 !h-2 !bg-amber-400" />
      <Handle type="target" position={Position.Left} id="l" className="!w-2 !h-2 !bg-amber-400" />
      <div className="text-amber-500 font-bold text-lg">?</div>
      <div className="text-amber-600 mt-0.5">空缺</div>
      {d.question && <div className="text-[10px] text-gray-500 mt-1 italic">{d.question}</div>}
      <Handle type="source" position={Position.Right} id="r" className="!w-2 !h-2 !bg-amber-400" />
      <Handle type="source" position={Position.Bottom} id="b" className="!w-2 !h-2 !bg-amber-400" />
    </div>
  )
}

// StepNode: a non-draggable derivation step card in the vertical flow
function StepNode({ data }: NodeProps) {
  const d = data as any
  const label = d.label ?? ''
  const description = d.description ?? ''
  const question = d.question ?? ''
  return (
    <div className="px-3 py-3 rounded-lg border-2 border-gray-200 bg-white shadow-sm text-xs text-left" style={{ minWidth: 180 }}>
      <Handle type="target" position={Position.Top} id="t" className="!w-2 !h-2 !bg-green-400" />
      <div className="font-semibold text-sm">{label}</div>
      {description && <div className="text-[10px] text-gray-500 mt-1 truncate">{description}</div>}
      <hr className="my-2" />
      {question && (
        <div className="text-[10px] text-gray-600"><span className="mr-1 text-yellow-500">💡</span>{question}</div>
      )}
      <div className="mt-2 text-center text-gray-400">⬇</div>
      <Handle type="source" position={Position.Bottom} id="b" className="!w-2 !h-2 !bg-green-400" />
    </div>
  )
}

// FilledNode: purple concept node that represents user-provided text
function FilledNode({ data }: NodeProps) {
  const d = data as any
  const label = d.label ?? ''
  return (
    <div className="px-3 py-3 rounded-lg border-2 border-purple-300 bg-purple-50 text-purple-800 shadow-sm text-xs min-w-[120px]" onDoubleClick={() => { d.onDoubleClick?.() }}>
      <Handle type="target" position={Position.Top} id={"t"} className="!w-2 !h-2 !bg-purple-400" />
      <div className="truncate font-semibold">{label}</div>
      <Handle type="source" position={Position.Bottom} id={"b"} className="!w-2 !h-2 !bg-purple-400" />
    </div>
  )
}

// ========== 文字输入节点（T 工具） ==========

function TextInputNode({ data }: NodeProps) {
  const d = data as any
  const [value, setValue] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (inputRef.current) inputRef.current.focus()
  }, [])

  const handleDone = () => {
    d.onDone?.(value || '未命名')
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
          if (e.key === 'Escape') d.onDone?.('')
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

const nodeTypes = {
  conceptNode: ConceptNode,
  gapNode: GapNode,
  textInputNode: TextInputNode,
  stepNode: StepNode,
  filledNode: FilledNode,
}

// ========== 主组件 ==========

interface ProcessCanvasProps {
  concept: Concept
  chain: ProcessChain | null
  allConcepts: Concept[]
  onComplete: (userFlow: string[]) => void
  onNavigate: (conceptId: string) => void
}

const NODE_W = 160
const NODE_H = 70
const GAP = 80
// vertical layout constants (tuned for a clean card-stack look)
const STEP_SPACING_Y = 180
const STEP_CARD_HEIGHT = 110
const GAP_CARD_HEIGHT = 90

function initialEdges(concept: Concept, nodes: Node[], chain: ProcessChain | null): Edge[] {
  const edges: Edge[] = []
  if (!chain) return edges
  // Build vertical derivation flow: Step_i -> Gap_i, Gap_i -> Step_(i+1)
  const steps = chain.steps
  for (let i = 0; i < steps.length; i++) {
    const stepId = `step_${steps[i].id}`
    const gapId = `gap_${steps[i].leads_to_id}`
    // Step -> Gap
    edges.push({
      id: `e_${stepId}_${gapId}`,
      source: stepId,
      target: gapId,
      type: 'smoothstep',
      animated: true,
      style: { stroke: '#94a3b8', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
    })
    // Gap -> Next Step (if exists)
    if (i < steps.length - 1) {
      const nextStepId = `step_${steps[i + 1].id}`
      edges.push({
        id: `e_${gapId}_${nextStepId}`,
        source: gapId,
        target: nextStepId,
        type: 'smoothstep',
        animated: true,
        style: { stroke: '#94a3b8', strokeWidth: 2 },
        markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
      })
    }
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

  // Gap editing start helper: switch a gap node into inline editing state
  // Uses setNodesRef to avoid circular dependency (setNodes from useNodesState is defined later)
  const onGapStartEditing = useCallback((gapId: string) => {
    setNodesRef.current(nds => nds.map(n => n.id === gapId ? { ...n, data: { ...n.data, isEditing: true, editingValue: '' } } : n))
  }, [])

  const initialNodesWithResize = useMemo((): Node[] => {
    const nodes: Node[] = []
    const knownIds = new Set(concept.depends_on)
    const knownConcepts = allConcepts.filter(c => knownIds.has(c.id))

    // Known concepts: render as headers on the top row (green headers)
    knownConcepts.forEach((c, i) => {
      nodes.push({
        id: `known_${c.id}`,
        type: 'conceptNode',
        position: { x: i * (NODE_W + GAP), y: 0 },
        data: { label: c.title, sub: c.problem?.slice(0, 20), isCurrent: false, onResize: handleResize },
      })
    })

    // Current concept sits to the right of knowns (shifted to the base y)
    const baseX = knownConcepts.length * (NODE_W + GAP)

    nodes.push({
      id: `current_${concept.id}`,
      type: 'conceptNode',
      position: { x: baseX, y: 30 },
      data: { label: concept.title, sub: '← 当前概念', isCurrent: true, onResize: handleResize },
    })

    // If there is a knowledge chain, render a vertical derivation stack: Step + Gap pairs
    if (chain) {
      const steps = chain.steps
      // Vertical spacing constants
      const UNIT_Y = STEP_SPACING_Y
      steps.forEach((s, idx) => {
        // Step node
        const stepY = 60 + (idx + 1) * UNIT_Y
        nodes.push({
          id: `step_${s.id}`,
          type: 'stepNode',
          position: { x: baseX, y: stepY - 20 },
          data: {
            label: s.label,
            description: s.description,
            question: s.question,
            onResize: handleResize,
          },
        })
        // Gap node (target concept)
        const targetConcept = allConcepts.find(c => c.id === s.leads_to_id)
        if (targetConcept) {
        nodes.push({
          id: `gap_${targetConcept.id}`,
          type: 'gapNode',
          position: { x: baseX, y: stepY + STEP_CARD_HEIGHT + 20 },
          data: {
            label: targetConcept.title,
            question: targetConcept.problem?.slice(0, 30) || '这里应该是什么？',
            isEditing: false,
            editingValue: '',
            onStartEditing: () => onGapStartEditing(`gap_${targetConcept.id}`),
            onDone: (val: string) => {
              // mutate a gap node into a filled node with the input value,
              // and wire up a quick re-edit path via double-click on the filled node
              const gapId = `gap_${targetConcept.id}`
              setNodes(nds => nds.map(n => {
                if (n.id === gapId) {
                  return {
                    ...n,
                    type: 'filledNode',
                    data: {
                      label: val,
                      onResize: handleResize,
                      onDoubleClick: () => {
                        // revert to gap editing for this gap
                        setNodes(prev => prev.map(m => m.id === gapId ? { ...m, type: 'gapNode', data: { ...m.data, isEditing: true, editingValue: val } } : m))
                      }
                    }
                  }
                }
                return n
              }))
            },
            onCancel: () => {
              // cancel editing
              setNodes(prev => prev.map(n => n.id === `gap_${targetConcept.id}` ? { ...n, data: { ...n.data, editingValue: '', isEditing: false } } : n))
            },
          },
        })
        }
      })
    }

    return nodes
  }, [concept, chain, allConcepts])

  const reactFlowWrapper = useRef<HTMLDivElement>(null)
  const [initialEdgesList] = useState(() => initialEdges(concept, initialNodesWithResize, chain))
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodesWithResize)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdgesList)
  setNodesRef.current = setNodes
  const [rfInstance, setRfInstance] = useState<ReactFlowInstance | null>(null)
  const [toolMode, setToolMode] = useState<ToolMode>('box')
  const lastPaneClickRef = useRef(0)

  // 骨架填充已移除：不再维护填充状态

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
          <svg width={16} height={16} className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" /></svg>
        </button>
        <div className="w-full h-px bg-gray-200 my-0.5" />
        {toolBtn('box',
          <svg width={14} height={14} className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v14a1 1 0 01-1 1H5a1 1 0 01-1-1V5z" /></svg>,
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
    const newLabel = window.prompt('重命名节点', (node.data as any).label)
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

  const handleDeleteNode = useCallback((nodeId: string) => {
    setNodes(nds => nds.filter(n => n.id !== nodeId))
    setEdges(eds => eds.filter(e => e.source !== nodeId && e.target !== nodeId))
  }, [setNodes, setEdges])

  // 骨架填充模式已移除，统一渲染自由画板模式

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
        <div className="text-[10px] text-gray-400">
          {nodes.length} 节点 · {edges.length} 连线 · Del 删除
        </div>
      </div>
    </div>
  )
}
