import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import cytoscape, { Core, ElementDefinition } from 'cytoscape'
// @ts-expect-error cytoscape-fcose has no types
import fcose from 'cytoscape-fcose'
import type { Concept, ConceptEdge } from '../types'
import { LEVEL_COLORS, EDGE_COLORS } from '../constants/graph'

cytoscape.use(fcose)

interface AdaptiveParams {
  idealEdgeLength: number
  nodeRepulsion: number
  gravity: number
  padding: number
  numIter: number
}

function calcAdaptiveLayoutParams(
  containerWidth: number,
  containerHeight: number,
  nodeCount: number
): AdaptiveParams {
  const diagonal = Math.sqrt(containerWidth * containerWidth + containerHeight * containerHeight)
  const idealEdgeLength = Math.max(60, Math.min(200, (diagonal / Math.sqrt(Math.max(nodeCount, 1))) * 1.2))
  const nodeRepulsion = Math.max(12000, Math.min(80000, idealEdgeLength * nodeCount * 5))
  const gravity = Math.max(0.02, Math.min(0.12, 50 / (nodeCount + 10)))
  const padding = Math.max(20, Math.min(80, Math.min(containerWidth, containerHeight) * 0.06))
  const numIter = Math.max(200, Math.min(1200, nodeCount * 15))
  return { idealEdgeLength, nodeRepulsion, gravity, padding, numIter }
}

interface KnowledgeGraphProps {
  concepts: Concept[]
  edges: ConceptEdge[]
  selectedConcept: Concept | null
  focusEnabled?: boolean
  linkMode?: boolean
  linkSource?: string | null
  onSelectConcept: (concept: Concept) => void
  onNavigate?: (conceptId: string) => void
  onDoubleTapConcept?: (concept: Concept) => void
  onBackgroundDoubleTap?: () => void
  onHoverConcept?: (payload: { concept: Concept; x: number; y: number; width: number; height: number }) => void
  onHoverLeave?: () => void
  onToggleLinkMode?: () => void
  onLinkStart?: (conceptId: string) => void
  onLinkEnd?: (sourceId: string, targetId: string) => void
  onLinkCancel?: () => void
  // Precision focus: supply a set of node IDs to focus the graph to a specific subset
  focusedNodeIds?: string[]
  // Container dimensions for adaptive layout
  containerWidth?: number
  containerHeight?: number
  // Exit focus mode callback
  onExitFocus?: () => void
}

export function KnowledgeGraph({
  concepts,
  edges,
  selectedConcept,
  focusEnabled = false,
  linkMode = false,
  linkSource = null,
  focusedNodeIds,
  containerWidth = 1200,
  containerHeight = 800,
  onSelectConcept,
  onNavigate,
  onDoubleTapConcept,
  onBackgroundDoubleTap,
  onHoverConcept,
  onHoverLeave,
  onToggleLinkMode,
  onLinkStart,
  onLinkEnd,
  onLinkCancel,
  onExitFocus
}: KnowledgeGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const cyRef = useRef<Core | null>(null)
  const onHoverConceptRef = useRef(onHoverConcept)
  const onHoverLeaveRef = useRef(onHoverLeave)
  const onDoubleTapConceptRef = useRef(onDoubleTapConcept)
  onHoverConceptRef.current = onHoverConcept
  onHoverLeaveRef.current = onHoverLeave
  onDoubleTapConceptRef.current = onDoubleTapConcept
  const lastBackgroundTapAtRef = useRef(0)
  const wasFocusedRef = useRef(false)
  const lastNodeTapRef = useRef({ id: '', time: 0 })
  const conceptsRef = useRef(concepts)
  const edgesRef = useRef(edges)
  conceptsRef.current = concepts
  edgesRef.current = edges
  const initialLayoutDoneRef = useRef(false)
  const linkModeRef = useRef(linkMode)
  const linkSourceRef = useRef(linkSource)
  linkModeRef.current = linkMode
  linkSourceRef.current = linkSource
  const [isReady, setIsReady] = useState(false)
  const [zoomLevel, setZoomLevel] = useState(1)

  // 结构签名：仅当节点 ID 集或边结构变化时变更——纯 content 编辑不触发布局
  const structuralKey = useMemo(
    () =>
      concepts.map(c => c.id).sort().join(',') + '|' +
      edges.map(e => `${e.source}-${e.target}-${e.type}`).sort().join(','),
    [concepts, edges]
  )

  const handleZoomIn = useCallback(() => {
    if (!cyRef.current) return
    cyRef.current.zoom(cyRef.current.zoom() * 1.3)
    setZoomLevel(cyRef.current.zoom())
  }, [])

  const handleZoomOut = useCallback(() => {
    if (!cyRef.current) return
    cyRef.current.zoom(cyRef.current.zoom() * 0.7)
    setZoomLevel(cyRef.current.zoom())
  }, [])

  const handleFit = useCallback(() => {
    if (!cyRef.current) return
    cyRef.current.fit(undefined, 50)
    setZoomLevel(cyRef.current.zoom())
  }, [])

  const handleSmartLayout = useCallback(() => {
    const cy = cyRef.current
    if (!cy) return
    const count = concepts.length
    if (count <= 1) {
      cy.fit(undefined, 50)
      setZoomLevel(cy.zoom())
      return
    }
    const w = containerWidth || containerRef.current?.clientWidth || 1200
    const h = containerHeight || containerRef.current?.clientHeight || 800
    const params = calcAdaptiveLayoutParams(w, h, count)
    try {
      const layout = cy.layout({
        name: 'fcose',
        quality: 'proof',
        animate: true,
        animationDuration: 400,
        nodeRepulsion: params.nodeRepulsion,
        idealEdgeLength: params.idealEdgeLength,
        gravity: params.gravity,
        numIter: params.numIter,
        tile: true,
        padding: params.padding,
      } as cytoscape.LayoutOptions)
      layout.one('layoutstop', () => {
        cy.fit(undefined, params.padding)
        setZoomLevel(cy.zoom())
      })
      layout.run()
    } catch (e) {
      console.warn('[KnowledgeGraph] smart layout failed:', e)
      cy.fit(undefined, 50)
      setZoomLevel(cy.zoom())
    }
  }, [concepts.length, containerWidth, containerHeight])

  // 初始化 Cytoscape（仅在首次数据到达时创建）
  useEffect(() => {
    if (cyRef.current) return
    if (!containerRef.current) return
    if (concepts.length === 0) return
    
    const elements: ElementDefinition[] = [
      ...concepts.map(c => ({
        group: 'nodes' as const,
        data: {
          id: c.id,
          label: c.title,
          level: c.level,
          category: c.category
        }
      })),
      ...edges.map(e => ({
        group: 'edges' as const,
        data: {
          id: e.id,
          source: e.source,
          target: e.target,
          edgeType: e.type
        }
      }))
    ]

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: [
        {
          selector: 'node',
          style: {
            'label': 'data(label)',
            'width': 50,
            'height': 50,
            'background-color': '#3b82f6',
            'color': '#374151',
            'font-size': '10px',
            'text-valign': 'bottom',
            'text-margin-y': 6,
            'text-wrap': 'wrap',
            'text-max-width': '80px',
            'shape': 'ellipse',
            'border-width': 2,
            'border-color': '#fff'
          }
        },
        {
          selector: 'node[level="1"]',
          style: { 'background-color': LEVEL_COLORS[1] }
        },
        {
          selector: 'node[level="2"]',
          style: { 'background-color': LEVEL_COLORS[2] }
        },
        {
          selector: 'node[level="3"]',
          style: { 'background-color': LEVEL_COLORS[3] }
        },
        {
          selector: 'node:selected',
          style: {
            'border-width': 4,
            'border-color': '#f59e0b',
            'background-color': '#f59e0b'
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 2,
            'line-color': '#d1d5db',
            'target-arrow-color': '#d1d5db',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'opacity': 0.7
          }
        },
        {
          selector: 'edge[edgeType="depends_on"]',
          style: {
            'line-color': EDGE_COLORS.depends_on,
            'target-arrow-color': EDGE_COLORS.depends_on,
            'width': 3,
            'opacity': 0.9
          }
        },
        {
          selector: 'edge[edgeType="leads_to"]',
          style: {
            'line-color': EDGE_COLORS.leads_to,
            'target-arrow-color': EDGE_COLORS.leads_to,
            'width': 3,
            'opacity': 0.9
          }
        },
        {
          selector: 'edge[edgeType="related"]',
          style: {
            'line-color': EDGE_COLORS.related,
            'target-arrow-color': EDGE_COLORS.related,
            'width': 1.5,
            'opacity': 0.5,
            'line-style': 'dashed'
          }
        }
      ],
      minZoom: 0.2,
      maxZoom: 4,
      wheelSensitivity: 0.15
    })

    // Stage 1: fast rough layout (immediate)
    try {
      const fastLayout = cy.layout({
        name: 'fcose',
        quality: 'default',
        animate: false,
        nodeRepulsion: 25000,
        idealEdgeLength: 160,
        gravity: 0.08,
        numIter: 200,
        tile: true,
        padding: 80
      } as cytoscape.LayoutOptions)
      fastLayout.one('layoutstop', () => {
        cy.fit(undefined, 50)
        setZoomLevel(cy.zoom())
      })
      fastLayout.run()
    } catch (e) {
      console.warn('[KnowledgeGraph] fast layout skipped:', e)
      // Fallback: basic fit even if layout fails
      try { cy.fit(undefined, 50) } catch {}
    }

    // Stage 2: refined layout during idle time
    if (typeof requestIdleCallback !== 'undefined') {
      requestIdleCallback(() => {
        const cy = cyRef.current
        if (!cy) return  // component unmounted
        try {
          const refineLayout = cy.layout({
            name: 'fcose',
            quality: 'proof',
            animate: true,
            animationDuration: 600,
            nodeRepulsion: 25000,
            idealEdgeLength: 160,
            gravity: 0.08,
            numIter: 1000,
            tile: true,
            padding: 80
          } as cytoscape.LayoutOptions)
          refineLayout.run()
        } catch (e) {
          console.warn('[KnowledgeGraph] refine layout skipped:', e)
        }
      }, { timeout: 3000 })
    }

    cy.on('zoom', () => {
      setZoomLevel(cy.zoom())
    })
    
    cy.on('tap', 'node', (evt) => {
      const nodeId = evt.target.id()
      const now = Date.now()
      const prev = lastNodeTapRef.current
      
      if (linkModeRef.current) {
        console.log('[KnowledgeGraph] node clicked in linkMode, linkSource:', linkSourceRef.current, 'nodeId:', nodeId)
        if (linkSourceRef.current) {
          if (linkSourceRef.current !== nodeId) {
            console.log('[KnowledgeGraph] calling onLinkEnd with:', linkSourceRef.current, nodeId)
            onLinkEnd?.(linkSourceRef.current, nodeId)
          }
        } else {
          console.log('[KnowledgeGraph] calling onLinkStart with:', nodeId)
          onLinkStart?.(nodeId)
        }
        lastNodeTapRef.current = { id: nodeId, time: now }
        return
      }
      
      if (prev.id === nodeId && now - prev.time < 400) {
        const concept = conceptsRef.current.find(c => c.id === nodeId)
        const handler = onDoubleTapConceptRef.current
        if (concept && handler) handler(concept)
        lastNodeTapRef.current = { id: '', time: 0 }
        return
      }
      lastNodeTapRef.current = { id: nodeId, time: now }
      const concept = concepts.find(c => c.id === nodeId)
      if (concept) {
        onSelectConcept(concept)
      }
    })

    cy.on('mouseover', 'node', (evt) => {
      const nodeId = evt.target.id()
      const concept = concepts.find(c => c.id === nodeId)
      if (!concept) return
      const position = evt.target.renderedPosition()
      const nodeEl = evt.target
      const width = nodeEl.renderedOuterWidth ? nodeEl.renderedOuterWidth() : 50
      const height = nodeEl.renderedOuterHeight ? nodeEl.renderedOuterHeight() : 30
      onHoverConceptRef.current?.({ concept, x: position.x, y: position.y, width, height })
    })

    cy.on('mouseout', 'node', () => {
      onHoverLeaveRef.current?.()
    })

    cy.on('tap', (evt) => {
      if (evt.target !== cy) return
      
      if (linkMode && linkSource) {
        onLinkCancel?.()
        return
      }
      
      const now = Date.now()
      const delta = now - lastBackgroundTapAtRef.current
      lastBackgroundTapAtRef.current = now
      if (delta > 0 && delta < 300) {
        onBackgroundDoubleTap?.()
      }
    })
    
    // doubleTap via Cytoscape gesture
    cy.on('doubleTap', 'node', (evt) => {
      const nodeId = evt.target.id()
      const concept = conceptsRef.current.find(c => c.id === nodeId)
      const handler = onDoubleTapConceptRef.current
      if (concept && handler) handler(concept)
    })


    
    cyRef.current = cy
    setIsReady(true)
    initialLayoutDoneRef.current = true
    
    return () => {
      cy.destroy()
      cyRef.current = null
    }
  }, [concepts.length])

  // 增量更新节点和边（不重建整个图）
  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    // 首次加载时布局由初始化 effect 负责，跳过增量布局避免 race condition
    if (!initialLayoutDoneRef.current) return

    const currentIds = new Set(cy.nodes().map(n => n.id()))
    const targetIds = new Set(concepts.map(c => c.id))

    // 删除不再存在的节点
    cy.nodes().forEach(n => {
      if (!targetIds.has(n.id())) {
        cy.remove(n)
      }
    })

    // 添加新节点
    concepts.forEach(c => {
      if (!currentIds.has(c.id)) {
        cy.add({
          group: 'nodes',
          data: {
            id: c.id,
            label: c.title,
            level: c.level,
            category: c.category
          }
        })
      }
    })

    // 更新边
    const targetEdgeKeys = new Set(edges.map(e => `${e.source}-${e.target}-${e.type}`))
    cy.edges().forEach(e => {
      const key = `${e.data('source')}-${e.data('target')}-${e.data('edgeType')}`
      if (!targetEdgeKeys.has(key)) {
        cy.remove(e)
      }
    })
    edges.forEach(e => {
      const key = `${e.source}-${e.target}-${e.type}`
      const exists = cy.edges().some(ce => `${ce.data('source')}-${ce.data('target')}-${ce.data('edgeType')}` === key)
      if (!exists) {
        cy.add({
          group: 'edges',
          data: {
            id: e.id,
            source: e.source,
            target: e.target,
            edgeType: e.type
          }
        })
      }
    })

    // 增量布局（快速，不精化 — 精化在 idle 回调中已处理）
    try {
      const layout = cy.layout({
        name: 'fcose',
        quality: 'default',
        animate: true,
        animationDuration: 400,
        nodeRepulsion: 20000,
        idealEdgeLength: 160,
        gravity: 0.08,
        numIter: 200,
        tile: true,
        padding: 40
      } as cytoscape.LayoutOptions)
      layout.run()
    } catch (e) {
      console.warn('[KnowledgeGraph] incremental layout skipped:', e)
    }
  }, [structuralKey])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return

    // NEW: precision focus mode based on a list of node IDs
    if (focusedNodeIds && focusedNodeIds.length > 0) {
      const focusSet = new Set<string>(focusedNodeIds)
      const firstId = focusedNodeIds[0]

      cy.nodes().forEach(n => {
        const id = n.id()
        const isInFocus = focusSet.has(id)
        const isFirst = id === firstId
        n.style({
          'display': isInFocus ? 'element' : 'none',
          'width': isFirst ? 72 : 60,
          'height': isFirst ? 72 : 60,
          'border-width': isFirst ? 5 : 3,
          'border-color': isFirst ? '#f59e0b' : '#fff',
          'background-color': isFirst ? '#f59e0b' : (LEVEL_COLORS[Number(n.data('level'))] || '#3b82f6'),
          'opacity': 1,
        })
      })

      cy.edges().forEach(e => {
        const srcInFocus = focusSet.has(e.data('source'))
        const tgtInFocus = focusSet.has(e.data('target'))
        const show = srcInFocus && tgtInFocus
        e.style({
          'display': show ? 'element' : 'none',
          'opacity': show ? 1 : 0,
          'width': show ? 2 : 0.5,
        })
      })

      const focusNodes = cy.nodes().filter(n => focusSet.has(n.id()))
      if (focusNodes.length > 0) {
        cy.fit(focusNodes, 60)
      }
      wasFocusedRef.current = true
      return
    }

    const isFocusedNow = Boolean(selectedConcept && focusEnabled)

    if (!selectedConcept || !focusEnabled) {
      cy.nodes().forEach(n => {
        const isSelected = selectedConcept && n.id() === selectedConcept.id
        n.style({
          'display': 'element',
          'width': 50,
          'height': 50,
          'border-width': isSelected ? 4 : 2,
          'border-color': isSelected ? '#f59e0b' : '#fff',
          'background-color': isSelected ? '#f59e0b' : (LEVEL_COLORS[Number(n.data('level'))] || '#3b82f6'),
          'opacity': 1
        })
      })

      cy.edges().forEach(e => {
        const sourceSelected = selectedConcept && e.data('source') === selectedConcept.id
        const targetSelected = selectedConcept && e.data('target') === selectedConcept.id
        e.style({
          'display': 'element',
          'opacity': !selectedConcept || sourceSelected || targetSelected ? 1 : 0.15,
          'width': !selectedConcept || sourceSelected || targetSelected ? 3 : 1
        })
      })

      if (wasFocusedRef.current) {
        cy.fit(undefined, 50)
        setZoomLevel(cy.zoom())
      }
      wasFocusedRef.current = false
      return
    }

    const selectedNode = cy.getElementById(selectedConcept.id)
    const connectedEdges = selectedNode.connectedEdges()
    const neighborNodes = connectedEdges.connectedNodes()
    const relatedNodeIds = new Set([selectedConcept.id, ...neighborNodes.map(n => n.id())])

    cy.nodes().forEach(n => {
      const isSelected = n.id() === selectedConcept.id
      const isRelated = relatedNodeIds.has(n.id())
      n.style({
        'display': isRelated ? 'element' : 'none',
        'width': isSelected ? 72 : 60,
        'height': isSelected ? 72 : 60,
        'border-width': isSelected ? 5 : 3,
        'border-color': isSelected ? '#f59e0b' : '#fff',
        'background-color': isSelected ? '#f59e0b' : (LEVEL_COLORS[Number(n.data('level'))] || '#3b82f6'),
        'opacity': 1
      })
    })

    cy.edges().forEach(e => {
      const sourceVisible = relatedNodeIds.has(e.data('source'))
      const targetVisible = relatedNodeIds.has(e.data('target'))
      const isVisible = sourceVisible && targetVisible
      const touchesSelected = e.data('source') === selectedConcept.id || e.data('target') === selectedConcept.id
      e.style({
        'display': isVisible ? 'element' : 'none',
        'opacity': isVisible ? 1 : 0,
        'width': touchesSelected ? 4 : 2
      })
    })

    cy.fit(neighborNodes.union(selectedNode), 60)
    wasFocusedRef.current = isFocusedNow
  }, [selectedConcept, focusEnabled, focusedNodeIds, structuralKey])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    
    cy.nodes().forEach(n => {
      const isLinkSource = linkModeRef.current && linkSourceRef.current && n.id() === linkSourceRef.current
      n.style({
        'border-width': isLinkSource ? 5 : 2,
        'border-color': isLinkSource ? '#3b82f6' : '#fff',
        'border-style': isLinkSource ? 'dashed' : 'solid'
      })
    })
  }, [linkMode, linkSource])

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="w-full h-full" />
      
      {!isReady && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-50">
          <div className="text-gray-500">加载知识图中...</div>
        </div>
      )}

      {isReady && (
        <div className="absolute flex flex-col gap-1" style={{ top: '12px', right: '12px' }}>
          {selectedConcept && (
            <button
              onClick={onExitFocus}
              className="w-8 h-8 bg-white rounded shadow flex items-center justify-center hover:bg-gray-100 text-gray-700"
              title="退出聚焦"
            >
              <svg width={14} height={14} className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
          <button
            onClick={handleSmartLayout}
            className="w-8 h-8 bg-white rounded shadow flex items-center justify-center hover:bg-gray-100 text-gray-700"
            title="自适应布局"
          >
            <svg width={16} height={16} className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
            </svg>
          </button>
        </div>
      )}

      {isReady && (
        <button
          onClick={() => {
            if (linkMode && linkSource) {
              onLinkCancel?.()
            } else {
              onToggleLinkMode?.()
            }
          }}
          className={`absolute top-3 right-[70px] w-8 h-8 rounded shadow flex items-center justify-center transition-colors ${
            linkMode 
              ? 'bg-blue-500 text-white hover:bg-blue-600' 
              : 'bg-white text-gray-700 hover:bg-gray-100'
          }`}
          title={linkMode ? '退出连线' : '连线模式'}
        >
          <svg width={16} height={16} className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.979-1.101l1.101-1.102a4 4 0 005.657-5.656l-4-4z" />
          </svg>
        </button>
      )}
      
      <div className="absolute bg-white/90 backdrop-blur rounded-lg shadow p-2.5 text-xs space-y-1.5" style={{ bottom: '16px', left: '16px' }}>
        <div className="font-medium text-gray-700 mb-1">关系图例</div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-0.5 bg-red-500" />
          <span className="text-gray-600">依赖 (depends_on)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-0.5 bg-green-500" />
          <span className="text-gray-600">引出 (leads_to)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-0.5 bg-gray-500" style={{ borderTop: '1px dashed #6b7280' }} />
          <span className="text-gray-600">相关 (related)</span>
        </div>
        <div className="border-t border-gray-200 my-1.5" />
        <div className="font-medium text-gray-700 mb-1">级别颜色</div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: LEVEL_COLORS[1] }} />
          <span className="text-gray-600">Level 1 - 基础</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: LEVEL_COLORS[2] }} />
          <span className="text-gray-600">Level 2 - 中级</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: LEVEL_COLORS[3] }} />
          <span className="text-gray-600">Level 3 - 高级</span>
        </div>
      </div>
    </div>
  )
}
