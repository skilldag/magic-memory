import { useEffect, useRef, useState, useCallback } from 'react'
import cytoscape, { Core, ElementDefinition } from 'cytoscape'
import fcose from 'cytoscape-fcose'
import type { Concept, ConceptEdge } from '../types'
import { LEVEL_COLORS, EDGE_COLORS } from '../constants/graph'

cytoscape.use(fcose)

interface KnowledgeGraphProps {
  concepts: Concept[]
  edges: ConceptEdge[]
  selectedConcept: Concept | null
  focusEnabled?: boolean
  onSelectConcept: (concept: Concept) => void
  onNavigate?: (conceptId: string) => void
  onBackgroundDoubleTap?: () => void
  onHoverConcept?: (payload: { concept: Concept; x: number; y: number; width: number; height: number }) => void
  onHoverLeave?: () => void
}

export function KnowledgeGraph({
  concepts,
  edges,
  selectedConcept,
  focusEnabled = false,
  onSelectConcept,
  onNavigate,
  onBackgroundDoubleTap,
  onHoverConcept,
  onHoverLeave
}: KnowledgeGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const cyRef = useRef<Core | null>(null)
  const onHoverConceptRef = useRef(onHoverConcept)
  const onHoverLeaveRef = useRef(onHoverLeave)
  onHoverConceptRef.current = onHoverConcept
  onHoverLeaveRef.current = onHoverLeave
  const lastBackgroundTapAtRef = useRef(0)
  const wasFocusedRef = useRef(false)
  const conceptsRef = useRef(concepts)
  const edgesRef = useRef(edges)
  conceptsRef.current = concepts
  edgesRef.current = edges
  const [isReady, setIsReady] = useState(false)
  const [zoomLevel, setZoomLevel] = useState(1)

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

  // 初始化 Cytoscape（仅在首次数据到达时创建）
  useEffect(() => {
    if (cyRef.current) return
    if (!containerRef.current) return
    if (concepts.length === 0) return
    
    const elements: ElementDefinition[] = [
      ...concepts.map(c => ({
        group: 'nodes',
        data: {
          id: c.id,
          label: c.title,
          level: c.level,
          category: c.category
        }
      })),
      ...edges.map(e => ({
        group: 'edges',
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
      layout: {
        name: 'fcose',
        animate: true,
        animationDuration: 500,
        nodeRepulsion: 5000,
        idealEdgeLength: 80,
        gravity: 0.3,
        nestingFactor: 0.5,
        tile: true,
        padding: 50
      },
      minZoom: 0.2,
      maxZoom: 4,
      wheelSensitivity: 0.15
    })

    cy.on('zoom', () => {
      setZoomLevel(cy.zoom())
    })
    
    cy.on('tap', 'node', (evt) => {
      const nodeId = evt.target.id()
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
      const now = Date.now()
      const delta = now - lastBackgroundTapAtRef.current
      lastBackgroundTapAtRef.current = now
      if (delta > 0 && delta < 300) {
        onBackgroundDoubleTap?.()
      }
    })
    
    cy.on('doubleTap', 'node', (evt) => {
      const nodeId = evt.target.id()
      if (onNavigate) {
        onNavigate(nodeId)
      }
    })
    
    cyRef.current = cy
    setIsReady(true)
    
    return () => {
      cy.destroy()
    }
  }, [concepts.length])

  // 增量更新节点和边（不重建整个图）
  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return

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

    // 运行增量布局
    const layout = cy.layout({
      name: 'fcose',
      animate: true,
      animationDuration: 300,
      nodeRepulsion: 5000,
      idealEdgeLength: 80,
      gravity: 0.3,
      nestingFactor: 0.5,
      tile: true,
      padding: 50
    })
    layout.run()
  }, [concepts, edges])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return

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
    const neighborhood = selectedNode.closedNeighborhood()
    const relatedNodeIds = new Set(neighborhood.nodes().map(n => n.id()))

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

    cy.fit(neighborhood, 60)
    wasFocusedRef.current = isFocusedNow
  }, [selectedConcept, focusEnabled])

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="w-full h-full" />
      
      {!isReady && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-50">
          <div className="text-gray-500">加载知识图中...</div>
        </div>
      )}

      {isReady && (
        <div className="absolute top-3 right-3 flex flex-col gap-1">
          <button
            onClick={handleZoomIn}
            className="w-8 h-8 bg-white rounded shadow flex items-center justify-center hover:bg-gray-100 text-gray-700 text-lg"
            title="放大"
          >
            +
          </button>
          <button
            onClick={handleZoomOut}
            className="w-8 h-8 bg-white rounded shadow flex items-center justify-center hover:bg-gray-100 text-gray-700 text-lg"
            title="缩小"
          >
            −
          </button>
          <button
            onClick={handleFit}
            className="w-8 h-8 bg-white rounded shadow flex items-center justify-center hover:bg-gray-100 text-gray-700"
            title="适应视图"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
            </svg>
          </button>
          <div className="text-center text-xs text-gray-400 mt-1">{Math.round(zoomLevel * 100)}%</div>
        </div>
      )}
      
      <div className="absolute bottom-4 left-4 bg-white/90 backdrop-blur rounded-lg shadow p-2.5 text-xs space-y-1.5">
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