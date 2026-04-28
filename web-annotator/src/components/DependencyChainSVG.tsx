import type { Concept } from '../types'

interface DependencyChainSVGProps {
  concept: Concept
  concepts: Concept[]
}

export function DependencyChainSVG({ concept, concepts }: DependencyChainSVGProps) {
  const chain: typeof concepts = []
  let current = concepts.find(c => c.id === concept.id)
  while (current && current.depends_on.length > 0) {
    const parentId = current.depends_on[0]
    const parent = concepts.find(c => c.id === parentId)
    if (parent) {
      chain.unshift(parent)
      current = parent
    } else break
  }

  const allNodes = [...chain, concept]
  const nodeW = 80, gap = 30
  const totalW = allNodes.length * nodeW + (allNodes.length - 1) * gap
  const startX = Math.max(10, (400 - totalW) / 2)
  const y = 60

  return (
    <svg viewBox="0 0 400 120" className="w-full h-auto">
      <g>
        {allNodes.map((node, i) => {
          const x = startX + i * (nodeW + gap)
          const isCurrent = node.id === concept.id
          return (
            <g key={node.id}>
              {i > 0 && (
                <line
                  x1={startX + (i - 1) * (nodeW + gap) + nodeW}
                  y1={y}
                  x2={x}
                  y2={y}
                  stroke="#d1d5db"
                  strokeWidth={1.5}
                  markerEnd="url(#arrow)"
                />
              )}
              <rect
                x={x}
                y={y - 18}
                width={nodeW}
                height={36}
                rx={8}
                fill={isCurrent ? '#3b82f6' : '#f3f4f6'}
                stroke={isCurrent ? '#2563eb' : '#e5e7eb'}
                strokeWidth={1}
              />
              <text
                x={x + nodeW / 2}
                y={y + 4}
                textAnchor="middle"
                fontSize={10}
                fill={isCurrent ? 'white' : '#374151'}
              >
                {node.title.length > 8 ? `${node.title.slice(0, 7)}…` : node.title}
              </text>
            </g>
          )
        })}
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#d1d5db" />
          </marker>
        </defs>
      </g>
    </svg>
  )
}
