import type { Concept, ConceptEdge } from '../types'

function deriveEdges(
  concepts: Concept[],
  files: { path: string; content: string }[],
): ConceptEdge[] {
  const seen = new Set<string>()
  const edges: ConceptEdge[] = []

  function addEdge(source: string, target: string) {
    if (source === target) return
    const pair = [source, target].sort().join('::')
    if (seen.has(pair)) return
    seen.add(pair)
    edges.push({ id: `e_${pair}`, source, target, type: 'related' as const })
  }

  const titleToId = new Map<string, string>()
  for (const c of concepts) {
    const plain = c.title.replace(/^\d+\s*/, '').toLowerCase()
    titleToId.set(c.title.toLowerCase(), c.id)
    if (plain !== c.title.toLowerCase()) titleToId.set(plain, c.id)
  }

  const contentById = new Map<string, string>(
    files.map(f => {
      const id = f.path.replace('.md', '').replace(/\//g, '-')
      return [id, f.content]
    }),
  )

  for (const c of concepts) {
    const content = contentById.get(c.id) || ''
    if (!content) continue
    for (const [title, otherId] of titleToId) {
      if (otherId === c.id) continue
      if (title.length >= 3 && content.toLowerCase().includes(title)) {
        addEdge(c.id, otherId)
      }
    }
  }

  return edges
}

if (typeof self !== 'undefined') {
  self.onmessage = (
    e: MessageEvent<{
      concepts: Concept[]
      files: { path: string; content: string }[]
    }>,
  ) => {
    const { concepts, files } = e.data
    const edges = deriveEdges(concepts, files)
    self.postMessage({ edges })
  }
}

export { deriveEdges as deriveEdgesSync }

export async function deriveEdgesInWorker(
  concepts: Concept[],
  files: { path: string; content: string }[],
): Promise<ConceptEdge[]> {
  try {
    const worker = new Worker(
      new URL('./deriveEdges.worker.ts', import.meta.url),
      { type: 'module' },
    )
    return new Promise((resolve, reject) => {
      worker.onmessage = (e: MessageEvent<{ edges: ConceptEdge[] }>) => {
        resolve(e.data.edges)
        worker.terminate()
      }
      worker.onerror = (err) => {
        reject(err)
        worker.terminate()
      }
      worker.postMessage({ concepts, files })
    })
  } catch {
    return deriveEdges(concepts, files)
  }
}
