import type { Concept } from '../src/types'

// In the real app, concept IDs are derived from file paths:
// file.path.replace('.md', '').replace(/\//g, '-')
function idFromPath(path: string): string {
  return path.replace('.md', '').replace(/\//g, '-')
}

function makeConcept(id: string, title: string, path: string): Concept {
  return { id, title, path, level: 1, category: '', problem: '', depends_on: [], leads_to: [], related: [], tags: [], lastModified: new Date() }
}

async function main() {
  let passed = 0
  let failed = 0

  function ok(condition: boolean, msg: string) {
    if (condition) { console.log(`  ok ${msg}`); passed++ }
    else { console.log(`  FAIL ${msg}`); failed++ }
  }

  const { deriveEdgesSync } = await import('../src/workers/deriveEdges.worker')

  console.log('\nTest 1: Cross-reference detection')
  {
    const paths = ['docs/attention.md', 'docs/transformer.md']
    const concepts = [
      makeConcept(idFromPath(paths[0]), 'Attention', paths[0]),
      makeConcept(idFromPath(paths[1]), 'Transformer', paths[1]),
    ]
    const files = [
      { path: paths[0], content: '# Attention\n\nTransformer is related to this concept.' },
      { path: paths[1], content: '# Transformer\nNothing about Attention here.' },
    ]
    const edges = deriveEdgesSync(concepts, files)
    ok(edges.length === 1, `Expected 1 edge, got ${edges.length}`)
    if (edges.length > 0) {
      ok(edges[0].source === idFromPath(paths[0]), `source=${edges[0].source}`)
      ok(edges[0].target === idFromPath(paths[1]), `target=${edges[0].target}`)
    }
  }

  console.log('\nTest 2: Self-reference should not create edge')
  {
    const path = 'docs/attention.md'
    const concepts = [makeConcept(idFromPath(path), 'Attention', path)]
    const files = [{ path, content: '# Attention\nAttention is the best.' }]
    const edges = deriveEdgesSync(concepts, files)
    ok(edges.length === 0, `Expected 0, got ${edges.length}`)
  }

  console.log('\nTest 3: No duplicate edges')
  {
    const paths = ['docs/attention.md', 'docs/transformer.md']
    const concepts = [
      makeConcept(idFromPath(paths[0]), 'Attention', paths[0]),
      makeConcept(idFromPath(paths[1]), 'Transformer', paths[1]),
    ]
    const files = [
      { path: paths[0], content: '# Attention\nMentions Transformer twice: Transformer and Transformer.' },
      { path: paths[1], content: '# Transformer' },
    ]
    const edges = deriveEdgesSync(concepts, files)
    ok(edges.length === 1, `Expected 1 edge, got ${edges.length}`)
  }

  console.log('\n========================')
  console.log(`Results: ${passed} passed, ${failed} failed`)
  console.log('========================')
  process.exit(failed > 0 ? 1 : 0)
}

main().catch(err => { console.error(err); process.exit(1) })
