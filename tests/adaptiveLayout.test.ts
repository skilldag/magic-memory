// tests/adaptiveLayout.test.ts

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
  const nodeRepulsion = Math.max(8000, Math.min(50000, idealEdgeLength * nodeCount * 3))
  const gravity = Math.max(0.02, Math.min(0.3, 100 / (nodeCount + 10)))
  const padding = Math.max(20, Math.min(80, Math.min(containerWidth, containerHeight) * 0.06))
  const numIter = Math.max(100, Math.min(800, nodeCount * 8))
  return { idealEdgeLength, nodeRepulsion, gravity, padding, numIter }
}

async function main() {
  let passed = 0
  let failed = 0

  function ok(condition: boolean, msg: string) {
    if (condition) { console.log(`  ok ${msg}`); passed++ }
    else { console.log(`  FAIL ${msg}`); failed++ }
  }

  // Test 1: normal case — 20 nodes in 1200x800 container
  const r1 = calcAdaptiveLayoutParams(1200, 800, 20)
  ok(r1.idealEdgeLength >= 60 && r1.idealEdgeLength <= 200, 'idealEdgeLength in range')
  ok(r1.nodeRepulsion >= 8000 && r1.nodeRepulsion <= 50000, 'nodeRepulsion in range')
  ok(r1.gravity >= 0.02 && r1.gravity <= 0.3, 'gravity in range')
  ok(r1.padding >= 20 && r1.padding <= 80, 'padding in range')
  ok(r1.numIter >= 100 && r1.numIter <= 800, 'numIter in range')

  // Test 2: small container — 5 nodes in 400x300
  const r2 = calcAdaptiveLayoutParams(400, 300, 5)
  ok(r2.idealEdgeLength >= 60, 'small container: edge length min clamped')
  ok(r2.padding >= 20, 'small container: padding min clamped')

  // Test 3: large graph — 100 nodes in 1920x1080
  const r3 = calcAdaptiveLayoutParams(1920, 1080, 100)
  ok(r3.idealEdgeLength <= 200, 'large graph: edge length max clamped')
  ok(r3.nodeRepulsion >= 8000, 'large graph: repulsion scales up')
  ok(r3.numIter <= 800, 'large graph: numIter max clamped')

  // Test 4: single node — 800x600, 1 node
  const r4 = calcAdaptiveLayoutParams(800, 600, 1)
  ok(r4.numIter >= 100, 'single node: min iter')
  ok(r4.gravity >= 0.02, 'single node: gravity min clamped')

  console.log(`\n${passed} passed, ${failed} failed`)
  if (failed > 0) process.exit(1)
}

main()
