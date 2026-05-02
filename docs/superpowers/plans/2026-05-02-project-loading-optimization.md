# Project Loading Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate main-thread blocking during project switch/create by batching file reads, offloading edge derivation to a Web Worker, deferring heavy layout/analysis, and trimming Zustand persist payload.

**Architecture:** Five independent changes — each reversible and testable in isolation. The data flow shifts from "load everything → setState once → render" to "stream in batches → render progressively → compute in background."

**Tech Stack:** React 19 + Zustand + Cytoscape + Vite (native Web Worker support)

---

### Task 1: Batch file scanning (readMdFilesBatched)

**Files:**
- Modify: `src/utils/fileSystem.ts`
- Test: `src/utils/__tests__/fileSystem.test.ts`

Adding an async-generator variant of `readMdFiles` that yields batches of files instead of returning them all at once. The original `readMdFiles` is kept for backward compatibility (used elsewhere).

- [ ] **Step 1: Write failing test for `readMdFilesBatched`**

```typescript
// src/utils/__tests__/fileSystem.test.ts
import { describe, it, expect } from 'vitest'

// Mock FileSystem API for testing
function createMockDirHandle(files: Record<string, string>): FileSystemDirectoryHandle {
  const entries = Object.entries(files).map(([name, content]) => [
    name,
    {
      kind: 'file',
      name,
      getFile: async () => ({ text: async () => content }),
    },
  ])
  return {
    entries: async function* () { yield* entries },
  } as any
}

describe('readMdFilesBatched', () => {
  it('should yield files in batches of the specified size', async () => {
    const files: Record<string, string> = {}
    for (let i = 0; i < 25; i++) {
      files[`doc-${i}.md`] = `# Doc ${i}\ncontent`
    }
    const handle = createMockDirHandle(files)
    const { readMdFilesBatched } = await import('../fileSystem')
    const batches: string[][] = []
    for await (const batch of readMdFilesBatched(handle, '', 10)) {
      batches.push(batch.map(f => f.path))
    }
    expect(batches.length).toBe(3) // 10 + 10 + 5
    expect(batches[0].length).toBe(10)
    expect(batches[1].length).toBe(10)
    expect(batches[2].length).toBe(5)
  })

  it('should skip non-markdown files', async () => {
    const files: Record<string, string> = {
      'readme.md': '# Readme',
      'notes.txt': 'plain text',
      'index.md': '# Index',
    }
    const handle = createMockDirHandle(files)
    const { readMdFilesBatched } = await import('../fileSystem')
    let total = 0
    for await (const batch of readMdFilesBatched(handle, '', 10)) {
      total += batch.length
    }
    expect(total).toBe(2) // only .md files
  })

  it('should skip empty files', async () => {
    const files: Record<string, string> = {
      'empty.md': '',
      'content.md': '# Has Content',
    }
    const handle = createMockDirHandle(files)
    const { readMdFilesBatched } = await import('../fileSystem')
    let total = 0
    for await (const batch of readMdFilesBatched(handle, '', 10)) {
      total += batch.length
    }
    expect(total).toBe(1)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/utils/__tests__/fileSystem.test.ts --reporter=verbose`
Expected: FAIL — "readMdFilesBatched is not defined"

- [ ] **Step 3: Implement `readMdFilesBatched`**

```typescript
// Add to src/utils/fileSystem.ts

export async function* readMdFilesBatched(
  dirHandle: FileSystemDirectoryHandle,
  pathPrefix = '',
  batchSize = 10,
): AsyncGenerator<{ path: string; content: string }[], void, undefined> {
  let batch: { path: string; content: string }[] = []
  for await (const [name, entry] of (dirHandle as any).entries()) {
    const entryPath = pathPrefix ? `${pathPrefix}/${name}` : name
    if (entry.kind === 'directory' && !name.startsWith('.')) {
      for await (const files of readMdFilesBatched(entry, entryPath, batchSize)) {
        batch.push(...files)
        if (batch.length >= batchSize) { yield batch; batch = [] }
      }
    } else if (entry.kind === 'file' && name.endsWith('.md')) {
      try {
        const file = await (entry as FileSystemFileHandle).getFile()
        const content = await file.text()
        if (content.trim()) {
          batch.push({ path: entryPath, content })
          if (batch.length >= batchSize) { yield batch; batch = [] }
        }
      } catch { /* skip unreadable */ }
    }
  }
  if (batch.length > 0) yield batch
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/utils/__tests__/fileSystem.test.ts --reporter=verbose`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/utils/fileSystem.ts src/utils/__tests__/fileSystem.test.ts
git commit -m "feat: add readMdFilesBatched async generator for progressive file scanning"
```

---

### Task 2: Refactor projectStore to use batch streaming

**Files:**
- Modify: `src/store/projectStore.ts`
- Modify: `src/types/project.ts` (if Project interface needs changes)

No test needed for this refactor — the behavior change is internal (same output, streaming input). Verify manually.

- [ ] **Step 1: Check current projectStore switchProject/createProject data flow**

Read `src/store/projectStore.ts` lines 215-273 (`switchProject`) and 118-192 (`createProject`). Note the pattern:
1. Load handle from IndexedDB
2. Call `readMdFiles(handle)` (returns all files at once)
3. Map files to concept objects
4. Call `deriveEdges()`
5. `useKnowledgeGraphStore.setState({ concepts, edges, ... })`

- [ ] **Step 2: Modify `switchProject` to use batch streaming**

Replace the single `readMdFiles()` call with the batch generator:

```typescript
// In src/store/projectStore.ts, modify switchProject:

switchProject: async (projectId: string) => {
  set({ isLoading: true, concepts: [], edges: [] })
  try {
    const { projects, currentProjectId } = get()
    const project = projects.find(p => p.id === projectId)
    if (!project) throw new Error('项目不存在')

    // Save current project snapshot
    if (currentProjectId && currentProjectId !== projectId) {
      const kg = (await import('./knowledgeGraphStore')).useKnowledgeGraphStore.getState()
      const snapshots = loadSnapshots()
      snapshots[currentProjectId] = {
        edges: kg.edges,
        reviewRecords: Array.from(kg.reviewRecords.entries()),
        annotations: kg.annotations,
        chains: kg.chains,
      }
      saveSnapshots(snapshots)
    }

    if (project.handleStoreId) {
      const handle = await loadHandle(project.handleStoreId)
      if (!handle) throw new Error('项目文件夹句柄已丢失，请重新选择')
      const ok = await ensurePermission(handle)
      if (!ok) throw new Error('请授权文件夹读取权限')

      // Batch streaming: read files progressively
      const allConcepts: Concept[] = []
      const allFiles: { path: string; content: string }[] = []
      const { readMdFilesBatched } = await import('../utils/fileSystem')
      const { deriveEdges } = await import('../utils/deriveEdges')
      const kgStore = (await import('./knowledgeGraphStore')).useKnowledgeGraphStore

      for await (const batch of readMdFilesBatched(handle)) {
        allFiles.push(...batch)
        const newConcepts = batch.map(file => ({
          id: file.path.replace('.md', '').replace(/\//g, '-'),
          title: file.path.replace('.md', '').split('/').pop() || file.path.replace('.md', ''),
          path: file.path,
          level: 1, category: '', problem: '',
          depends_on: [], leads_to: [], related: [], tags: [],
          lastModified: new Date(),
        }))
        allConcepts.push(...newConcepts)
        // Push progressively to store
        kgStore.setState({ concepts: [...allConcepts], isLoading: true })
      }

      // All files loaded — restore edges from snapshot or derive
      const snapshot = loadSnapshots()[projectId]
      const derivedEdges = snapshot?.edges?.length
        ? snapshot.edges
        : deriveEdges(allConcepts, allFiles)

      kgStore.setState({
        concepts: allConcepts,
        edges: derivedEdges,
        reviewRecords: new Map(snapshot?.reviewRecords || []),
        annotations: snapshot?.annotations || [],
        chains: snapshot?.chains || [],
        isLoading: false,
      })
    }

    set({
      currentProjectId: projectId,
      projects: get().projects.map(p =>
        p.id === projectId ? { ...p, lastOpenedAt: new Date().toISOString() } : p
      ),
      isLoading: false,
    })
  } catch (error) {
    set({ error: error instanceof Error ? error.message : '切换项目失败', isLoading: false })
  }
}
```

- [ ] **Step 3: Modify `createProject` similarly for batch streaming**

```typescript
// In src/store/projectStore.ts, modify createProject:

createProject: async (name: string, handle: FileSystemDirectoryHandle) => {
  set({ isLoading: true, error: null, isScanning: true })
  try {
    const handleStoreId = generateId()
    const { projects } = get()
    const existing = projects.find(p => p.name === name)
    if (existing) {
      set({ currentProjectId: existing.id, isLoading: false, isScanning: false })
      return existing
    }

    await saveHandle(handleStoreId, handle)

    const project: Project = {
      id: generateId(),
      name,
      handleStoreId,
      createdAt: new Date().toISOString(),
      lastOpenedAt: new Date().toISOString(),
    }

    const ok = await ensurePermission(handle)
    if (!ok) {
      set({ error: '没有文件夹读取权限', isLoading: false, isScanning: false })
      return null
    }

    // Batch streaming
    const allConcepts: Concept[] = []
    const allFiles: { path: string; content: string }[] = []
    const { readMdFilesBatched } = await import('../utils/fileSystem')
    const { deriveEdges } = await import('../utils/deriveEdges')
    const kgStore = (await import('./knowledgeGraphStore')).useKnowledgeGraphStore

    for await (const batch of readMdFilesBatched(handle)) {
      allFiles.push(...batch)
      const newConcepts = batch.map(file => ({
        id: file.path.replace('.md', '').replace(/\//g, '-'),
        title: file.path.replace('.md', '').split('/').pop() || file.path.replace('.md', ''),
        path: file.path,
        level: 1, category: '', problem: '',
        depends_on: [], leads_to: [], related: [], tags: [],
        lastModified: new Date(),
      }))
      allConcepts.push(...newConcepts)
      kgStore.setState({ concepts: [...allConcepts], isLoading: true })
    }

    const snapshot = loadSnapshots()[project.id]
    const derivedEdges = snapshot?.edges?.length ? snapshot.edges : deriveEdges(allConcepts, allFiles)

    kgStore.setState({
      concepts: allConcepts,
      edges: derivedEdges,
      reviewRecords: new Map(snapshot?.reviewRecords || []),
      annotations: snapshot?.annotations || [],
      chains: snapshot?.chains || [],
      isLoading: false,
    })

    const updated = [...projects, project]
    set({ projects: updated, currentProjectId: project.id, isLoading: false, isScanning: false })
    localStorage.setItem('magic-memory-projects', JSON.stringify(updated))

    try {
      await fetch('/api/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, handleStoreId }),
      })
    } catch {}

    return project
  } catch (error) {
    set({ error: error instanceof Error ? error.message : '创建项目失败', isLoading: false, isScanning: false })
    return null
  }
}
```

- [ ] **Step 4: Verify manually — check no regressions**

Run: `npx tsc --noEmit`
Expected: No type errors (pre-existing errors in server files are OK)
Then: Test in browser — create a project, switch projects, verify concepts appear progressively and no console errors.

- [ ] **Step 5: Commit**

```bash
git add src/store/projectStore.ts
git commit -m "refactor: stream project file scanning in progressive batches"
```

---

### Task 3: Web Worker for deriveEdges

**Files:**
- Create: `src/workers/deriveEdges.worker.ts`
- Modify: `src/store/projectStore.ts`
- Test: `src/workers/__tests__/deriveEdges.worker.test.ts`

- [ ] **Step 1: Write failing test for the worker**

```typescript
// src/workers/__tests__/deriveEdges.worker.test.ts
import { describe, it, expect } from 'vitest'

describe('deriveEdges Worker', () => {
  it('should compute edges via worker and return them', async () => {
    const concepts = [
      { id: 'doc-A', title: 'A', path: 'A.md', level: 1, category: '', problem: '', depends_on: [], leads_to: [], related: [], tags: [] },
      { id: 'doc-B', title: 'B', path: 'B.md', level: 1, category: '', problem: '', depends_on: [], leads_to: [], related: [], tags: [] },
    ]
    const files = [
      { path: 'A.md', content: '# A\n\nB is related to this concept.' },
      { path: 'B.md', content: '# B\n\nNothing about A here.' },
    ]

    const { deriveEdgesInWorker } = await import('../deriveEdges.worker')
    const edges = await deriveEdgesInWorker(concepts, files)
    expect(edges.length).toBeGreaterThan(0)
    expect(edges[0].source).toBe('doc-A')
    expect(edges[0].target).toBe('doc-B')
    expect(edges[0].type).toBe('related')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/workers/__tests__/deriveEdges.worker.test.ts --reporter=verbose`
Expected: FAIL — "deriveEdgesInWorker is not defined"

- [ ] **Step 3: Create the Web Worker file**

```typescript
// src/workers/deriveEdges.worker.ts
import type { Concept, ConceptEdge } from '../types'

// Pure computation function (moved from utils/deriveEdges.ts)
function deriveEdges(concepts: Concept[], files: { path: string; content: string }[]): ConceptEdge[] {
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

  const contentById = new Map<string, string>(files.map(f => {
    const id = f.path.replace('.md', '').replace(/\//g, '-')
    return [id, f.content]
  }))

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

// Worker message handler
self.onmessage = (e: MessageEvent<{ concepts: Concept[]; files: { path: string; content: string }[] }>) => {
  const { concepts, files } = e.data
  const edges = deriveEdges(concepts, files)
  self.postMessage({ edges })
}

// Exported so tests/direct callers can use without Web Worker
export { deriveEdges as deriveEdgesSync }
export async function deriveEdgesInWorker(
  concepts: Concept[],
  files: { path: string; content: string }[],
): Promise<ConceptEdge[]> {
  try {
    const worker = new Worker(new URL('./deriveEdges.worker.ts', import.meta.url), { type: 'module' })
    return new Promise((resolve, reject) => {
      worker.onmessage = (e: MessageEvent<{ edges: ConceptEdge[] }>) => {
        resolve(e.data.edges)
        worker.terminate()
      }
      worker.onerror = (err) => { reject(err); worker.terminate() }
      worker.postMessage({ concepts, files })
    })
  } catch {
    // Fallback for file:// and unsupported environments
    return deriveEdges(concepts, files)
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/workers/__tests__/deriveEdges.worker.test.ts --reporter=verbose`
Expected: PASS

- [ ] **Step 5: Wire worker into projectStore (replace sync deriveEdges)**

In `src/store/projectStore.ts`:

```typescript
// Replace the direct deriveEdges import/call with the worker version

// At top:
// Remove: const { deriveEdges } = await import('../utils/deriveEdges')
// Add:
const { deriveEdgesInWorker } = await import('../workers/deriveEdges.worker')

// For both switchProject and createProject, replace:
//   const derivedEdges = deriveEdges(allConcepts, allFiles)
// With:
const derivedEdges = await deriveEdgesInWorker(allConcepts, allFiles)
```

- [ ] **Step 6: Remove the now-unused duplicate utility if no other callers**

Check callers:
```bash
grep -r "deriveEdges" src/ --include="*.ts" --include="*.tsx" -l
```

If the only caller was `projectStore.ts` (which now uses the worker), keep `utils/deriveEdges.ts` for backward compatibility but note it as deprecated.

- [ ] **Step 7: Verify**

Run: `npx tsc --noEmit`
Expected: No type errors related to the changes

- [ ] **Step 8: Commit**

```bash
git add src/workers/deriveEdges.worker.ts src/workers/__tests__/deriveEdges.worker.test.ts src/store/projectStore.ts
git commit -m "feat: offload deriveEdges to Web Worker, remove main-thread blocking"
```

---

### Task 4: Zustand persist slim-down

**Files:**
- Modify: `src/store/knowledgeGraphStore.ts`

- [ ] **Step 1: Review current partialize**

Read lines 273-278 in `src/store/knowledgeGraphStore.ts` — currently excludes nothing (persists entire state).

- [ ] **Step 2: Modify partialize to exclude heavy fields**

```typescript
// In src/store/knowledgeGraphStore.ts, update the persist config:

partialize: (state) => ({
  concepts: state.concepts.map(c => ({
    id: c.id, title: c.title,
    level: c.level, category: c.category,
    problem: c.problem, gap_anticipate: c.gap_anticipate,
    depends_on: c.depends_on, leads_to: c.leads_to, related: c.related,
    path: c.path, tags: c.tags,
    lastModified: c.lastModified,
    metadata: c.metadata,
    alias: c.alias,
    // Excluded: content, elements, hierarchy, process, process
  })),
  edges: state.edges,
  reviewRecords: Array.from(state.reviewRecords.entries()),
  annotations: state.annotations,
}),
```

- [ ] **Step 3: Verify ConceptDetailPanel content fallback**

Read `src/components/ConceptDetailPanel.tsx` lines 45-58 to confirm it already handles `concept.content` being undefined by calling `loadDocContent()`. Should look like:

```typescript
useEffect(() => {
    if (action !== 'read') return
    if (concept.content) {           // ← will be undefined after persist slim-down
      setDocContent(concept.content)
      setDocContent(concept.content)
      setDocLoading(false)
      return
    }
    setDocContent(null)
    setDocLoading(true)
    loadDocContent(concept.path).then(content => {  // ← this fallback is already in place
      if (content) setDocContent(content)
      setDocLoading(false)
    })
  }, [action, concept.id, concept.path, concept.content])
```

No changes needed in ConceptDetailPanel — the fallback path already exists.

- [ ] **Step 4: Use `concept.content` only as in-memory runtime field**

`updateConceptContent` in the store sets `concept.content` on the runtime state. Since it's excluded from persist, it won't be serialized — but it will be available in memory during the session. This is correct behavior (content reloads from file on next session).

- [ ] **Step 5: Verify**

Run: `npx tsc --noEmit`
Expected: No type errors

- [ ] **Step 6: Commit**

```bash
git add src/store/knowledgeGraphStore.ts
git commit -m "perf: slim Zustand persist payload by excluding content/elements from serialization"
```

---

### Task 5: Two-stage Cytoscape layout

**Files:**
- Modify: `src/components/KnowledgeGraph.tsx`

- [ ] **Step 1: Read the current initialization effect**

Read `src/components/KnowledgeGraph.tsx` lines 100-325. The fcose layout on line 214-233 uses `quality: 'proof'` and `numIter: 2000`.

- [ ] **Step 2: Replace with two-stage layout**

Replace lines 213-236:

```typescript
    // 阶段 1: 快速粗略布局（立即执行）
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
        padding: 80,
      } as cytoscape.LayoutOptions)
      fastLayout.one('layoutstop', () => {
        cy.fit(undefined, 50)
        setZoomLevel(cy.zoom())
      })
      fastLayout.run()
    } catch (e) {
      console.warn('[KnowledgeGraph] fast layout failed:', e)
    }

    // 阶段 2: 空闲时精化布局
    if (typeof requestIdleCallback !== 'undefined') {
      requestIdleCallback(() => {
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
            padding: 80,
          } as cytoscape.LayoutOptions)
          refineLayout.run()
        } catch (e) {
          console.warn('[KnowledgeGraph] refine layout failed:', e)
        }
      }, { timeout: 3000 })
    }
```

- [ ] **Step 3: Also optimize the incremental layout (structuralKey effect)**

Replace lines 384-402 — the incremental layout on structuralKey changes:

```typescript
    // 增量布局（快速，不精化）
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
        padding: 40,
      } as cytoscape.LayoutOptions)
      layout.run()
    } catch (e) {
      console.warn('[KnowledgeGraph] incremental layout skipped:', e)
    }
```

- [ ] **Step 4: Verify**

Run: `npx tsc --noEmit`
Expected: No type errors
Then: Test in browser — observe graph appears almost instantly with a rough layout, then smoothly animates to a refined layout.

- [ ] **Step 5: Commit**

```bash
git add src/components/KnowledgeGraph.tsx
git commit -m "perf: two-stage Cytoscape layout — fast first render, idle-time refinement"
```

---

### Task 6: analyzeGraph cache in SummaryPanel

**Files:**
- Modify: `src/components/SummaryPanel.tsx`

- [ ] **Step 1: Convert current useMemo to cached variant**

Replace the current `useMemo` call:

```typescript
// Before (line 31):
// const analysis = useMemo(() => analyzeGraph(concepts, edges), [concepts, edges])

// After:
const analysisCacheRef = useRef(new Map<string, GraphAnalysis>())

const structuralKey = useMemo(
  () => concepts.map(c => c.id).sort().join(',') + '|' +
         edges.map(e => `${e.source}-${e.target}-${e.type}`).sort().join(','),
  [concepts, edges]
)

const analysis = useMemo(() => {
  const key = structuralKey
  const cached = analysisCacheRef.current.get(key)
  if (cached) return cached

  const result = analyzeGraph(concepts, edges)

  // LRU: keep max 5 entries
  if (analysisCacheRef.current.size >= 5) {
    const firstKey = analysisCacheRef.current.keys().next().value
    analysisCacheRef.current.delete(firstKey)
  }
  analysisCacheRef.current.set(key, result)
  return result
}, [structuralKey, concepts, edges])
```

- [ ] **Step 2: Verify imports**

Ensure `useRef` is already imported at the top of the file (likely already is). Ensure `GraphAnalysis` type is imported from `../utils/graphAnalysis`.

- [ ] **Step 3: Verify**

Run: `npx tsc --noEmit`
Expected: No type errors

- [ ] **Step 4: Commit**

```bash
git add src/components/SummaryPanel.tsx
git commit -m "perf: cache analyzeGraph results with LRU to avoid redundant DFS traversals"
```

---

### Task 7: Clean up — remove duplicate graphAnalysis.ts

**Files:**
- Read: `src/utils/graphAnalysis.ts` and `src/utils/graphSummary.ts` (they are duplicates)
- Modify: Callers of `graphSummary.ts` → redirect to `graphAnalysis.ts`
- Deprecate: `src/utils/graphSummary.ts`

- [ ] **Step 1: Find all callers of graphSummary**

```bash
grep -r "graphSummary" src/ --include="*.ts" --include="*.tsx"
```

- [ ] **Step 2: Redirect callers to graphAnalysis**

For each caller, replace imports:
```typescript
// Before:
import { analyzeGraph, formatSummaryToString } from '../utils/graphSummary'
// After:
import { analyzeGraph } from '../utils/graphAnalysis'
// formatSummaryToString → formatAnalysisToString (same function, different name in graphAnalysis)
import { formatAnalysisToString } from '../utils/graphAnalysis'
```

- [ ] **Step 3: Mark graphSummary.ts as deprecated**

Add a comment at the top:
```typescript
// DEPRECATED: This file is a duplicate of graphAnalysis.ts.
// All callers should import from graphAnalysis.ts instead.
// Will be removed in a future cleanup.
```

- [ ] **Step 4: Verify**

Run: `npx tsc --noEmit`
Expected: No type errors

- [ ] **Step 5: Commit**

```bash
git add src/
git commit -m "chore: deprecate duplicate graphSummary.ts, redirect callers to graphAnalysis.ts"
```

---

## Self-Review

**1. Spec coverage:**
- Section 4.1 (batch file scanning) → Task 1 + Task 2 ✓
- Section 4.2 (Web Worker for deriveEdges) → Task 3 ✓
- Section 4.3 (two-stage layout) → Task 5 ✓
- Section 4.4 (persist slim-down) → Task 4 ✓
- Section 4.5 (analyzeGraph cache) → Task 6 ✓
- Section: merge duplicate graphAnalysis/Summary → Task 7 ✓

**2. Placeholder scan:** No TBD, TODO, "implement later", or vague placeholders found. All steps contain complete code.

**3. Type consistency:**
- `deriveEdgesInWorker(concepts, files)` signature matches across Task 3's worker export and Task 3's projectStore wiring ✓
- `readMdFilesBatched` yield type `{ path: string; content: string }[]` consistent across Task 1 and Task 2 ✓
- `structuralKey` computation consistent between Task 5 (KnowledgeGraph uses the same pattern) and Task 6 ✓
- `partialize` excluded fields match `Concept` type definition in `types/index.ts` ✓
