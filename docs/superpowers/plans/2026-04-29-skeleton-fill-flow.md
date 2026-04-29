# 骨架填充流程 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace blank ProcessCanvas with a guided skeleton canvas on first entry — show gap nodes with guiding questions, let users fill them by dragging candidate concepts, and support question-driven knowledge growth.

**Architecture:** Extend existing ProcessCanvas with a `skeletonMode` that pre-populates gap nodes from ProcessChain steps. Add question mechanism as a lightweight dialog that persists questions to the store. Breadcrumb + tree progress overlay in KnowledgeGraphView header.

**Tech Stack:** React 19, TypeScript, @xyflow/react (ReactFlow), Zustand, Tailwind CSS 4

---

### File Inventory

| File | Action | Responsibility |
|------|--------|---------------|
| `src/types/index.ts` | Modify | Add `BaseQuestion`, `ConceptQuestion`, `hierarchy` to `Concept` |
| `src/data/mockGraphData.ts` | Modify | Add `baseQuestions` to mock concepts |
| `src/utils/processComparison.ts` | Modify | Add `generateSkeletonNodes()` |
| `src/store/knowledgeGraphStore.ts` | Modify | Add `questions`, `canvasHistory`, `setConceptPanelMode` |
| `src/components/ProcessCanvas.tsx` | Modify | Add `skeletonMode` with gap nodes + candidate area + question button |
| `src/components/KnowledgeGraphView.tsx` | Modify | Add breadcrumb bar, pass skeleton state to ProcessCanvas |
| `src/components/QuestionDialog.tsx` | Create | Question input and management dialog |

---

### Task 1: Extend Types

**Files:**
- Modify: `src/types/index.ts`

Add `BaseQuestion`, `ConceptQuestion`, `ConceptQuestionContext`, extend `Concept` with `hierarchy` and `baseQuestions`.

- [ ] **Step 1: Add new interfaces and extend Concept**

Insert these after the `ConceptElement` interface block (around line 124):

```typescript
// ========== 骨架填充类型 ==========

export interface BaseQuestion {
  id: string
  conceptId: string
  question: string
  targetConceptId?: string
  hint?: string
  order: number
}

export interface ConceptQuestion {
  id: string
  conceptId: string
  question: string
  context: {
    location: 'skeleton' | 'canvas' | 'comparison'
    stepId?: string
  }
  status: 'open' | 'converted_to_concept' | 'converted_to_step' | 'resolved'
  convertedTo?: {
    type: 'concept' | 'step'
    targetId: string
  }
  createdAt: Date
}

export interface CanvasHistoryItem {
  conceptId: string
  view: 'skeleton' | 'canvas'
}
```

Then extend the `Concept` interface — add these fields at the end of the existing Concept interface (before the closing `}`):

```typescript
  hierarchy?: {
    parentId: string | null
    level: number
    order: number
  }
  baseQuestions?: BaseQuestion[]
```

- [ ] **Step 2: Verify types compile**

Run: `npx tsc --noEmit`
Expected: No type errors.

- [ ] **Step 3: Commit**

```bash
git add src/types/index.ts
git commit -m "feat: add BaseQuestion, ConceptQuestion, hierarchy types for skeleton fill flow"
```

---

### Task 2: Add Mock BaseQuestions

**Files:**
- Modify: `src/data/mockGraphData.ts`

Add `baseQuestions` to mock concepts so the skeleton has data to display.

- [ ] **Step 1: Import BaseQuestion**

At the top of `mockGraphData.ts`, add `BaseQuestion` to the import:

```typescript
import type { Concept, ConceptEdge, ProcessChain, ConceptElement, BaseQuestion } from '../types'
```

- [ ] **Step 2: Add mock base questions**

Add this after `mockElements` (around line 91):

```typescript
export const mockBaseQuestions: Record<string, BaseQuestion[]> = {
  '0': [
    { id: 'bq_0_1', conceptId: '0', question: '要让模型跑推理，首先需要知道什么？', targetConceptId: '0', hint: '想想推理一个请求需要哪些前提信息', order: 1 },
    { id: 'bq_0_2', conceptId: '0', question: '配置分为哪几个子系统？', hint: '从配置的结构去想', order: 2 },
    { id: 'bq_0_3', conceptId: '0', question: '配置错误会导致什么后果？', hint: '想想初始化的连锁反应', order: 3 },
  ],
  '1': [
    { id: 'bq_1_1', conceptId: '1', question: 'vLLM 必须在什么硬件上执行推理？', targetConceptId: '1', order: 1 },
    { id: 'bq_1_2', conceptId: '1', question: 'Device 抽象解决了什么问题？', hint: '想想多硬件支持', order: 2 },
  ],
  '2': [
    { id: 'bq_2_1', conceptId: '2', question: '模型权重在 GPU 上以什么形式存在？', targetConceptId: '2', order: 1 },
    { id: 'bq_2_2', conceptId: '2', question: 'Tensor 和 PyTorch Tensor 有什么关系？', hint: 'vLLM 有自己的封装', order: 2 },
  ],
}
```

- [ ] **Step 3: Update mockConcepts to include baseQuestions**

Find the concept with `id: '0'` in mockConcepts and add `baseQuestions` field:

```typescript
{
  id: '0',
  title: 'VllmConfig',
  // ...existing fields...
  baseQuestions: mockBaseQuestions['0'],
}
```

Do the same for id `'1'` and id `'2'`.

- [ ] **Step 4: Update getMockGraphData to include baseQuestions**

Find the `getMockGraphData` function and make sure `baseQuestions` is included in the returned concepts (it should be since it's on the concept objects).

- [ ] **Step 5: Verify types compile**

Run: `npx tsc --noEmit`
Expected: No type errors.

- [ ] **Step 6: Commit**

```bash
git add src/data/mockGraphData.ts
git commit -m "feat: add mock baseQuestions for skeleton fill flow"
```

---

### Task 3: Add skeleton node generation utility

**Files:**
- Modify: `src/utils/processComparison.ts`

Add `generateSkeletonNodes()` that creates gap nodes with guiding questions from the process chain.

- [ ] **Step 1: Add SkeletonNode type and generator**

Add these after the `getGapConceptIds` function (end of file):

```typescript
export interface SkeletonNodeDef {
  id: string
  type: 'gap' | 'known' | 'current'
  label: string
  question: string      // 引导问题
  correctConceptId?: string  // 这个空缺期望填哪个概念
  hint?: string
}

/**
 * Generate skeleton nodes for first-entry canvas.
 * Returns an ordered list of skeleton node definitions.
 */
export function generateSkeletonNodes(
  concept: Concept,
  chain: ProcessChain | null,
  allConcepts: Concept[]
): SkeletonNodeDef[] {
  const nodes: SkeletonNodeDef[] = []

  if (chain) {
    chain.steps.forEach((step, idx) => {
      if (step.leads_to_id === concept.id) {
        // This step IS the current concept — add as 'current'
        nodes.push({
          id: `current_${concept.id}`,
          type: 'current',
          label: concept.title,
          question: step.question || '当前概念',
          correctConceptId: concept.id,
        })
      } else {
        // This step is a process step — add as 'gap'
        const targetConcept = step.leads_to_id
          ? allConcepts.find(c => c.id === step.leads_to_id)
          : null
        nodes.push({
          id: `gap_${step.id}`,
          type: 'gap',
          label: step.label,
          question: step.question || '这里应该是什么概念？',
          correctConceptId: step.leads_to_id,
          hint: step.hint,
        })
      }
    })
  } else {
    // Fallback: generic chain
    const generic = generateGenericChain(concept.id, allConcepts)
    generic.steps.forEach((step) => {
      if (step.leads_to_id === concept.id) {
        nodes.push({
          id: `current_${concept.id}`,
          type: 'current',
          label: concept.title,
          question: step.question || '当前概念',
          correctConceptId: concept.id,
        })
      } else {
        nodes.push({
          id: `gap_${step.id}`,
          type: 'gap',
          label: step.label,
          question: step.question || '这里应该是什么？',
          correctConceptId: step.leads_to_id,
        })
      }
    })
  }

  return nodes
}
```

- [ ] **Step 2: Verify types compile**

Run: `npx tsc --noEmit`
Expected: No type errors.

- [ ] **Step 3: Commit**

```bash
git add src/utils/processComparison.ts
git commit -m "feat: add generateSkeletonNodes for skeleton fill canvas"
```

---

### Task 4: Extend Store with question state

**Files:**
- Modify: `src/store/knowledgeGraphStore.ts`

Add questions array, canvasHistory, conceptPanelMode tracking.

- [ ] **Step 1: Extend the store interface**

Add to the `KnowledgeGraphStore` interface (after existing fields around line 15):

```typescript
  // 骨架填充模式
  conceptPanelMode: boolean       // true = first-entry skeleton mode
  questions: ConceptQuestion[]
  canvasHistory: CanvasHistoryItem[]
  skeletonCompleted: Set<string>  // concept IDs that completed skeleton

  setConceptPanelMode: (mode: boolean) => void
  addQuestion: (q: Omit<ConceptQuestion, 'id' | 'createdAt'>) => void
  markSkeletonCompleted: (conceptId: string) => void
  pushHistory: (item: CanvasHistoryItem) => void
  popHistory: () => CanvasHistoryItem | undefined
```

- [ ] **Step 2: Add implementation**

In the store creator, add initial state:

```typescript
conceptPanelMode: true,
questions: [],
canvasHistory: [],
skeletonCompleted: new Set(),
```

Add methods:

```typescript
setConceptPanelMode: (mode) => {
  set({ conceptPanelMode: mode })
},

addQuestion: (q) => {
  const question: ConceptQuestion = {
    ...q,
    id: `q_${Date.now()}`,
    createdAt: new Date(),
  }
  set(state => ({ questions: [...state.questions, question] }))
},

markSkeletonCompleted: (conceptId) => {
  set(state => {
    const next = new Set(state.skeletonCompleted)
    next.add(conceptId)
    return { skeletonCompleted: next }
  })
},

pushHistory: (item) => {
  set(state => ({
    canvasHistory: [...state.canvasHistory, item]
  }))
},

popHistory: () => {
  const { canvasHistory } = get()
  if (canvasHistory.length === 0) return undefined
  const popped = canvasHistory[canvasHistory.length - 1]
  set({ canvasHistory: canvasHistory.slice(0, -1) })
  return popped
},
```

- [ ] **Step 3: Add imports**

Make sure `ConceptQuestion` and `CanvasHistoryItem` are imported:

```typescript
import type { Concept, ConceptEdge, ReviewRecord, UserAnnotation, ProcessChain, ProcessState, ConceptQuestion, CanvasHistoryItem } from '../types'
```

- [ ] **Step 4: Verify types compile**

Run: `npx tsc --noEmit`
Expected: No type errors.

- [ ] **Step 5: Commit**

```bash
git add src/store/knowledgeGraphStore.ts
git commit -m "feat: add questions, canvasHistory, skeleton mode to store"
```

---

### Task 5: Implement skeleton mode in ProcessCanvas

**Files:**
- Modify: `src/components/ProcessCanvas.tsx`

Add `skeletonMode` prop. When true, render gap nodes pre-populated from `generateSkeletonNodes()` with a question overlay on each gap, a candidate concept bar at the bottom, and a submit button that validates against `correctConceptId`.

- [ ] **Step 1: Add new props**

Modify `ProcessCanvasProps`:

```typescript
interface ProcessCanvasProps {
  concept: Concept
  chain: ProcessChain | null
  allConcepts: Concept[]
  onComplete: (userFlow: string[]) => void
  onNavigate: (conceptId: string) => void
  /** 首次进入时启用骨架填充模式 */
  skeletonMode?: boolean
  /** 骨架模式的引导问题列表 */
  skeletonNodes?: SkeletonNodeDef[]
  /** 提交骨架填充结果 */
  onSkeletonSubmit?: (results: { gapId: string; filledConceptId: string | null }[]) => void
  /** 打开提问 */
  onOpenQuestion?: () => void
}
```

Add imports:

```typescript
import type { SkeletonNodeDef } from '../utils/processComparison'
```

- [ ] **Step 2: Add skeleton state**

Inside the `ProcessCanvas` function, add state:

```typescript
// 骨架填充模式的状态
const [skeletonFills, setSkeletonFills] = useState<Map<string, string | null>>(new Map())
const [skeletonResults, setSkeletonResults] = useState<{
  gapId: string; correct: boolean; filledLabel: string | null
}[] | null>(null)
```

- [ ] **Step 3: Render skeleton mode overlay**

Before the main return, add skeleton mode handler:

```typescript
if (skeletonMode && skeletonNodes) {
  const gapNodes = skeletonNodes.filter(n => n.type === 'gap')
  const knownNodes = skeletonNodes.filter(n => n.type === 'known')
  const filledCount = gapNodes.filter(g => skeletonFills.get(g.id) != null).length

  return (
    <div className="flex flex-col h-full">
      {/* 骨架画板 */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="max-w-3xl mx-auto space-y-4">
          {/* 进度条 */}
          <div className="flex items-center gap-2 text-xs text-gray-500 mb-4">
            <span>填充进度</span>
            <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 rounded-full transition-all"
                style={{ width: `${gapNodes.length > 0 ? (filledCount / gapNodes.length) * 100 : 0}%` }}
              />
            </div>
            <span>{filledCount}/{gapNodes.length}</span>
          </div>

          {/* 步骤卡片 */}
          <div className="space-y-3">
            {skeletonNodes.map((node, idx) => {
              const filledId = skeletonFills.get(node.id)
              const filledConcept = filledId ? allConcepts.find(c => c.id === filledId) : null
              const isCorrect = skeletonResults?.find(r => r.gapId === node.id)

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
                <div key={node.id} className={`p-4 rounded-lg border-2 transition-colors ${
                  isCorrect
                    ? isCorrect.correct ? 'border-emerald-300 bg-emerald-50' : 'border-red-300 bg-red-50'
                    : filledId
                    ? 'border-blue-300 bg-blue-50'
                    : 'border-amber-200 bg-amber-50/50 border-dashed'
                }`}>
                  <div className="flex items-start gap-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="w-5 h-5 rounded-full bg-gray-200 text-xs flex items-center justify-center text-gray-600 font-medium">
                          {idx + 1}
                        </span>
                        <span className="text-xs font-medium text-gray-500">{node.label}</span>
                      </div>
                      <div className="text-sm text-gray-700 ml-7">
                        <span className="italic">{node.question}</span>
                      </div>
                      {node.hint && (
                        <div className="text-xs text-gray-400 ml-7 mt-1">
                          💡 {node.hint}
                        </div>
                      )}
                      {/* 填充状态 */}
                      <div className="ml-7 mt-2">
                        {filledConcept ? (
                          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-white border border-blue-200 text-sm text-blue-700">
                            <span>✓</span>
                            <span>{filledConcept.title}</span>
                            <button
                              onClick={() => {
                                setSkeletonFills(prev => { const m = new Map(prev); m.delete(node.id); return m })
                                setSkeletonResults(null)
                              }}
                              className="text-gray-400 hover:text-red-500 ml-1"
                            >
                              ✕
                            </button>
                          </div>
                        ) : (
                          <span className="text-xs text-amber-500">等待填充...</span>
                        )}
                      </div>
                    </div>

                    {/* 验证结果 */}
                    {isCorrect && (
                      <div className={`shrink-0 text-lg ${isCorrect.correct ? 'text-emerald-500' : 'text-red-500'}`}>
                        {isCorrect.correct ? '✓' : '✗'}
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
              .filter(c => c.id !== concept.id && !skeletonFills.has(`gap_${c.id}`) && !Array.from(skeletonFills.values()).includes(c.id))
              .map(c => (
                <button
                  key={c.id}
                  draggable
                  onDragStart={(e) => {
                    e.dataTransfer.setData('text/plain', c.id)
                  }}
                  className="px-2.5 py-1.5 text-xs font-medium rounded-md border border-gray-200 bg-white text-gray-700 hover:border-blue-300 hover:bg-blue-50 cursor-grab active:cursor-grabbing transition-colors"
                >
                  {c.title}
                </button>
              ))}
            <button
              onClick={onOpenQuestion}
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
          <button
            onClick={() => {
              // Validate each gap
              const results = gapNodes.map(g => {
                const filled = skeletonFills.get(g.id)
                const correct = filled === g.correctConceptId
                return { gapId: g.id, correct, filledLabel: allConcepts.find(c => c.id === filled)?.title ?? null }
              })
              setSkeletonResults(results)
              onSkeletonSubmit?.(gapNodes.map(g => ({
                gapId: g.id,
                filledConceptId: skeletonFills.get(g.id) ?? null,
              })))
            }}
            disabled={filledCount === 0}
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
            <button
              onClick={() => {
                setSkeletonFills(new Map())
                setSkeletonResults(null)
              }}
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
```

- [ ] **Step 4: Add drop handler**

Add a `useEffect` to handle drag-and-drop onto gap cards. Since we're using native HTML drag and drop, add:

```typescript
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
    if (target) { e.preventDefault() }
  }

  document.addEventListener('drop', handleDrop)
  document.addEventListener('dragover', handleDragOver)
  return () => {
    document.removeEventListener('drop', handleDrop)
    document.removeEventListener('dragover', handleDragOver)
  }
}, [])
```

Also add `data-gap-id` to gap card div — modify the card div:

```typescript
<div key={node.id} data-gap-id={node.id} className={`p-4 rounded-lg border-2 transition-colors ...`}>
```

- [ ] **Step 5: Verify types compile**

Run: `npx tsc --noEmit`
Expected: No type errors.

- [ ] **Step 6: Commit**

```bash
git add src/components/ProcessCanvas.tsx src/utils/processComparison.ts
git commit -m "feat: add skeletonMode to ProcessCanvas with gap nodes, drag-drop candidate area, and validation"
```

---

### Task 6: Wire skeleton mode in KnowledgeGraphView

**Files:**
- Modify: `src/components/KnowledgeGraphView.tsx`

Add breadcrumb bar, determine when to show skeleton vs canvas, manage question dialog state.

- [ ] **Step 1: Add state and determine skeleton mode**

Add state variables:

```typescript
const [showQuestionDialog, setShowQuestionDialog] = useState(false)
const conceptPanelMode = useKnowledgeGraphStore(s => s.conceptPanelMode)
const skeletonCompleted = useKnowledgeGraphStore(s => s.skeletonCompleted)
const setConceptPanelMode = useKnowledgeGraphStore(s => s.setConceptPanelMode)
const markSkeletonCompleted = useKnowledgeGraphStore(s => s.markSkeletonCompleted)
const canvasHistory = useKnowledgeGraphStore(s => s.canvasHistory)
const pushHistory = useKnowledgeGraphStore(s => s.pushHistory)
const popHistory = useKnowledgeGraphStore(s => s.popHistory)
const addQuestion = useKnowledgeGraphStore(s => s.addQuestion)
```

Determine if skeleton mode should be active:

```typescript
// Show skeleton if it's the first time entering this concept
const shouldShowSkeleton = processConcept !== null && 
  !skeletonCompleted.has(processConcept.id) && 
  conceptPanelMode
```

- [ ] **Step 2: Add breadcrumb bar**

Before the process canvas block, add a breadcrumb bar when in process mode:

```typescript
{processMode && (
  <div className="shrink-0 flex items-center gap-1 px-4 py-1.5 border-b border-gray-100 bg-gray-50 text-xs text-gray-500">
    <button onClick={() => { setProcessMode(false); setProcessConcept(null) }} className="hover:text-blue-600 transition-colors">
      图谱
    </button>
    {canvasHistory.map((h, i) => (
      <span key={i} className="flex items-center gap-1">
        <span className="text-gray-300 mx-1">›</span>
        <button
          onClick={() => {
            const c = concepts.find(c2 => c2.id === h.conceptId)
            if (c) { setProcessConcept(c); setSelectedConcept(c) }
          }}
          className="hover:text-blue-600 transition-colors"
        >
          {concepts.find(c2 => c2.id === h.conceptId)?.title ?? h.conceptId}
          <span className="text-gray-400 ml-0.5">{h.view === 'skeleton' ? '[填充]' : '[画板]'}</span>
        </button>
      </span>
    ))}
    {processConcept && (
      <span className="flex items-center gap-1">
        <span className="text-gray-300 mx-1">›</span>
        <span className="text-gray-700 font-medium">
          {processConcept.title}
          <span className="text-gray-400 ml-0.5">{shouldShowSkeleton ? '[填充]' : '[画板]'}</span>
        </span>
      </span>
    )}
  </div>
)}
```

- [ ] **Step 3: Update handleEnterProcess**

Modify `handleEnterProcess` to push to history:

```typescript
const handleEnterProcess = (concept: Concept) => {
  setProcessConcept(concept)
  setProcessMode(true)
  pushHistory({ conceptId: concept.id, view: shouldShowSkeleton ? 'skeleton' : 'canvas' })
}
```

- [ ] **Step 4: Pass skeleton props to ProcessCanvas**

Find the `<ProcessCanvas>` usage and add the new props:

```typescript
<ProcessCanvas
  concept={processConcept}
  chain={processChain}
  allConcepts={concepts}
  onComplete={(flow) => {
    useKnowledgeGraphStore.getState().updateProcessState(processConcept.id, {
      user_flow: flow,
      filled: true,
      compared: false,
    })
  }}
  onNavigate={handleNavigate}
  skeletonMode={shouldShowSkeleton}
  skeletonNodes={shouldShowSkeleton && processChain ? generateSkeletonNodes(processConcept, processChain, concepts) : undefined}
  onSkeletonSubmit={(results) => {
    markSkeletonCompleted(processConcept.id)
    // Save the flow
    const filledIds = results.map(r => r.filledConceptId).filter(Boolean) as string[]
    useKnowledgeGraphStore.getState().updateProcessState(processConcept.id, {
      user_flow: filledIds,
      filled: true,
      compared: true,
    })
  }}
  onOpenQuestion={() => setShowQuestionDialog(true)}
/>
```

Add the import for `generateSkeletonNodes`:

```typescript
import { generateGenericChain, generateSkeletonNodes } from '../utils/processComparison'
```

- [ ] **Step 5: Add QuestionDialog**

Add at the bottom of the return, before the closing `</div>`:

```typescript
{showQuestionDialog && processConcept && (
  <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
    <div className="bg-white rounded-xl shadow-xl p-6 w-full max-w-md mx-4">
      <h3 className="text-sm font-semibold text-gray-900 mb-3">关于「{processConcept.title}」的疑问</h3>
      <textarea
        autoFocus
        className="w-full border border-gray-200 rounded-lg p-3 text-sm resize-none h-24 outline-none focus:border-blue-400"
        placeholder="输入你的问题..."
        id="question-input"
      />
      <div className="flex items-center justify-end gap-2 mt-3">
        <button onClick={() => setShowQuestionDialog(false)}
          className="px-3 py-1.5 text-xs font-medium text-gray-600 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors">
          取消
        </button>
        <button onClick={() => {
          const input = document.getElementById('question-input') as HTMLTextAreaElement
          if (input?.value?.trim()) {
            addQuestion({
              conceptId: processConcept.id,
              question: input.value.trim(),
              context: { location: 'skeleton' },
              status: 'open',
            })
            input.value = ''
            setShowQuestionDialog(false)
          }
        }}
          className="px-3 py-1.5 text-xs font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 transition-colors">
          提交问题
        </button>
      </div>
      <p className="text-xs text-gray-400 mt-2">问题会沉淀到问题集，可后续转化为新概念或流程步骤</p>
    </div>
  </div>
)}
```

- [ ] **Step 6: Verify types compile**

Run: `npx tsc --noEmit`
Expected: No type errors.

- [ ] **Step 7: Commit**

```bash
git add src/components/KnowledgeGraphView.tsx
git commit -m "feat: wire skeleton mode in KnowledgeGraphView with breadcrumb bar and question dialog"
```

---

### Task 7: Add question concept conversion

**Files:**
- Modify: `src/components/KnowledgeGraphView.tsx`

Add a "question bank" view in the right panel when a concept is selected and has questions.

- [ ] **Step 1: Add question bank tab to ConceptDetailPanel**

Read `src/components/ConceptDetailPanel.tsx` and add a new tab "question bank" (or add it as a section in the existing "explore" tab).

After the `actions` array (around line 80), add a new action:

```typescript
{ key: 'questions', label: '问题集', desc: questions.length > 0 ? `${questions.length} 个问题` : '暂无问题' },
```

Add state:

```typescript
const questions = useKnowledgeGraphStore(s => s.questions)
  .filter(q => q.conceptId === concept.id)
const addQuestion = useKnowledgeGraphStore(s => s.addQuestion)
```

- [ ] **Step 2: Add questions tab content**

Add after the 'read' tab (around line 248):

```typescript
{action === 'questions' && (
  <div className="px-5 py-4 space-y-3">
    <div className="flex items-center gap-2 mb-2">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">问题集</h3>
      <span className="text-xs text-gray-400">({questions.length})</span>
    </div>
    {questions.length === 0 && (
      <p className="text-xs text-gray-400 text-center py-8">暂无问题</p>
    )}
    {questions.map(q => (
      <div key={q.id} className="p-3 rounded-lg border border-gray-200 bg-white">
        <p className="text-sm text-gray-800">{q.question}</p>
        <div className="flex items-center gap-2 mt-2">
          <span className={`px-1.5 py-0.5 text-[10px] rounded ${
            q.status === 'open' ? 'bg-amber-100 text-amber-700' :
            q.status === 'converted_to_concept' ? 'bg-green-100 text-green-700' :
            'bg-gray-100 text-gray-600'
          }`}>
            {q.status === 'open' ? '待处理' : q.status === 'converted_to_concept' ? '已转为概念' : q.status === 'converted_to_step' ? '已转为步骤' : '已解决'}
          </span>
          <span className="text-[10px] text-gray-400">
            {new Date(q.createdAt).toLocaleDateString()}
          </span>
        </div>
        {q.status === 'open' && (
          <div className="flex items-center gap-2 mt-2 pt-2 border-t border-gray-100">
            <button
              onClick={() => {
                // Create a new concept from this question
                const concept = useKnowledgeGraphStore.getState().addConcept({
                  title: q.question.slice(0, 30),
                  alias: [],
                  level: 1,
                  category: 'user',
                  problem: q.question,
                  depends_on: [q.conceptId],
                  leads_to: [],
                  related: [],
                  content: `# ${q.question}\n\n> 来自用户提问\n\n## 问题\n${q.question}\n\n## 来源\n在「${concept.title}」的推导过程中提出。`,
                  path: `./docs/user/question-${q.id}.md`,
                  tags: ['user-generated'],
                })
                useKnowledgeGraphStore.getState().addEdge(q.conceptId, concept.id, 'leads_to')
                // Update question status
                useKnowledgeGraphStore.setState(state => ({
                  questions: state.questions.map(x =>
                    x.id === q.id ? { ...x, status: 'converted_to_concept' as const, convertedTo: { type: 'concept' as const, targetId: concept.id } } : x
                  )
                }))
              }}
              className="text-xs text-blue-600 hover:text-blue-800 transition-colors"
            >
              转为新概念
            </button>
          </div>
        )}
      </div>
    ))}
    {/* 提问入口 */}
    <div className="pt-2">
      <textarea
        className="w-full border border-gray-200 rounded-lg p-2.5 text-sm resize-none h-16 outline-none focus:border-blue-400"
        placeholder="提出新的问题..."
        id="panel-question-input"
      />
      <button
        onClick={() => {
          const input = document.getElementById('panel-question-input') as HTMLTextAreaElement
          if (input?.value?.trim()) {
            addQuestion({
              conceptId: concept.id,
              question: input.value.trim(),
              context: { location: 'canvas' },
              status: 'open',
            })
            input.value = ''
          }
        }}
        className="mt-2 px-3 py-1.5 text-xs font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 transition-colors"
      >
        提问
      </button>
    </div>
  </div>
)}
```

- [ ] **Step 3: Verify types compile**

Run: `npx tsc --noEmit`

- [ ] **Step 4: Commit**

```bash
git add src/components/ConceptDetailPanel.tsx
git commit -m "feat: add question bank tab to ConceptDetailPanel with concept conversion"
```

---

### Verification

After all tasks are complete:

- [ ] Run `npx tsc --noEmit` — no type errors
- [ ] Run `npm run dev` — app starts without errors
- [ ] Manual test: double-click a concept → skeleton canvas shows with gap nodes
- [ ] Manual test: drag a candidate concept to a gap → gap fills
- [ ] Manual test: submit skeleton → validation feedback
- [ ] Manual test: click "question" button → dialog opens
- [ ] Manual test: ask a question → appears in question bank tab
- [ ] Manual test: convert question to concept → new node in graph
