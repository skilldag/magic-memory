# Concept 索引持久化与文档分离 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Concept 类型中的 `content` 字段移除，Concept 变瘦为图关系索引节点，文档正文通过 `path` 按需加载；将 concepts/edges 持久化到 localStorage；docs/ 目录按 category 重构。

**Architecture:** 
- Concept 类型移除 `content`，保留 `path` 作为唯一文档引用
- Zustand persist 增加 concepts/edges 持久化（统一存储，不区分内置/自定义）
- docs/ 目录从 `level-1/2/3` 改为 `category/` 结构
- 文档加载失败时显示空状态 + "请求 LLM 生成"按钮

**Tech Stack:** TypeScript, Zustand, React 19, localStorage

---

### Task 1: Concept 类型移除 content 字段

**Files:**
- Modify: `magic-memory/src/types/index.ts:66-108`

- [ ] **Step 1: 从 Concept interface 移除 content 行**

```typescript
// 修改前 (types/index.ts)
export interface Concept {
  // ...
  content: string          // Markdown内容
  path: string           // 文件路径
  // ...
}

// 修改后
export interface Concept {
  // ...
  path: string           // 文件路径 — 唯一文档引用
  // ...
}
```

- [ ] **Step 2: 检查 LSP 诊断无残留引用报错**

Run: LSP diagnostics on `magic-memory/src/types/index.ts`
Expected: clean, no errors

- [ ] **Step 3: Commit**

```bash
git add magic-memory/src/types/index.ts
git commit -m "refactor: remove content field from Concept type"
```

---

### Task 2: 从 mock 概念数据中移除 content

**Files:**
- Modify: `magic-memory/src/data/mockGraphData.ts` (mockConcepts 数组中每个对象移除 content 行)

- [ ] **Step 1: 从每个 mockConcepts 条目中删除 content 字段**

对于概念 ID '0' 到 '50'，每个删除 `content: \`...\`` 块。这是一个机械操作 — 每个概念删掉从 `content: \`# ${title}` 到闭合反引号的行。

共 51 个概念，每个 content 块约 10-30 行。可以用正则批量替换：

查找模式：每个概念条目中的 `content: \`# ...\n...\n...\`,`

具体方式：逐概念删除从 `content:` 开始到 `path:` 上一行的内容。

示例改动（概念 '0'）：
```diff
-   content: `# VllmConfig - 配置中心
- 
- VllmConfig 是 vLLM 的统一配置入口...
- ...`,
    path: './docs/level-1/00-egg.md',
```

注意保留 `path` 字段不变（后续 Task 5 会更新路径）。

- [ ] **Step 2: 运行 LSP 诊断**

Run: LSP diagnostics on `magic-memory/src/data/mockGraphData.ts`
Expected: clean, no errors

- [ ] **Step 3: Commit**

```bash
git add magic-memory/src/data/mockGraphData.ts
git commit -m "refactor: remove inline content from mock concepts"
```

---

### Task 3: 修复 getWhatIsSummary ——不再依赖 concept.content

**Files:**
- Modify: `magic-memory/src/utils/knowledgeGraph.ts:28-38`

当前 `getWhatIsSummary` 从 `concept.content` 提取纯文本摘要。改为使用 `concept.problem` + `concept.gap_anticipate` + `elements` description 拼接摘要。

- [ ] **Step 1: 修改 getWhatIsSummary**

```typescript
// 修改后
export function getWhatIsSummary(concept: Concept) {
  const parts: string[] = []

  if (concept.problem) {
    parts.push(concept.problem)
  }
  if (concept.gap_anticipate) {
    parts.push(concept.gap_anticipate)
  }
  if (concept.elements && concept.elements.length > 0) {
    parts.push(concept.elements.map(e => e.description).join('；'))
  }

  const summary = parts.join(' ').trim()
  if (!summary) return ''
  return summary.length > 120 ? `${summary.slice(0, 120)}...` : summary
}
```

- [ ] **Step 2: LSP 诊断确认无报错**

Run: LSP diagnostics on `magic-memory/src/utils/knowledgeGraph.ts`
Expected: clean

- [ ] **Step 3: Commit**

```bash
git add magic-memory/src/utils/knowledgeGraph.ts
git commit -m "refactor: getWhatIsSummary uses problem/gap/elements instead of content"
```

---

### Task 4: 修复 createConceptWithEdges ——不再生成内嵌 content

**Files:**
- Modify: `magic-memory/src/store/knowledgeGraphStore.ts:177-213`

当前 `createConceptWithEdges` 生成内嵌 content（含 Markdown）。改为生成 path 而非 content。用户新增概念没有对应的 .md 文件，所以 path 指向 `./docs/user/<id>.md`，后续通过 "请求 LLM" 按钮写入。

- [ ] **Step 1: 修改 createConceptWithEdges 中的 content 为 path**

```typescript
// 修改前
const concept: Concept = {
  // ...
  content: input.content || `# ${input.title}\n\n## 问题\n${input.problem || `与「${source.title}」关联`}\n\n## 来源\n通过探索模式关联添加。`,
  path: `./docs/user/${input.title.toLowerCase().replace(/\s+/g, '-')}.md`,
  // ...
}

// 修改后
const concept: Concept = {
  // ...
  path: `./docs/user/${Date.now()}-${input.title.toLowerCase().replace(/\s+/g, '-')}.md`,
  // ...
}
```

移除 content 字段赋值。如果将来有 content 参数需要补文档内容，留一个 TODO 注释说明通过 LLM 生成。

- [ ] **Step 2: LSP 诊断**

Run: LSP diagnostics on `magic-memory/src/store/knowledgeGraphStore.ts`
Expected: clean

- [ ] **Step 3: Commit**

```bash
git add magic-memory/src/store/knowledgeGraphStore.ts
git commit -m "refactor: createConceptWithEdges generates path instead of inline content"
```

---

### Task 5: 将 concepts/edges 加入 Zustand persist

**Files:**
- Modify: `magic-memory/src/store/knowledgeGraphStore.ts` (partialize, merge, loadGraph)

- [ ] **Step 1: 修改 partialize 加入 concepts 和 edges**

```typescript
// 找到 partialize 配置，修改为：
partialize: (state) => ({
  concepts: state.concepts,
  edges: state.edges,
  reviewRecords: Array.from(state.reviewRecords.entries()),
  annotations: state.annotations,
}),
```

- [ ] **Step 2: 修改 merge 函数优先取持久化数据**

```typescript
merge: (persisted: any, current) => ({
  ...current,
  concepts: persisted?.concepts ?? current.concepts,
  edges: persisted?.edges ?? current.edges,
  reviewRecords: new Map(persisted?.reviewRecords || []),
  annotations: persisted?.annotations ?? [],
})
```

- [ ] **Step 3: 修改 loadGraph —— 有持久化数据时跳过覆盖**

```typescript
loadGraph: async () => {
  set({ isLoading: true, error: null })
  try {
    const currentState = get()
    // 如果已经有持久化数据，不覆盖
    if (currentState.concepts.length > 0 && currentState.edges.length > 0) {
      set({ isLoading: false })
      return
    }
    const data = getMockGraphData()
    await new Promise(resolve => setTimeout(resolve, 300))
    set({
      concepts: data.concepts,
      edges: data.edges,
      chains: data.chains ?? [],
      isLoading: false
    })
  } catch (error) {
    set({
      error: error instanceof Error ? error.message : 'Unknown error',
      isLoading: false
    })
  }
},
```

- [ ] **Step 4: LSP 诊断**

Run: LSP diagnostics on `magic-memory/src/store/knowledgeGraphStore.ts`
Expected: clean

- [ ] **Step 5: Commit**

```bash
git add magic-memory/src/store/knowledgeGraphStore.ts
git commit -m "feat: persist concepts and edges to localStorage"
```

---

### Task 6: 移动 docs 文件到 category 目录结构

**Files:**
- Create: `magic-memory/public/docs/Foundation/`, `Model/`, `Performance/`, `Scheduling/`, `Serving/`, `Advanced/`, `Optimization/`, `Infrastructure/` 目录
- Move: `magic-memory/public/docs/level-1/*.md` → `Foundation/`
- Move: `magic-memory/public/docs/level-2/*.md` (按 category 分类)
- Move: `magic-memory/public/docs/level-3/*.md` (按 category 分类)
- Delete: 空的 `level-1/`, `level-2/`, `level-3/` 目录

**概念到 category 的映射：**

| level | category | 概念 ID | 目标目录 |
|-------|----------|---------|---------|
| 1 | Foundation | 0-9 | `Foundation/` |
| 2 | Model | 10-15, 20-25, 29 | `Model/` |
| 2 | Performance | 26-28 | `Performance/` |
| 2 | Model (attention 子概念) | 16-19 | `Model/attention/` |
| 3 | Advanced | 30-33, 46 | `Advanced/` |
| 3 | Scheduling | 34-37, 39 | `Scheduling/` |
| 3 | Serving | 40-45 | `Serving/` |
| 3 | Optimization | 38, 48-49 | `Optimization/` |
| 3 | Infrastructure | 47, 50 | `Infrastructure/` |

文件命名规则：保持原有编号前缀 + 英文短横线命名，如 `00-vllm-config.md`，`16-paged-attention.md`。

- [ ] **Step 1: 创建所有目标目录**

```bash
mkdir -p magic-memory/public/docs/Foundation
mkdir -p magic-memory/public/docs/Model/attention
mkdir -p magic-memory/public/docs/Performance
mkdir -p magic-memory/public/docs/Scheduling
mkdir -p magic-memory/public/docs/Serving
mkdir -p magic-memory/public/docs/Advanced
mkdir -p magic-memory/public/docs/Optimization
mkdir -p magic-memory/public/docs/Infrastructure
```

- [ ] **Step 2: 按 category 移动文件**

```bash
# Foundation (level 1, id 0-9)
mv magic-memory/public/docs/level-1/00-egg.md magic-memory/public/docs/Foundation/00-vllm-config.md
mv magic-memory/public/docs/level-1/01-candle.md magic-memory/public/docs/Foundation/01-device.md
mv magic-memory/public/docs/level-1/02-duck.md magic-memory/public/docs/Foundation/02-tensor.md
mv magic-memory/public/docs/level-1/03-ear.md magic-memory/public/docs/Foundation/03-logger.md
mv magic-memory/public/docs/level-1/04-boat.md magic-memory/public/docs/Foundation/04-vllm-core.md
mv magic-memory/public/docs/level-1/05-hook.md magic-memory/public/docs/Foundation/05-gpu-allocator.md
mv magic-memory/public/docs/level-1/06-spoon.md magic-memory/public/docs/Foundation/06-error-handling.md
mv magic-memory/public/docs/level-1/07-crutch.md magic-memory/public/docs/Foundation/07-init.md
mv magic-memory/public/docs/level-1/08-gourd.md magic-memory/public/docs/Foundation/08-foundation-layer.md
mv magic-memory/public/docs/level-1/09-balloon.md magic-memory/public/docs/Foundation/09-kv-cache.md

# Model (level 2, category=Model)
# 公共概念
mv magic-memory/public/docs/level-2/10-baseball.md magic-memory/public/docs/Model/10-model-registry.md
mv magic-memory/public/docs/level-2/11-chopsticks.md magic-memory/public/docs/Model/11-model-loader.md
mv magic-memory/public/docs/level-2/12-highchair.md magic-memory/public/docs/Model/12-model.md
mv magic-memory/public/docs/level-2/13-umbrella.md magic-memory/public/docs/Model/13-model-runner.md
mv magic-memory/public/docs/level-2/14-rose.md magic-memory/public/docs/Model/14-embedding.md
mv magic-memory/public/docs/level-2/15-parrot.md magic-memory/public/docs/Model/15-transformer-layers.md
mv magic-memory/public/docs/level-2/20-cigarette.md magic-memory/public/docs/Model/20-sampler.md
mv magic-memory/public/docs/level-2/21-crocodile.md magic-memory/public/docs/Model/21-sampling-params.md
mv magic-memory/public/docs/level-2/22-twins.md magic-memory/public/docs/Model/22-logits.md
mv magic-memory/public/docs/level-2/23-earplugs.md magic-memory/public/docs/Model/23-token.md
mv magic-memory/public/docs/level-2/24-alarm.md magic-memory/public/docs/Model/24-decode-step.md
mv magic-memory/public/docs/level-2/25-erhu.md magic-memory/public/docs/Model/25-forward-pass.md
mv magic-memory/public/docs/level-2/29-uncle.md magic-memory/public/docs/Model/29-weights-loading.md
# attention 子概念 (id 16-19)
mv magic-memory/public/docs/level-2/16-pomegranate.md magic-memory/public/docs/Model/attention/16-paged-attention.md
mv magic-memory/public/docs/level-2/17-microscope.md magic-memory/public/docs/Model/attention/17-block-table.md
mv magic-memory/public/docs/level-2/18-money.md magic-memory/public/docs/Model/attention/18-cache-block.md
mv magic-memory/public/docs/level-2/19-medicine.md magic-memory/public/docs/Model/attention/19-kv-cache-manager.md

# Performance (level 2, category=Performance)
mv magic-memory/public/docs/level-2/26-river.md magic-memory/public/docs/Performance/26-gpu-memory-pool.md
mv magic-memory/public/docs/level-2/27-headphones.md magic-memory/public/docs/Performance/27-flash-attention.md
mv magic-memory/public/docs/level-2/28-bully.md magic-memory/public/docs/Performance/28-quantization.md

# Advanced (level 3, category=Advanced)
mv magic-memory/public/docs/level-3/30-mitsubishi.md magic-memory/public/docs/Advanced/30-speculative-decoding.md
mv magic-memory/public/docs/level-3/31-yam.md magic-memory/public/docs/Advanced/31-draft-token.md
mv magic-memory/public/docs/level-3/32-fan.md magic-memory/public/docs/Advanced/32-verifier.md
mv magic-memory/public/docs/level-3/33-stars.md magic-memory/public/docs/Advanced/33-n-gram-proposer.md
mv magic-memory/public/docs/level-3/46-pomegranate.md magic-memory/public/docs/Advanced/46-multi-lora.md

# Scheduling (level 3, category=Scheduling)
mv magic-memory/public/docs/level-3/34-vegetable.md magic-memory/public/docs/Scheduling/34-continuous-batching.md
mv magic-memory/public/docs/level-3/35-coral.md magic-memory/public/docs/Scheduling/35-scheduler.md
mv magic-memory/public/docs/level-3/36-deer.md magic-memory/public/docs/Scheduling/36-prefill.md
mv magic-memory/public/docs/level-3/37-pheasant.md magic-memory/public/docs/Scheduling/37-decode.md
mv magic-memory/public/docs/level-3/39-sword.md magic-memory/public/docs/Scheduling/39-request-queue.md

# Serving (level 3, category=Serving)
mv magic-memory/public/docs/level-3/40-commander.md magic-memory/public/docs/Serving/40-vllm-engine.md
mv magic-memory/public/docs/level-3/41-lizard.md magic-memory/public/docs/Serving/41-engine-api.md
mv magic-memory/public/docs/level-3/42-corn.md magic-memory/public/docs/Serving/42-vllm-serving.md
mv magic-memory/public/docs/level-3/43-rock.md magic-memory/public/docs/Serving/43-openai-api.md
mv magic-memory/public/docs/level-3/44-cobra.md magic-memory/public/docs/Serving/44-grpc.md
mv magic-memory/public/docs/level-3/45-master.md magic-memory/public/docs/Serving/45-websocket.md

# Optimization (level 3, category=Optimization)
mv magic-memory/public/docs/level-3/38-woman.md magic-memory/public/docs/Optimization/38-prefix-caching.md
mv magic-memory/public/docs/level-3/48-loofah.md magic-memory/public/docs/Optimization/48-prefix-lookup.md
mv magic-memory/public/docs/level-3/49-wetdog.md magic-memory/public/docs/Optimization/49-cache-eviction.md

# Infrastructure (level 3, category=Infrastructure)
mv magic-memory/public/docs/level-3/47-driver.md magic-memory/public/docs/Infrastructure/47-gpu-driver.md
mv magic-memory/public/docs/level-3/50-minivan.md magic-memory/public/docs/Infrastructure/50-distributed.md
```

- [ ] **Step 3: 删除空目录**

```bash
rmdir magic-memory/public/docs/level-1
rmdir magic-memory/public/docs/level-2
rmdir magic-memory/public/docs/level-3
```

- [ ] **Step 4: Commit**

```bash
git add magic-memory/public/docs/
git commit -m "reorg: move docs to category-based directory structure"
```

---

### Task 7: 更新 mock 概念中所有 path 指向新目录

**Files:**
- Modify: `magic-memory/src/data/mockGraphData.ts`

每个 mock 概念有 `path` 指向旧目录如 `'./docs/level-1/00-egg.md'`，改为新路径如 `'./docs/Foundation/00-vllm-config.md'`。

- [ ] **Step 1: 替换所有 mockConcepts 中的 path 值**

共 51 个 path 需要更新。映射关系同 Task 6 的文件移动映射。

示例：
```
'./docs/level-1/00-egg.md'      → './docs/Foundation/00-vllm-config.md'
'./docs/level-1/01-candle.md'    → './docs/Foundation/01-device.md'
'./docs/level-2/10-baseball.md'  → './docs/Model/10-model-registry.md'
'./docs/level-3/30-mitsubishi.md' → './docs/Advanced/30-speculative-decoding.md'
...（共 51 个）
```

- [ ] **Step 2: LSP 诊断确认**

Run: LSP diagnostics on `magic-memory/src/data/mockGraphData.ts`
Expected: clean

- [ ] **Step 3: Commit**

```bash
git add magic-memory/src/data/mockGraphData.ts
git commit -m "fix: update concept paths to match new docs directory structure"
```

---

### Task 8: 修复 ConceptDetailPanel ——从 path 加载文档，无 fallback

**Files:**
- Modify: `magic-memory/src/components/ConceptDetailPanel.tsx:255-258`

- [ ] **Step 1: 移除 concept.content fallback**

```typescript
// 修改前 (line 255-258)
<DocumentViewer document={{
  id: concept.id, title: concept.title, path: concept.path,
  content: docContent ?? concept.content, level: concept.level, category: concept.category,
  tags: concept.tags, lastModified: concept.lastModified, metadata: concept.metadata,
}} />

// 修改后
<DocumentViewer document={{
  id: concept.id, title: concept.title, path: concept.path,
  content: docContent ?? '', level: concept.level, category: concept.category,
  tags: concept.tags, lastModified: concept.lastModified, metadata: concept.metadata,
}} />
```

- [ ] **Step 2: 确认 docContent 加载失败时显示空状态**

检查 `DocumentViewer.tsx` 中当 `document.content` 为空时的表现。如果为空字符串，应显示空白内容区域（目前 marked('') 应返回空 HTML）。确认不需要额外调整。

- [ ] **Step 3: LSP 诊断**

Run: LSP diagnostics on `magic-memory/src/components/ConceptDetailPanel.tsx`
Expected: clean

- [ ] **Step 4: Commit**

```bash
git add magic-memory/src/components/ConceptDetailPanel.tsx
git commit -m "fix: ConceptDetailPanel loads doc from path, no content fallback"
```

---

### Task 9: 添加"请求 LLM 生成"按钮（文档缺失时）

**Files:**
- Modify: `magic-memory/src/components/ConceptDetailPanel.tsx` (read tab 中，当 docContent 为 null 时显示按钮)

- [ ] **Step 1: 在 read tab 中添加缺失状态 UI**

在 `<DocumentViewer>` 之前添加条件判断：如果 `docContent` 为 null 且已加载完毕（非 loading 状态），显示空状态和按钮。

```typescript
// 在 action === 'read' 的渲染块中，在 DocumentViewer 之前添加：
{/* 新增：文档缺失状态 */}
{docContent === null && !loading && (
  <div className="flex flex-col items-center justify-center py-16 text-gray-400">
    <svg className="w-12 h-12 mb-3 text-gray-200" ... />
    <p className="text-sm mb-4">还没有对应的文档内容</p>
    <button
      onClick={handleRequestLLM}
      className="px-4 py-2 bg-blue-500 text-white text-sm rounded-lg hover:bg-blue-600 transition-colors"
    >
      请求 LLM 生成
    </button>
  </div>
)}
```

需要加的 state：
```typescript
const [loading, setLoading] = useState(false)
```

注意：这里的 `loading` 是文档加载中状态。`docContent === null && !loading` 表示"已经尝试加载过但没有内容"。

修改 useEffect：
```typescript
useEffect(() => {
  if (action !== 'read') return
  setDocContent(null)
  setLoading(true)
  loadDocContent(concept.path).then(content => {
    if (content) setDocContent(content)
    setLoading(false)
  })
}, [action, concept.id, concept.path])
```

添加 handleRequestLLM：
```typescript
const handleRequestLLM = useCallback(async () => {
  // TODO: 调用 LLM API 生成文档内容并写入 path
  // 当前只展示 toast 提示
  alert('LLM 生成功能待接入')
}, [concept])
```

- [ ] **Step 2: LSP 诊断**

Run: LSP diagnostics on `magic-memory/src/components/ConceptDetailPanel.tsx`
Expected: clean

- [ ] **Step 3: Commit**

```bash
git add magic-memory/src/components/ConceptDetailPanel.tsx
git commit -m "feat: add LLM generation button for missing concept docs"
```

---

### Task 10: 更新 remaining components 中对 concept.content 的引用

**Files:**
- Modify: `magic-memory/src/components/ExploreDialog.tsx:157` (检查 aiData.content 是否等于 concept.content)
- Modify: `magic-memory/src/components/QuickExploreDialog.tsx:47` (同上)

- [ ] **Step 1: 检查 ExploreDialog.tsx 和 QuickExploreDialog.tsx 中的 content 引用**

如果 `data.content` / `aiData.content` 来自 API 返回的独立字段（非 Concept.content），则不需要修改。
按 grep 结果分析：

```
ExploreDialog.tsx:157  —  content: aiData.content  → 来自 LLM API 响应，不是 Concept 字段，无需修改
QuickExploreDialog.tsx:47 — content: data.content  → 同上，来自 API 响应，无需修改
```

这些引用的是后端 LLM 返回的 content，不依赖 Concept 类型。

- [ ] **Step 2: 全局确认无其他 concept.content 残留**

Run: `grep -r "concept\.content" magic-memory/src/ --include="*.ts" --include="*.tsx"`
Expected: 0 matches (全部清理完毕)

- [ ] **Step 3: 确认编译通过**

Run: `cd magic-memory && npx tsc --noEmit`
Expected: exit code 0

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove remaining concept.content references"
```
