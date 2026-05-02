# 项目切换/创建加载性能优化设计

> 用户感知到的"快"，不是总耗时少，而是 UI 先响应。

---

## 1. 问题定义

### 1.1 症状

切换或创建项目时，UI 卡住 2-5 秒无响应，用户不知道系统是否在工作。

### 1.2 根因链路

切换/创建项目时，主线程上有一条同步阻塞链路：

```
readMdFiles(递归扫描所有 .md，逐文件读取全文)
  → 构建 N 个 Concept 对象
  → deriveEdges(O(N²) 全文交叉引用匹配)
  → Zustand setState → persist 序列化到 localStorage
  → React 重渲染:
      → Cytoscape fcose 布局 (quality:proof, 2000 iter, O(N²))
      → SummaryPanel analyzeGraph() (DFS 最长路径, O(N²))
```

### 1.3 影响范围

- `projectStore.ts`: `createProject()` 和 `switchProject()` 方法
- `utils/fileSystem.ts`: `readMdFiles()` 文件扫描
- `utils/deriveEdges.ts`: 交叉引用边推导
- `components/KnowledgeGraph.tsx`: Cytoscape 初始化 + fcose 布局
- `components/SummaryPanel.tsx`: `analyzeGraph()` 计算
- `store/knowledgeGraphStore.ts`: Zustand persist 序列化

---

## 2. 设计目标

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 首屏可见时间（TTI 前） | ~3s | ~300ms |
| 全量加载完成时间 | ~3s | ~3s（不变） |
| 主线程阻塞时长 | ~2s | <100ms |
| localStorage 体积 | ~500KB | ~100KB |
| 操作后续响应（分析结果缓存命中） | ~500ms | <10ms |

核心目标不是减少总耗时，而是**不阻塞主线程**。

---

## 3. 架构变更

### 3.1 数据流对比

**优化前（同步阻塞）**:

```
用户触发切换项目
  → loadHandle (IDB)
  → ensurePermission
  → readMdFiles (同步读取全部)
  → buildConcepts (同步遍历)
  → deriveEdges (同步 O(N²))
  → setState (一次性)
  → React 重渲染所有组件
  → fcose 布局 (完整精度)
  → analyzeGraph (完整分析)
  → UI 可用
```

**优化后（渐进式）**:

```
用户触发切换项目
  → loadHandle (IDB)
  → ensurePermission
  → readMdFiles 分批 yield (每批 10 个)
  → setState 每批推入 (第一批 ~200ms)
  → React 渲染骨架图
  → 第二批、第三批... 持续追加
  → [后台] deriveEdges Web Worker
  → [后台] 第二阶段 fcose 精布局
  → [后台] analyzeGraph 惰性计算
  → UI 一直可用
```

---

## 4. 详细设计

### 4.1 分批文件扫描

**文件**: `utils/fileSystem.ts`

将 `readMdFiles()` 改为 async generator，每读 10 个 .md 文件 yield 一批。

```typescript
// 新增
export async function* readMdFilesBatched(
  dirHandle: FileSystemDirectoryHandle,
  pathPrefix = '',
  batchSize = 10,
): AsyncGenerator<{ path: string; content: string }[], void, undefined> {
  let batch: { path: string; content: string }[] = []
  let count = 0
  for await (const [name, entry] of (dirHandle as any).entries()) {
    const entryPath = pathPrefix ? `${pathPrefix}/${name}` : name
    if (entry.kind === 'directory' && !name.startsWith('.')) {
      for await (const file of readMdFilesBatched(entry, entryPath, batchSize)) {
        batch.push(...file)
        count += file.length
        if (batch.length >= batchSize) {
          yield batch
          batch = []
        }
      }
    } else if (entry.kind === 'file' && name.endsWith('.md')) {
      try {
        const file = await (entry as FileSystemFileHandle).getFile()
        const content = await file.text()
        if (content.trim()) {
          batch.push({ path: entryPath, content })
          count++
          if (batch.length >= batchSize) {
            yield batch
            batch = []
          }
        }
      } catch { /* skip unreadable files */ }
    }
  }
  if (batch.length > 0) yield batch
}
```

**文件**: `store/projectStore.ts`

`switchProject()` 和 `createProject()` 中使用 async generator 消费，每批推一次 store。

```typescript
switchProject: async (projectId: string) => {
  set({ isLoading: true, concepts: [], edges: [] })  // 清空旧数据，触发加载态
  // ... 验证权限、加载 handle ...

  const concepts: Concept[] = []
  for await (const batch of readMdFilesBatched(handle)) {
    const newConcepts = batch.map(file => ({
      id: file.path.replace('.md', '').replace(/\//g, '-'),
      title: file.path.replace('.md', '').split('/').pop() || file.path.replace('.md', ''),
      path: file.path,
      level: 1, category: '', problem: '',
      depends_on: [], leads_to: [], related: [], tags: [],
      lastModified: new Date(),
    }))
    concepts.push(...newConcepts)
    // 每批推入 store，UI 逐步渲染
    useKnowledgeGraphStore.setState({ concepts: [...concepts], isLoading: true })
  }

  // 全量数据就位后，将 isLoading 置 false，后台推导 edges
  useKnowledgeGraphStore.setState({ isLoading: false })
  deriveEdgesInWorker(concepts, allFiles)
}
```

### 4.2 deriveEdges Web Worker

**新建**: `src/workers/deriveEdges.worker.ts`

```typescript
// Web Worker: 纯计算，接收 concepts + files，返回 edges
self.onmessage = (e: MessageEvent<{ concepts: Concept[]; files: { path: string; content: string }[] }>) => {
  const { concepts, files } = e.data
  const edges = deriveEdges(concepts, files)
  self.postMessage({ edges })
}
```

**文件**: `store/projectStore.ts`

在 `switchProject()` 中启动 worker，收到结果后更新 store。

```typescript
function deriveEdgesInWorker(concepts: Concept[], files: { path: string; content: string }[]) {
  try {
    const worker = new Worker(new URL('../workers/deriveEdges.worker.ts', import.meta.url))
    worker.postMessage({ concepts, files })
    worker.onmessage = (e) => {
      useKnowledgeGraphStore.setState({ edges: e.data.edges })
      worker.terminate()
    }
  } catch {
    // Fallback: Web Worker 不可用时（如 file:// 协议），同步执行
    import('../utils/deriveEdges').then(({ deriveEdges }) => {
      const edges = deriveEdges(concepts, files)
      useKnowledgeGraphStore.setState({ edges })
    })
  }
}
```

### 4.3 两阶段 Cytoscape 布局

**文件**: `components/KnowledgeGraph.tsx`

初始化 layout 拆为两个阶段：

```typescript
// 阶段 1: 快速粗略布局
try {
  const fastLayout = cy.layout({
    name: 'fcose',
    quality: 'default',
    animate: false,
    nodeRepulsion: 25000,
    idealEdgeLength: 160,
    gravity: 0.08,
    numIter: 200,  // 原 2000 → 200
    tile: true,
    padding: 80,
  } as cytoscape.LayoutOptions)
  fastLayout.run()
  cy.fit(undefined, 50)
  setZoomLevel(cy.zoom())
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
      /* ignore */
    }
  }, { timeout: 3000 })
}
```

增量更新布局（`structuralKey` effect）保持原有 `quality: 'default'` + `numIter: 300`。

### 4.4 Zustand persist 瘦身

**文件**: `store/knowledgeGraphStore.ts`

```typescript
persist(
  (set, get) => ({ ... }),
  {
    name: 'knowledge-graph-storage',
    partialize: (state) => ({
      concepts: state.concepts.map(c => ({
        id: c.id, title: c.title, alias: c.alias,
        level: c.level, category: c.category,
        problem: c.problem, gap_anticipate: c.gap_anticipate,
        depends_on: c.depends_on, leads_to: c.leads_to, related: c.related,
        path: c.path, tags: c.tags,
        lastModified: c.lastModified,
        metadata: c.metadata,
        // 排除: content, elements, hierarchy, process
      })),
      edges: state.edges,
      reviewRecords: Array.from(state.reviewRecords.entries()),
      annotations: state.annotations,
    }),
  }
)
```

content 通过 `loadDocContent()` 按需加载（已有实现 — `ConceptDetailPanel.tsx` 中 `action === 'read'` 时自动从文件 fetch）。elements、hierarchy、process 作为计算/运行时数据不需要持久化。

> **兼容性验证**：`ConceptDetailPanel.tsx` 第 45-58 行已存在 `concept.content` 回退到 `loadDocContent()` 的逻辑，persist 排除 content 后该回退路径自动生效，无需额外改动。

### 4.5 analyzeGraph 缓存

**文件**: `components/SummaryPanel.tsx`

```typescript
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

  // LRU 缓存：最多保留 5 个
  if (analysisCacheRef.current.size >= 5) {
    const firstKey = analysisCacheRef.current.keys().next().value
    analysisCacheRef.current.delete(firstKey)
  }
  analysisCacheRef.current.set(key, result)
  return result
}, [structuralKey, concepts, edges])
```

---

## 5. 文件变更清单

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `utils/fileSystem.ts` | 新增函数 | 新增 `readMdFilesBatched()` async generator |
| `store/projectStore.ts` | 重构 | `switchProject()`/`createProject()` 改用 batch 流式处理 |
| `workers/deriveEdges.worker.ts` | **新建** | Web Worker 文件 |
| `components/KnowledgeGraph.tsx` | 重构 | 两阶段布局 (fast + idle refine) |
| `store/knowledgeGraphStore.ts` | 修改 | `partialize` 瘦身，排除 content 等大字段 |
| `components/SummaryPanel.tsx` | 修改 | 添加 LRU 缓存，惰性分析 |

---

## 6. 不做的事

- 不将 localStorage 迁移到 IndexedDB（改动量过大，且当前数据量 localStorage 足够）
- 不引入虚拟列表/虚拟图谱（概念数 < 200 时收益不大）
- 不修改 `ClusterView` 的 Louvain 计算（仅在 cluster 视图使用，非关键路径）
- 不需要 Service Worker（浏览器环境限制）
- 不重构非项目切换相关的其他性能问题

---

## 7. 验证标准

1. 切换已有项目（50 个 .md 文件）：首屏 < 500ms，全量加载完成 < 5s
2. 创建新项目（50 个 .md 文件）：首屏 < 500ms，全量加载完成 < 5s
3. 切换后立即操作（点击节点/缩放等）：无卡顿
4. 关闭再打开应用：localStorage 读写 < 50ms
5. deriveEdges 结果与优化前一致
