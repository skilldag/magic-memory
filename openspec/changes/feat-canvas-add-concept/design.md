# Design: 图谱画布双击添加概念 + 手动关系更新

## Feature A: 双击空白区添加概念节点

### 交互流程

```
[用户操作]                    [系统响应]
─────────────────────────────────────────────
1. 双击图谱空白区             → onBackgroundDoubleTap 触发
2.                            → 弹出 AddConceptDialog（modal）
3. 输入概念名，按 Enter        →
4.                            → 调用 store.addConcept({
                                 title, level: 1,
                                 category: '用户自定义',
                                 path: auto-generated,
                                 tags: [], metadata: { status: 'draft' }
                               })
5.                            → KnowledgeGraph 增量 useEffect 触发
                               → cy.add() 添加新节点
                               → fcose 增量布局
6.                            → 新节点自动选中
                               → 右侧面板展示空文档编辑区
7. （可选）用户输入文档内容     → 编辑完成后可点击"更新关系"
```

### 双击检测

已有实现（`KnowledgeGraph.tsx:248-256`）：

```typescript
cy.on('tap', (evt) => {
  if (evt.target !== cy) return          // 仅在背景点击时触发
  const now = Date.now()
  const delta = now - lastBackgroundTapAtRef.current
  lastBackgroundTapAtRef.current = now
  if (delta > 0 && delta < 300) {        // 300ms 内两次 tap = 双击
    onBackgroundDoubleTap?.()
  }
})
```

双击节点进入过程画板的检测使用不同的时间窗（400ms）和节点过滤，两者无冲突。

### AddConceptDialog 组件

轻量弹窗，只包含概念名输入，不包含关系配置（关系通过后续的"更新关系"功能处理）：

```
┌──────────────────────────────────┐
│  添加新概念                        │
│                                  │
│  概念名称                         │
│  ┌──────────────────────────────┐│
│  │                              ││
│  └──────────────────────────────┘│
│                                  │
│         [取消]  [确认添加]        │
└──────────────────────────────────┘
```

### store 接入

```typescript
// KnowledgeGraphView.tsx
const handleAddConcept = (title: string) => {
  const concept = useKnowledgeGraphStore.getState().addConcept({
    title,
    level: 1,
    category: '',
    problem: '',
    depends_on: [],
    leads_to: [],
    related: [],
    path: `./docs/user/${Date.now()}-${title.toLowerCase().replace(/\s+/g, '-')}.md`,
    tags: [],
    metadata: { status: 'draft' },
  })
  // 选中新创建的概念
  useKnowledgeGraphStore.getState().selectConcept(concept)
}
```

增量更新在 `KnowledgeGraph.tsx` 的 useEffect 中自动处理。

---

## Feature B: 手动关系和图谱更新

### 交互

在 `ConceptDetailPanel` 的文档编辑 Tab 中，编辑器下方新增按钮：

```
┌──────────────────────────────────┐
│  ┌────────────────────────────┐  │
│  │ # Attention                │  │
│  │ ---                        │  │
│  │ depends_on: [KVCacheManager│  │
│  │ leads_to: [FlashAttention] │  │
│  │ ---                        │  │
│  │ 文档正文...                 │  │
│  └────────────────────────────┘  │
│                                  │
│  [💾 保存]  [🔄 更新关系和图谱]   │
│                                  │
│  当前关系:                        │
│  depends_on: KVCacheManager      │
│  leads_to:   FlashAttention      │
└──────────────────────────────────┘
```

### 更新逻辑

```typescript
function reparseRelations(conceptId: string) {
  const store = useKnowledgeGraphStore.getState()
  const concept = store.concepts.find(c => c.id === conceptId)
  if (!concept?.content) return

  // 1. 解析 frontmatter
  const { meta } = parseFrontmatter(concept.content)
  const dependsOnIds = matchTitlesToIds(meta.depends_on || [], store.concepts)
  const leadsToIds = matchTitlesToIds(meta.leads_to || [], store.concepts)
  const relatedIds = matchTitlesToIds(meta.related || [], store.concepts)

  // 2. 更新概念的引用数组
  const updatedConcept = {
    ...concept,
    depends_on: dependsOnIds,
    leads_to: leadsToIds,
    related: relatedIds,
  }

  // 3. 计算新旧边集 diff
  const newEdgeSet = new Set<string>()
  const newEdges: ConceptEdge[] = []

  const addEdge = (source: string, target: string, type: ConceptEdge['type']) => {
    const key = `${source}|${target}|${type}`
    if (!newEdgeSet.has(key)) {
      newEdgeSet.add(key)
      newEdges.push({ id: `${source}-${type}-${target}`, source, target, type })
    }
  }

  dependsOnIds.forEach(t => {
    addEdge(conceptId, t, 'depends_on')
    addEdge(t, conceptId, 'leads_to')
  })
  leadsToIds.forEach(t => {
    addEdge(conceptId, t, 'leads_to')
    addEdge(t, conceptId, 'depends_on')
  })
  relatedIds.forEach(t => addEdge(conceptId, t, 'related'))

  const oldEdgeKeys = new Set(store.edges.map(e => `${e.source}|${e.target}|${e.type}`))
  const keptEdges = store.edges.filter(e => {
    // 保留不涉及本概念的边
    if (e.source !== conceptId && e.target !== conceptId) return true
    // 涉及本概念的边需在新边集中存在
    return newEdgeSet.has(`${e.source}|${e.target}|${e.type}`)
  })

  // 4. 更新 store → 触发增量布局
  useKnowledgeGraphStore.setState({
    concepts: store.concepts.map(c => c.id === conceptId ? updatedConcept : c),
    edges: [...keptEdges, ...newEdges],
  })
}
```

### store 改动

在 `knowledgeGraphStore.ts` 中新增 action：

```typescript
reparseRelations: (conceptId: string) => {
  // 上述逻辑
}
```

也可直接在 `ConceptDetailPanel` 中作为事件处理函数实现（更轻量，不增加 store 的 API surface）。

---

## 分离 content 和 structure 的 effect 依赖

当前 `KnowledgeGraph.tsx` 的增量更新 effect 依赖 `[concepts, edges]`，这意味着只改 `content` 字段也会触发无效的图布局。

**改动：将依赖改为结构签名（structural key）：**

```typescript
const structuralKey = useMemo(
  () =>
    concepts.map(c => c.id).sort().join(',') + '|' +
    edges.map(e => `${e.source}-${e.target}-${e.type}`).sort().join(','),
  [concepts, edges]
)

useEffect(() => {
  // 增量节点/边更新 + layout
}, [structuralKey])
```

这样 `updateConceptContent` 只改 content 字段时，`structuralKey` 不变，不会触发布局。而 `addConcept`/`reparseRelations` 改变了 id 集或边集，key 变化，正常触发增量更新。
