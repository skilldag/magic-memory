# KEY CONCEPTS 条目删除 — 设计文档

> 在语义对齐面板中，允许用户临时忽略或永久删除 KEY CONCEPTS 段落中的单个条目，解决对齐结果中冗余/不相关术语的干扰问题。

---

## 1. 现状分析

### 1.1 已有基础设施

| 能力 | 状态 | 位置 |
|------|------|------|
| KEY CONCEPTS 段落解析 | ✅ 已实现 | `utils/alignment.ts` → `parseKeyConcepts()` |
| 语义对齐面板 | ✅ 已实现 | `components/AlignmentPanel.tsx` |
| 对齐结果展示 (NodeRow) | ✅ 已实现 | `components/AlignmentPanel.tsx` → `NodeRow` 组件 |
| 对齐草稿缓存 | ✅ 已实现 | `store/knowledgeGraphStore.ts` → `alignmentDrafts` |
| 文档内容更新 | ✅ 已实现 | `store/knowledgeGraphStore.ts` → `updateConceptContent()` |
| 数据持久化 | ✅ 已实现 | `store/knowledgeGraphStore.ts` → `persistToServer()` |

### 1.2 缺失能力

- **无法忽略冗余条目**：用户对某些 KEY CONCEPTS 条目已经很熟悉（如"Transformer"），每次对齐都出现造成干扰，无法将其从当前结果中移除
- **无法删除错误条目**：某些自动提取的 KEY CONCEPTS 条目不准确或与概念无关，无法从原文中永久删除
- **无已忽略状态管理**：临时忽略的条目没有独立存储，重新对齐后会再次出现
- **无 KEY CONCEPTS 编辑入口**：修改 KEY CONCEPTS 段落需要手动编辑原始 markdown 文件

---

## 2. 设计目标

- **临时忽略**：用户在缺少/额外节点列表中可临时忽略某个条目，重新对齐后保留忽略状态
- **永久删除**：用户可选择从原文 KEY CONCEPTS 段落中永久删除某个条目
- **已忽略管理**：可查看已忽略的条目列表，并支持恢复
- **状态稳定**：忽略状态随对齐草稿持久化，刷新页面后保留

---

## 3. 数据模型

### 3.1 AlignmentDraft 扩展

现有 `alignmentDrafts` 的 value 类型扩展 `ignoredTerms` 字段：

```typescript
// knowledgeGraphStore.ts
alignmentDrafts: Map<string, {
  userText: string
  hasAligned: boolean
  result: GraphAlignmentResult | null
  ignoredTerms: string[]  // ← 新增
}>
```

### 3.2 现有类型无变更

`GraphAlignmentResult`、`AlignedNodePair`、Concept 类型保持不变。忽略逻辑在展示层过滤。

---

## 4. 组件设计

### 4.1 NodeRow 增强

```
┌─── NodeRow (现有样式) ──────────────────────────────┐
│ [●] KVCacheManager             已理解               │
│                                ⨯  🗑   ← 新增      │
└──────────────────────────────────────────────────────┘
```

每个 `NodeRow` 在 hover 时右侧显示两个 icon 按钮：

- **⨯（忽略）**：仅对该条目在当前对齐 session 中生效
  - 将该词条加入 `ignoredTerms[]`
  - 行移入"已忽略"折叠区
  - 重新计算对齐统计（matched/missing/extra 数量、nodeCoverage、掌握分）
  - 调用 `updateMastery()` 更新掌握分

- **🗑（删除）**：从原文中永久删除
  - 调用 `removeKeyConceptFromContent(conceptId, term)` 修改文档内容
  - 更新 `concept.content` 通过 `updateConceptContent()`
  - 自动持久化到服务器
  - 从对齐结果中移除该行
  - 重新计算对齐统计

**交互规则：**
- 两个按钮仅在 `node.status === 'missing'` 时显示（"仅原文有"的条目）
- `matched` 状态的条目只显示 ⨯ 忽略按钮（原文匹配成功的不应被删除）
- `extra` 状态的条目是用户自己写的，与 KEY CONCEPTS 无关，两个按钮都不显示

### 4.2 "已忽略"折叠区

在"原文有但你的描述中未出现的术语"（missing）区域下方新增：

```
▼ 已忽略 (2)
  ┌─── NodeRow ──────────────────────────────────────┐
  │ [●] Block Table              已忽略     [恢复]    │
  └───────────────────────────────────────────────────┘
  ┌─── NodeRow ──────────────────────────────────────┐
  │ [●] COW 机制                  已忽略     [恢复]    │
  └───────────────────────────────────────────────────┘
```

- 点击"恢复"将该术语从 `ignoredTerms` 中移除，行移回原 missing 区域
- 重新计算对齐统计
- 该区域在 `ignoredTerms.length === 0` 时隐藏

### 4.3 KEY CONCEPTS 段落编辑器

在底部新增"编辑 KEY CONCEPTS"入口（在"图对比概览"区域），点击展开一个内联编辑器，直接编辑文档中 KEY CONCEPTS 段落的原始文本：

```
[编辑 KEY CONCEPTS 段落]  ← 按钮

▼ KEY CONCEPTS 段落编辑器（点击展开）
┌─────────────────────────────────────────────┐
│ Block Table CacheBlock COW 映射 ...         │
│                                             │
└─────────────────────────────────────────────┘
[保存修改]
```

- 编辑器预填当前 KEY CONCEPTS 段落文本
- 保存后更新 `concept.content` 并重新执行对齐
- 这是永久删除的替代入口（批量编辑场景）

---

## 5. 核心函数

### 5.1 `removeKeyConceptFromContent()`

位置：`utils/alignment.ts`

```typescript
export function removeKeyConceptFromContent(
  content: string,
  termToRemove: string
): string | null
```

行为：
1. 用 `parseKeyConcepts()` 定位 KEY CONCEPTS 段落
2. 从段落文本中移除 `termToRemove`（精确匹配空格分隔的完整 term）
3. 返回修改后的完整 content
4. 如果段落中无该 term 或段落不存在，返回 null

### 5.2 `recomputeStats()`

在 `AlignmentPanel` 组件内，当 `ignoredTerms` 变化或被忽略条目被修改时，重新计算统计值并更新 mastery score。

---

## 6. 流程

### 6.1 临时忽略流程

```
用户 hover NodeRow → 点击 ⨯
  → term 加入 alignmentDrafts[concept.id].ignoredTerms[]
  → 该行移入"已忽略"折叠区
  → 重新过滤 result.nodes（排除 ignoredTerms 中的条目）
  → 重新计算 stats（nodeCoverage, matchedCount, missingCount等）
  → updateMastery() 更新掌握分
  → 对齐草稿自动持久化到 store
```

### 6.2 永久删除流程

```
用户 hover NodeRow → 点击 🗑
  → 确认对话框："确定从原文永久删除「{term}」？"
  → 确认 → removeKeyConceptFromContent(concept.content, term)
  → updateConceptContent(concept.id, newContent)
  → 重新执行 compareTexts(userText, newContent, ...)
  → 更新 result 和 stats
  → persistToServer() 持久化
```

### 6.3 恢复流程

```
用户在"已忽略"区点击某条的"恢复"
  → term 从 ignoredTerms[] 中移除
  → 该行从"已忽略"区移回原 missing 区域
  → 重新计算 stats
  → updateMastery()
```

### 6.4 重新对齐流程

```
用户点击"执行图对齐"
  → 调用 compareTexts()，传入原始 content（不含任何忽略过滤）
  → 在结果过滤阶段排除 alignmentDrafts[concept.id].ignoredTerms
  → 显示过滤后的结果
```

---

## 7. 边界情况

| 场景 | 处理 |
|------|------|
| 忽略后重新对齐 | `ignoredTerms` 保留，对齐结果中过滤掉被忽略项 |
| 从原文删除一个词后重新对齐 | 该词已从 KEY CONCEPTS 段落移除，自然不再出现 |
| 所有条目都被忽略/删除 | 显示"当前无 KEY CONCEPTS 条目"空态 |
| 切换到其他概念再切回 | `ignoredTerms` 随 `alignmentDrafts` 保留（按 concept.id 隔离） |
| 无 KEY CONCEPTS 段落 | 两个按钮 disable，tooltip: "文档中无 KEY CONCEPTS 段落" |
| 删除后 KEY CONCEPTS 段落为空 | 段落保留（空行），不删除段标记 |
| 删除某个 term 时原文中不存在 | 按钮 disable，或操作时静默失败 |
| 多个 term 有相同子串 | 精确匹配完整 term（空格分隔），不误删 |

---

## 8. 实现范围

### 文件变更清单

| 文件 | 变更 |
|------|------|
| `src/utils/alignment.ts` | 新增 `removeKeyConceptFromContent()` 导出函数 |
| `src/components/AlignmentPanel.tsx` | NodeRow 增强（hover 按钮）、新增"已忽略"折叠区、KEY CONCEPTS 编辑器入口、recomputeStats 函数 |
| `src/store/knowledgeGraphStore.ts` | `AlignmentDraft` 类型扩展 `ignoredTerms` 字段（隐式，通过 setAlignmentDraft 支持） |

### 不需变更的文件

| 文件 | 原因 |
|------|------|
| `src/types/index.ts` | 无类型修改必要，`ConceptElement` 与 KEY CONCEPTS 无关 |
| `src/store/knowledgeGraphStore.ts` | 现有 `alignmentDrafts` Map 已是 `any` 类型足够灵活 |
| `src/components/ConceptDetailPanel.tsx` | 不影响概念详情面板 |
| `src/components/DocumentViewer.tsx` | VIEW 模式显示文档内容，不涉及交互删除 |

---

## 9. 验收标准

- [ ] 在 missing 节点上 hover 时显示 ⨯ 和 🗑 按钮
- [ ] 点击 ⨯ 后该节点移入"已忽略"区域，统计数值重新计算
- [ ] 点击"恢复"后节点回到原区域，统计恢复
- [ ] 点击 🗑 后显示确认对话框
- [ ] 确认删除后，原文 KEY CONCEPTS 段落移除该词，对齐结果更新
- [ ] 切换到其他概念再切回，ignoredTerms 保留
- [ ] 无 KEY CONCEPTS 段落时按钮不可见/disable
- [ ] 所有统计数值（覆盖率、掌握分）在忽略/删除后保持合理
