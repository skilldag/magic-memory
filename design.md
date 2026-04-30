# 问题驱动推导学习 — 设计文档

> 记忆是充分思考的结果。所有设计都应让用户不得不思考，而非不得不记忆。

---

## 1. 核心理念

### 1.1 认知链条

```
过程描述 → 发现矛盾 → 形成问题 → 推导方案 → 揭晓概念
```

人理解一个陌生概念的自然顺序不是"查定义"，而是：

1. **感知过程** — 这个概念参与什么流程？在哪一步？
2. **明确角色** — 在这步中它负责做什么？
3. **发现问题** — 过程中有什么矛盾/瓶颈？
4. **推导方案** — 如果是你，你会怎么解决？
5. **揭晓概念** — 哦，这个方案就叫 Attention / PagedAttention / ...

概念不是知识的原子，而是一个问题的命名方案。

### 1.2 要素与概念的同构性

要素（element）和概念（concept）是同一个知识单元在不同粒度下的呈现。要素过大时可提升为独立概念，概念过小时可降级为要素。

```
上层概念：KVCacheManager
  ├─ 要素：Block Table 映射
  ├─ 要素：CacheBlock 管理        ← 在上一层是要素
  └─ 要素：内存分配策略
                         ↓ 放大粒度
                  独立概念：Block Table
                    ├─ 要素：逻辑块 → 物理块映射
                    ├─ 要素：引用计数
                    └─ 要素：COW 机制
```

### 1.3 验证不是回忆，是再造

用户理解了 Attention，不是因为他能列出 Q/K/V/softmax/scale 五个要素，而是因为他看到"模型需要关注不同的词"这个问题时，能推导出"需要权重 → 需要三元组 → 需要归一化"这条推理链。

---

## 2. 知识图谱交互

### 2.1 概念节点行为

| 手势 | 行为 | 说明 |
|------|------|------|
| 单击 | 选中概念 | 右侧面板展示概念摘要、要素、关联 |
| 双击 | 进入全屏画板 | 左侧切换到 ReactFlow 过程画板 |

单手是理解的概念：

- **单击 = 查看**（选中、浏览、关联）
- **双击 = 推导**（进入过程画板，自己梳理流程）

双击检测实现：监听 Cytoscape `tap` 事件，同一节点 400ms 内两次 tap 即为双击。

### 2.2 右侧面板结构

面板固定 420px (xl: 480px)，四个标签页：

| 标签 | 内容 | 触发条件 |
|------|------|----------|
| 🖱️ 梳理过程 | 提示"双击图谱节点进入画板" | 始终可用 |
| ⚖️ 对照验证 | ComparisonPanel | 需先提交过程梳理 |
| 🔗 探索关联 | 前置/后置/相关概念 | 始终可用 |
| 📖 查阅文档 | DocumentViewer | 始终可用 |

### 2.3 图谱节点状态

节点视觉编码（考虑引入）：

```
○ 灰暗     → 从未点击过
◐ 微光     → 梳理过但未完成
◑ 脉动光   → ⚡ 有差异待探索
● 稳定光   → 推导完成，已掌握
```

---

## 3. 过程画板（ProcessCanvas）

### 3.1 入口

双击知识图谱中的概念节点 → 左侧全屏切换为 ReactFlow 画板。

顶部显示当前概念名 + "过程梳理画板" + "← 返回图谱" 按钮。

### 3.2 画板结构

```
┌── 顶部工具栏 ──────────────────────────┐
│ ← 返回图谱   当前概念    过程梳理画板     │
├── 候选概念区 ───────────────────────────┤
│ 关联概念 (N) → 点击添加到画布            │
│ [VllmConfig] [Device] [Tensor] ...      │
│ [+ 自定义节点] ↘ 输入名称 → [添加]      │
├── ReactFlow 画布 ──────────────────────┤
│                                         │
│  绿色节点 = 已知前置概念                  │
│  蓝色节点 = 当前概念（高亮）              │
│  虚线节点 = 空缺（可填充）                │
│  自由拖拽 / 连线 / 缩放 / 平移            │
│                                         │
├── 底部工具栏 ───────────────────────────┤
│ [自动排列]          [提交梳理]  3 节点·2 连线│
└──────────────────────────────────────────┘
```

### 3.3 候选概念过滤

候选概念只展示与当前概念相关的（不展示全部 50+ 概念）：

- 同一过程链中的其他概念
- `depends_on` / `leads_to` / `related` 中的概念
- 同 level 或同 category 的概念

### 3.4 自定义节点

用户可创建任意文字节点（非预定义概念），用于表达自己的理解。

### 3.5 过程链的两种来源

| 来源 | 说明 | 示例 |
|------|------|------|
| 预定义 chain | 手动编写的过程链，精确描述 | 推理启动流程（5 步） |
| 动态生成链 | 从 depends_on → self → leads_to 自动组装 | 所有无预定义链的概念 |

动态生成链确保**每个概念点击后都有推导流程可梳理**。

### 3.6 提交与对照

用户点击"提交梳理"→ 将当前画布的节点布局作为 user_flow 提交 → 自动跳转到右侧面板的"对照验证"标签 → 显示 ComparisonPanel。

---

## 4. 对照验证（ComparisonPanel）

### 4.1 数据流

```
用户提交画布布局 → user_flow (nodeId 有序列表)
                  ↓
          generateReferenceFlow(concept, chains)
                  ↓
          diffFlows(user_flow, reference_steps)
                  ↓
          DiffItem[] (match / missing / extra)
                  ↓
          getGapConceptIds → 跳转到新概念
```

### 4.2 对照展示

```
 ┌──────────────┐
 │ 匹配  │ 遗漏 │ 多余 │
 │   3   │  2   │  0   │
 └──────────────┘
      推导覆盖率: 60%

 ┌──────────────────────────────┐
 │ ✓ VllmConfig    匹配         │
 │ ✓ Device        匹配         │
 │ ⚡ Tensor        遗漏 → [探索]│
 │ ✓ GpuAllocator  匹配         │
 │ ⚡ ModelLoader   遗漏 → [探索]│
 └──────────────────────────────┘
```

- ⚡ 遗漏项可点击 → 跳转到该概念的梳理画板
- 覆盖率 < 50% → 红色，建议回看
- 50-80% → 琥珀色，可巩固
- > 80% → 绿色，掌握良好

---

## 4.2 语义对齐（Semantic Alignment）

### 4.2.1 原理

将用户的理解文本与概念的文档正文各自建图，然后对比两张图的结构差异来定位知识缺口。整个管线不依赖 LLM，纯图算法驱动。

```
用户输入文本                        概念文档正文
     │                                   │
     ▼                                   ▼
   清理 markdown 工件 ─── scrub() ─── 清理 markdown 工件
     │                                   │
     ▼                                   ▼
   术语提取（中文短语 + 英文技术词）
     │                                   │
     ▼                                   ▼
   KG 概念匹配                            KG 概念匹配
   （标题/别名模糊匹配）                    （标题/别名模糊匹配）
     │                                   │
     ▼                                   ▼
   共现图 + 标签传播社区发现               共现图 + 标签传播社区发现
     │                                   │
     └───────────────┬───────────────────┘
                     ▼
              图结构对齐
           （节点集 + 边集对比）
                     │
             ┌───────┴───────┐
             ▼               ▼
          matched         missing
         （已理解）       （知识缺口）
```

### 4.2.2 管线步骤

| 步骤 | 算法 | 输出 |
|------|------|------|
| 文本清洗 | 去代码块、ASCII 图、emoji、链接 | 纯净文本 |
| 术语提取 | 标点分句 + 中文短语(n≥2) + 英文技术词 | 候选术语集 |
| KG 概念匹配 | 标题/别名模糊匹配（大小写不敏感） | 标注 KG 节点 |
| 共现图 | 句内共现建无向边 | Graph<Node, Edge> |
| 社区发现 | 标签传播（Label Propagation, 30 轮） | 概念社区分组 |
| 标签选择 | KG 概念 > 位置加权(Title×3/冒号×2.5/列表×2) > 度 | 每组一个知识标签 |
| 非概念过滤 | 疑问句 / 章节标题 / 碎片移除 | 干净知识点 |
| 图对齐 | 节点集 × 边集 ± 模糊匹配(charJaccard) | 共现/缺失/多余 |

### 4.2.3 四种算法对比

从 attention.md 和 kv-cache.md 的实验得出的结论：

| 算法 | 机制 | 适合 | 不适合 |
|------|------|------|--------|
| TextRank | PageRank 排序术语 | 长文关键词提取 | 短文档（噪声大） |
| 位置加权 | 标题×3、冒号前×2.5、列表×2 | 结构清晰的 markdown | 无结构的纯文本 |
| 社区发现 | 标签传播分组 + 度选标签 | 通用分组 | 章节标题污染 |
| StructuralRank | 社区分组 + KG/位置加权选标签 | **通用最佳** | Long-tail噪音 |

### 4.2.4 StructuralRank

核心洞察：**位置加权能准确识别"概念"，社区发现能正确合并"相关术语"**。

```
输入文本
    │
    ▼
[位置加权] ──→ 候选术语得分表（Query=5.5, 注意力机制=4.0, ...）
    │
    ▼
[社区发现] ──→ 术语分组（{Query, 你想查的词, Q (Query)}）
    │
    ▼
[标签选择] ──→ 每组选 KG 概念 > 位置加权最高分
    │
    ▼
知识点列表：[注意力机制, Q (Query), K (Key), V (Value), ...]
```

优先级规则：
1. **KG 概念**（图谱已注册的概念名）
2. **位置加权高分**（标题、冒号前、列表项中出现的术语）
3. **原标题**（兜底）

### 4.2.5 文件

| 文件 | 职责 |
|------|------|
| `utils/alignment.ts` | 所有图算法（extractTerms → buildConceptGraphFromText → structuralExtract → alignGraphs） |
| `components/AlignmentPanel.tsx` | 对齐 UI：文本输入、执行、结果展示 |
| `tests/alignment.test.ts` | 基本对齐测试 |
| `tests/compare-methods.test.ts` | 五种方法对比 |

---

## 5. 验证框架：用户如何证明理解

### 5.1 理解的多维度验证

一个人证明理解了一个概念，不是靠"回忆要素"——而是靠：

| 维度 | 验证方式 | 在 UI 中的体现 |
|------|----------|---------------|
| 要素提取 | 能否列出概念的关键要素 | 过程画板中填充空缺节点 |
| 结构理解 | 能否正确组织要素的层级/顺序 | 画板中的节点连线与排序 |
| 边界判断 | 能否区分"属于/不属于" | 候选概念选择（不选无关的） |
| 应用推导 | 能否在新场景中推演 | 场景题（未来迭代） |

### 5.2 提示递进（Graduated Prompting）

当用户想不起来时，逐级提供线索：

| 层级 | 提示 | 得分 |
|------|------|------|
| 0 | 无提示 | 100% |
| 1 | 给范围（"想想有哪几类"） | 85% |
| 2 | 给关键词（"model / cache / scheduler"） | 70% |
| 3 | 给关联概念（"GpuAllocator 和哪个配置相关"） | 55% |
| 4 | 展示答案 | 30% |

提示层级被记录到 `ReviewRecord`，在下一次复习时重点考察最薄弱的要素。

---

## 6. 数据模型

### 6.1 新增类型

```typescript
// 概念要素
interface ConceptElement {
  name: string
  description: string
  type: 'core_field' | 'design_pattern' | 'key_insight' | 'boundary' | 'relation'
  order: number
}

// 过程步骤
interface ProcessStep {
  id: string
  label: string
  description: string
  question: string
  hint: string
  leads_to_type: 'element' | 'concept'
  leads_to_id?: string
  is_core: boolean
}

// 过程链
interface ProcessChain {
  id: string
  name: string
  steps: ProcessStep[]
}

// 梳理状态
interface ProcessState {
  user_flow: string[]
  llm_flow: string[]
  gaps: string[]
  filled: boolean
  compared: boolean
}
```

### 6.2 现有类型扩展

```typescript
// Concept 新增
process?: {
  chain_id: string
  step_index: number
  role: string
}
elements?: ConceptElement[]

// ReviewRecord 新增
process_state?: ProcessState
```

---

## 7. 技术实现

### 7.1 依赖

- **@xyflow/react** v12: 过程画板（拖拽流程图）
- **Cytoscape** (已有): 知识图谱（概念网络）
- **Zustand** (已有): 状态管理

### 7.2 关键文件

| 文件 | 职责 |
|------|------|
| `components/KnowledgeGraph.tsx` | Cytoscape 图渲染 + 单击/双击事件 |
| `components/KnowledgeGraphView.tsx` | 布局容器（左画板/右面板）+ processMode |
| `components/ConceptDetailPanel.tsx` | 右侧面板四个标签 + 对照 |
| `components/ProcessCanvas.tsx` | ReactFlow 全屏画板 |
| `components/ComparisonPanel.tsx` | 用户 vs 参考流程对照 |
| `utils/processComparison.ts` | generateReferenceFlow, diffFlows, generateGenericChain |
| `types/index.ts` | ConceptElement, ProcessStep, ProcessChain, ProcessState |
| `utils/alignment.ts` | 语义对齐图算法管线 |

### 7.3 过程链的两种生成方式

```typescript
// 预定义链（来自 mock 数据）
mockProcessChains: [
  { id: 'inference-startup', name: '推理启动流程', steps: [...] }
]

// 动态生成链（任意概念兜底）
generateGenericChain(conceptId, concepts)
// 输出: [depends_on...] → [self] → [leads_to...]
```

---

## 8. 不做的事

- 不做新手引导/教学提示（用户自己发现双击进入画板）
- 不保留全局 explore/review 模式切换
- 不依赖双指缩放等移动端手势（桌面优先）
- 不实现传统自评分数（2/3/5 自评已移除）
- 不改动现有的三个 hover 按钮（AI/?/手动）

---

## 9. 迭代方向

- **要素验证模式**：在 ProcessCanvas 中嵌入要素回忆/填空
- **场景推导题**：给一个具体场景，用户在画板上推演
- **多人梳理对比**：多个用户的推导结果对比
- **AI 辅助补全**：用户拖入部分节点后，AI 建议补全
- **提示层级持久化**：记录用户在每个要素上消耗的提示层级
