# 知识图谱索引生成设计

> 自动/手动双模式的知识图谱索引生成，取代硬编码 mock 数据。

---

## 核心理念

图谱索引不是预先生成的静态数据，而是通过用户操作逐步生长的结构。文档是知识载体，索引是图结构，两者分离但通过 path 关联。

---

## 1. 启动状态

首次打开时，localStorage 中没有索引数据，图谱为空。提示用户设置 docs 目录路径并选择模式。

```
┌──────────────────────────────────────┐
│                                      │
│       还没有知识图谱索引               │
│                                      │
│  文档目录路径: [__________________]   │
│                                      │
│  [自动扫描文档建索引]                  │
│  [手动添加概念]                        │
│                                      │
└──────────────────────────────────────┘
```

用户输入本地路径（如 `/Users/meetai/source/magic-memory/docs`）后选择模式。

---

## 2. 自动模式（Auto Scan）

### 流程

```
用户输入 docs 路径
  ↓
POST /api/scan-docs { path }
  ↓
后端递归遍历目录下所有 .md 文件
  ↓
对每个文件：
  有 frontmatter → 直接解析
                   (id, title, depends_on, leads_to, ...)
  无 frontmatter → LLM 分析文件内容
                   生成概念属性 + 关系
                   自动补写 frontmatter 到原文件
  ↓
合并构建 ConceptIndex[] + 推导边
  ↓
返回 { concepts, edges, path }
  ↓
前端存入 localStorage → 渲染图谱
```

### frontmatter 格式

```yaml
---
id: "0"
title: "VllmConfig - 配置中心"
alias: ["配置", "Config"]
level: 1
category: Foundation
depends_on: []
leads_to: ["1", "2", "10"]
related: ["7"]
problem: "vLLM 如何统一管理所有配置？"
gap_anticipate: "配置为什么需要分 model/cache/scheduler 三类？"
---
```

### LLM 无 frontmatter 兜底

对没有 frontmatter 的 .md 文件，将内容发给 LLM，要求生成：

```json
{
  "title": "推断的标题",
  "alias": ["别名"],
  "level": 1,
  "category": "推断分类",
  "depends_on_titles": ["前置概念标题"],
  "leads_to_titles": ["引出概念标题"],
  "related_titles": [],
  "problem": "核心问题",
  "gap_anticipate": "常见认知缺口"
}
```

前端将标题匹配到已有概念 ID。LLM 返回的结果同时作为 frontmatter 写回到原 .md 文件，下次扫描直接读取。

---

## 3. 手动模式（Manual Add）

### 流程

```
用户输入概念名
  ↓
前端收集上下文 → 已有概念列表（title + problem）
  ↓
POST /api/manual-add-concept
  {
    title: "用户输入的概念名",
    context: [
      { id: "0", title: "VllmConfig", problem: "..." },
      ...
    ]
  }
  ↓
LLM 分析，返回结构化数据:
  {
    title: "完整标题",
    problem: "核心问题",
    gap_anticipate: "认知缺口",
    depends_on_titles: ["前置概念1", "前置概念2"],
    leads_to_titles: ["引出概念"],
    related_titles: [],
    elements: [
      { name: "要素名", description: "描述", type: "core_field", order: 1 }
    ],
    content: "# 标题\n\n## 问题\n..."
  }
  ↓
前端将标题匹配到已有概念 ID
  ↓
分配新 ID → 创建 ConceptIndex + Edges
  ↓
写入文档到 {docs_path}/{category}/{id}-{title}.md
  ↓
更新 localStorage → 图谱刷新
```

### 匹配策略

LLM 返回 `depends_on_titles` 时使用标题文字匹配，模糊匹配到已有概念 ID：

```typescript
function matchTitleToId(title: string, concepts: ConceptIndex[]): string | null {
  const exact = concepts.find(c => c.title.includes(title) || title.includes(c.title))
  if (exact) return exact.id
  // 模糊匹配：去掉空格、特殊字符后比较
  const normalized = title.replace(/[\s\-_]/g, '').toLowerCase()
  return concepts.find(c =>
    c.title.replace(/[\s\-_]/g, '').toLowerCase().includes(normalized)
  )?.id ?? null
}
```

### 文档缺失处理

概念有 `path` 但文件不存在时，文档区显示空状态 + [请求 LLM 生成] 按钮。LLM 生成完整文档正文后写入 `.md` 文件。

---

## 4. 数据模型

### 索引层（localStorage）

```typescript
interface ConceptIndex {
  id: string
  title: string
  alias?: string[]
  level: number
  category: string
  problem?: string
  gap_anticipate?: string
  depends_on: string[]
  leads_to: string[]
  related: string[]
  elements?: ConceptElement[]
  process?: { chain_id: string; step_index: number; role: string }
  path: string
  tags: string[]
}

// 边不从属数据存储，运行时从 depends_on/leads_to/related 推导
interface ConceptEdge {
  id: string
  source: string
  target: string
  type: 'depends_on' | 'leads_to' | 'related'
}
```

### 文档层（文件系统）

```
{docs_path}/
├── Foundation/
│   ├── 00-vllm-config.md
│   ├── 01-device.md
│   └── ...
├── Model/
│   ├── 10-model-registry.md
│   └── attention/
│       └── 16-paged-attention.md
├── user/              ← 手动添加的概念
│   └── 52-flash-attention.md
└── ...
```

### 持久化

| 数据 | 存储位置 | 同步方式 |
|------|---------|---------|
| ConceptIndex[] | localStorage | Zustand persist |
| Edges | 运行时推导 | 不持久化 |
| 文档正文 | 文件系统 | docLoader.fetch 按需加载 |

---

## 5. 依赖移除

- 移除 `mockGraphData.ts` 作为默认数据源
- `loadGraph()` 检查 localStorage → 有则加载，无则显示空白
- `mockConcepts`、`mockEdges`、`mockProcessChains`、`mockElements` 不再使用
- `conceptParser.ts` 保留并增强，用于自动扫描时的 frontmatter 解析

---

## 6. 后端 API 新增

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/scan-docs` | POST | 扫描目录，解析 frontmatter，返回 concepts + edges |
| `/api/manual-add-concept` | POST | LLM 生成概念结构，返回 concept data |
| `/api/generate-doc` | POST | LLM 生成完整文档正文 |
| `/api/write-doc` | POST | 写入 .md 文件到指定路径 |

---

## 7. 不做的

- 不预置任何 mock 概念数据
- 不区分内置概念和用户概念
- 不做索引版本管理（统一存 localStorage，重置靠手动清空）
