# Concept 索引持久化与文档分离设计

> 将 mock 硬编码数据转换为可持久化的图关系索引节点，指向按 category 组织的文档文件。

---

## 问题

1. **Concept 过重**：`content` 字段内嵌完整 Markdown 正文，与图索引数据耦合
2. **无持久化**：用户新增的概念刷新丢失，只有 reviewRecords 和 annotations 被持久化
3. **文档无结构**：docs/ 按 level-1/2/3 编号组织，不反映概念的内在聚合（category / 子概念）
4. **数据源单一**：所有概念来自 mockGraphData.ts 硬编码，无法动态更新

---

## 设计

### 1. Concept 瘦身

移除 `content` 字段，Concept 只保留图关系索引字段。正文内容通过 `path` 指向外部文件按需加载。

```typescript
interface Concept {
  // 身份标识
  id: string
  title: string
  alias?: string[]
  level: number        // 1 | 2 | 3
  category: string     // Foundation | Model | Performance | ...

  // 问题驱动（教学用）
  problem?: string
  gap_anticipate?: string

  // 图关系索引
  depends_on: string[]
  leads_to: string[]
  related: string[]
  elements?: ConceptElement[]
  process?: { chain_id: string; step_index: number; role: string }

  // 层级（放大/缩小同构性）
  hierarchy?: { parentId: string | null; level: number; order: number }

  // 文档引用（唯一指向正文）
  path: string

  // 元数据
  tags: string[]
  lastModified: Date
  metadata?: { author?: string; version?: string; status?: 'draft' | 'review' | 'approved' }
}
```

### 2. 持久化方案

`knowledgeGraphStore` 的 Zustand persist 加入 concepts 和 edges。

```typescript
// store/knowledgeGraphStore.ts
partialize: (state) => ({
  concepts: state.concepts,
  edges: state.edges,
  reviewRecords: Array.from(state.reviewRecords.entries()),
  annotations: state.annotations,
})

merge: (persisted, current) => ({
  ...current,
  concepts: persisted?.concepts ?? current.concepts,
  edges: persisted?.edges ?? current.edges,
  reviewRecords: new Map(persisted?.reviewRecords || []),
  annotations: persisted?.annotations ?? [],
})
```

加载策略：
- 首次打开 → 从 mockGraphData 初始化，写入 localStorage
- 后续打开 → 直接读取 localStorage，不覆盖
- 手动重置 → 通过 "重置" 按钮清除 localStorage 重新加载

### 3. 文档目录结构

按 category 组织，子概念放入子目录。

```
docs/
├── Foundation/
│   ├── 00-vllm-config.md
│   ├── 01-device.md
│   ├── 02-tensor.md
│   ├── 03-logger.md
│   ├── 04-vllm-core.md
│   ├── 05-gpu-allocator.md
│   ├── 06-error-handling.md
│   ├── 07-init.md
│   ├── 08-foundation-layer.md
│   └── 09-kv-cache.md
├── Model/
│   ├── 10-model-registry.md
│   ├── 11-model-loader.md
│   ├── 12-model.md
│   ├── 13-model-runner.md
│   ├── 14-embedding.md
│   ├── 15-transformer-layers.md
│   ├── 20-sampler.md
│   ├── 21-sampling-params.md
│   ├── 22-logits.md
│   ├── 23-token.md
│   ├── 24-decode-step.md
│   ├── 25-forward-pass.md
│   ├── 29-weights-loading.md
│   ├── attention/                    ← 子概念聚合
│   │   ├── 16-paged-attention.md
│   │   ├── 17-block-table.md
│   │   ├── 18-cache-block.md
│   │   └── 19-kv-cache-manager.md
│   └── ...
├── Performance/
│   ├── 26-gpu-memory-pool.md
│   ├── 27-flash-attention.md
│   └── 28-quantization.md
├── Scheduling/
│   ├── 34-continuous-batching.md
│   ├── 35-scheduler.md
│   ├── 36-prefill.md
│   ├── 37-decode.md
│   └── 39-request-queue.md
├── Serving/
│   ├── 40-vllm-engine.md
│   ├── 41-engine-api.md
│   ├── 42-vllm-serving.md
│   ├── 43-openai-api.md
│   ├── 44-grpc.md
│   └── 45-websocket.md
├── Advanced/
│   ├── 30-speculative-decoding.md
│   ├── 31-draft-token.md
│   ├── 32-verifier.md
│   ├── 33-n-gram-proposer.md
│   └── 46-multi-lora.md
├── Optimization/
│   ├── 38-prefix-caching.md
│   ├── 48-prefix-lookup.md
│   └── 49-cache-eviction.md
└── Infrastructure/
    ├── 47-gpu-driver.md
    └── 50-distributed.md
```

子概念判定：`Concept.hierarchy.parentId !== null` → 放入父概念 category 下的子目录。
用户自定义概念：存入 `docs/user/` 目录，后续可考虑按 category 归类。

### 4. 文档加载链路

```
ConceptDetailPanel / DocumentViewer
  → concept.path → docLoader.fetch()
  → 成功: 渲染 Markdown 正文
  → 失败: 显示空白 + [请求 LLM 生成] 按钮
  → LLM 生成后: 写回到对应 path
```

docLoader 逻辑不变：`getDocUrl(path)` 将 `./docs/...` 转为 `/docs/...` 进行 fetch。

### 5. 不做的

- 内置概念版本号（统一存，重置靠手动）
- docs/ 文件变更自动检测
- 服务端路径切换（保持前端 docLoader fetch 模式）

---

## 数据迁移

1. 从 Concept 类型移除 `content` 字段
2. 调整 mockGraphData.ts 中已有概念的 path 指向新目录（如 `./docs/Foundation/00-vllm-config.md`）
3. 调整 `knowledgeGraphStore.partialize/merge`
4. 调整 `ConceptDetailPanel` / `DocumentViewer` 从 path 加载而非 content
5. 移动 docs/ 目录下的 .md 文件到新结构
