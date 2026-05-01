# Magic Memory

> 基于数字锚点与问题驱动推导的 vLLM 架构学习系统

---

## 项目概述

Magic Memory 是一个专为 **vLLM 架构学习**设计的记忆与理解系统，结合了两种互补的学习方法：

| 方法 | 解决什么问题 | 文件 |
|------|-------------|------|
| **数字锚点记忆法** | 快速记忆 50+ vLLM 概念的名称和位置 | [memory.md](./memory.md) |
| **问题驱动推导法** | 深入理解概念的设计原因和内部原理 | [design.md](./design.md) |

---

## 核心功能

### 🧠 数字锚点记忆系统

将 50 个 vLLM 核心概念对应到 0-50 的数字锚点上，通过"数字 → 锚点图像 → 概念"的三步联想进行记忆。

- **三级难度递进**：Level 1 (0-9 基础) → Level 2 (10-29 核心) → Level 3 (30-50 高级)
- **概念分类组织**：Foundation、Model、Attention、Performance、Scheduling、Serving、Infrastructure
- **完整数据流链路**：从 API 请求到 Token 输出的完整路径记忆
- **自测验证**：每个 Level 配套自测题，支持数字→概念、概念→数字双向回忆

### 🔍 问题驱动推导学习

通过"过程描述 → 发现矛盾 → 形成问题 → 推导方案 → 揭晓概念"的认知链条深入理解。

- **知识图谱交互**：50+ 概念节点的关系网络，支持单击查看、双击进入过程画板
- **过程画板 (ProcessCanvas)**：使用 ReactFlow 拖拽搭建概念推导流程
- **对照验证 (ComparisonPanel)**：用户推导结果 vs 参考流程自动对比，定位知识缺口
- **渐进式提示**：回忆困难时逐级提供线索，记录薄弱环节

### 📚 Web 标注阅读器

基于 Bun + React 的 Web 应用，用于浏览和标注 vLLM 知识文档。

- **文档浏览**：支持查看、搜索、筛选 Markdown 文档
- **注释系统**：支持评论、问题、建议、纠正四种注释类型
- **导入导出**：JSON / Markdown / HTML 格式

---

## 项目结构

```
magic-memory/
├── memory.md                # 数字锚点记忆法（核心记忆参考）
├── design.md                # 问题驱动推导设计文档
├── docs/                    # vLLM 概念文档库（按分类组织）
│   ├── Foundation/          # Level 1: 基础设施
│   ├── Model/               # Level 2: 模型执行
│   │   └── attention/       #   └─ PagedAttention 系列
│   ├── Performance/         #   └─ 性能优化
│   ├── Scheduling/          # Level 3: 调度
│   ├── Serving/             #   └─ 服务化
│   ├── Optimization/        #   └─ 高级优化
│   ├── Infrastructure/      #   └─ 基础设施
│   └── Advanced/            #   └─ 高级特性
├── magic-memory/            # Web 标注阅读器
│   ├── src/                 # React 前端源码
│   ├── server.ts            # Bun 后端服务
│   └── start.sh             # 一键启动脚本
└── openspec/                # OpenSpec 变更管理
```

### 文档分类体系

| 分类 | 覆盖概念 | Level |
|------|---------|-------|
| Foundation | VllmConfig, Device, Tensor, Logger, GpuAllocator, Error Handling, Init, KV Cache | Level 1 (0-9) |
| Model | ModelRegistry, ModelLoader, ModelRunner, Embedding, Transformer, Sampler, Token, Forward Pass | Level 2 (10-29) |
| Attention | PagedAttention, Block Table, CacheBlock, KVCacheManager | Level 2 (16-19) |
| Performance | GPU Memory Pool, FlashAttention, Quantization | Level 2 (26-28) |
| Scheduling | Continuous Batching, Scheduler, Prefill/Decode, Prefix Caching, Request Queue | Level 3 (34-39) |
| Serving | vllm-engine, Engine API, vllm-serving, OpenAI API, gRPC, WebSocket | Level 3 (40-45) |
| Advanced | Speculative Decoding, Draft Token, Verifier, N-gram | Level 3 (30-33) |
| Infrastructure | GPU Driver, Distributed | Level 3 (47, 50) |

---

## 快速开始

### 使用记忆系统

```bash
# 直接从记忆参考开始
open memory.md

# 按 Level 渐进学习
# Level 1 → docs/Foundation/
# Level 2 → docs/Model/ + docs/Performance/
# Level 3 → docs/Scheduling/ + docs/Serving/

# 自测验证
# docs/Level1-自测验证.md
# docs/Level2-自测验证.md
# docs/Level3-自测验证.md
```

### 启动 Web 标注阅读器

```bash
cd magic-memory
./start.sh
```

访问 http://localhost:3000

---

## 学习路径建议

```
第1周: Level 1 (0-9) 基础设施
  → 记忆锚点 → 阅读 Foundation 文档 → 自测验证

第2周: Level 2 (10-29) 模型执行
  → Model 加载流程 → PagedAttention → Sampler → Forward
  
第3周: Level 3 (30-50) 高级特性
  → Speculative Decoding → Scheduling → Serving
  
第4周: 整合复习
  → 默写数据流路径 → 概念关联图 → 场景实战
```

---

## 相关文档

| 文档 | 说明 |
|------|------|
| [memory.md](./memory.md) | 数字锚点记忆法 - 快速记忆 50+ 概念 |
| [design.md](./design.md) | 问题驱动推导 - 深入理解设计原理 |
| [知识体系化框架](./docs/知识体系化框架.md) | 概念之间的关系和组织方式 |
| [vLLM概念推演过程](./docs/vLLM概念推演过程.md) | 从问题到概念的完整推演 |
| [定期复盘机制](./docs/定期复盘机制.md) | 间隔复习和巩固计划 |

---

## 技术栈

- **Web 前端**: React 19 + TypeScript + Vite + Tailwind CSS 4
- **状态管理**: Zustand
- **流程图**: ReactFlow
- **后端**: Bun + TypeScript
- **文档**: Markdown
