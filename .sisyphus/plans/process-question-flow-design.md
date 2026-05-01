# 过程画板问题驱动推导 — 设计文档

> 基于 @design.md 的 梳理过程 功能重设计

## 1. 问题

当前「梳理过程」标签页只提示"双击图谱节点进入画板"，ProcessCanvas 展示水平排列的 known → current → gap 节点，用户通过拖拽排列来梳理流程。但这种方式缺乏问题驱动——用户只是排列节点，而非真正推导概念。

## 2. 设计目标

- ProcessCanvas 成为**问题驱动推导画板**
- 每个推导步骤展示其**问题**，用户通过回答问题来推导概念
- 不提供候选列表、不 AI 推荐——完全由用户自己推导
- 右侧面板联动展示概念文档（要素 + 专业文本）

## 3. 布局：垂直卡片栈

ReactFlow 画布从水平流改为**垂直排列的推导单元卡片栈**。

每个推导单元：

```
┌─ 推导单元 ──────────────────────────┐
│  配置读取                            │  ← process label
│  Engine 启动前读取 VllmConfig...      │  ← description
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│  💡 vLLM 启动时，第一个需要的是什么？ │  ← question
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│  ⬇                                  │
│  ┌─ 概念 ──────────────────────┐    │
│  │ [点击输入你的答案...]        │    │  ← gap (空缺)
│  └────────────────────────────┘    │
└────────────────────────────────────┘
```

## 4. 交互

| 交互 | 行为 |
|------|------|
| 点击空缺 | 空缺区域变为内联 `<input>`，自动聚焦 |
| 输入回车 | 文本保存，空缺变为「用户填充节点」(紫色) |
| 双击用户节点 | 可重新编辑 |
| Esc | 取消编辑，恢复空缺 |
| 拖拽 | 保持 ReactFlow 拖拽能力 |
| 自动排列 | 恢复垂直排列布局 |

## 5. 节点类型

| 类型 | 颜色 | 说明 |
|------|------|------|
| 步骤节点 (stepNode) | 灰色/蓝色 | 显示 process label + description + question |
| 空缺节点 (gapNode) | 黄色虚线 | 显示 "?" + "点击输入答案" |
| 已填节点 (filledNode) | 紫色 | 用户输入的文本，可双击重新编辑 |
| 已知概念 (conceptNode) | 绿色 | 前置已知概念（depends_on），直接展示 |

## 6. 右侧面板联动

用户在 ProcessCanvas 模式下时：
- 右侧面板保持显示 ConceptDetailPanel
- 当用户点击空缺开始编辑时，右侧「查阅文档」标签定位到该步骤对应的概念文档
- 如该步骤无对应概念（纯用户自定义），显示空状态

## 7. 提交与对照

用户点击「提交梳理」→ 收集所有已填概念文本 → 与参考流程对比：

| 情况 | 对照结果 |
|------|---------|
| 用户输入匹配正确概念名 | ✅ match |
| 用户输入了不同概念名 | ⚡ mismatch（显示用户文本 vs 正确概念） |
| 用户未填 | 标记为 missing |

## 8. 实现范围

本次实现：

### ProcessCanvas 改造（核心）
- 新增 `stepNode` 自定义节点（process label + description + question）
- 改造 `gapNode` 为可点击编辑（click → inline input）
- 新增 `filledNode`（用户已填概念，紫色）
- 布局从水平改为垂直排列
- 节点生成逻辑从 `initialNodesWithResize` 改为垂直卡片布局

### 右侧面板增强
- 查阅文档 tab 展示 ConceptElements（按 type 分组）
- 要素卡片：name + description + type badge

### 非本次范围
- 对照验证 ComparisonPanel 的增强（mismatch 对比）
- 在线 AI 辅助
- 画板缩放/平移优化

## 9. 关键文件变更

| 文件 | 变更 |
|------|------|
| `components/ProcessCanvas.tsx` | 新增 stepNode 类型，改造 gapNode 为可编辑，垂直布局 |
| `components/ConceptDetailPanel.tsx` | 查阅文档 tab 增加 ConceptElements 展示 |
| `types/index.ts` | 无变更（已有 ConceptElement, ProcessStep 类型） |
