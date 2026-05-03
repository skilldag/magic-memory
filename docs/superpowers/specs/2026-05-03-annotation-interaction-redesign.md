# Annotation Interaction Redesign

> 选中文本后添加评论功能的完整交互重设计

## 概述

重新设计文档阅读器中「选中文本 → 添加标注」的交互流程，从当前的底部浮动栏改为选区附近 Popover + 独立弹窗输入，并增加文本高亮标记和预览功能。

## 现状问题

1. **评论内容为空** — 选中文本后点"添加评论"直接创建 `content: ''` 的空 annotation，无输入对话框
2. **交互笨重** — 底部浮动栏遮挡文档内容，且显示选中文本全文，冗余
3. **无视觉反馈** — 已有 annotation 在文档中不可见，无法定位
4. **功能分散** — QuestionDialog 独立于评论流程，体验不一致

## 设计方案

### 1. SelectionPopover（选区 Popover）

选中文本后，在选区上方弹出紧凑菜单。

**触发：** `onMouseUp` 检测到非空选择 → 弹出 Popover

**定位：** 
- 基于 `getBoundingClientRect()` 定位到选区上方居中
- 边界检测，空间不足时自动下移
- 小箭头指向选区

**内容布局：**

```
┌─────────────────────────────┐
│ 💬 评论  ❓ 提问  ✏️ 纠正    │
│             🔗 概念提升      │
│                      [✕] 关闭 │
└─────────────────────────────┘
```

- 左侧：三种标注类型入口按钮
- 右侧：概念提升（次要操作）
- 关闭按钮 / 点击外部关闭 / Escape 关闭
- 保持紧凑，不重复显示选中文本

### 2. AnnotationDialog（统一标注弹窗）

所有标注类型共用同一个弹窗，取代现有的 QuestionDialog。

**布局：**

```
┌─────────────────────────────────┐
│  添加注释                    ✕   │
├─────────────────────────────────┤
│  选中的文本:                     │
│ ┌─────────────────────────────┐ │
│ │ "..."（只读展示）            │ │
│ └─────────────────────────────┘ │
│                                 │
│  类型: ○ 评论 ○ 提问 ○ 建议 ○ 纠正 │
│                                 │
│  内容:                           │
│ ┌─────────────────────────────┐ │
│ │  （多行 textarea）           │ │
│ └─────────────────────────────┘ │
│                                 │
│  [🤖 AI 解答] [📌 提升为概念]     │
│                                 │
│          [取消]    [提交]        │
└─────────────────────────────────┘
```

**行为：**
- 标题和按钮文案根据选中类型动态变化
- 选中文本只读展示，不可编辑
- 类型可在弹窗中切换，切换时更新标题
- 「AI 解答」仅在提问类型时可用
- 「提升为概念」在所有类型均可用
- 提交时通过 `annotationStore.addAnnotation()` 创建含实际内容的 annotation

### 3. Annotation Highlights（文本高亮）

渲染文档时，根据 annotations 的 `position.start/end` 在 HTML 中插入 `<mark>` 标签。

**颜色映射：**

| 类型 | 背景色 | 装饰 |
|------|--------|------|
| comment | `#dbeafe` (blue-100) | 下划线 |
| question | `#e9d5ff` (purple-100) | 波浪下划线 |
| suggestion | `#dcfce7` (green-100) | 虚线下划线 |
| correction | `#fee2e2` (red-100) | 实线下划线 |
| resolved/closed | 对应颜色半透明 | 降低不透明度 |

**实现：** DocumentViewer 中使用 `useMemo` 对 `htmlContent` 做后处理，根据 annotation 偏移在纯文本对应位置插入 `<mark data-ann-id="...">`。

### 4. AnnotationPreview（高亮预览浮层）

点击文档中的高亮文本时弹出小型预览。

**布局：**

```
┌──────────────────────────────────┐
│ 💬 评论     User · 3天前   [已解决] │
│──────────────────────────────────│
│ 内容预览（最多3行，超出截断）       │
│                                  │
│  ↩ 2 条回复                       │
│                      [查看详情 →] │
└──────────────────────────────────┘
```

**交互：**
- 点击高亮 → Preview 弹出在标记位置附近
- 显示：类型图标、作者、时间、状态标签
- 内容最多 3 行
- 「查看详情」→ `selectAnnotation(ann)` + 打开 AnnotationPanel 并滚动到对应位置
- 点击外部关闭

### 5. AnnotationPanel 联动

- Preview 点击「查看详情」→ 右侧 AnnotationPanel 打开并定位到对应条目
- AnnotationPanel 中 `selectedAnnotation` 变化 → 文档中对应高亮文本闪烁提示
- 双向联动确保用户不会迷失

## 技术方案

### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/components/SelectionPopover.tsx` | **新建** | 选区 Popover 组件 |
| `src/components/AnnotationDialog.tsx` | **新建** | 统一标注输入弹窗 |
| `src/components/AnnotationPreview.tsx` | **新建** | 高亮预览浮层 |
| `src/components/DocumentViewer.tsx` | **修改** | 移除底部栏，集成 Popover + 高亮 + 预览 |
| `src/components/AnnotationPanel.tsx` | **微调** | 选中联动 |
| `src/components/QuestionDialog.tsx` | **删除** | 由 AnnotationDialog 替代 |
| `src/index.css` | **修改** | 添加高亮标记样式 |

### 数据流

```
选中文本 → SelectionPopover → 点击按钮 → AnnotationDialog
  → 填写内容 → annotationStore.addAnnotation()
  → DocumentViewer 重新渲染高亮
  → 新的 <mark> 标签出现在文档中

点击 <mark> 标签 → AnnotationPreview → "查看详情"
  → selectAnnotation() + AnnotationPanel 打开
```

### 状态管理

- `annotationStore.ts` — 现有 API 完全满足，无需新增方法
- `DocumentViewer` 中新增局部状态：
  - `popoverPosition: { x, y } | null`
  - `previewAnnotation: Annotation | null`
  - `previewPosition: { x, y } | null`

### 样式

所有高亮样式通过 CSS class 控制：

```css
.ann-highlight {
  cursor: pointer;
  border-radius: 2px;
  transition: background-color 0.15s;
}
.ann-highlight:hover { opacity: 0.8; }
.ann-comment { background-color: #dbeafe; text-decoration: underline; }
.ann-question { background-color: #e9d5ff; text-decoration: wavy underline; }
.ann-suggestion { background-color: #dcfce7; text-decoration: dashed underline; }
.ann-correction { background-color: #fee2e2; text-decoration: solid underline; }
.ann-resolved { opacity: 0.5; }
```

## 不在此范围

- 后端 API 改动 — annotation 仍然通过 Zustand persist 持久化
- Types 定义改动 — Annotation 类型已满足需求
- 知识图谱概念提升流程改动
- 批量操作或筛选功能

## 验收标准

1. 选中文本 → Popover 出现在选区附近，不遮挡关键内容
2. 点击"评论" → 弹出 AnnotationDialog → 填写内容 → 提交 → annotation 创建成功 → 文档中立即出现高亮
3. 已有 annotation 在文档加载时正确高亮显示
4. 点击高亮文本 → 预览浮层弹出 → "查看详情" → 右侧面板定位到对应条目
5. 提问类型可选择 AI 解答（调用 /api/ask-question）
6. 所有类型可选择提升为概念
7. Escape / 点击外部关闭所有浮层和弹窗
