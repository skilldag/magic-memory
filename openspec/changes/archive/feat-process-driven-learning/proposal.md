# Proposal: 问题驱动推导学习流程

## Problem

当前 UI 有两个全局模式：「探索」和「学习」，但真实学习过程不是二分的。

**当前问题明细：**

1. **被动接收** — 点击概念节点后直接展示文档内容，用户处于阅读状态而非思考状态
2. **回忆型验证** — quiz 模式问"用一句话回答解决了什么问题"，本质是要求回忆而非推导
3. **脱离过程** — 概念被孤立展示，用户看不到它在推理流程中的位置和角色
4. **自评不准确** — 复习质量靠用户自评 2/3/5，没有客观的验证手段

核心矛盾：**"记忆是充分思考的结果"——所有设计都应让用户不得不思考，而非不得不记忆。**

## Solution

### 设计原则

1. **入口是问题，不是名字** — 用户看到概念前先看到它要解决的问题
2. **过程先于概念** — 概念是流程中的一步，理解流程才能理解概念为什么存在
3. **空缺驱动参与** — 不完整的流程骨架本身就是触发（affordance），不需要文字引导
4. **推导即学习** — 用户自己推导要素，再与 LLM 版本比对，"差异"即学习点
5. **点亮而非判错** — 差异项用发光吸引注意，而非用红色标注错误

### 核心交互流程

```
点击节点 → 展示过程骨架（有空缺）
              ↓
        用户拖拽/填写空缺
              ↓
        提交推导 → LLM 生成对照版本
              ↓
        比对 → 差异项"发光"
              ↓
        点击差异项 → 探索新概念/补漏
              ↓
        概念从"暗"变"亮"
```

### 详细的五阶段

#### 阶段 0：触发（点击节点）

用户在图谱上点击任意概念节点 → 不展示文档，展示**该概念参与的过程骨架**。

骨架是一个有明确空缺的流程图：

```
推理启动流程：

  [VllmConfig] → [Device] → [____] → [GpuAllocator] → [____]
                               ↑                         ↑
                            空缺 1                     空缺 2
```

空缺处等待填充。下方提供候选概念池供拖拽：

```
可拖拽概念：
  ⚪ Tensor    ⚪ Scheduler    ⚪ Sampler    ⚪ KVCache
```

或用户自行输入自己的推导。

**设计依据**：空缺是本能级别的触发——人天生想补全不完整的图案。不需要文字说明。

#### 阶段 1：过程梳理（用户推导）

用户拖拽或填写自己的流程版本。系统不做对错判断，只记录。

每一步用户完成填充后，可选"添加步骤说明"（用户自己的理解）。

```
你梳理的流程：
① VllmConfig → 配置读取
② Device → 确定运行设备       ← 你写的说明："决定在 GPU 还是 CPU 上跑"
③ Tensor → 准备数据容器       ← 你拖进去的
④ GpuAllocator → 分配显存
⑤ ______                      ← 卡住了，感觉还差一步
```

卡住本身是学习信号——用户意识到自己不知道。

#### 阶段 2：LLM 生成对照

用户点击"生成对照"→ LLM 基于概念内容和知识图谱关系生成参考版本：

```
LLM 参考版本：
① VllmConfig → 配置读取          ✓
② Device → 设备初始化            ✓
③ Tensor → 张量准备              ✓
④ GpuAllocator → 显存分配        ✓
⑤ ModelLoader → 权重加载    ⚡ 你没写

⚡ 你漏了「模型加载」——配置了设备就要把模型放上去
```

#### 阶段 3：比对与点亮

差异项不显示为"错误"，而显示为**发光点（⚡）**，点击后展开：

```
⚡ ModelLoader — 权重加载

你梳理的流程里没有这一步。
但想想：配置了设备和显存，模型权重本身怎么进去的？

这指向一个新概念：
「ModelLoader」— 负责把 HuggingFace 权重加载到 vLLM 的设备上

[ 开始梳理 ModelLoader ]
```

点击"开始梳理 ModelLoader"→ 跳转到 ModelLoader 的过程梳理面板。学习从一条链跳到另一条链。

#### 阶段 4：状态持久化

每个概念的"梳理状态"持久化到 ReviewRecord：

```typescript
interface ProcessState {
  user_flow: string[]                    // 用户填写的步骤 ID 列表
  llm_flow: string[]                     // LLM 参考步骤 ID 列表
  gaps: string[]                         // 差异项（漏掉的概念 ID）
  filled: boolean                        // 是否已完成梳理
  compared: boolean                      // 是否已完成比对
}
```

图谱节点视觉编码：

```
○ 灰暗（未点击过）     → 还没碰过
◐ 微光（梳理了一半）   → 填了部分空缺
◑ 脉动光（比对了）     → ⚡ 有差异待探索
● 稳定光（无差异）     → 已掌握
```

---

## Data Model 变更

### 新增类型

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
  label: string                          // "配置读取"
  description: string                    // "Engine 启动前读取 VllmConfig"
  question: string                       // 这一步引出的问题
  hint: string                           // 推导线索
  leads_to_type: 'element' | 'concept'   // 指向要素还是新概念
  leads_to_id?: string                   // 目标 ID
  is_core: boolean                       // 是否核心步骤（空缺必须填）
}

// 过程链
interface ProcessChain {
  id: string
  name: string                           // "推理启动流程"
  steps: ProcessStep[]
}
```

### Concept 类型扩展

```typescript
interface Concept {
  // ...现有字段保持不变
  process?: {
    chain_id: string                     // 所属过程链 ID
    step_index: number                   // 在链中的位置
    role: string                         // 在这一步中的角色
  }
  elements?: ConceptElement[]
}
```

---

## UI 组件变更

### 删除

- `KnowledgeGraphView.tsx` 第 156-161 行：全局 explore/review 模式切换按钮

### 修改

- `KnowledgeGraphView.tsx`：点击节点行为从"展示文档"改为"展示过程骨架"
- `ConceptDetailPanel.tsx`：替换 exploreTabs/learnTabs 为 process 梳理面板

### 新增

- `ProcessCanvas.tsx`：过程骨架展示 + 拖拽填充组件
- `ComparisonPanel.tsx`：LLM 对照 + 差异点亮组件
- `GapExplorer.tsx`：差异项展开 + 跳转新概念梳理

---

## 阶段划分

### Phase 1：Process Canvas（核心交互）

- 点击节点展示过程骨架 + 空缺
- 拖拽概念填充空缺
- 手动输入补充步骤
- 提交推导

### Phase 2：LLM 对照 + 比对

- 调取 LLM 生成参考流程
- 差异项"发光"展示
- 点击差异项展开说明

### Phase 3：概念跳转 + 状态持久化

- 差异项跳转到新概念梳理
- 节点状态持久化到 store
- 图谱节点视觉编码

---

## 不做的事

- 不做新手引导（用户自己发现点击行为）
- 不保留全局 explore/review 模式
- 不做传统自测评分（2/3/5 自评取消）
- 不改变现有三个 hover 按钮（AI/?/手动）
