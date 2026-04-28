import type { Concept, ConceptEdge } from '../types'

/**
 * vLLM 知识图谱模拟数据
 * 基于数字锚点记忆法的概念层级
 */

export const mockConcepts: Concept[] = [
  // Level 1: 基础层 (0-9)
  {
    id: '0',
    title: 'VllmConfig - 配置中心',
    alias: ['配置', 'Config'],
    level: 1,
    category: 'Foundation',
    problem: 'vLLM 如何统一管理所有配置？',
    gap_anticipate: '配置为什么需要分 model/cache/scheduler 三类？',
    depends_on: [],
    leads_to: ['1', '2', '10'],
    related: ['7'],
    content: `# VllmConfig - 配置中心

鸡蛋是源头，所有配置的"蛋黄"。VllmConfig 包含 model/cache/scheduler 三个配置。

## 核心功能
- 模型配置：模型路径、参数、量化设置
- 缓存配置：KV Cache 大小、分配策略
- 调度配置：批处理策略、并发限制

## 使用示例
\`\`\`python
config = VllmConfig(
    model="meta-llama/Llama-2-7b",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9
)
\`\`\``,
    path: './docs/level-1/00-egg.md',
    tags: ['config', 'foundation'],
    lastModified: new Date('2024-01-15')
  },
  {
    id: '1',
    title: 'Device - GPU设备抽象',
    alias: ['设备', 'GPU'],
    level: 1,
    category: 'Foundation',
    problem: 'vLLM 如何统一管理不同的 GPU 设备？',
    gap_anticipate: '为什么需要 Device trait 而不是直接用 CUDA？',
    depends_on: ['0'],
    leads_to: ['2', '5'],
    related: ['47'],
    content: `# Device - GPU设备抽象

蜡烛点亮 GPU，Device trait 是照亮系统的第一层抽象。

## 设计思想
- 抽象 GPU 操作接口
- 支持多设备类型（CUDA, ROCm, CPU）
- 提供统一的内存管理 API`,
    path: './docs/level-1/01-candle.md',
    tags: ['device', 'gpu', 'foundation'],
    lastModified: new Date('2024-01-15')
  },
  {
    id: '2',
    title: 'Tensor - 张量抽象',
    alias: ['张量', '数据'],
    level: 1,
    category: 'Foundation',
    problem: '如何统一处理不同来源的 Tensor？',
    gap_anticipate: 'Tensor 和 PyTorch Tensor 有什么区别？',
    depends_on: ['1'],
    leads_to: ['14', '25'],
    related: [],
    content: `# Tensor - 张量抽象

张量像鸭子浮在水面上，漂浮在 GPU 内存上。

## 核心属性
- shape: 维度
- dtype: 数据类型
- device: 设备位置
- data_ptr: 内存指针`,
    path: './docs/level-1/02-duck.md',
    tags: ['tensor', 'data', 'foundation'],
    lastModified: new Date('2024-01-15')
  },
  {
    id: '3',
    title: 'Logger/Tracing - 日志追踪',
    alias: ['日志', '监控'],
    level: 1,
    category: 'Foundation',
    problem: '如何有效地记录和追踪系统状态？',
    gap_anticipate: '为什么日志系统如此重要？',
    depends_on: [],
    leads_to: [],
    related: [],
    content: `# Logger/Tracing - 日志追踪

日志被耳朵监听，记录系统状态。`,
    path: './docs/level-1/03-ear.md',
    tags: ['logging', 'observability'],
    lastModified: new Date('2024-01-15')
  },
  {
    id: '4',
    title: 'vllm-core - 核心库',
    alias: ['核心'],
    level: 1,
    category: 'Foundation',
    problem: '核心库承载哪些关键功能？',
    gap_anticipate: 'core 和 engine 有什么区别？',
    depends_on: [],
    leads_to: ['5', '6', '7', '8'],
    related: [],
    content: `# vllm-core - 核心库

帆船承载 vLLM 核心库一切。`,
    path: './docs/level-1/04-boat.md',
    tags: ['core', 'foundation'],
    lastModified: new Date('2024-01-15')
  },
  {
    id: '5',
    title: 'GpuAllocator - 内存分配器',
    alias: ['分配器', '显存'],
    level: 1,
    category: 'Foundation',
    problem: '如何高效管理有限的 GPU 显存？',
    gap_anticipate: '为什么需要专门的分配器而不是 malloc？',
    depends_on: ['1', '4'],
    leads_to: ['9', '18', '19'],
    related: [],
    content: `# GpuAllocator - GPU内存分配器

钩子钩住 GPU 显存分配。`,
    path: './docs/level-1/05-hook.md',
    tags: ['memory', 'allocator', 'gpu'],
    lastModified: new Date('2024-01-15')
  },
  {
    id: '6',
    title: 'Error Handling - 错误处理',
    alias: ['错误', '异常'],
    level: 1,
    category: 'Foundation',
    problem: '如何统一处理各种错误情况？',
    gap_anticipate: 'VllmError 和普通 Exception 有什么区别？',
    depends_on: ['4'],
    leads_to: [],
    related: [],
    content: `# Error Handling - 错误处理

勺子舀出错误 VllmError。`,
    path: './docs/level-1/06-spoon.md',
    tags: ['error', 'exception'],
    lastModified: new Date('2024-01-15')
  },
  {
    id: '7',
    title: 'Init - 初始化系统',
    alias: ['初始化'],
    level: 1,
    category: 'Foundation',
    problem: '系统如何安全地启动和初始化？',
    gap_anticipate: 'init_logger 有什么特殊之处？',
    depends_on: ['3', '4'],
    leads_to: ['40'],
    related: ['0'],
    content: `# Init - 初始化系统

初始化需要支撑，像拐杖。init_logger 支撑整个系统。`,
    path: './docs/level-1/07-crutch.md',
    tags: ['init', 'startup'],
    lastModified: new Date('2024-01-15')
  },
  {
    id: '8',
    title: 'Foundation Layer - 基础层',
    alias: ['底层'],
    level: 1,
    category: 'Foundation',
    problem: '基础层提供哪些底层能力？',
    gap_anticipate: 'Foundation Layer 和 vllm-core 是什么关系？',
    depends_on: ['4'],
    leads_to: ['9'],
    related: [],
    content: `# Foundation Layer - 基础层

基础层像葫芦装所有底层功能。`,
    path: './docs/level-1/08-gourd.md',
    tags: ['foundation', '底层'],
    lastModified: new Date('2024-01-15')
  },
  {
    id: '9',
    title: 'KV Cache - 键值缓存',
    alias: ['缓存', 'KV'],
    level: 1,
    category: 'Foundation',
    problem: '为什么 KV Cache 对 LLM 如此重要？',
    gap_anticipate: 'KV Cache 和传统缓存有什么区别？',
    depends_on: ['5', '8'],
    leads_to: ['16', '17', '18', '19'],
    related: ['26'],
    content: `# KV Cache - 键值缓存

气球膨胀如缓存增长。KV Cache 是 Transformer 模型的核心缓存。`,
    path: './docs/level-1/09-balloon.md',
    tags: ['cache', 'kv', 'foundation'],
    lastModified: new Date('2024-01-15')
  },

  // Level 2: 核心执行层 (10-29)
  {
    id: '10',
    title: 'ModelRegistry - 模型注册',
    alias: ['注册表'],
    level: 2,
    category: 'Model',
    problem: '如何管理众多模型架构？',
    gap_anticipate: '为什么模型名要限制在10个字符内？',
    depends_on: ['0'],
    leads_to: ['11', '12'],
    related: [],
    content: `# ModelRegistry - 模型注册

模型注册如棒球规则有限制（10个字符内命名）。`,
    path: './docs/level-2/10-baseball.md',
    tags: ['model', 'registry'],
    lastModified: new Date('2024-01-16')
  },
  {
    id: '11',
    title: 'ModelLoader - 模型加载器',
    alias: ['加载器'],
    level: 2,
    category: 'Model',
    problem: '如何加载各种格式的模型权重？',
    gap_anticipate: 'HuggingFace 和 Safetensors 格式有什么区别？',
    depends_on: ['10'],
    leads_to: ['12', '29'],
    related: [],
    content: `# ModelLoader - 模型加载器

筷子夹取权重，11=夹取。`,
    path: './docs/level-2/11-chopsticks.md',
    tags: ['model', 'loader'],
    lastModified: new Date('2024-01-16')
  },
  {
    id: '12',
    title: 'Model - 模型主体',
    alias: ['模型'],
    level: 2,
    category: 'Model',
    problem: '模型如何执行前向传播？',
    gap_anticipate: 'Model 和 ModelRunner 有什么区别？',
    depends_on: ['10', '11'],
    leads_to: ['13', '14'],
    related: [],
    content: `# Model - 模型主体

模型如婴儿被加载到 Model Runner 上。`,
    path: './docs/level-2/12-highchair.md',
    tags: ['model', 'inference'],
    lastModified: new Date('2024-01-16')
  },
  {
    id: '13',
    title: 'ModelRunner - 模型运行器',
    alias: ['运行器', 'Runner'],
    level: 2,
    category: 'Model',
    problem: '如何协调模型的完整执行流程？',
    gap_anticipate: 'ModelRunner 和 Model 是什么关系？',
    depends_on: ['12'],
    leads_to: ['14', '24'],
    related: [],
    content: `# ModelRunner - 模型运行器

ModelRunner 如雨伞保护前向传播。`,
    path: './docs/level-2/13-umbrella.md',
    tags: ['model', 'runner'],
    lastModified: new Date('2024-01-16')
  },
  {
    id: '14',
    title: 'Embedding - 词嵌入',
    alias: ['嵌入'],
    level: 2,
    category: 'Model',
    problem: '如何将词转换为向量？',
    gap_anticipate: 'Embedding 和 Transformer 是什么关系？',
    depends_on: ['2', '12'],
    leads_to: ['15'],
    related: [],
    content: `# Embedding - 词嵌入

玫瑰芳香嵌入向量。`,
    path: './docs/level-2/14-rose.md',
    tags: ['embedding', 'nlp'],
    lastModified: new Date('2024-01-16')
  },
  {
    id: '15',
    title: 'Transformer Layers - Transformer层',
    alias: ['Transformer', '层'],
    level: 2,
    category: 'Model',
    problem: '多层 Transformer 如何堆叠和执行？',
    gap_anticipate: 'Attention 层和 FFN 层的执行顺序？',
    depends_on: ['14'],
    leads_to: ['16'],
    related: [],
    content: `# Transformer Layers - 层叠

鹦鹉学舌对应多层 Transformer。`,
    path: './docs/level-2/15-parrot.md',
    tags: ['transformer', 'layers'],
    lastModified: new Date('2024-01-16')
  },
  {
    id: '16',
    title: 'PagedAttention - 分页注意力',
    alias: ['分页', 'Attention'],
    level: 2,
    category: 'Model',
    problem: '如何高效管理长序列的 KV Cache？',
    gap_anticipate: '为什么叫"分页"注意力？和传统 Attention 有什么区别？',
    depends_on: ['9', '15'],
    leads_to: ['17'],
    related: ['19'],
    content: `# PagedAttention - 分页注意力

石榴多籽 = 多分页，KV 分页管理是 vLLM 的核心技术。`,
    path: './docs/level-2/16-pomegranate.md',
    tags: ['attention', 'paging', '核心技术'],
    lastModified: new Date('2024-01-16')
  },
  {
    id: '17',
    title: 'Block Table - 块表映射',
    alias: ['块表'],
    level: 2,
    category: 'Model',
    problem: '如何管理虚拟块到物理块的映射？',
    gap_anticipate: 'Block Table 和 Page Table 有什么区别？',
    depends_on: ['16'],
    leads_to: ['18'],
    related: [],
    content: `# Block Table - 块表映射

Block Table 如显微镜看物理块映射。`,
    path: './docs/level-2/17-microscope.md',
    tags: ['block', 'paging', 'mmu'],
    lastModified: new Date('2024-01-16')
  },
  {
    id: '18',
    title: 'CacheBlock - 缓存块',
    alias: ['块'],
    level: 2,
    category: 'Model',
    problem: '如何组织和管理缓存块？',
    gap_anticipate: 'CacheBlock 大小固定还是可变的？',
    depends_on: ['9', '17'],
    leads_to: ['19'],
    related: [],
    content: `# CacheBlock - 缓存块

CacheBlock 如钞票管理显存资源。`,
    path: './docs/level-2/18-money.md',
    tags: ['cache', 'block'],
    lastModified: new Date('2024-01-16')
  },
  {
    id: '19',
    title: 'KVCacheManager - 缓存管理器',
    alias: ['管理器'],
    level: 2,
    category: 'Model',
    problem: '如何统筹管理所有 KV Cache？',
    gap_anticipate: 'CacheManager 和 GpuAllocator 有什么区别？',
    depends_on: ['9', '16', '18'],
    leads_to: ['20'],
    related: [],
    content: `# KVCacheManager - 缓存管理器

KVCacheManager 如药剂师分配缓存。`,
    path: './docs/level-2/19-medicine.md',
    tags: ['cache', 'manager'],
    lastModified: new Date('2024-01-16')
  },
  {
    id: '20',
    title: 'Sampler - 采样器',
    alias: ['采样'],
    level: 2,
    category: 'Model',
    problem: '如何从 logits 选择下一个 token？',
    gap_anticipate: 'greedy 和 random sampling 有什么区别？',
    depends_on: ['19'],
    leads_to: ['21', '22'],
    related: [],
    content: `# Sampler - 采样器

Sampler 如点燃决定下一个 token。`,
    path: './docs/level-2/20-cigarette.md',
    tags: ['sampling', 'decoding'],
    lastModified: new Date('2024-01-16')
  },
  {
    id: '21',
    title: 'Sampling Params - 采样参数',
    alias: ['参数'],
    level: 2,
    category: 'Model',
    problem: '如何配置采样行为？',
    gap_anticipate: 'temperature/top_k/top_p 是什么？',
    depends_on: ['20'],
    leads_to: ['22'],
    related: [],
    content: `# Sampling Params - 采样参数

temperature/top_k/top_p 如鳄鱼参数凶猛。`,
    path: './docs/level-2/21-crocodile.md',
    tags: ['sampling', 'params'],
    lastModified: new Date('2024-01-16')
  },
  {
    id: '22',
    title: 'Logits - 对数几率',
    alias: ['logits'],
    level: 2,
    category: 'Model',
    problem: '如何计算和处理模型输出？',
    gap_anticipate: 'Logits 和 probabilities 有什么区别？',
    depends_on: ['21'],
    leads_to: ['23'],
    related: [],
    content: `# Logits - 对数几率

Logits 如双胞胎成对出现。`,
    path: './docs/level-2/22-twins.md',
    tags: ['logits', 'output'],
    lastModified: new Date('2024-01-16')
  },
  {
    id: '23',
    title: 'Token - 词元',
    alias: ['词元', 'token'],
    level: 2,
    category: 'Model',
    problem: '什么是 token？',
    gap_anticipate: '中文和英文的 token 有什么区别？',
    depends_on: ['22'],
    leads_to: ['24'],
    related: [],
    content: `# Token - 词元

一个 Token 如耳塞塞住信息。`,
    path: './docs/level-2/23-earplugs.md',
    tags: ['token', 'vocabulary'],
    lastModified: new Date('2024-01-16')
  },
  {
    id: '24',
    title: 'Decode Step - 解码步骤',
    alias: ['解码'],
    level: 2,
    category: 'Model',
    problem: '如何逐步生成 token？',
    gap_anticipate: 'autoregressive generation 是什么？',
    depends_on: ['13', '23'],
    leads_to: ['25'],
    related: [],
    content: `# Decode Step - 解码步骤

解码如闹钟每步滴答产生 token。`,
    path: './docs/level-2/24-alarm.md',
    tags: ['decode', 'generation'],
    lastModified: new Date('2024-01-16')
  },
  {
    id: '25',
    title: 'Forward Pass - 前向传播',
    alias: ['前向'],
    level: 2,
    category: 'Model',
    problem: '数据如何在模型中流动？',
    gap_anticipate: 'Forward 和 Backward 有什么区别？',
    depends_on: ['2', '24'],
    leads_to: ['26', '27'],
    related: [],
    content: `# Forward Pass - 前向传播

前向传播如二胡弦动。`,
    path: './docs/level-2/25-erhu.md',
    tags: ['forward', 'inference'],
    lastModified: new Date('2024-01-16')
  },
  {
    id: '26',
    title: 'GPU Memory Pool - 显存池',
    alias: ['显存池'],
    level: 2,
    category: 'Performance',
    problem: '如何高效利用 GPU 显存？',
    gap_anticipate: 'memory pool 和直接 malloc 有什么区别？',
    depends_on: ['9', '25'],
    leads_to: ['27', '28'],
    related: [],
    content: `# GPU Memory Pool - 显存池

GPU 内存如河流流动。`,
    path: './docs/level-2/26-river.md',
    tags: ['memory', 'gpu', 'pool'],
    lastModified: new Date('2024-01-16')
  },
  {
    id: '27',
    title: 'FlashAttention - 快速注意力',
    alias: ['FlashAttn'],
    level: 2,
    category: 'Performance',
    problem: '如何加速 Attention 计算？',
    gap_anticipate: 'FlashAttention 为什么比标准 Attention 快？',
    depends_on: ['25', '26'],
    leads_to: ['28'],
    related: [],
    content: `# FlashAttention - 快速注意力

FlashAttention 如耳机快速聆听处理。`,
    path: './docs/level-2/27-headphones.md',
    tags: ['attention', 'performance', 'optimization'],
    lastModified: new Date('2024-01-16')
  },
  {
    id: '28',
    title: 'Quantization - 量化',
    alias: ['量化', 'INT8', 'FP8'],
    level: 2,
    category: 'Performance',
    problem: '如何减少模型内存和加速推理？',
    gap_anticipate: 'INT8 和 FP16 有什么区别？量化会损失精度吗？',
    depends_on: ['26', '27'],
    leads_to: ['29'],
    related: [],
    content: `# Quantization - 量化

量化如恶霸暴力压缩精度。`,
    path: './docs/level-2/28-bully.md',
    tags: ['quantization', 'performance', 'model-compression'],
    lastModified: new Date('2024-01-16')
  },
  {
    id: '29',
    title: 'Weights Loading - 权重加载',
    alias: ['权重'],
    level: 2,
    category: 'Model',
    problem: '如何高效加载模型权重？',
    gap_anticipate: 'load_hf_model 做了什么？',
    depends_on: ['11', '28'],
    leads_to: ['30'],
    related: [],
    content: `# Weights Loading - 权重加载

load_hf_model 如二舅帮忙加载。`,
    path: './docs/level-2/29-uncle.md',
    tags: ['weights', 'loading'],
    lastModified: new Date('2024-01-16')
  },

  // Level 3: 高级特性层 (30-50)
  {
    id: '30',
    title: 'Speculative Decoding - 投机解码',
    alias: ['推测解码', 'SpecDecode'],
    level: 3,
    category: 'Advanced',
    problem: '如何加速自回归解码？',
    gap_anticipate: '为什么推测解码能加速？需要额外的模型吗？',
    depends_on: ['29'],
    leads_to: ['31', '32', '33'],
    related: [],
    content: `# Speculative Decoding - 投机解码

推测解码如三菱标志三分支。`,
    path: './docs/level-3/30-mitsubishi.md',
    tags: ['speculative', 'decoding', 'optimization'],
    lastModified: new Date('2024-01-17')
  },
  {
    id: '31',
    title: 'Draft Token - 起草Token',
    alias: ['草稿'],
    level: 3,
    category: 'Advanced',
    problem: '如何快速生成候选 token？',
    gap_anticipate: 'draft token 一定被接受吗？',
    depends_on: ['30'],
    leads_to: ['32'],
    related: [],
    content: `# Draft Token - 起草 token

Draft token 如山药生长。`,
    path: './docs/level-3/31-yam.md',
    tags: ['speculative', 'draft'],
    lastModified: new Date('2024-01-17')
  },
  {
    id: '32',
    title: 'Verifier - 验证器',
    alias: ['验证'],
    level: 3,
    category: 'Advanced',
    problem: '如何验证 draft token 的正确性？',
    gap_anticipate: 'Verifier 和 main model 是什么关系？',
    depends_on: ['30', '31'],
    leads_to: ['33'],
    related: [],
    content: `# Verifier - 验证器

验证器扇出正确 token。`,
    path: './docs/level-3/32-fan.md',
    tags: ['speculative', 'verify'],
    lastModified: new Date('2024-01-17')
  },
  {
    id: '33',
    title: 'N-gram Proposer - N元提议',
    alias: ['N-gram'],
    level: 3,
    category: 'Advanced',
    problem: '如何用简单方法快速提议 token？',
    gap_anticipate: 'N-gram 和神经网络哪个更快？',
    depends_on: ['30'],
    leads_to: ['34'],
    related: [],
    content: `# N-gram Proposer - N元提议

N-gram 如星空群星。`,
    path: './docs/level-3/33-stars.md',
    tags: ['speculative', 'ngram'],
    lastModified: new Date('2024-01-17')
  },
  {
    id: '34',
    title: 'Continuous Batching - 连续批处理',
    alias: ['批处理', 'Batching'],
    level: 3,
    category: 'Scheduling',
    problem: '如何最大化 GPU 利用率？',
    gap_anticipate: 'static batching 和 continuous batching 有什么区别？',
    depends_on: ['33'],
    leads_to: ['35'],
    related: [],
    content: `# Continuous Batching - 连续批处理

批处理如蔬菜条条有序。`,
    path: './docs/level-3/34-vegetable.md',
    tags: ['batching', 'scheduling', 'performance'],
    lastModified: new Date('2024-01-17')
  },
  {
    id: '35',
    title: 'Scheduler - 调度器',
    alias: ['调度'],
    level: 3,
    category: 'Scheduling',
    problem: '如何调度和分配请求？',
    gap_anticipate: 'scheduler 和 batcher 有什么区别？',
    depends_on: ['34'],
    leads_to: ['36', '37', '38'],
    related: [],
    content: `# Scheduler - 调度器

调度器如珊瑚礁分支多。`,
    path: './docs/level-3/35-coral.md',
    tags: ['scheduler', 'scheduling'],
    lastModified: new Date('2024-01-17')
  },
  {
    id: '36',
    title: 'Prefill - 预填充',
    alias: ['预填充'],
    level: 3,
    category: 'Scheduling',
    problem: '如何处理首次 token 生成？',
    gap_anticipate: 'Prefill 和 Decode 哪个更耗时间？',
    depends_on: ['35'],
    leads_to: ['37'],
    related: [],
    content: `# Prefill - 预填充

Prefill 如鹿冲锋快速。`,
    path: './docs/level-3/36-deer.md',
    tags: ['prefill', 'scheduling'],
    lastModified: new Date('2024-01-17')
  },
  {
    id: '37',
    title: 'Decode - 解码',
    alias: ['自回归解码'],
    level: 3,
    category: 'Scheduling',
    problem: '如何高效进行自回归生成？',
    gap_anticipate: 'Decode 阶段有什么特点？',
    depends_on: ['35', '36'],
    leads_to: ['38'],
    related: [],
    content: `# Decode - 解码

Decode 如野鸡慢走解码。`,
    path: './docs/level-3/37-pheasant.md',
    tags: ['decode', 'scheduling'],
    lastModified: new Date('2024-01-17')
  },
  {
    id: '38',
    title: 'Prefix Caching - 前缀缓存',
    alias: ['前缀缓存'],
    level: 3,
    category: 'Optimization',
    problem: '如何复用相同前缀的 KV Cache？',
    gap_anticipate: 'prefix caching 适合什么场景？',
    depends_on: ['35'],
    leads_to: ['39', '48'],
    related: [],
    content: `# Prefix Caching - 前缀缓存

女性记前缀能力强。`,
    path: './docs/level-3/38-woman.md',
    tags: ['caching', 'prefix', 'optimization'],
    lastModified: new Date('2024-01-17')
  },
  {
    id: '39',
    title: 'Request Queue - 请求队列',
    alias: ['队列'],
    level: 3,
    category: 'Scheduling',
    problem: '如何管理待处理的请求？',
    gap_anticipate: '队列的优先级如何确定？',
    depends_on: ['38'],
    leads_to: ['40'],
    related: [],
    content: `# Request Queue - 请求队列

请求队列如剑林排列。`,
    path: './docs/level-3/39-sword.md',
    tags: ['queue', 'scheduling'],
    lastModified: new Date('2024-01-17')
  },
  {
    id: '40',
    title: 'vllm-engine - 引擎',
    alias: ['Engine'],
    level: 3,
    category: 'Serving',
    problem: '如何协调整个推理流程？',
    gap_anticipate: 'Engine 和 ModelRunner 有什么区别？',
    depends_on: ['7', '39'],
    leads_to: ['41', '42'],
    related: [],
    content: `# vllm-engine - 引擎

Engine 司令指挥全流程。`,
    path: './docs/level-3/40-commander.md',
    tags: ['engine', 'core'],
    lastModified: new Date('2024-01-17')
  },
  {
    id: '41',
    title: 'Engine API - 引擎API',
    alias: ['API'],
    level: 3,
    category: 'Serving',
    problem: '如何提供统一的接口？',
    gap_anticipate: 'Engine API 和 OpenAI API 有什么区别？',
    depends_on: ['40'],
    leads_to: ['42', '43'],
    related: [],
    content: `# Engine API - 引擎 API

API 如蜥蜴爬行接口层。`,
    path: './docs/level-3/41-lizard.md',
    tags: ['api', 'interface'],
    lastModified: new Date('2024-01-17')
  },
  {
    id: '42',
    title: 'vllm-serving - 服务层',
    alias: ['Serving'],
    level: 3,
    category: 'Serving',
    problem: '如何提供在线服务？',
    gap_anticipate: 'serving 和 engine 有什么区别？',
    depends_on: ['40', '41'],
    leads_to: ['43', '44', '45'],
    related: [],
    content: `# vllm-serving - 服务层

Serving 如玉米棒包万物。`,
    path: './docs/level-3/42-corn.md',
    tags: ['serving', 'server'],
    lastModified: new Date('2024-01-17')
  },
  {
    id: '43',
    title: 'OpenAI API - OpenAI兼容',
    alias: ['OpenAI'],
    level: 3,
    category: 'Serving',
    problem: '如何兼容 OpenAI 接口？',
    gap_anticipate: '需要认证吗？如何处理 rate limit？',
    depends_on: ['41', '42'],
    leads_to: ['44'],
    related: [],
    content: `# OpenAI API - OpenAI 协议

OpenAI 兼容如石山稳固。`,
    path: './docs/level-3/43-rock.md',
    tags: ['openai', 'api', 'compatibility'],
    lastModified: new Date('2024-01-17')
  },
  {
    id: '44',
    title: 'gRPC - gRPC协议',
    alias: ['grpc'],
    level: 3,
    category: 'Serving',
    problem: '为什么需要 gRPC 而不是只用 HTTP？',
    gap_anticipate: 'gRPC 和 HTTP 哪个更快？',
    depends_on: ['42'],
    leads_to: ['45'],
    related: [],
    content: `# gRPC - gRPC 协议

gRPC 如眼镜蛇快速。`,
    path: './docs/level-3/44-cobra.md',
    tags: ['grpc', 'protocol'],
    lastModified: new Date('2024-01-17')
  },
  {
    id: '45',
    title: 'WebSocket - 实时推送',
    alias: ['WS'],
    level: 3,
    category: 'Serving',
    problem: '如何实现实时 token 流式输出？',
    gap_anticipate: 'WebSocket 和 SSE 哪个适合流式？',
    depends_on: ['42', '44'],
    leads_to: ['46'],
    related: [],
    content: `# WebSocket - WebSocket

WS 实时推流如师傅传功。`,
    path: './docs/level-3/45-master.md',
    tags: ['websocket', 'streaming'],
    lastModified: new Date('2024-01-17')
  },
  {
    id: '46',
    title: 'Multi-LoRA - 多LoRA',
    alias: ['LoRA'],
    level: 3,
    category: 'Advanced',
    problem: '如何同时服务多个 LoRA 适配器？',
    gap_anticipate: 'LoRA 和全参数微调有什么区别？',
    depends_on: ['45'],
    leads_to: ['47'],
    related: [],
    content: `# Multi-Lora - 多 Lora

多 Lora 如石榴多籽。`,
    path: './docs/level-3/46-pomegranate.md',
    tags: ['lora', 'finetuning', 'multi-tenant'],
    lastModified: new Date('2024-01-17')
  },
  {
    id: '47',
    title: 'GPU Driver - GPU驱动',
    alias: ['驱动'],
    level: 3,
    category: 'Infrastructure',
    problem: '如何和 GPU 驱动交互？',
    gap_anticipate: 'CUDA Driver 和 CUDA Runtime 有什么区别？',
    depends_on: ['1', '46'],
    leads_to: ['48'],
    related: [],
    content: `# GPU Driver - GPU 驱动

驱动如司机驾驶 GPU。`,
    path: './docs/level-3/47-driver.md',
    tags: ['gpu', 'driver', 'cuda'],
    lastModified: new Date('2024-01-17')
  },
  {
    id: '48',
    title: 'Prefix Lookup - 前缀查找',
    alias: ['查找'],
    level: 3,
    category: 'Optimization',
    problem: '如何快速查找已缓存的前缀？',
    gap_anticipate: '前缀查找用什么数据结构？',
    depends_on: ['38', '47'],
    leads_to: ['49'],
    related: [],
    content: `# Prefix Lookup - 前缀查找

前缀查找如丝瓜络过滤。`,
    path: './docs/level-3/48-loofah.md',
    tags: ['prefix', 'lookup', 'optimization'],
    lastModified: new Date('2024-01-17')
  },
  {
    id: '49',
    title: 'Cache Eviction - 缓存驱逐',
    alias: ['驱逐'],
    level: 3,
    category: 'Optimization',
    problem: '缓存满时如何选择驱逐对象？',
    gap_anticipate: 'LRU 和 LFU 哪个更适合 KV Cache？',
    depends_on: ['48'],
    leads_to: ['50'],
    related: [],
    content: `# Cache Eviction - 缓存驱逐

缓存驱逐如湿狗甩水。`,
    path: './docs/level-3/49-wetdog.md',
    tags: ['cache', 'eviction', 'optimization'],
    lastModified: new Date('2024-01-17')
  },
  {
    id: '50',
    title: 'Distributed - 分布式',
    alias: ['分布式', '多卡'],
    level: 3,
    category: 'Infrastructure',
    problem: '如何支持多 GPU 分布式推理？',
    gap_anticipate: 'Tensor Parallel 和 Pipeline Parallel 有什么区别？',
    depends_on: ['49'],
    leads_to: [],
    related: [],
    content: `# Distributed - 分布式

分布式如五菱装更多节点。`,
    path: './docs/level-3/50-minivan.md',
    tags: ['distributed', 'multi-gpu', 'infrastructure'],
    lastModified: new Date('2024-01-17')
  }
]

const edgeIdSet = new Set<string>()

export const mockEdges: ConceptEdge[] = [
  // 只从 leads_to 生成边，depends_on 是语义反向（避免每条关系产生两条边）
  ...mockConcepts.flatMap(concept =>
    concept.leads_to.map(targetId => {
      const id = `e_${concept.id}_${targetId}_leads_to`
      if (edgeIdSet.has(id)) return null
      edgeIdSet.add(id)
      return {
        id,
        source: concept.id,
        target: targetId,
        type: 'leads_to' as const
      }
    })
  ).filter(Boolean) as ConceptEdge[]
]

/**
 * Get knowledge graph data
 */
export function getMockGraphData() {
  return {
    concepts: mockConcepts,
    edges: mockEdges
  }
}