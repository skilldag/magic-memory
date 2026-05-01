# 4 - 帆船 → vllm-core

> **[← 返回目录](../level-1/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 4 | ⛵ 帆船 | vllm-core |

**记忆口诀**: 帆船承载 vLLM 核心库一切

---

## vLLM 概念

**vllm-core** 是 vLLM 的核心基础库：

```
vllm/
├── vllm-core/     # 核心配置、日志、错误
├── vllm-device/  # 设备抽象
├── vllm-kernels/ # GPU 内核
└── ...
```

**记忆故事**:
- 帆船载万物 = core 装所有底层

---

## 核心模块

| 模块 | 功能 | 记忆 |
|------|------|------|
| vllm-core | 配置、日志、错误 | 基础 |
| vllm-device | GPU 抽象 | 设备 |
| vllm-kernels | CUDA 内核 | 计算 |
| vllm-model | 模型加载 | 模型 |

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [VllmConfig](./00-egg.md) | 0 | 鸡蛋 |
| [Device](./01-candle.md) | 1 | 蜡烛 |

---

*帆船虽小，能载万物。core 虽简，能撑全架构。*