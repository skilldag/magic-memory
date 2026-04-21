# 1 - 蜡烛 → Device 抽象

> **[← 返回目录](../level-1/README.md)**

## 快速记忆

| 数字 | 锚点 | 概念 |
|------|------|------|
| 1 | 🕯️ 蜡烛 | Device 抽象 |

**记忆口诀**: 蜡烛点亮 GPU，Device trait 是照亮系统的第一层

---

## vLLM 概念

**Device** 是 vLLM 的设备抽象层，类似于 CUDA 的抽象：

```rust
pub trait Device {
    /// 设备类型
    fn device_type(&self) -> DeviceType;

    /// 设备 ID
    fn device_id(&self) -> i32;

    /// 分配显存
    fn allocate(&self, size: usize) -> *mut void;

    /// 同步
    fn synchronize(&self);
}
```

**记忆故事**:
- 蜡烛照亮黑暗 = Device 照亮 GPU
- 1 是第一个抽象层 = Device 是底层第一层

---

## 深入理解

### DeviceType 枚举

| 类型 | 说明 | 记忆 |
|------|------|------|
| **CPU** | CPU 设备 | 通用计算 |
| **CUDA** | NVIDIA GPU | 主流推理 |
| **ROCm** | AMD GPU | AMD 显卡 |
| **CPU** | 聚合设备 | 多卡聚合 |

### 核心能力

```
Device 抽象
    ├── 内存管理 (allocate/free)
    ├── 计算调度 (dispatch)
    ├── 流管理 (Stream)
    └── 同步控制 (synchronize)
```

---

## 实际使用

```rust
// 获取当前设备
let device = Device::new_cuda(0);

// 在设备上分配内存
let ptr = device.allocate(kv_cache_size);

// 设备同步
device.synchronize();
```

---

## 关联记忆

| 相关概念 | 数字 | 锚点 |
|----------|------|------|
| [VllmConfig](./00-egg.md) | 0 | 鸡蛋 |
| [Tensor](./02-duck.md) | 2 | 鸭子 |
| [GpuAllocator](./05-hook.md) | 5 | 钩子 |

---

*蜡烛虽小，照亮前行。Device 虽薄，承载全部计算。*