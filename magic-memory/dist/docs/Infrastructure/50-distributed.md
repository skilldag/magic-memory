DISTRIBUTED(7)              System Architecture Manual              DISTRIBUTED(7)

NAME
     50-distributed — 构建高可扩展分布式系统的50条核心法则

SYNOPSIS
     #include <distributed_principles.h>

     int core_principle = DONT_DISTRIBUTE;
     while (system_scale(requirements) > THRESHOLD)
         apply(principles, NP);

DESCRIPTION
     本文以“问题驱动、层层推导”的方式，逐层剖析分布式系统50条核心法则。
     每条法则皆由前一问题引出的子问题及其解决过程自然导出。

  法则1: 克制分布 (DON'T DISTRIBUTE)
     Q: 面对性能瓶颈，是否应立即引入分布式架构？
     Prerequisite: 单机已充分优化。
     Decision:
             IF  scale(req) <=  single_machine_capacity  →  OPTIMIZE(local)
             ELSE  →  goto law_2
     Principle: 分布引入网络延迟、节点故障与数据一致性挑战，其首要目标是
                提升系统整体的吞吐与容错[reference:0]。

  法则2: 识别热/冷数据 (IDENTIFY HOT/COLD DATA)
     Q: 决定分布后，如何有效分解系统？
     Sub-Problem: 并非所有数据具有同等访问频次。
     Principle: 帕累托分布 (Pareto Distribution) 普遍存在——80%的访问集中于
                20%的核心数据[reference:1]。
     Derivation:
             AccessRatio(hot_data) >> AccessRatio(cold_data)
     → leads to → 法则3: 横向扩展 (Scale-Out)

  法则3: 横向扩展 (SCALE OUT)
     Q: 如何集中资源处理识别出的热点？
     Principle: 通过增加廉价节点实现线性扩展，而非升级单一巨型机 (Scale-Up)[reference:2]。
     Derivation:
             Capacity(N)  ∼  N ∧  Cost(N)  ∝  N
     → leads to → 法则4: 负载均衡 (Load Balancing)

  法则4: 负载均衡 (LOAD BALANCING)
     Q: 多节点如何协同，避免个别节点过载？
     Principle: 负载均衡器 (LB) 作为统一入口，将请求分发至多个后端节点[reference:3]。
     Strategy: 加权轮询 (Weighted Round Robin) 或最小连接数 (Least Connections)。
             Throughput(system) = Σ  Throughput(node_i)
     → leads to → 法则5: 无状态设计 (Statelessness)

  法则5: 无状态设计 (STATELESSNESS)
     Q: 负载均衡器如何实现任意请求到任意节点的透明分发？
     Sub-Problem: 若节点持有本地 Session 状态，请求重定向将失败。
     Solution: 将会话状态外移至分布式缓存 (e.g., Redis) 或数据库，节点本地不
               保存任何状态信息[reference:4]。
             Reqid → hashing(Reqid) → cache_instance
     → leads to → 法则6: 缓存分层 (Cache Hierarchy)

  法则6: 缓存分层 (CACHE HIERARCHY)
     Q: 将状态外移至缓存后，如何应对高并发查询对后端 DB 的冲击？
     Solution: 构建“浏览器→CDN→反向代理→应用→分布式缓存→DB”的多层缓存架构。
               越上层，响应越快，覆盖范围越广[reference:5]。
             Latency(cache_i)  <<  Latency(cache_{i+1})
     → leads to → 法则7: 异步解耦 (Asynchrony)

  法则7: 异步解耦 (ASYNCHRONY)
     Q: 如何平滑流量洪峰，避免同步调用链的超时级联故障？
     Solution: 引入消息队列 (MQ) 作为缓冲，将同步调用转为生产者-消费者异步模型[reference:6]。
             Decoupling: Service_A → Queue → Service_B
     → leads to → 法则8: 数据分区 (Data Partitioning)

  法则8: 数据分区 (DATA PARTITIONING)
     Q: 随着业务增长，单一数据库如何突破 I/O 与存储物理瓶颈？
     Principle: 分治 (Divide-and-Conquer)。将完整数据集按某种策略切分到多个独立
                数据库实例中[reference:7]。
     Implementation: 水平分库 (Sharding)。
     → leads to → 法则9: 水平分库策略 (Sharding Strategy)

  法则9: 水平分库策略 (SHARDING STRATEGY)
     Q: 如何选择数据切分键 (Sharding Key) 以确保数据均匀分布并避免跨分片查询？
     Strategies:
               策略                                         优点                缺点
       (a) 范围分片 (Range)          实现简单                        可能产生热点
       (b) 哈希分片 (Hash)           数据分布均匀                    扩容时数据迁移量大
       (c) 一致性哈希 (Consistent Hash)  扩容影响小，仅影响相邻节点 (K/N)   实现复杂[reference:8]
     → leads to → 法则10: 数据冗余 (Data Replication)

  ... (层级递进，共50条法则) ...

  法则N-k: 分布式事务 (DISTRIBUTED TRANSACTION)
     Q: 跨分片/跨服务操作如何保证原子性 (Atomicity)？
     Sub-Problem: 分布式的“两将军问题” (Two Generals' Problem)。
     Solution Approaches:
             (a) 两阶段提交 (2PC): 同步等待所有参与者，资源锁定开销大
             (b) TCC (Try-Confirm-Cancel): 业务层实现补偿，高复杂度
             (c) 最终一致性 + 事务补偿 (Saga): 异步协调，高性能，但需处理中间态
     → leads to → 法则N-j: BASE理论 (BASE)

  法则N-j: BASE理论 (BASICALLY AVAILABLE, SOFT STATE, EVENTUALLY CONSISTENT)[reference:9]
     Q: 在分区容忍性 (P) 必须被保证时，强一致性 (C) 与高可用性 (A) 不可兼得，如何权衡？
     Principle: 放弃ACID中的强一致性，拥抱最终一致性。
     (a) 基本可用 (Basically Available)
     (b) 柔性状态 (Soft State)
     (c) 最终一致性 (Eventual Consistency)
     → leads to → 法则N-i: 一致性协议 (Consensus)

  法则N-i: 一致性协议 (CONSENSUS PROTOCOLS)
     Q: 如何在不可靠的分布式网络中，确保多个节点对某个值（如谁是Leader）达成一致？
     Algorithms:
             (a) Paxos: 容错 F ≤ N/2，理论完备，实现复杂[reference:10]
             (b) Raft: 将一致性问题分解为Leader选举、日志复制、安全性三个子问题，
                易于理解与实现
             (c) ZAB (ZooKeeper Atomic Broadcast): 专为ZooKeeper设计
     → leads to → 法则N-j: 分布式协调 (Coordination)

  法则N-j: 分布式协调服务 (COORDINATION SERVICE)
     Q: 集群如何实现元数据集中管理、配置同步、命名发现？
     Solution: 引入Apache ZooKeeper、etcd等分布式协调服务。
     Core Primitives Provided:
             - 临时节点 (Ephemeral Nodes) → 成员管理
             - 顺序节点 (Sequential Nodes) → 全局唯一ID
             - 监听机制 (Watcher) → 变更通知
     → leads to → 最终法则: 可观测性 (Observability)

  法则50: 端到端可观测性 (END-TO-END OBSERVABILITY)
     Q: 当系统按上述法则演进成一个包含成百上千节点的复杂巨系统后，如何洞察其内部
        状态、快速定位故障？
     Method: 统一采集“三大支柱”信号，构建全链路可观测性平台。
             (a) 日志 (Logs): 记录离散事件
             (b) 指标 (Metrics): 量化性能，如 P50 (中位数) 延迟[reference:11][reference:12]
             (c) 链路追踪 (Tracing): 通过TraceID串联一次完整请求调用链
     Concept: 单一窗口 (Single Pane of Glass)。将三支柱数据关联，实现从宏观指标
               (如P99延迟毛刺) 到微观链路 (单次请求瓶颈) 的穿透式分析。

SEE ALSO
     scalability_rules(5), cap_theorem(7), sharding(7), consensus(3),
     raft(1), docker(1), kubernetes(1)
