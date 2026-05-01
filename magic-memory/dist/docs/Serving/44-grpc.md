GRPC(7) gRPC Programmer's Manual GRPC(7)

NAME
gRPC -- 高性能、跨语言的远程过程调用框架

DESCRIPTION
gRPC 是 Google 发起的开源 RPC 框架，核心目标：让一台机器上的程序
像调用本地函数一样调用另一台机器上的方法（只函数名、参数、返回类型必须
事先声明）。其关键技术选型为：

• IDL/序列化：Protocol Buffers (protobuf)，二进制编码。
• 传输协议：HTTP/2，多路复用长连接。
• 支持四种调用模式：一元、服务端流、客户端流、双向流。

本文采用“问题 → 解决 → 引出的下一问题”的链式推导方式，逐层剖析
gRPC 的核心原理。

Q₀: 如何让不同语言编写的服务互相调用？
┌ 痛点：调用方与被调用方可能分属不同语言，网络两端需要统一的
│ 数据表示和方法约定。
│
└ 解决：定义一套语言无关的接口描述 (IDL)
│
│ .proto 文件 ──protoc 编译器──▶ 各语言 stub / skeleton
│
├ 定义服务 + 消息格式：protobuf 同时承担 IDL 和序列化双重角色
│
└ 引出下一问题：IDL 只解决了“如何描述”的问题，真正在线路上传输时，
数据以什么编码格式发送才能既小又快？

Q₁: 网络传输数据如何紧凑编码以节省带宽和 CPU？
┌ 痛点：JSON 文本格式体积大，解析慢，类型校验弱。
│
└ 解决：Protocol Buffers 二进制编码
│
├ 原理：用字段编号代替字段名，配合 varint 等技巧，消息自描述性信息
│ 不进入载荷
│
├ 效果：
│ • 数据体积比 JSON 小 3-5 倍
│ • 序列化速度比 JSON 快 4-8 倍
│ • 强类型契约，编译期类型安全
│
├ 编码示例：
│ message Point { int32 x = 1; int32 y = 2; }
│ → 二进制帧仅含字段号 + 值，不传 "x"/"y" 字符。
│
└ 引出下一问题：数据体已经足够紧凑，但网络上同时传输大量调用
时，如果每个请求都独占一个 TCP 连接，连接开销和延迟仍然是
瓶颈。传输层如何支撑高并发？

Q₂: 传输层如何以最少的连接支撑海量并发 RPC 调用？
┌ 痛点：HTTP/1.1 每个请求独立连接或受队头阻塞制约，高并发场景
│ 连接开销大，吞吐上限低。
│
└ 解决：基于 HTTP/2 构建传输层
│
├ 1. 多路复用 (Multiplexing)
│ • 单个 TCP 连接上并发交错传输多个 Stream (双向字节流)
│ • 每个 Stream 由一个 31-bit Stream Identifier 标识
│ • 任一端点可交错发送来自不同 Stream 的 Frame
│
├ 2. 二进制分帧 (Binary Framing)
│ • 最小通信单位：帧 = 9 字节帧头 + 变长载荷
│ • 帧头定义：Length(24) + Type(8) + Flags(8) + StreamID(31)
│ • 10 种帧类型：HEADERS、DATA、SETTINGS、PING、GOAWAY 等
│ (gRPC 主要使用 HEADERS、DATA、CONTINUATION、RST_STREAM)
│
├ 3. 头部压缩 (HPACK)
│ • 静态/动态表 + Huffman 编码，请求头体积减少 50%-70%
│
├ 4. 流控 (Flow Control)
│ • Connection 级与 Stream 级 WINDOW_UPDATE 机制
│ • 避免发送方压垮接收方缓冲区
│
└ 引出下一问题：HTTP/2 提供了通用的帧基于多路复用传输层，但 RPC
还需要明确的“请求-响应”配对、超时、状态码、消息边界等语义。
gRPC 如何在 HTTP/2 之上定义这些 RPC 专用语义？

Q₃: gRPC 如何在 HTTP/2 帧之上封装一次完整的 RPC 调用？
┌ 痛点：HTTP/2 仅提供通用帧传输能力，RPC 需要清晰界定一次调用的
│ 生命周期、消息边界、超时与状态。
│
└ 解决：gRPC 定义一套长度前缀消息格式与帧组合规范
│
├ 消息格式：Length-Prefixed Message
│ ┌─────────────────────────────────────┐
│ │ Compressed(1) | Length(31-bit) | Payload │
│ └─────────────────────────────────────┘
│ • 压缩标志为 1 时，Payload 使用 HEADERS 帧中声明的算法压缩
│ （grpc-encoding: gzip/deflate/snappy 等）
│
├ 请求 (Request)
│ ① HEADERS 帧 → 携带伪头部 :method=POST, :path=
│ "/{package}.{Service}/{Method}"，以及 grpc-timeout,
│ content-type=application/grpc, te=trailers 等自定义头
│ ② 0 个或多个 DATA 帧 → 每个 DATA 帧承载一条 Length-Prefixed
│ Message；最后一个 DATA 帧带 END_STREAM 标志
│
├ 响应 (Response)
│ ① HEADERS 帧 → :status=200, grpc-encoding, content-type
│ ② 0 个或多个 DATA 帧 → Length-Prefixed Message 负载
│ ③ TRAILERS 帧 → grpc-status, grpc-message（调用状态码和消息）
│ • Trailer 独立发送的原因：流式模式下，服务器在全部消息
│ 发送完之前无法确定最终状态码
│
├ 超时处理：deadline 通过 grpc-timeout 头或配置传递，
│ 超时后 RST_STREAM 帧终止调用。
│
└ 引出下一问题：协议层已支持“单个请求-单个响应”的基本模式，但如果
数据量大或需要实时推送，单请求-单响应的模式如何扩展？

Q₄: 如何支持大数据传输和实时双向通信？
┌ 痛点：一元 RPC 要求客户端发送完整请求后等待服务端返回完整响应，
│ 不适合大文件上传、实时行情推送、聊天等场景。
│
└ 解决：基于 HTTP/2 双向流特性，定义四种服务方法类型
│
├ 1. Unary RPC (一元)
│ rpc GetUser(Req) returns (Resp);
│ → 1 Request → 1 Response，类比本地函数调用
│
├ 2. Server Streaming (服务端流式)
│ rpc ListFeatures(Req) returns (stream Resp);
│ → 1 Request → N Response (Stream)，服务端持续推送消息序列
│ 典型场景：股票行情推送，单服务器支撑 10 万+ 并发订阅
│
├ 3. Client Streaming (客户端流式)
│ rpc Upload(stream Req) returns (Resp);
│ → N Request (Stream) → 1 Response
│ 典型场景：日志批量上传、文件上传
│
├ 4. Bidirectional Streaming (双向流式)
│ rpc Chat(stream Msg) returns (stream Msg);
│ → N Request (Stream) ↔ M Response (Stream)
│ 两端流独立操作，可交错读写，消息顺序在各自流内保序
│ 典型场景：实时聊天、视频会议
│
└ 引出下一问题：海量 RPC 调用在生产环境下，如何保证高吞吐、
低延迟，并避免单连接成为瓶颈？

Q₅: 生产环境下如何优化 gRPC 的吞吐与延迟？
┌ 痛点：虽然单连接可支撑大量并发流，但过度复用单连接可能导致
│ 流控排队、负载不均等问题。
│
└ 解决：通道复用 / 连接池 / 负载均衡 / 心跳保活 / 异步调用
│
├ 1. Channel 复用
│ • 一个 gRPC Channel 对应一条 HTTP/2 长连接
│ • 不要每次调用都创建新 Channel（涉及 Socket → TCP →
│ TLS → HTTP/2 建联，开销显著）
│ • Channel 线程安全，多线程 / 多协程可并发复用
│
├ 2. 连接池与并发控制
│ • 默认单连接最大并发流约 100（可配置 settings
│ SETTINGS_MAX_CONCURRENT_STREAMS）
│ • 达到上限后新调用排队 → 高负载场景建议启用
│ EnableMultipleHttp2Connections 或使用 Channel Pool
│
├ 3. 负载均衡
│ • 客户端负载均衡：Resolver 发现端点 → Balancer 选择后端
│ • 策略：round_robin / pick_first / 权重 / 最少连接
│ • 服务端可通过 Service Config 下发负载策略
│
├ 4. 连接保活
│ • keepalive ping 防止空闲连接被中间设备销毁
│ • 参数：GRPC_ARG_KEEPALIVE_TIME_MS, GRPC_ARG_KEEPALIVE_TIMEOUT_MS
│ • 弱网环境下合理配置可将重连次数降低 60%
│
└ 5. 异步非阻塞调用
• 同步调用阻塞当前线程直到收到响应，适用于低 QPS 场景
• 异步 / 回调模式可不阻塞线程，提升吞吐
• 客户端使用 StreamObserver 或 Future/CompletableFuture 模式

SUMMARY: 核心原理推导链
Q₀: 跨语言调用 → IDL (protobuf)
│ ↓
Q₁: 数据编码 → 二进制序列化 (varint + 字段号)
│ ↓
Q₂: 传输并发 → HTTP/2 多路复用 + 分帧 + HPACK
│ ↓
Q₃: RPC 语义 → 长度前缀消息 + HEADERS/DATA/TRAILERS 帧组合
│ ↓
Q₄: 流式通信 → 四种服务方法（Unary/Server/Client/Bidi Streaming）
│ ↓
Q₅: 生产优化 → Channel 复用 + 连接池 + 负载均衡 + 心跳保活

SEE ALSO
protoc(1), HTTP/2 RFC 7540, protocol-buffers(7), grpc-go(1),
grpc-java(1), Envoy Proxy documentation.

AUTHOR
gRPC 由 Google Inc. 发起并开源，现为 CNCF 孵化项目。

VERSION
This page describes gRPC protocol as of v1.x 系列。

