WEBSOCKET(7) — Linux Programmer's Manual
NAME
WebSocket — a full-duplex, message-oriented, application-layer protocol over a single TCP connection, standardized as RFC 6455

SYNOPSIS
text
ws://host[:port]/path
wss://host[:port]/path          (over TLS)
DESCRIPTION
WebSocket provides a persistent bidirectional communication channel between a client and a server with minimal per-message overhead (2–14 bytes), replacing the high-overhead, client-initiated-only model of HTTP with a stateful, frame-based message stream.

text
WebSocket  ::=  Opening Handshake  →  Data Framing  →  Closing Handshake
                   (HTTP Upgrade)       (TCP/TLS)         (Close Frames)
Q0: WHY NOT HTTP?
PROBLEM
HTTP/1.1 is a client-initiated, request-response protocol where the server cannot push data unsolicited. To simulate "real-time" updates, applications resort to polling or long-polling, both of which carry significant overhead:

text
HTTP Short Polling:
  Client ──GET──► Server
  Client ◄──200── Server          (each request: ~600 bytes of headers)
  Client ──GET──► Server
  Client ◄──200── Server
  ...

HTTP Long Polling:
  Client ──GET──► Server ──(hold)──► (data arrives) ──► Client ◄──200── Server
  Client ──GET──► Server ──(hold)──► ...
Overhead Analysis (per message):

Protocol	Header size	Direction	Connection	Latency
HTTP polling	500–2000 bytes	Client→Server only	New per request	RTT × polling interval
HTTP long-polling	500–2000 bytes	Client→Server only	Held open	RTT + hold time
WebSocket (post-handshake)	2–14 bytes	Bidirectional	Single persistent	~0 (warm)
If an application polls more than once per second, the bandwidth waste from HTTP headers alone exceeds the payload size for most messages.Modern HTTP/2 and HTTP/3 add multiplexing and header compression but retain the request-response paradigm — they remain fundamentally unsuitable for true bidirectional push.

text
Problem:  HTTP cannot push. Polling wastes bandwidth.
           │
           ▼
Q1:  How does WebSocket establish a full-duplex channel
     while maintaining backward compatibility with HTTP
     infrastructure (proxies, firewalls, CDNs)?
Q1: THE OPENING HANDSHAKE — HTTP UPGRADE
SOLUTION
WebSocket bootstraps over HTTP/1.1 using the protocol upgrade mechanism: the client sends a specially formed HTTP GET request; the server responds with 101 Switching Protocols. After this single exchange, HTTP is discarded and the TCP connection carries only WebSocket frames.

Client Request (HTTP Upgrade)
text
GET /chat HTTP/1.1
Host: server.example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13
Origin: http://example.com

Key headers:

Upgrade: websocket — signals intent to switch protocols.

Sec-WebSocket-Key — 16-byte random nonce, Base64-encoded. Proves the server understands WebSocket.

Sec-WebSocket-Version — always 13 (RFC 6455).

Server Response
text
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=

Sec-WebSocket-Accept Derivation
text
Sec-WebSocket-Accept  =  Base64( SHA-1( Sec-WebSocket-Key + GUID ) )

where:
  GUID  =  "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"   (fixed magic string)

This magic GUID ensures the response is not a cached HTTP reply — it mathematically confirms the server intentionally performed the WebSocket handshake, preventing accidental or cross-protocol connections.

text
Handshake establishes the channel, but HTTP is now gone.
           │
           ▼
Q2:  Once HTTP is out of the picture, how does WebSocket
     structure data on the raw TCP stream?
Q2: DATA FRAMING — FROM BYTE STREAM TO MESSAGES
PROBLEM
TCP provides an undifferentiated byte stream with no message boundaries. Sending two 100-byte messages on raw TCP may arrive as one 200-byte chunk, three chunks of 80/70/50, or any other fragmentation pattern. The application must implement its own delimiters to reconstruct messages.

SOLUTION
WebSocket adds a frame layer: every unit of transmission is a self-describing frame with type, length, and boundary markers. This converts TCP's opaque byte stream into a message-oriented protocol.

Frame Format (RFC 6455 §5.2)
text
  0               1               2               3
  0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |F|R|R|R| opcode (4) |M| Payload len(7)| Extended payload (16/64)
 |I|S|S|S|              |A|              | (if len=126 or 127)
 |N|V|V|V|              |S|              |
 | |1|2|3|              |K|              |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 | Masking Key (0 or 4 bytes, if MASK=1)                        |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 | Payload Data (n bytes)                                       |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Field Definitions
text
FIN  (1 bit)   ::=  1 ⇒ final fragment of message
                    0 ⇒ more fragments follow

RSV1/2/3 (3 bits) ::=  0 unless a negotiated extension defines meaning

OPCODE (4 bits)  ::=  %x0  → Continuation frame
                      %x1  → Text frame       (UTF-8 validated)
                      %x2  → Binary frame
                      %x3–7→ Reserved (non-control)
                      %x8  → Connection Close
                      %x9  → Ping
                      %xA  → Pong
                      %xB–F→ Reserved (control)

text
MASK  (1 bit)    ::=  1 ⇒ payload masked (client→server: mandatory)
                      0 ⇒ payload unmasked (server→client: optional)

Payload Length  ::=  if len ∈ [0,  125]    → 7-bit value
                     if len ∈ [126, 2^16-1] → code 126, then 16-bit extended
                     if len ∈ [2^16, 2^64-1]→ code 127, then 64-bit extended

Masking Key (4 bytes) ::=  random 32-bit value
   Payload_Data[i]  =  Frame_Payload[i]  XOR  Masking_Key[i MOD 4]

Why Client→Server Masking?
The mask prevents cache poisoning attacks: a malicious script could send a frame that a misconfigured intermediate proxy interprets as an HTTP request, injecting content into the proxy cache. Masking ensures every client frame is cryptographically randomized on the wire, making such attacks computationally infeasible.

Per-Message Overhead Accounting
text
Overhead(server→client)  =  2 bytes  (FIN+RSV+OPCODE+MASK+len=7bit)
                          =  4 bytes  (for 126 ≤ len ≤ 65535)
                          = 10 bytes  (for len ≥ 65536)

Overhead(client→server)  =  Above + 4 bytes (masking key)

A "Hello" text message (~5 bytes payload) requires a 7-byte WebSocket frame vs. a ~600-byte HTTP request — a 98.8% overhead reduction.

text
Frames solve the message boundary problem on a single TCP stream.
           │
           ▼
Q3:  How does WebSocket handle messages larger than a single
     frame, and how are control signals embedded in the stream?
Q3: FRAGMENTATION & CONTROL FRAMES
Message Fragmentation
A logical message may span multiple frames via FIN=0 chaining:

text
Message  ::=  Frame(FIN=0, OPCODE≠0)  +
              Frame(FIN=0, OPCODE=0)  +
              ...                     +
              Frame(FIN=1, OPCODE=0)

Example (streaming unknown-length JSON):
  Frame 0:  FIN=0  OPCODE=%x1 (Text)  payload="{\"users\":["
  Frame 1:  FIN=0  OPCODE=%x0 (Cont)  payload="{\"id\":1},"
  Frame 2:  FIN=1  OPCODE=%x0 (Cont)  payload="]}"

Control Frame Interleaving
Control frames (Close, Ping, Pong) may be interleaved between fragmented message frames — they terminate the fragmentation sequence if not handled correctly.

text
Stream multiplexing on a single connection:
  [TEXT fin=0] [PING] [CONT fin=1]     ←  Control frames injectable
Close Frame (OPCODE %x8)
text
Close Frame  ::=  FIN=1, OPCODE=%x8
                  [Status Code (2 bytes, big-endian)]
                  [Reason (UTF-8, optional)]

Status Codes:
  1000  →  Normal closure
  1001  →  Endpoint going away
  1002  →  Protocol error
  1003  →  Unsupported data type
  1008  →  Policy violation
  1011  →  Internal server error

Ping/Pong (OPCODE %x9 / %xA)
text
Ping Frame  ::=  FIN=1, OPCODE=%x9, [Application Data, max 125 bytes]
Pong Frame  ::=  FIN=1, OPCODE=%xA, [Same Application Data (echoed)]

Pong frames must be sent as soon as practicable and must echo the exact payload of the triggering Ping. Either endpoint may initiate a Ping. Applications use this for latency measurement and connection health checks.

text
Control frames handle lifecycle and health.
           │
           ▼
Q4:  How does the application know the connection is still alive
     when the network goes silent?
Q4: CONNECTION LIVENESS — HEARTBEATS & ZOMBIE DETECTION
PROBLEM
TCP connections can silently die: a Wi-Fi disconnect, an elevator ride, or a NAT timeout may cause a connection that appears open on both ends to be dead. Without an explicit RST or FIN, the TCP socket stays open — but no data flows. These are zombie connections.

The Three Heartbeat Layers
text
Layer                    | Mechanism              | Visibility     | Latency
-------------------------|------------------------|----------------|--------
Protocol (RFC 6455)      | Ping/Pong frames       | C++/server only| ~RTT
Application              | JSON {"type":"ping"}   | JavaScript     | ~RTT + processing
Transport (TCP)          | SO_KEEPALIVE           | OS kernel      | Default 7200s (!)

Protocol-level Ping/Pong
The most efficient mechanism: 2-byte overhead, handled at the protocol layer. But browsers do not expose ping/pong to JavaScript — the browser responds to server pings automatically at the RFC level, invisible to application code. Browser applications must implement application-level keepalive messages instead.

Configuring Heartbeat Interval
text
Heartbeat_Interval  ≤  0.75  ×  min(Proxy_Timeout)

Common proxy idle timeouts:
  Nginx         →  60 s
  AWS ALB       →  60 s
  Cloudflare    →  100 s
  Google Cloud  →  30 s

Thus, behind Nginx: send heartbeats every ~45 seconds.

Zombie Cost at Scale
text
10,000 zombie connections × 5 KB/connection  =  50 MB  (wasted memory)
10,000 zombie connections × 1 fd/connection  =  10,000 (wasted file descriptors)

text
Liveness is handled. Now we know how one connection works.
           │
           ▼
Q5:  WebSocket gives us a raw bidirectional pipe.  What does
     it NOT give us, and what must the application provide?
Q5: THE APPLICATION-LAYER GAP
PROBLEM
After the handshake completes, the developer holds something closer to a raw TCP socket:

text
WebSocket DOES NOT provide:
  ❌  Request-response pairing
  ❌  Per-message status codes
  ❌  Content-type negotiation
  ❌  Built-in acknowledgments (did the other side receive my message?)
  ❌  Automatic reconnection
  ❌  State synchronization on reconnect
  ❌  Authentication per message
Everything the application needs beyond a raw bidirectional pipe — message routing, acknowledgments, reconnection, state sync — must be built on top of the protocol, or adopted from an existing higher-level library.

Reconnection & State Sync
Connection loss is a certainty, not an edge case. When the client reconnects, server and client states have diverged:

text
Approaches:

(1)  Sequence Numbers
     Server assigns monotonic ID per message.
     On reconnect: client sends last_seen_ID, server replays delta.
     Cost: server must buffer messages.

(2)  Event Sourcing
     Server stores full event log.
     On reconnect: client replays from its last checkpoint.
     Cost: storage grows; requires compaction/snapshotting.

(3)  Full State Sync
     Server sends complete current state on reconnect.
     Cost: O(state_size) bandwidth; wasteful for small deltas.

text
Application semantics are the developer's concern.
           │
           ▼
Q6:  Once scaled to production, what are the real bottlenecks —
     and how do they compound?
Q6: SCALING — FROM ONE TO MILLIONS
Connection Capacity
A well-tuned Linux server can hold 500K+ idle WebSocket connections:

text
Resource Ceilings:
  File descriptors   →  default 1,024 → set RLIMIT_NOFILE to 1M+
  Memory             →  2–10 KB per idle connection → 500K ≈ 2.5 GB
  CPU                →  near zero at idle

The Real Killer: Connection Churn, Not Count
Opening a new WebSocket connection requires:

text
TCP handshake     →  1 RTT
TLS negotiation   →  1-2 RTTs + RSA/ECDHE CPU (1–5 ms per handshake)
HTTP Upgrade      →  1 request/response round
A server handling 10,000 new connections per second burns 10–50 seconds of CPU time per second on TLS alone — mathematically impossible on a single core.

text
Scale_Limiting_Factor  :=  TLS_Handshake_CPU_Cost  ×  New_Connections_Per_Second
Past ~100K Connections: Horizontal Scaling
Beyond ~100K active connections or ~50K messages/sec on a single server, horizontal scaling becomes necessary:

text
Architecture Requirements:
  - Sticky sessions (session affinity) at the load balancer
  - Redis Pub/Sub for cross-server message routing
  - Graceful connection migration on scale-down
  - Heartbeat coordination across fleet
This is a stateful protocol — unlike stateless HTTP, a WebSocket connection is tied to a specific server. Load balancing requires session affinity, and cross-server message delivery requires a pub/sub backbone.

EXIT STATUS / CLOSE CODES
text
Code   Name                  Description
1000   Normal Closure         Purpose fulfilled; clean shutdown.
1001   Going Away             Endpoint navigated away or server restarted.
1002   Protocol Error         RSV bit set without extension negotiation.
1003   Unsupported Data       Received frame type not understood.
1008   Policy Violation       Application-level reason (e.g., auth failure).
1011   Internal Error         Unexpected server condition.
SEE ALSO
RFC 6455 — The WebSocket Protocol
RFC 7936 — WebSocket Protocol Errata
RFC 8441 — Bootstrapping WebSockets with HTTP/2
http(7), tcp(7), tls(7)

COLOPHON
This page is a conceptual reference for the WebSocket protocol as defined by RFC 6455. It describes the layered design — from HTTP upgrade to frame encoding, fragmentation, control signaling, heartbeat management, and scaling constraints — in a systematic, problem-to-solution derivation.

text
WebSocket Core Architecture (Summary):

Q0: HTTP can't push. Polling wastes bandwidth.
  → Q1: HTTP Upgrade Handshake (101 Switching Protocols)
    → Q2: Frame Layer (2-14 byte headers, OPCODE-based typing)
      → Q3: Fragmentation + Control Frame Interleaving
        → Q4: Ping/Pong Heartbeats (protocol vs application vs TCP)
          → Q5: Application-Layer Gap (state sync, auth, reconnection)
            → Q6: Horizontal Scaling (churn > count, sticky sessions)
