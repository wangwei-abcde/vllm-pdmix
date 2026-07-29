# A2-A3 跨 DP 卡死根因分析

> **注意：以下结论部分为代码分析+日志推导的假设，尚未通过修复验证。** 标注 `[假设]` 的部分需要进一步确认。

## 背景

- **组网拓扑**：
  ```
  Server A (边侧)              Server B (云侧)
  ┌────────────────┐          ┌────────────────┐
  │ Edge0 (A2 NPU) │  RoCE    │ Cloud0 (A3 NPU)│  ← DP0
  │ Edge1 (A2 NPU) │  RoCE    │ Cloud1 (A3 NPU)│  ← DP1
  └────────────────┘          └────────────────┘

  通信组：
  - DP0: Edge0 + Cloud0 (跨节点, RoCE)
  - DP1: Edge1 + Cloud1 (跨节点, RoCE)
  - EP_edge: Edge0 + Edge1 (同节点, HCCS)
  - EP_cloud: Cloud0 + Cloud1 (同节点, HCCS)
  - alt_device_group (per DP): P2P 通信器 (跨节点, RoCE)
  ```

- **关键事实**：EP_cloud 通信（all_gather/ALLTOALL）在同节点内走 **HCCS**，与边云 P2P 的 **RoCE** 物理链路分离。
- A2-A2 同构环境正常，A2-A3 异构环境非必现卡死
- 现象：多请求第 5 步卡死，单请求第二次也可能卡死

---

## 1. 云侧 MoE 卡死

### 1.1 现象

云侧 2 个 EP rank 全部卡在 `token_dispatcher.py` 中，具体位置：

[token_dispatcher.py#L685](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py#L685)

```python
torch.npu.synchronize()
```

或 [token_dispatcher.py#L677-L679](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py#L677-L679)

```python
global_input_tokens_local_experts_indices = torch.repeat_interleave(
    self.expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel()
)
```

### 1.2 卡死链路

```
_prepare_dispatch_tokens_ep()
  └─ gather_from_sequence_parallel_region(num_local_tokens_per_expert, group=self.ep_group)
       └─ HCCL all_gather (EP_cloud 通信器，同节点 Cloud0 + Cloud1 → HCCS)
            └─ torch.npu.synchronize()  ← 卡死
```

> **重要**：在本组网中，EP_cloud 只有 2 个 rank 且在同一台服务器，all_gather 使用 HCCS 直连（与边云 P2P 的 RoCE 物理链路分离），EP 通信本身不会阻塞。

EP 通信组初始化（Cloud0 + Cloud1，同节点）：

[token_dispatcher.py#L76-L78](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py#L76-L78)

```python
@property
def ep_group(self):
    """Get expert model parallel group."""
    return get_ep_group().device_group
```

[token_dispatcher.py#L80-L82](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py#L80-L82)

```python
@property
def ep_rank(self):
    return get_ep_group().rank_in_group
```

[token_dispatcher.py#L84-L86](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py#L84-L86)

```python
@property
def ep_size(self):
    return get_ep_group().world_size
```

> 注：本组网中 EP_cloud = {Cloud0, Cloud1}，同节点，走 HCCS。与边云 alt_device_group 的 RoCE P2P 使用不同的物理链路。

### 1.3 为什么 EP_HCCS 不阻塞却仍然卡死？`torch.npu.synchronize()` 是全局屏障

**关键认知**：`torch.npu.synchronize()` 是**全局同步原语**——它会等待 `torch.npu.current_stream()` 上所有已提交的 NPU 操作完成。而 GPU 上可能存在多个 stream，不同通信域的操作可能写在同一个 Device 的不同 stream 上。

在本组网中，卡死的真正原因不是 EP 通信（HCCS），而是云侧 NPU 的 **alt_device_group 流上有一个迟迟无法完成的 RoCE irecv**：

```
Cloud0 NPU 上的操作流:
  ┌─────────────────────────────────────────────────────┐
  │ stream_hccl_alt (alt_device_group, RoCE P2P):      │
  │   irecv(0→1, RoCE) ← PENDING!  等 Edge0 的 isend    │
  │                          数据在 RoCE 层完成传输      │
  │                                                     │
  │ stream_hccl_ep (EP_cloud, HCCS):                     │
  │   all_gather(HCCS) ← 已完成（同节点直连，极快）       │
  │   ALLTOALL(HCCS)                                     │
  │                                                     │
  │ stream_compute:                                      │
  │   ...MoE kernels... ← 依赖 stream_hccl_alt 的 irecv  │
  │                                                     │
  │ torch.npu.synchronize() ← 等所有 stream ← 卡死！     │
  └─────────────────────────────────────────────────────┘
```

**`torch.npu.synchronize()` 对 `torch.npu.current_stream()` 来说是 `c10d::Synchronize`，但它会阻塞直到设备上所有排队的操作完成**。因此，即使 EP 的 HCCS all_gather 已经完成，alt_device_group 上那个尚未完成的 RoCE irecv 仍然会卡住全局同步。

### 1.4 真正的卡死链路

```
T1  边侧: DECODE_FIRST → isend(0→1, RoCE, alt_group) 发 hidden_states
    云侧: passive_core 收 DECODE_FIRST ZMQ → 立即发布 DECODE_LAST (BUG!)
    
T2  边侧: 收 DECODE_LAST ZMQ → irecv(1→0, RoCE, alt_group) 等待云侧 isend
    云侧: worker 开始处理 DECODE_FIRST
    
T3  云侧: irecv(0→1, RoCE, alt_group) ← 应匹配 T1 的 isend
          但 RoCE 传输层有延迟（跨节点 RTT 大，或有 RNR 重试）
          irecv kernel 在 NPU 上 PENDING
    
T4  云侧: MoE → EP all_gather(HCCS) → 完成（CPU 返回）
          ALLTOALL(HCCS) → 完成
    
T5  云侧: torch.npu.synchronize() ← 全局屏障
          等 T3 的 RoCE irecv → 迟迟不完成 → 卡死
```

**为什么 irecv 迟迟不完成？**

RoCE（RDMA over Converged Ethernet）的两阶段完成语义：
1. 发送端：isend → RDMA Send/Write → 数据到达接收端 NIC
2. 接收端：irecv → 匹配到达的数据 → 写入 receiver buffer → 完成

如果步骤 1 时接收端尚未 posted irecv，RoCE 会产生 **RNR (Receiver Not Ready) NAK**。HCCL/HCOMM 层会重试，但每次重试有退避延迟。一旦接收端 posted irecv，重试的 RDMA Send 应该能匹配。但由于跨节点网络延迟 + RNR 重试开销，这个等待时间可能很长。

在极端情况下（恰好边侧也卡在 irecv 等待、云侧 worker 被调度延迟、或 RoCE QP 资源紧张），这个等待会变成死锁。

**验证方法**：在对 `torch.npu.synchronize()` 之前用 `torch.cuda.utilization()` 或 HCCL trace 检查 alt_device_group 上是否有未完成的通信操作。

### 1.5 为什么 A2-A2 不卡

在 A2-A2 同构环境，边云都在同一节点内，alt_device_group 的 P2P 走的是 **HCCS 直连**而非 RoCE：

| | A2-A2 (同节点) | A2-A3 (跨节点) |
|---|---|---|
| alt_device_group P2P | HCCS（纳秒级，无 RNR） | RoCE（微秒级，有 RNR/重试） |
| isend→irecv 完成延迟 | 几乎瞬时 | 可能数百微秒到毫秒 |
| torch.npu.synchronize() | 能顺利完成 | 被 pending RoCE irecv 阻塞 |

HCCS 是芯片间直连，发送完成后数据立即可被接收端读取，不存在 RoCE 的 RNR 重试机制。

---

## 2. 边侧 DECODE 收发问题

### 2.1 现象

边侧在云侧还未执行 DECODE_FIRST 时就收到了 DECODE_LAST，并继续往下执行 MoE 层，用错误的数据完成推理。

### 2.2 问题链路

#### 步骤一：云侧收到 DECODE_FIRST 后立即发布 DECODE_LAST

[passive_core.py#L612-L614](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/v1/engine/passive_core.py#L612-L614)

```python
if batch.scheduler_output.total_num_scheduled_tokens > 0:
    if batch.scheduler_output.batch_type == BatchType.DECODE_FIRST:
        self._maybe_publish_post_out(batch.scheduler_output)  # ← 立即发布！
```

注意对比：PREFILL_FIRST 是**延迟发布**的，等 cloud worker 完成后再发：

[passive_core.py#L615-L623](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/v1/engine/passive_core.py#L615-L623)

```python
    elif (
        batch.scheduler_output.batch_type == BatchType.PREFILL_FIRST
        and (slice_info is None or slice_info.is_last_slice)
    ):
        head_token = getattr(batch.scheduler_output, "head_token", None)
        if head_token:
            self._pending_post_out_by_head_token[head_token] = (  # ← 延迟发布
                batch.scheduler_output
            )
```

`_maybe_publish_post_out` 将 `DECODE_FIRST` 改写为 `DECODE_LAST` 并通过 ZMQ PUSH/PULL 发回边侧：

[passive_core.py#L625-L655](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/v1/engine/passive_core.py#L625-L655)

```python
def _maybe_publish_post_out(self, scheduler_output: SchedulerOutput) -> None:
    """Rewrite + publish a head-segment batch as a tail-segment one
    on the POST_OUT (cloud → edge) channel.
    """
    bt = scheduler_output.batch_type
    if bt == BatchType.PREFILL_FIRST:
        tail = replace(scheduler_output, batch_type=BatchType.PREFILL_LAST)
    elif bt == BatchType.DECODE_FIRST:
        tail = replace(scheduler_output, batch_type=BatchType.DECODE_LAST)
    # ...
    self._pp_pd_channel.publish(tail)  # ZMQ → edge
```

#### 步骤二：边侧收到 DECODE_LAST，等待 isend handle 完成

[worker.py#L564-L577](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/worker/worker.py#L564-L577)

```python
if self.model_runner._edge_cloud_enabled:
    bt = scheduler_output.batch_type
    if bt in (
        BatchType.PREFILL_FIRST, BatchType.DECODE_FIRST,
        BatchType.PREFILL_LAST, BatchType.DECODE_LAST,
    ):
        ch = self._hidden_channel_for(scheduler_output)
        self._wait_pp_send_work(ch)  # ← 等待之前的 DECODE isend handle
```

`_wait_pp_send_work` 等的是 **DECODE_FIRST 的 isend handle**：

[worker.py#L503-L516](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/worker/worker.py#L503-L516)

```python
def _wait_pp_send_work(self, channel: HiddenChannelType | None = None) -> None:
    # ...
    handles = self._pp_send_work_by_channel.pop(channel.value, [])
    for handle in handles:
        handle.wait()  # ← 仅等待本地 HCCL isend 完成！
```

**关键：`handle.wait()` 仅等待 HCCL isend 本地完成（数据从源 buffer 拷贝出去、操作提交到网络层），不等待远端 peer 执行匹配的 irecv。**

#### 步骤三：边侧进入 _edge_tail 执行 irecv

[worker.py#L739-L776](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/worker/worker.py#L739-L776)

```python
def _execute_model_edge_tail(self, scheduler_output, layer_slice_info):
    channel = self._hidden_channel_for(scheduler_output)
    tensor_dict, comm_handles, comm_postprocess = edge_cloud_broadcast_recv(
        num_tokens=scheduler_output.total_num_scheduled_tokens,
        channel=channel,   # ← HiddenChannelType.DECODE
        sp_chunk=...
    )
    # ...
    intermediate_tensors = AsyncIntermediateTensors(
        tensor_dict, comm_handles=comm_handles,
        comm_postprocess=comm_postprocess,
    )
    output = self.model_runner.execute_model(
        scheduler_output, intermediate_tensors, ...
    )
```

#### 步骤四：DECODE_FIRST 的 isend 和 DECODE_LAST 的 irecv 共用同一个 HCCL communicator

两者都通过 `_hidden_channel_for` 映射到 `HiddenChannelType.DECODE`：

[worker.py#L613-L622](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/worker/worker.py#L613-L622)

```python
def _hidden_channel_for(self, scheduler_output):
    bt = scheduler_output.batch_type
    if bt in (BatchType.PREFILL_FIRST, BatchType.PREFILL_LAST):
        return HiddenChannelType.PREFILL_1
    if bt in (BatchType.DECODE_FIRST, BatchType.DECODE_LAST):
        return HiddenChannelType.DECODE   # ← 同一个！
```

`HiddenChannelType.DECODE` 映射到 `alt_device_group`：

[parallel_state.py#L787-L794](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/distributed/parallel_state.py#L787-L794)

```python
if channel == HiddenChannelType.DECODE:
    return pp_group.alt_device_group  # ← 同一个 HCCL communicator
```

DECODE_FIRST 的 isend 使用 `alt_device_group`：

[parallel_state.py#L1000-L1001](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/distributed/parallel_state.py#L1000-L1001)

```python
handle = torch.distributed.isend(
    value, dst=pp_group.ranks[dst], group=group
)
# src=边(0), dst=云(1), group=alt_device_group
```

DECODE_LAST 的 irecv 也使用 `alt_device_group`：

[parallel_state.py#L1133-L1134](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/distributed/parallel_state.py#L1133-L1134)

```python
handle = torch.distributed.irecv(
    recv_view, src=pp_group.ranks[src], group=group
)
# src=云(1), dst=边(0), group=alt_device_group
```

### 2.3 数据混淆时序 [假设]

> **以下时序中"irecv 匹配到脏数据"是推断，尚未通过精确日志（如对比 head_sum）直接确认脏数据来源。**

```
时间线:  边侧(A2)                              云侧(A3)
─────────────────────────────────────────────────────────────────
T1       isend(0→1, alt_group)                 
         DECODE_FIRST 数据发出                  
         handle.wait() 本地完成 ✓              
                                                
T2       ← ZMQ: DECODE_LAST                    收到 DECODE_FIRST ZMQ
                                               立即发布 DECODE_LAST
                                               (不等 worker 执行!)
                                                
T3       _wait_pp_send_work(DECODE)            云侧仍在忙 PREFILL 的
         handle 早已完成 → 通过 ✓               all_to_all，未执行:
                                               - irecv(0→1) 消费 isend
                                               - isend(1→0) 发回数据
                                                
T4       irecv(1→0, alt_group) ←              （云侧 isend 还未发出）
         等待匹配...                            
                                                
         此时 alt_device_group 上有:
         - 边侧 isend(0→1) 的残留数据（云侧未消费）
         - 边侧 irecv(1→0) 等待匹配
         
         [假设] 在 A2-A3 跨节点 HCCL 中，同一 communicator 
           内的数据可能被错误匹配
         [假设] irecv 读到了旧数据 ✓ (脏数据)
```

**[假设] 待确认的脏数据来源**：

1. **自消费（self-consumption）**：边侧 `isend(0→1)` 的数据在 HCCL 内部缓冲区中，被同一 communicator 上的 `irecv(1→0)` 错误匹配。但 (src,dst) 方向不同，理论上不应匹配。

2. **上一轮残留**：上一轮 DECODE 云侧 `isend(1→0)` 的数据在 HCCL 缓冲区中未被完全清理，被新 `irecv(1→0)` 重复匹配。

3. **HCCL tag 回绕**：多次 P2P 操作后 HCCL 内部 tag 回绕，新旧操作的 tag 相同导致误匹配。

4. **根本不需要匹配**：irrev 可能处于等待状态，边侧后续的 `execute_model` 在 `wait_for_comm()` 时阻塞，导致边侧也卡住——而非用脏数据继续执行。

**验证方法**：通过对比边侧 `[FP-EDGE-HEAD-GATHERED]` 的 `head_sum` 与 `[FP-EDGE-TAIL-RECV]` 接收数据的 `head_sum`，确认是否真的收到了与发送端不同的数据。

### 2.4 为什么 A2-A2 不卡但 A2-A3 卡 [假设]

| 环境 | P2P 通信方式 | 通道隔离 | 数据混淆 |
|------|-------------|---------|---------|
| A2-A2 | 同节点 HCCS 直连 | HCCS 不同物理通道隔离性好 | 不会混淆 |
| A2-A3 | 跨节点 RoCE 网络 | 同一物理链路，HCCL 内部缓冲区可能残留 | **会混淆** |

A2-A2 同节点的 `isend(0→1)` 和 `irecv(1→0)` 走不同的 HCCS 物理通道，数据不会混淆。A2-A3 跨节点走同一 RoCE 链路，HCCL 内部缓冲管理中旧数据可能被新 irecv 匹配。

> **[假设] 以上物理通道差异的解释是推测性的，缺少 HCCL 内部实现文档支持。** 另一种可能是 A2-A3 的 Worker 进程调度时序差异（IPC/进程间通信开销），导致 DECODE_FIRST enqueue → worker dequeue 的延迟更大，使边侧更容易在云侧处理完成前收到 DECODE_LAST。

### 2.5 边侧为何能用脏数据继续执行 [假设]

> **目前观察到边侧 `execute_model` 能执行 MoE 层并打印完成日志。但实际数据是否正确未确认。**

1. `irecv` **可能**匹配到了 HCCL 内部缓冲中的旧数据（上一轮 DECODE 云侧 asend 的残留）
2. `AsyncIntermediateTensors.wait_for_comm()` 等待 irecv handle 完成后，tensor 数据就绪
3. 边侧 `model_runner.execute_model()` 使用这些张量执行 MoE 层
4. 由于脏数据的 shape 和 dtype 合法（都是 num_tokens=1, hidden_size 一致），不会触发 shape 错误
5. 因此边侧能"成功"往下执行，但用的是错误数据

**不确定点**：
- 如果 `irecv` 实际处于等待状态（未匹配到任何数据），边侧的 `execute_model` 会在首次访问 `intermediate_tensors.tensors` → `wait_for_comm()` 时阻塞
- 需要确认边侧日志中 `[HANG] edge tail recv EXIT` 之后到 `Execute model after` 之间的时间差是否正常

---

## 3. 修复方案（待验证）

将 DECODE_FIRST 也纳入延迟发布队列，与 PREFILL_FIRST 一样的处理方式。**当前已 revert，尚未在实际环境中验证修复效果。**

### 3.1 延迟存储

```python
# passive_core.py L612-623 (修改后)
if batch.scheduler_output.total_num_scheduled_tokens > 0:
    if (
        batch.scheduler_output.batch_type == BatchType.DECODE_FIRST
        or (
            batch.scheduler_output.batch_type == BatchType.PREFILL_FIRST
            and (slice_info is None or slice_info.is_last_slice)
        )
    ):
        head_token = getattr(batch.scheduler_output, "head_token", None)
        if head_token:
            self._pending_post_out_by_head_token[head_token] = (
                batch.scheduler_output
            )
```

### 3.2 扩展 ack 处理

```python
# passive_core.py L518 (修改后)
if result.get("batch_type") not in (
    BatchType.PREFILL_FIRST, BatchType.DECODE_FIRST
):
    continue
```

### 3.3 效果

修复后的时序：

```
边侧                                   云侧
────────────────────────────────────────────────────────────
isend(0→1) DECODE_FIRST                
                                       收到 DECODE_FIRST ZMQ
                                       enqueue worker
                                       存 pending_post_out
                                       (不立即发布!)
                                       
                                       worker 执行 DECODE_FIRST
                                       irecv(0→1) 消费 isend 数据 ✓
                                       isend(1→0) 发出 DECODE_LAST 数据
                                       worker ack → drain_worker_completion_acks
                                       发布 DECODE_LAST ZMQ ↑
                                       
收到 DECODE_LAST ZMQ                    
_wait_pp_send_work → 通过              
irecv(1→0) → 匹配云侧 isend ✓
拿到正确数据 ✓
```

---

## 4. 相关文件汇总

| 文件 | 关键行 | 作用 |
|------|--------|------|
| `vllm_ascend/v1/engine/passive_core.py` | L612-623 | DECODE_FIRST 立即发布 DECODE_LAST（Bug 位置） |
| `vllm_ascend/v1/engine/passive_core.py` | L626-655 | `_maybe_publish_post_out`: DECODE_FIRST → DECODE_LAST 改写 |
| `vllm_ascend/v1/engine/passive_core.py` | L497-534 | `_drain_worker_completion_acks` + `_pending_post_out_by_head_token` |
| `vllm_ascend/worker/worker.py` | L503-516 | `_wait_pp_send_work`: handle.wait() 仅本地完成 |
| `vllm_ascend/worker/worker.py` | L613-622 | `_hidden_channel_for`: DECODE_FIRST/LAST → 同一通道 |
| `vllm_ascend/worker/worker.py` | L624-704 | `_execute_model_edge_head`: isend DECODE_FIRST 数据 |
| `vllm_ascend/worker/worker.py` | L739-812 | `_execute_model_edge_tail`: irecv DECODE_LAST 数据 |
| `vllm_ascend/distributed/parallel_state.py` | L787-794 | DECODE 通道 → alt_device_group |
| `vllm_ascend/distributed/parallel_state.py` | L1000-1001 | isend on alt_device_group |
| `vllm_ascend/distributed/parallel_state.py` | L1133-1134 | irecv on alt_device_group |
| `vllm_ascend/ops/fused_moe/token_dispatcher.py` | L76-86 | EP group 初始化（Cloud0 + Cloud1，同节点） |
| `vllm_ascend/ops/fused_moe/token_dispatcher.py` | L645-647 | EP_ALLGATHER（返回后卡在 synchronize） |
| `vllm_ascend/ops/fused_moe/token_dispatcher.py` | L685 | torch.npu.synchronize()（全局屏障，被 pending RoCE irecv 阻塞） |
| `vllm_ascend/distributed/parallel_state.py` | L787-794 | 各通信通道到 device_group 的映射（alt_device_group 走 RoCE） |
| `vllm_ascend/distributed/parallel_state.py` | L1000-1001 | isend on alt_device_group (跨节点 RoCE) |
| `vllm_ascend/distributed/parallel_state.py` | L1133-1134 | irecv on alt_device_group (跨节点 RoCE) |

---

## 5. 待验证项

| # | 验证项 | 方法 | 优先级 |
|---|--------|------|--------|
| 1 | DECODE_LAST 延迟发布修复是否消除卡死 | 重新应用修复 commit，运行多请求测试 | 高 |
| 2 | 云侧 `torch.npu.synchronize()` 是否被 alt_device_group pending irecv 阻塞 | 在 synchronize 前 log 每个 stream 的状态；用 HCCL_ENTRY_LOG_ENABLE 确认 pending 操作 | 高 |
| 3 | Cloud0 的 irecv 是否收到了数据（RNR 重试是否成功） | 在云侧 irecv 前后加 tensor sum dump，对比边侧 isend 的 head_sum | 高 |
| 4 | A2-A2 vs A2-A3 差异是 HCCS vs RoCE 的延迟差异 | 对比两边 isend→irecv 完成耗时的日志 | 中 |
| 5 | `torch.npu.synchronize()` 是否真的全局同步所有 stream | 阅读 NPU 文档确认 synchronize 语义（Device vs Stream 级别） | 中 |

---

## 6. 底层协议诊断日志

### 6.1 确认通信组物理拓扑

在 `parallel_state.py` 中添加日志，输出各通信组的 rank 列表和所在节点 IP：

```python
# parallel_state.py 中添加
import socket

def _log_group_info(group_name: str, group):
    """在初始化通信组时打印拓扑信息"""
    world_size = torch.distributed.get_world_size(group=group)
    rank = torch.distributed.get_rank(group=group)
    hostname = socket.gethostname()
    logger.info(
        f"[PD-TOPO] {group_name}: "
        f"world_size={world_size}, local_rank={rank}, "
        f"hostname={hostname}"
    )
```

在各类通信组初始化后调用：
- EP group: `ep_group` — 预计 {Cloud0, Cloud1} 同节点
- DP group: `dp_group` — 预计 {Edge0, Cloud0} 或包含所有 4 个 rank
- alt_device_group: — 预计 per-DP 的 P2P 通信器
- TP group: `tp_group`（如果启用）

### 6.2 确认底层传输协议（HCCS vs RoCE）

HCCL 本身没有直接的 Python API 查询当前 operation 使用的协议。验证方法：

#### 方法 A：HCCL Entry Log

```bash
# 设置环境变量启用 HCCL API 日志
export HCCL_ENTRY_LOG_ENABLE=1
export HCCL_ENTRY_LOG_FILE=/tmp/hccl_entry_rank_${RANK}.log

# 日志中会显示每次通信操作的详细信息，包括 communicator name 和传输阶段
```

日志中关键字段：
- `[HcclRankGraphGetLinks]` — 两个 rank 之间的物理链路列表，包含 link 协议类型
- `[CalcLevel1ChannelRequest]` — 算法选择的传输协议（RoCE / HCCS）
- `[Channel] CreateChannel` — 实际创建的 Channel 协议

#### 方法 B：NPU Profiling Trace

```bash
# 启用 HCCL profiling
export HCCL_EXEC_TIMEOUT=1800
export HCCL_PROFILE_LEVEL=1  # 或 2 获取更详细信息

# 或在代码中：
# token_dispatcher.py 的 synchronize 之前
import torch_npu
torch_npu.npu.profiler.profile(
    activities=[torch_npu.npu.profiler.ProfilerActivity.CPU,
                torch_npu.npu.profiler.ProfilerActivity.NPU],
    record_shapes=True,
) as prof:
    # ... MoE 代码 ...
    torch.npu.synchronize()  # ← 卡死位置

# 从 trace 中可以看到：
# - 哪些 HCCL kernel 在哪个 stream 上执行
# - 每个 kernel 的完成状态
# - pending 的 RoCE 操作
```

#### 方法 C：确认 alt_device_group 实际链路类型

```python
# parallel_state.py 中，创建 alt_device_group 时添加
def _create_alt_device_group(pp_group):
    # ... existing code ...
    
    # 判断 P2P 链路类型
    local_rank = torch.distributed.get_rank(group=pp_group.alt_device_group)
    remote_rank = pp_group.ranks[1] if local_rank == pp_group.ranks[0] else pp_group.ranks[0]
    
    logger.info(
        f"[PD-P2P-LINK] alt_device_group: "
        f"local_rank={local_rank}, remote_rank={remote_rank}, "
        f"local_hostname={socket.gethostname()}"
        # 可通过 getenv("HCCL_CONNECT_TIMEOUT") 等环境变量间接推断
    )
```

### 6.3 诊断 `torch.npu.synchronize()` 阻塞原因

在 [token_dispatcher.py](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py) 中添加：

```python
# torch.npu.synchronize() 之前
logger.info(
    f"[PD-MOE-SYNC-BEFORE] rank={torch.distributed.get_rank()}, "
    f"batch_type={getattr(scheduler_output, 'batch_type', '?')}"
)

# 可选：检查 stream 状态
import torch_npu
default_stream = torch_npu.npu.current_stream()
logger.info(
    f"[PD-MOE-STREAM] default_stream={default_stream}, "
    f"device={torch_npu.npu.current_device()}"
)

torch.npu.synchronize()  # ← 卡死位置

logger.info(
    f"[PD-MOE-SYNC-AFTER] rank={torch.distributed.get_rank()}"
)
```

### 6.4 确认 RoCE RNR 重试是否发生

在 HCOMM 层面，可通过以下方式观察：

```bash
# 方法 1: 检查 RDMA 统计（需要 root）
# 在云侧节点上
cat /sys/class/infiniband/*/ports/*/counters/*rnr* 

# 方法 2: HCCL 调试日志
export HCCL_DEBUG_CONFIG="rdma"        # 如果支持 RDMA 调试
export HCOMM_RDMA_LOG_LEVEL=2          # HCOMM RDMA 日志级别

# 方法 3: 在代码中添加 HCCL stream 完成检查
# 在 token_dispatcher.py 中 synchronize 之前
# 尝试用 event 检测 alt_device_group stream 是否已完成
event = torch.npu.Event()
event.record(stream=alt_device_stream)  # 需要拿到 alt stream 引用
if not event.query():  # 非阻塞检查
    logger.warning("[PD-MOE-PENDING] alt_device_group stream not done!")
```

### 6.5 综合诊断脚本

```bash
#!/bin/bash
# run_with_diag.sh — 启用所有诊断日志运行

export HCCL_ENTRY_LOG_ENABLE=1
export HCCL_ENTRY_LOG_FILE=/tmp/hccl_entry_rank_${RANK}.log
export HCCL_EXEC_TIMEOUT=1800
export HCOMM_RDMA_LOG_LEVEL=2

# vLLM 日志级别
export VLLM_LOGGING_LEVEL=DEBUG

# 运行
python -m vllm.entrypoints.openai.api_server \
    --model <model_path> \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    ...
```

---

## 7. 分析方面全面汇总

### 7.1 死锁根因：`.cpu().tolist()` 触发 Device 级全局同步

**结论**: 云侧 MoE 的 EP_ALLGATHER 后调用 `.cpu().tolist()` 触发隐式 `torch.npu.synchronize()`（Device 级全局同步），此时 NPU 上还有未完成的边云 PP irecv（RoCE），导致死锁。

**证据** — [token_dispatcher.py#L675-L676](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py#L675-L676)：

```python
_hang_logger.error("[HANG] EP_ALLGATHER repeat_interleave ENTER: ep_rank=%s ravel=%s",
                  _hang_ep, num_global_tokens_per_local_expert.ravel().cpu().tolist())
```

以及 [token_dispatcher.py#L684-L686](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py#L684-L686)：

```python
_hang_logger.error("[HANG] EP_ALLGATHER npu_sync ENTER: ep_rank=%s", _hang_ep)
_hang_sys.stderr.flush()
torch.npu.synchronize()
```

---

### 7.2 `torch.npu.synchronize()` 是 Device 级全局同步

**结论**: `torch.npu.synchronize()` 调用底层 `aclrtSynchronizeDevice()`，等待该 NPU 上**所有 stream** 上的所有操作完成。云侧 EP 通信（HCCS stream）和边云 PP 通信（RoCE stream）共享同一个 NPU Device，全局同步点会相互阻塞。

**证据**: 组网中 Cloud0 的 NPU 上同时存在：
- `stream_hccl_ep`: EP_cloud all_gather（HCCS，同节点快）
- `stream_hccl_alt`: alt_device_group irecv（RoCE，跨节点可能延迟）
- `stream_compute`: MoE 计算 kernel

`torch.npu.synchronize()` 等所有 stream，alt_device_group 上的 pending RoCE irecv 阻塞了全局同步。

---

### 7.3 "假执行"流程：边侧 CPU 日志正常，NPU 实际未执行

**结论**: 边侧 DECODE_LAST 的 CPU 日志显示 model_runner.execute_model 正常完成，但 NPU 侧所有 kernel 因流依赖被阻塞（等待 irecv 完成）。

**证据** — [gpu_worker.py#L73-L114](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm/vllm/v1/worker/gpu_worker.py#L73-L114)：

```python
class AsyncIntermediateTensors(IntermediateTensors):
    """IntermediateTensors with lazy comm synchronization"""
    def wait_for_comm(self) -> None:
        if self._comm_waited:
            return
        # handle.wait() 只插入流依赖（event barrier），不阻塞 CPU
        if self._comm_handles:
            for handle in self._comm_handles:
                handle.wait()  # ← 0ms 返回，只插入依赖到 compute stream
        self._comm_waited = True
```

**证据** — [worker.py#L739-L813](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/worker/worker.py#L739-L813)：

```python
# 异步接收云侧张量，不等待通信完成
tensor_dict, comm_handles, comm_postprocess = edge_cloud_broadcast_recv(...)
intermediate_tensors = AsyncIntermediateTensors(
    tensor_dict, comm_handles=comm_handles, comm_postprocess=comm_postprocess,
)
# execute_model 传入未完成的张量，CPU 侧照常执行、打印日志
output = self.model_runner.execute_model(scheduler_output, intermediate_tensors, ...)
```

**完整"假执行"时序**：

```
CPU 侧（看似正常）                          NPU 侧（实际被卡住）
────────────────────────                          ───────────────────
edge_cloud_broadcast_recv()
  └─ hcclBatchIsendIrecv() → 返回          irecv 任务入队 HCCL recv stream

[HANG] edge tail recv EXIT ✓              irecv 等待 RoCE 数据到达
                                          （云侧从未发出 isend → 等不到）

AsyncIntermediateTensors.__init__()
  └─ handle.wait() → 0ms 返回 ✓          event barrier 插入：
  └─ [PP-EVT] WAIT-DONE dt=0.0ms ✓       compute stream 等 hccl stream

model_runner.execute_model()
  └─ LayerNorm(irecv_data) → kernel 发射 ✓  NPU kernel 排队，被 barrier 挡住
  └─ Linear → kernel 发射 ✓                NPU kernel 排队，被 barrier 挡住
  └─ ...
  └─ sample(logits) → kernel 发射 ✓         NPU kernel 排队，被 barrier 挡住
  └─ 返回 AsyncModelRunnerOutput ✓          所有 kernel 都在 NPU 上等 irecv

[PD-DIAG] C. edge tail OUTPUT ✓            ← log 正常打印

引擎继续下一个循环...                        NPU 上空转等待
需要读取 sampled_token_ids 时
  └─ tensor.cpu() / .item()               ← 触发隐式 synchronize
  └─ CPU 卡死 💀                              NPU 上第一个 kernel 永远无法执行
                                            （irecv event 永远不触发）
```

---

### 7.4 `handle.wait()` 只插入流依赖（非 CPU 阻塞）

**结论**: PyTorch 的 `work.wait()` 在 NPU 上调用 `aclrtStreamWaitEvent()`，只在 compute stream 上插入 event barrier，不阻塞 CPU 线程。`handle.wait()` 返回时间约 0ms。

**证据** — [gpu_worker.py#L87-L108](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm/vllm/v1/worker/gpu_worker.py#L87-L108)：

```python
def wait_for_comm(self) -> None:
    if self._comm_waited:
        return
    import time as _time
    _t0 = _time.monotonic()
    if self._comm_handles:
        for handle in self._comm_handles:
            handle.wait()  # ← CPU 不阻塞
    _dt_ms = (_time.monotonic() - _t0) * 1000
    # 日志输出 dt=0.0ms，证明 handle.wait() 不阻塞 CPU
    _logging.getLogger("vllm").error(
        "[PP-EVT] WAIT-DONE dt=%.1fms handles=%d",
        _dt_ms, len(self._comm_handles or []),
    )
```

其底层调用链：`ProcessGroupHCCL::wait()` → `aclrtStreamWaitEvent(compute_stream, hccl_stream_event)`，不调用 `aclrtSynchronizeStream()`，故 CPU 不等待。

---

### 7.5 A2 vs A3 通信路径差异（HCCL 底层）

**结论**: A3 走 DPU 新路径（`HcclSendNext` → `SendExec` → `HcclExecOp`），A2 走老路径（`HcclSendInner`，直接 kernel）。DPU 新路径多一级调度延迟。

**证据** — [send_op.cc#L48-L72](file:///Users/wangwei/Desktop/project/hccl/src/ops/send/send_op.cc#L48-L72)：

```cpp
HcclResult HcclSend(void *sendBuf, uint64_t count, HcclDataType dataType,
                    uint32_t destRank, HcclComm comm, aclrtStream stream) {
    if (IsHostDpu(comm)) {
        return HcclSendNext(sendBuf, count, dataType, destRank, comm, stream);  // A3: DPU新路径
    }
    if (GetHcommVersion() < CANN_VERSION(9, 0, 0)) {
        return HcclSendInner(sendBuf, count, dataType, destRank, comm, stream); // 老版本
    }
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType != DevType::DEV_TYPE_950) {
        return HcclSendInner(...);  // A2: 老路径，直接 kernel
    }
    return HcclSendNext(...);  // A3(950): DPU新路径
}
```

**证据** — [send_op.cc#L193-L223](file:///Users/wangwei/Desktop/project/hccl/src/ops/send/send_op.cc#L193-L223)：`SendExec` 中通过 `Selector` → `HcclExecOp` 下发操作，比 A2 的 `HcclSendInner` 多一层调度。

---

### 7.6 A2-A2 不会死锁的原因

**结论**: A2-A2 同构环境中，irecv 因通信延迟低（HCCL 老路径，无 DPU 调度），在 `.cpu().tolist()` 隐式同步前已就绪完成，不会阻塞全局同步。

**对比分析**：

| 维度 | A2-A2 | A2-A3 |
|------|-------|-------|
| HCCL 路径 | `HcclSendInner`（直接 kernel） | `HcclSendNext`（DPU 调度） |
| 通信延迟 | 极低（纳秒级） | 较高（DPU 调度 + RoCE RNR） |
| P2P 链路 | HCCS 直连 | RoCE 跨节点 |
| RNR 重试 | 无 | 可能触发 |
| `.cpu().tolist()` 时 irecv 状态 | 已完成 ✓ | PENDING ✗ |

---

### 7.7 组网拓扑与通信域物理链路

**结论**: 云侧 EP 通信（Cloud0↔Cloud1）走同节点 HCCS，边云 PP 通信（Edge↔Cloud）走跨节点 RoCE，物理链路分离但因 Device 级全局同步产生冲突。

```
Server A (边侧)              Server B (云侧)
┌────────────────┐          ┌────────────────┐
│ Edge0 (A2 NPU) │  RoCE    │ Cloud0 (A3 NPU)│  ← DP0
│ Edge1 (A2 NPU) │  RoCE    │ Cloud1 (A3 NPU)│  ← DP1
└────────────────┘          └────────────────┘

通信组：
- DP0: Edge0 + Cloud0 (跨节点, RoCE)
- DP1: Edge1 + Cloud1 (跨节点, RoCE)
- EP_edge: Edge0 + Edge1 (同节点, HCCS)
- EP_cloud: Cloud0 + Cloud1 (同节点, HCCS)
- alt_device_group (per DP): P2P 通信器 (跨节点, RoCE)
```

---

### 7.8 云侧 EP 报边云通信异常的原因

**结论**: `torch.npu.synchronize()` 是 Device 级全局同步，会等所有 stream。云侧 EP 代码中的同步点迫使 NPU 等待边云 PP 通信的 irecv（在 alt_device_group stream 上），而 irecv 因 RoCE 延迟/RNR 未完成，导致卡在 EP 代码位置（`token_dispatcher.py`），但实际阻塞原因是边云 RoCE 通信未就绪。

---

### 7.9 `num_local_experts > 1` 时的卡死位置

**结论**: 即使 `num_local_experts > 1`，代码走 `repeat_interleave` 分支而非显式 `torch.npu.synchronize()`，但 [token_dispatcher.py#L675-L676](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py#L675-L676) 中的 `.ravel().cpu().tolist()` 同样触发隐式 Device 级同步，效果与 `torch.npu.synchronize()` 等价。

```python
_hang_logger.error("[HANG] EP_ALLGATHER repeat_interleave ENTER: ep_rank=%s ravel=%s",
                  _hang_ep, num_global_tokens_per_local_expert.ravel().cpu().tolist())
#                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                          .cpu() 触发隐式 aclrtSynchronizeDevice()
```

---

### 7.10 死锁完整链路

```
T1  边侧: DECODE_FIRST → isend(0→1, RoCE, alt_group) 发 hidden_states
    云侧: passive_core 收 DECODE_FIRST ZMQ → 立即发布 DECODE_LAST (BUG!)

T2  边侧: 收 DECODE_LAST ZMQ → irecv(1→0, RoCE, alt_group) 等待云侧 isend
    云侧: worker 开始处理 DECODE_FIRST

T3  云侧: irecv(0→1, RoCE, alt_group) ← 应匹配 T1 的 isend
         但 RoCE 传输层有延迟（跨节点 RTT + RNR 重试）
         irecv 在 NPU alt stream 上 PENDING

T4  云侧: MoE → EP all_gather(HCCS) → 完成（CPU 返回）
         ALLTOALL(HCCS) → 完成

T5  云侧: .cpu().tolist() 或 torch.npu.synchronize() ← Device 级全局屏障
         等 T3 的 RoCE irecv → 迟迟不完成 → 死锁 💀

T6  边侧: 引擎下一循环 → 需要 sampled_token_ids
         .cpu() / .item() 触发隐式 synchronize → 卡死 💀
```

死锁闭环：
```
云侧等边侧 isend (irecv on alt stream, T3)
  → 边侧 isend 在 DECODE_LAST 中，已提交但被流依赖阻塞
  → 边侧流依赖等云侧 isend（下一个 batch 的 PP 通信）
  → 云侧被 synchronize 卡死无法发出 isend
  → 死锁闭环 🔒
```

---

## 8. 深入剖析：为什么边侧 isend 没发出，边侧还能往下执行？

### 8.1 核心：HCCL 的 isend 是全异步操作

**关键认知**：`torch.distributed.isend()` 只是把发送操作**提交到 HCCL stream 的队列**中，CPU 立即返回。数据的实际 DMA 传输由 NPU 上的 HCCL 引擎异步完成。

**证据** — [parallel_state.py#L999-L1004](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/distributed/parallel_state.py#L999-L1004)：

```python
with torch.npu.stream(_pps):        # 切换到 HCCL 专用 stream
    handle = torch.distributed.isend(
        value, dst=pp_group.ranks[dst], group=group
    )
# ← isend() 在此处已返回！数据尚未传输到远端
value.record_stream(_pps)
handles.append(handle)
```

`isend()` 返回的 `handle` 只是一个"稍后可以查询完成状态"的句柄，不表示传输已完成。

### 8.2 `handle.wait()` 也不是 CPU 阻塞——只是插入流依赖

**证据 1** — [worker.py#L503-L516](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/worker/worker.py#L503-L516)：

```python
def _wait_pp_send_work(self, channel=None):
    handles = self._pp_send_work_by_channel.pop(channel.value, [])
    for handle in handles:
        handle.wait()  # ← 仅等待本地 HCCL isend 完成，CPU 不阻塞！
```

`handle.wait()` → `ProcessGroupHCCL::wait()` → `aclrtStreamWaitEvent(compute_stream, hccl_stream_event)`

它的作用是：在 **compute stream 上插入一个 event barrier**，告诉 NPU "后续的 compute kernel 必须等 hccl stream 上的 isend 完成后才能执行"。CPU 线程不等待，立即返回（0ms）。

**证据 2** — [gpu_worker.py#L87-L108](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm/vllm/v1/worker/gpu_worker.py#L87-L108) 确认了这一点，日志输出 `dt=0.0ms`：

```python
def wait_for_comm(self) -> None:
    _t0 = time.monotonic()
    if self._comm_handles:
        for handle in self._comm_handles:
            handle.wait()       # ← 0ms 返回
    _dt_ms = (time.monotonic() - _t0) * 1000
    logger.error("[PP-EVT] WAIT-DONE dt=%.1fms", _dt_ms)  # 输出 dt=0.0ms
```

### 8.3 `blockingWait_` 默认是 false

**证据** — PyTorch `ProcessGroupHCCL.cpp` 中的代码（通过 Grep 确认）：

```cpp
// ProcessGroupHCCL.cpp 第 1059-1116 行
const char* blockingWait = getenv(HCCL_BLOCKING_WAIT);
// ...
if (blockingWait != nullptr) {
    blockingWait_ = true;  // 只有设置环境变量才启用
}
// 默认 blockingWait_ = false

// Work 创建时继承 blockingWait_:
work->blockingWait_ = blockingWait_;  // 第 3966, 4188 行

// Work::wait() 中:
if (blockingWait_) {
    // 只有设置了 HCCL_BLOCKING_WAIT=1 才会真正 CPU 阻塞等待
    // 默认不进入此分支
}
```

### 8.4 边侧从 DECODE_FIRST → DECODE_LAST 的完整异步流水线

下面是边侧时间线上，CPU 侧和 NPU 侧分别发生了什么：

```
时间  CPU 侧动作                                   NPU 侧实际状态
─────────────────────────────────────────────────────────────────────────
T1    收到 DECODE_FIRST ZMQ
      _execute_model_edge_head() 开始
      model_runner.execute_model(segment_a)
        → LayerNorm kernel 发射                   NPU: kernel 在 compute stream 排队
        → Linear kernel 发射                      NPU: kernel 排队执行
        → 返回 IntermediateTensors
      
T2    edge_cloud_send_tensor_dict()
        isend(hidden_states, dst=cloud, alt_group)
          → HCCL 提交发送任务到 _pps stream        NPU: isend task 入队 hccl stream
          → handle 立即返回                        NPU: isend 尚未开始 DMA
        _record_pp_send_work(handles, DECODE)
      
T3    返回 ModelRunnerOutput(req_ids=[])
      边侧引擎调度，发布 DECODE_LAST 给云侧
      
T4    下一轮循环: 收到 DECODE_LAST ZMQ
      execute_model() 入口
          _hidden_channel_for → DECODE
          _wait_pp_send_work(DECODE)
            handle.wait()
              → aclrtStreamWaitEvent(compute, hccl)
              → CPU 0ms 返回                         NPU: event barrier 插入
                                                     compute stream 后续 kernel 等 hccl stream
                                                     但 isend 还在 hccl stream 上排队！
                                                     
T5    _execute_model_edge_tail()
      edge_cloud_broadcast_recv()
        irecv(recv_buf, src=cloud, alt_group)
          → HCCL 提交接收任务到 _pps stream           NPU: irecv task 入队 hccl stream
          → handle 立即返回                           hccl stream 上: isend → irecv 串行
      
T6    AsyncIntermediateTensors.__init__()
        # 注意: .tensors 还没有被访问
        # wait_for_comm() 还没触发
      
T7    model_runner.execute_model(scheduler_output,
          intermediate_tensors, ...)
        → 访问 intermediate_tensors.tensors
          → __getattribute__("tensors") 
          → wait_for_comm()
            → handle.wait()                          CPU 0ms 返回
            → 只在 compute stream 插入 barrier
        → LayerNorm(tensor_data) → kernel 发射       NPU: kernel 在 compute stream 排队
        → Linear → kernel 发射                       但被 event barrier 挡住！
        → MoE → kernel 发射                          等 hccl stream 完成 isend + irecv
        → sample → kernel 发射
        → 返回 AsyncModelRunnerOutput                所有 kernel 都在等待！
      
T8    CPU 日志: "edge tail OUTPUT ✓"                NPU: 空转，hccl stream 上的 isend 还在等
                                                     远端匹配（云侧还没发 isend）
```

**关键洞察**：T4 中的 `handle.wait()` 只是对**当前 batch 之前**的那个 isend handle 调用的。但 `handle.wait()` 返回后，NPU 上的 isend 可能**还没执行完**——它只是在 hccl stream 排队队列中。CPU 侧从 T1 到 T8 全程没有任何同步点（直到后续 `.cpu()` / `.item()` 才真正同步）。

### 8.5 总结：为什么边侧能"正常"往下执行

| CPU 侧做了什么 | 为什么能执行 | NPU 侧实际状态 |
|---|---|---|
| `isend()` | 异步提交，立即返回 | isend 在 hccl stream 排队 |
| `handle.wait()` | 只插入 event barrier，CPU 不阻塞 | isend 可能还没开始 DMA |
| `irecv()` | 异步提交，立即返回 | irecv 在 hccl stream 排队（串行在 isend 后） |
| `AsyncIntermediateTensors.__init__()` | 只是保存 handle，不等待 | tensor 数据不可用 |
| `execute_model()` 中 `.tensors` 访问 | `wait_for_comm()` 只插入 stram 依赖 | kernel 排队但被 barrier 挡住 |
| CPU 日志打印 "execute_model after" | 所有 CPU 操作都是异步提交 | **所有 NPU kernel 都没执行！** |

这就是"假执行"——CPU 侧一切正常，NPU 侧实际在等通信完成。

---

## 9. Device 级全局同步的代码证据

### 9.1 `torch.npu.synchronize()` = `aclrtSynchronizeDevice()`

**证据** — PyTorch 源码中的调用链（Grep 结果）：

```cpp
// npu/Module.cpp 第 760 行
c10_npu::npuSynchronizeDevice();

// npu/Module.cpp 第 1706 行  
c10_npu::npuSynchronizeDevice();

// npu/Module.cpp 第 2337 行 - Python 绑定
{"_npu_synchronize", (PyCFunction)THNPModule_npuSynchronize, METH_NOARGS, nullptr},
```

Python 层的 `torch.npu.synchronize()` 最终调用 C++ 的 `npuSynchronizeDevice()`，底层是 `aclrtSynchronizeDevice()`。

**对比**：
```cpp
// Stream 级别同步 - 仅同步当前 stream
{"synchronize", (PyCFunction)THNPStream_synchronize, METH_NOARGS, nullptr},
// → aclrtSynchronizeStream(stream)

// Device 级别同步 - 同步该设备上所有 stream
torch.npu.synchronize()  →  aclrtSynchronizeDevice()
```

### 9.2 隐式 Device 级同步：`.cpu()` / `.item()` / `.tolist()`

任何将 NPU tensor 搬到 CPU 的操作都会隐式触发 `aclrtSynchronizeDevice()`，因为这需要确保数据已经在 NPU 侧计算完成、可以被 DMA 到 CPU。

```python
# token_dispatcher.py#L675-L676
num_global_tokens_per_local_expert.ravel().cpu().tolist()
# .cpu() 触发隐式 aclrtSynchronizeDevice()
```

这就是云侧 MoE 卡死的真正触发点。

### 9.3 `blockingWait_` 默认 false

```cpp
// ProcessGroupHCCL.cpp
bool blockingWait_ = false;  // 默认

// 只有设置环境变量才启用
const char* blockingWait = getenv("HCCL_BLOCKING_WAIT");
if (blockingWait != nullptr) {
    blockingWait_ = (std::stoi(blockingWait) != 0);
}

// work 创建时继承
work->blockingWait_ = blockingWait_;
```

因此 `handle.wait()` 默认不会调用任何阻塞 API（如 `aclrtSynchronizeStream`），只是插入 event barrier。

### 9.4 关键发现：DECODE 和 PREFILL 的 PP 通信共享同一个 stream

**证据** — [parallel_state.py#L826-L832](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/distributed/parallel_state.py#L826-L832)：

```python
def _get_pp_comm_stream() -> Any:
    """Lazy-init the dedicated PP communication stream."""
    global _pp_comm_stream          # ← 全局单例！
    if _pp_comm_stream is None:
        import torch_npu
        _pp_comm_stream = torch_npu.npu.Stream()
    return _pp_comm_stream
```

`_pp_comm_stream` 是**全局单例**。DECODE 通信（alt_device_group）和 PREFILL 通信（device_group）的 isend/irecv 全部走**同一个 `_pps` stream**。

**证据** — [parallel_state.py#L966-L970](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/distributed/parallel_state.py#L966-L970) 和 [#L979-L980](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/distributed/parallel_state.py#L979-L980)：

```python
_pps = _get_pp_comm_stream()     # ← 每次 isend 都用同一个 stream
_pps.wait_stream(torch.npu.current_stream())  # ← 关键：_pps 等 compute stream 完成
with torch.npu.stream(_pps):
    handle = torch.distributed.isend(value, dst=..., group=group)
```

每次 isend 之前都会调用 `_pps.wait_stream(compute)`，告诉 `_pps` stream：**等 compute stream 完成后才能开始 isend**。

### 9.5 死锁根因：compute stream ↔ _pps stream 的循环等待

#### 9.5.1 场景还原

```
时间线：
────────────────────────────────────────────────────────────────────────
步骤1  边侧: DECODE_FIRST 到达
       _execute_model_edge_head():
         compute stream: execute_model(segment_a) → 产出 tensor
         _pps.wait_stream(compute)               → _pps 等 compute 完成
         _pps: isend(decode, cloud, alt_group)    → 提交 HCCL 发送

步骤2  引擎调度: 返回 ModelRunnerOutput（空 sampled_tokens）
       passive_core 收到 ZMQ → 立即发布 DECODE_LAST ZMQ
       
步骤3  边侧: DECODE_LAST 到达
       execute_model 入口:
         _wait_pp_send_work(DECODE):
           handle.wait()                           → compute stream 等 _pps！
           → aclrtStreamWaitEvent(compute, hccl_isend_event)
       _execute_model_edge_tail():
         _pps: irecv(decode, cloud, alt_group)     → 提交 HCCL 接收
         compute: model forward（假执行）           → 所有 kernel 排队在 barrier 后

步骤4  边侧下一个 batch（假设是 PREFILL_FIRST）到达
       _execute_model_edge_head():
         compute: execute_model(segment_a)          → kernel 排队（compute 还在等 barrier）
         _pps.wait_stream(compute)                  → _pps 等 compute 完成
         → compute 在等 _pps（步骤3的 handle.wait）
         → _pps 在等 compute（wait_stream）
         → 🔒 循环等待死锁！
────────────────────────────────────────────────────────────────────────
```

#### 9.5.2 循环等待的精确描述

```
compute stream:  [DECODE_FIRST forward] → [handle.wait() barrier: 等 _pps 的 isend] → [排队中...]
                         ✅ 已完成              ⏸ 阻塞中

_pps stream:     [wait_stream(compute)] → [DECODE isend(alt_group)] → [DECODE irecv(alt_group)] → [wait_stream(compute)]
                         ✅ 已满足              ✅ 已提交                    ✅ 已提交                 ⏸ 阻塞中
                                                                                     ^^^^^^^^^^^^^^^^
                                                                                     等 compute 完成
                                                                                     但 compute 在等 _pps！
```

这就是经典的 **stream 级循环等待死锁**：
- `compute` stream 在 `handle.wait()` 处等 `_pps` stream 的 isend 完成
- `_pps` stream 在 `wait_stream(compute)` 处等 `compute` stream 完成
- 两个 stream 互相等待 → 死锁

#### 9.5.3 云侧 MoE 为什么被卡住

云侧执行 PREFILL_FIRST：

```
云侧 PREFILL_FIRST:
  _pps: irecv(prefill, edge, device_group)          → 等边侧 PREFILL isend
  compute: wait_for_comm() → [模型层] → [MoE] → [EP all_gather]
  mc2 stream: EP all_gather(HCCS)                   → 完成
  torch.npu.synchronize() = aclrtSynchronizeDevice() → 等所有 stream
```

云侧的 `irecv(prefill, edge)` 在等边侧的 PREFILL isend。但边侧的 PREFILL isend 还没提交到 `_pps` stream（因为边侧 `_pps` stream 卡在 `wait_stream(compute)` 处）。所以云侧的 irecv 永远等不到，导致云侧的 `torch.npu.synchronize()` 被阻塞。

**整个死锁链路**：
```
边侧 compute 等 _pps  isend → 
边侧 _pps 等 compute →
→ 边侧 _pps 卡死，PREFILL isend 永远无法提交 →
→ 云侧 PREFILL irecv 永远等不到 →
→ 云侧 synchronize 等所有 stream（包括 hccl stream 上 pending 的 irecv）→ 卡死
```

#### 9.5.4 为什么 A2-A2 没有这个问题

A2-A2 中，HCCL 走旧路径（`HcclSendInner` / `HcclRecvInner`），isend kernel 在 stream 上完成极快（纳秒级）。这意味着：

```
步骤3 handle.wait() barrier → isend 已瞬间完成 → barrier 立即解除
步骤4 _pps.wait_stream(compute) → compute 早已完成 → 不阻塞
```

整个流程快于 PREFILL_FIRST 的到达时间，循环等待来不及形成。

A2-A3 中，A3 走 DPU 路径（`HcclRecvNext`），irecv WQE posting 有额外延迟。边侧 A2 的 isend 虽然 kernel 快，但网络侧（RoCE RNR 重试）可能延长 isend 的"实际完成"时间。如果 isend 的 NPU 完成事件被延迟（即使是网络层延迟），`handle.wait()` barrier 无法及时解除，导致 compute stream 持续等待。

| | A2-A2 | A2-A3 |
|---|---|---|
| isend 路径 | HcclSendInner (直接 kernel) | HcclSendInner (直接 kernel) |
| irecv 路径 | HcclRecvInner (直接 kernel) | HcclRecvNext (DPU 调度) |
| isend 完成延迟 | ~ns | ~ns 本地，但网络层可能 RNR 延迟 |
| handle.wait barrier 解除 | 瞬时 | 可能被网络层延迟拖慢 |
| 循环等待形成 | 不可能 | 可能 |

---

## 10. 无侵入诊断日志方案

> **原则**：以下所有诊断代码均为纯观测性操作——`time.monotonic()` 计时、`torch.npu.Event().record()` 记录进度、`event.query()` 非阻塞查询、`logger.error()` 打印。**不使用 synchronize()、.cpu()、.item() 等任何阻塞性操作**，不会改变原有执行时序。

### 10.1 边侧：验证 isend 时 compute stream 是否已完成

在 [worker.py#L690-L694](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/worker/worker.py#L690-L694) 的 isend 提交位置添加：

```python
# ==== 无侵入诊断: 边侧 isend 提交时的 stream 状态 ====
import torch_npu as _diag_npu

# 在 compute stream(default) 上记录 event，标记 execute_model 完成位置
_diag_compute_ev = _diag_npu.npu.Event()
_diag_compute_ev.record()  # record on current (compute) stream

# isend 提交
self._record_pp_send_work(
    edge_cloud_send_tensor_dict(_gathered, channel=channel,
                                num_tokens=scheduler_output.total_num_scheduled_tokens),
    channel=channel,
)

# 在 hccl stream (_pps) 上记录 event，标记 isend 入队位置
_diag_hccl_ev = _diag_npu.npu.Event()
with _diag_npu.npu.stream(_get_pp_comm_stream()):
    _diag_hccl_ev.record()

# 非阻塞查询: compute stream 是否已完成
_diag_compute_done = _diag_compute_ev.query()
logger.error("[DIAG-ISEND] compute_stream_done=%s dp_rank=%s bt=%s",
             _diag_compute_done, self.model_runner.dp_rank,
             scheduler_output.batch_type.value)
# 预期: _diag_compute_done 的值表明了 isend 提交时数据是否已产出
# 如果 False → isend 要等 compute stream，可能导致发送延迟
```

### 10.2 边侧：DECODE_LAST 入口时检查上一轮 isend 是否真正完成

在 [worker.py#L577](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/worker/worker.py#L577) 的 `_wait_pp_send_work` 之后添加：

```python
# ==== 无侵入诊断: 上一轮 isend 在 NPU 上是否已完成 ====
# _wait_pp_send_work 已经调用了 handle.wait()（插入流依赖）
# 但 isend 可能在 hccl stream 上尚未执行
_diag_ev = _diag_npu.npu.Event()
_diag_ev.record()  # record on current stream after handle.wait()

# 这里不调用 query()，因为刚 record 的 event 肯定 False
# 改为检查: handle.wait() 耗时（侧证）
import time as _diag_time
_diag_t0 = _diag_time.monotonic()
# ... (handle.wait() 已在 _wait_pp_send_work 中调用)
_diag_dt = (_diag_time.monotonic() - _diag_t0) * 1e6  # us
logger.error("[DIAG-PP-SEND-WAIT] dt_us=%.1f dp_rank=%s bt=%s",
             _diag_dt, self.model_runner.dp_rank,
             scheduler_output.batch_type.value)
# 预期: dt ≈ 0us。如果 > 100us 说明有异常等待
```

### 10.3 云侧：synchronize 前检查所有 stream 状态（核心诊断）

在 [token_dispatcher.py#L684-L686](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py#L684-L686) 的 `torch.npu.synchronize()` 之前添加：

```python
# ==== 无侵入诊断: synchronize 前检查各 stream 状态 ====
import torch_npu as _diag_npu
import time as _diag_time

# 1. 在 default compute stream 上 record event 并立即 query
_diag_t0 = _diag_time.monotonic()
_diag_compute_ev = _diag_npu.npu.Event()
_diag_compute_ev.record()
_diag_compute_done = _diag_compute_ev.query()
_diag_dt_us = (_diag_time.monotonic() - _diag_t0) * 1e6

logger.error("[DIAG-MOE-BEFORE-SYNC] rank=%s compute_done=%s query_dt_us=%.1f",
             torch.distributed.get_rank(), _diag_compute_done, _diag_dt_us)

# 2. 如果能拿到 alt_device_group 的 stream，同样检查
# 注: alt_device_group stream 可能需要从 ProcessGroup 内部获取
# 替代方案: 在当前 stream 上 record 第二个 event
_diag_ev2 = _diag_npu.npu.Event()
_diag_ev2.record()
_diag_ev2_done = _diag_ev2.query()
logger.error("[DIAG-MOE-BEFORE-SYNC2] rank=%s ev2_done=%s",
             torch.distributed.get_rank(), _diag_ev2_done)

# 3. 打印 EP 通信组信息
_ep_world = get_ep_group().world_size
_ep_rank = get_ep_group().rank_in_group
logger.error("[DIAG-MOE-EP-INFO] ep_world=%d ep_rank=%d device=%d",
             _ep_world, _ep_rank, torch_npu.npu.current_device())
```

**关键解读**：
- `compute_done=False`：compute stream 上还有未完成的 kernel → synchronize 会等它们
- `compute_done=True` 但 synchronize 仍然卡死 → 说明有其他 stream (如 alt hccl stream) 有未完成操作
- `query_dt_us`：如果非常大（>100us）可能暗示资源争抢

### 10.4 云侧：irecv 前后的时间戳对比

在 [worker.py#L898-L907](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/worker/worker.py#L898-L907) 的 cloud recv 处添加：

```python
import time as _diag_time

_diag_t_recv0 = _diag_time.monotonic()
logger.error("[DIAG-CLOUD-RECV-TIME] ENTER dp_rank=%s t=%.6f bt=%s",
             _hang_cld_rank, _diag_t_recv0, scheduler_output.batch_type.value)

tensor_dict, comm_handles, comm_postprocess = edge_cloud_broadcast_recv(
    num_tokens=scheduler_output.total_num_scheduled_tokens,
    channel=channel,
    sp_chunk=do_sp_chunk and merge_payload,
    src=0,
)

_diag_dt_recv_us = (_diag_time.monotonic() - _diag_t_recv0) * 1e6
logger.error("[DIAG-CLOUD-RECV-TIME] EXIT dp_rank=%s dt_us=%.1f",
             _hang_cld_rank, _diag_dt_recv_us)
# 预期: dt_us < 100us (irecv 提交很快)
# 如果 > 1000us: DPU 路径或内存分配有异常延迟
```

### 10.5 边侧：isend 提交前后时间戳对比

在 [worker.py#L690-L702](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/worker/worker.py#L690-L702) 的 edge send 处添加：

```python
import time as _diag_time

_diag_t_send0 = _diag_time.monotonic()
self._record_pp_send_work(
    edge_cloud_send_tensor_dict(_gathered, channel=channel,
                                num_tokens=scheduler_output.total_num_scheduled_tokens),
    channel=channel,
)
_diag_dt_send_us = (_diag_time.monotonic() - _diag_t_send0) * 1e6
logger.error("[DIAG-EDGE-SEND-TIME] dp_rank=%s dt_us=%.1f bt=%s",
             self.model_runner.dp_rank, _diag_dt_send_us,
             scheduler_output.batch_type.value)
# 预期: dt_us < 500us (isend 提交 + dict 操作)
```

### 10.6 云侧：execute_model 全链路时间戳

在 [worker.py#L943-L946](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/worker/worker.py#L943-L946) 的 cloud execute_model 前后添加：

```python
import time as _diag_time

_diag_t_exec0 = _diag_time.monotonic()
logger.error("[DIAG-CLOUD-EXEC] ENTER dp_rank=%s t=%.6f bt=%s tokens=%d",
             getattr(self.model_runner, "dp_rank", "?"), _diag_t_exec0,
             scheduler_output.batch_type.value,
             scheduler_output.total_num_scheduled_tokens)

output = self.model_runner.execute_model(
    scheduler_output, intermediate_tensors,
    layer_slice_info=layer_slice_info,
)

_diag_dt_exec_ms = (_diag_time.monotonic() - _diag_t_exec0) * 1000
logger.error("[DIAG-CLOUD-EXEC] EXIT dp_rank=%s dt_ms=%.1f",
             getattr(self.model_runner, "dp_rank", "?"), _diag_dt_exec_ms)
# 如果这行没有打印 → 卡在 execute_model 内部（MoE synchronize）
```

### 10.7 诊断结果对照表

根据上述日志的输出组合，判断卡死根因：

| compute_done | sync前dt | exec EXIT | 边侧send dt | 根因判断 |
|---|---|---|---|---|
| False | 小 | 无 | — | 云侧 compute stream 未完成 |
| True | 小 | 无 | — | 云侧 alt stream irecv PENDING |
| True | 大 | 无 | — | 云侧资源争抢 |
| — | — | 有 | 大 | 边侧 isend 提交慢 |
| — | — | 有 | 正常 | 云侧 compute 计算慢（非通信阻塞） |

### 10.8 HCCL Entry Log 辅助确认

```bash
# 启用 HCCL API 级别日志（对性能有轻微影响，但不会改变时序）
export HCCL_ENTRY_LOG_ENABLE=1
export HCCL_ENTRY_LOG_FILE=/tmp/hccl_entry_rank${RANK}.log

# 运行后分析日志:
# 1. grep HcclSend 边侧日志 → 找到 isend 入口时间戳
# 2. grep HcclRecv 云侧日志 → 找到 irecv 入口时间戳
# 3. 计算时间差: 如果 isend 时间 < irecv 时间，且间隔 > 1ms
#    → 说明 isend 提交早于 irecv，可能触发 RoCE RNR
# 4. 检查 IsHostDpu 关键字 → 确认 A3 是否走 DPU 路径
```

---

## 11. 定位总结

### 11.1 已通过代码证据确认的事实

#### 11.1.1 边的阻塞机制

边侧 DECODE_LAST 执行时的阻塞链：

[gpu_worker.py#L93-L94](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm/vllm/v1/worker/gpu_worker.py#L93-L94)：

```python
for handle in self._comm_handles:
    handle.wait()  # → event.block(compute)
```

边的 DECODE_LAST 调用链插入**两道 event barrier**：
1. `_wait_pp_send_work(DECODE)` → `handle.wait()` on DECODE_FIRST isend handle
2. `wait_for_comm()` → `handle.wait()` on DECODE_LAST irecv handle

结果：
```
edge compute: [wait isend event] → [wait irecv event] → [kernel 全部排队]
```

`handle.wait()` 调用 `ProcessGroupHCCL::WorkHCCL::synchronizeInternal`：

[ProcessGroupHCCL.cpp#L895-L900](file:///Users/wangwei/Desktop/project/pytorch/torch_npu/csrc/distributed/ProcessGroupHCCL.cpp#L895-L900)：

```cpp
auto currentStream = c10_npu::getCurrentNPUStream(devices_[i].index());
(*hcclEndEvents_)[i].block(currentStream);  // 仅插入 stream 依赖，CPU 立即返回
```

#### 11.1.2 边的 isend 不产生网络流量 — 与云侧完全隔离

HCCL P2P send **全部走 rendezvous 协议**，在接收端 post recv 前不会发送数据：

[alg_data_trans_wrapper.cc#L248-L273](file:///Users/wangwei/Desktop/project/hccl/src/ops/op_common/template/wrapper/alg_data_trans_wrapper.cc#L248-L273)：

```cpp
// SendWrite: 先等 ACK，再传数据
HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK, execTimeout)
// ↑ 到这里就阻塞了 → 永远不会执行下面的 HcommWrite
HcommWriteOnThread(thread, remoteInputCopy, remoteOutputCopy)
HcommChannelNotifyRecord(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL)
```

**结论：边的 DECODE isend 在云侧 post 匹配 irecv 之前，不产生任何网络流量。云侧 NIC、DPU 完全无感知。**

#### 11.1.3 EP all_gather (HCCS) 和 PP send/recv (RoCE) 的传输层完全隔离

| | EP all_gather | PP send/recv |
|---|---|---|
| Transport | `TRANS_TYPE_P2P` (CCU/HCCS) | `TRANS_TYPE_IBV_EXP` (RoCE) |
| 硬件 | CCU 引擎 | RDMA NIC |
| Dispatcher | 独立 per-channel | 独立 per-channel |
| NotifyPool | 独立 per-channel | 独立 per-channel |

代码证据：[aicpu_ts_hccs_channel.cc#L205](file:///Users/wangwei/Desktop/project/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_hccs_channel.cc#L205)、[aicpu_ts_roce_channel.cc#L335](file:///Users/wangwei/Desktop/project/hcomm/src/base_comm/resources/endpoint_pairs/channels/aicpu/aicpu_ts_roce_channel.cc#L335)。

**结论：软件层完全隔离，不存在交叉阻塞。**

#### 11.1.4 A2/A3 路径分流

A3 (DEV_TYPE_950) 所有通信操作走 DPU 路径：

[send_op.cc#L58-L68](file:///Users/wangwei/Desktop/project/hccl/src/ops/send/send_op.cc#L58-L68)：

```cpp
if (deviceType != DevType::DEV_TYPE_950) {
    return HcclSendInner(...);  // A2 → HCOMM 内核直通
}
return HcclSendNext(...);  // A3 → DPU 调度
```

[all_gather_op.cc#L33-L41](file:///Users/wangwei/Desktop/project/hccl/src/ops/all_gather/all_gather_op.cc#L33-L41)（通过 Grep 确认）：

```cpp
DevType deviceType = DevType::DEV_TYPE_COUNT;
CHK_RET(hrtGetDeviceType(deviceType));
if (deviceType != DevType::DEV_TYPE_950) {
    return HcclAllGatherInner(...);  // A2 → HCOMM 内核直通
}
// A3 → selector → DPU 算法（如 InsAllGatherMeshNhrDPU）
```

#### 11.1.5 DPU all_gather 的阻塞点

A3 DPU all_gather 最终调用 `HcommWaitResponse` 等待 DPU 响应：

[ins_temp_all_gather_nhr_dpu.cc#L91-L92](file:///Users/wangwei/Desktop/project/hccl/src/ops/all_gather/template/aicpu/ins_temp_all_gather_nhr_dpu.cc#L91-L92)：

```cpp
if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr),
    recvData, 0, &recvMsgId) != 0) { ... }
```

`HcommWaitResponse` 实现为**轮询共享内存 flag**：

[aicpu_ts_sync_data_a_adpt.cc#L82-L102](file:///Users/wangwei/Desktop/project/hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_sync_data_a_adpt.cc#L82-L102)：

```cpp
int32_t HcommWaitResponse(MsgHandle handle, void *dst, size_t sizeByte, uint32_t *msgId) {
    uint8_t flagReadValue{0};
    // 轮询等待 DPU 在共享内存中设置 flag
    while (flagReadValue == 0) {
        ret = memcpy_s(&flagReadValue, 1, srcFlagPtr, 1);
        // ... 超时检查 ...
    }
}
```

### 11.2 诊断断点结果

#### 11.2.1 云侧 PREFILL recv 数据已到达

DIAG-CLOUD-RECV 断点（[worker.py#L438-L452](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/worker/worker.py#L438-L452)）输出：
- `first_val=... OK` → PREFILL 数据传输成功，`.cpu().item()` 隐式同步通过
- 说明 wait_for_comm 的 event barrier 已解除

#### 11.2.2 卡死位置确认：EP all_gather

用户确认卡死在 `gather_from_sequence_parallel_region`（[token_dispatcher.py#L662](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py#L662)）中的 EP all_gather。

卡死场景下 `[DIAG-EP-CHK] after_ag` 的 `ev0_done=False`：
- `ev0` 在 all_gather 调用**前** record
- `ev0_done=False` 说明 compute stream 上 all_gather 之前的 kernel 在 all_gather 返回时仍未完成
- 正常场景下 `ev0_done=True`（all_gather 返回时 compute 已推进）

#### 11.2.3 边侧算子数据

边侧 A2 profiling：`hcom_send` → `wait_notify`
- 确认是 rendezvous 等待云侧 post 匹配 irecv

### 11.3 尚未确定的事项

| 问题 | 状态 |
|---|---|
| 云侧 MoE 卡死是否与边的 DECODE 因果相关 | **无代码证据支持**。HCCL rendezvous 协议确保边 isend 不产生网络流量，HCOMM 软件层所有资源（Transport/Dispatcher/NotifyPool）完全隔离 |
| 延后 DECODE_LAST 发布时间为什么能避免卡死 | 可能是改变了时序使 A3 DPU all_gather 恰好成功，未触发 DPU 内部问题；或是避免了边的 _pps stream 被多个 DECODE 操作堵塞，间接改变了整体调度时序 |
| A3 DPU all_gather 为什么卡死 | 需要 DPU 固件层面分析。`HcommWaitResponse` 轮询等待 DPU 响应，DPU 未置 flag 导致卡死 |

### 11.4 下一步：HCCL 底层日志诊断

启用 HCCL Entry Log 和心跳机制，从通信层视角确认：

```bash
# 缩短超时到 60 秒（默认 1836 秒）
export HCCL_EXEC_TIMEOUT=60

# 记录每次通信算子调用（所有 rank，所有通信域）
export HCCL_ENTRY_LOG_ENABLE=1

# 可选：模块级详细日志
export HCCL_DEBUG_CONFIG="ALG,TASK,RESOURCE"
```

复现后分析 `~/ascend/log/run/plog/`：

| 检索命令 | 目的 |
|---|---|
| `grep "Entry-HcclAllGather"` | 确认 EP 通信域两 rank 的 all_gather 调用是否匹配 |
| `grep "Entry-HcclSend\|Entry-HcclRecv"` | 边云 alt_device_group 上 send/recv 调用时序 |
| `grep "HeartbeatAbnormal"` | 心跳是否检测到 STUCK/LOST/ERROR_CQE |
| `grep "TaskExecStage.*Timeout"` | 超时后 task exception 指向哪个通信域、哪个算子 |

[task_exec_stage.md#L14-L66](file:///Users/wangwei/Desktop/project/hccl/docs/zh/user_guide/fault_diagnosis/task_exec_stage.md#L14-L66) 描述了 HCCL 心跳机制可检测的三种异常：

| 异常类型 | 检测机制 |
|---|---|
| ERROR CQE | 定期轮询 RoCE 驱动重传超时事件 |
| **STUCK** | 每隔 1/3 HCCL_EXEC_TIMEOUT 轮询算子入/出次数，检测进程卡死 |
| LOST | 30s 未收到远端心跳报文 |

### 11.5 token_dispatcher.py 中 A2/A3 Python 层差异

`TokenDispatcherWithAll2AllV`（卡死所在的类）**在 Python 层完全不区分 A2/A3**：

[token_dispatcher.py#L116-L117](file:///Users/wangwei/v0.20.2_lwd_update_dpds_0714/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py#L116-L117)：

```python
# 只有 TokenDispatcherWithMC2 有 A2/A3 区分
self.need_extra_args = get_ascend_device_type() in [AscendDeviceType.A3, AscendDeviceType.A5]
```

`TokenDispatcherWithAll2AllV._preprocess` 无任何 `get_ascend_device_type` 调用。A2/A3 差异完全在底层 HCCL 通信路径（Inner vs DPU）。
