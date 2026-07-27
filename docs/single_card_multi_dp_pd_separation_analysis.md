# 单卡多 DP 适配 PD 分离调度框架分析

> 基于 [`SharedModelWorkerProc.worker_busy_loop`](../../vllm/v1/executor/shared_model_multiproc_executor.py#L319-L606) 入口，分析非MoE 模型在 `--edge-npu-count 1 --dp 2` 下适配 PD 分离调度所需的修改。

---

## 1. `_dispatch` 是什么，对应原 `worker.py` 的哪个接口

### 1.1 架构层次

```
EngineCore (dp_rank 0)                            EngineCore (dp_rank 1)
  │ scheduler.schedule()                             │ scheduler.schedule()
  │ execute_model(scheduler_output)                  │ execute_model(scheduler_output)
  ▼                                                  ▼
MultiprocExecutor.collective_rpc("execute_model", ...)
  │ enqueue to rpc_broadcast_mq[0]                   │ enqueue to rpc_broadcast_mq[1]
  ▼                                                  ▼
┌──────────────────────────────────────────────────────────────┐
│              SharedModelWorkerProc.worker_busy_loop            │
│                                                                │
│  for k, mq in rpc_broadcast_mqs:   # round-robin N 个 MQ      │
│    method, args = mq.dequeue()                                  │
│    self._dispatch(k, method, args)  ← 这是关键!                │
│      │                                                         │
│      │  virtual_worker = self.worker[k]                         │
│      │  func = getattr(virtual_worker, method)                  │
│      │  output = func(*args)                                    │
│      ▼                                                         │
│    handle_output(k, output) → response_mqs[k].enqueue(...)     │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 对应关系

`_dispatch` 对应的是标准 `WorkerProc.worker_busy_loop`（[multiproc_executor.py:1091-1200](../../vllm/v1/executor/multiproc_executor.py#L1091-L1200)）中的这段核心逻辑：

```python
# 标准 WorkerProc.worker_busy_loop (单 worker, 单 MQ):
method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue()
func = getattr(self.worker, method)          # ← 等价于 _dispatch 的 getattr
output = func(*args, **kwargs)               # ← 等价于 _dispatch 的 func(*args)
self.handle_output(output)                    # ← 等价于 handle_output(k, output)
```

区别在于：
- 标准 `WorkerProc`：**1 个 MQ → 1 个 worker → 1 个 NPU**
- `SharedModelWorkerProc`：**N 个 MQ → N 个 virtual worker → 1 个 NPU**（需要 round-robin 轮询 N 个 MQ）

`_dispatch` 不是对应 `worker.py`（即 `NPUWorker`）的任何接口。`worker.py` 中的 `NPUWorker.execute_model` 是 `_dispatch` 调用链的**下游**：

```
_dispatch (WorkerProc 层，等价于标准 WorkerProc.busy_loop 中的 getattr+func())
  → virtual_worker[k].execute_model_batched_pre   (SharedModelEdgeWorker)
    → model_runner.execute_model_pre              (BatchedModelRunner)
```

### 1.3 `_dispatch` 中 `execute_model_batched_pre` 的条件

```python
# shared_model_multiproc_executor.py line 634-635
if method == "execute_model" and hasattr(
        virtual_worker, "execute_model_batched_pre"):
    output = virtual_worker.execute_model_batched_pre(
        args[0] if args else None)
```

两个条件同时满足时走 batched 路径：

| 条件 | True 的情景 |
|------|------------|
| `method == "execute_model"` | EngineCore 发送的 `execute_model` RPC |
| `hasattr(..., "execute_model_batched_pre")` | worker 类是 `SharedModelEdgeWorker`（edge-cloud shared model 模式） |

`SharedModelEdgeWorker` 的使用条件（见 [platform.py:510-519](../../vllm-ascend/../vllm-ascend/vllm_ascend/platform.py#L510-L519)）：

```
edge_npu_count == 1 AND dp_size > 1 AND is_edge_node
  → is_shared_model_edge = True
  → parallel_config.worker_cls = "SharedModelEdgeWorker"
```

**非MoE 单卡也走这个路径**：判断只依赖拓扑，不依赖模型类型。

### 1.4 `virtual_worker` 何时是 `SharedModelEdgeWorker`

当 `--edge-npu-count 1 --dp 2` 且当前进程是 Edge 侧时。

| 场景 | `worker_cls` |
|------|-------------|
| `--edge-npu-count 1 --dp 2`（边侧） | `SharedModelEdgeWorker` |
| `--edge-npu-count 2 --dp 2`（边侧） | `NPUWorker`（标准） |
| 云侧（任意配置） | `NPUWorker`（标准） |

---

## 2. 多卡流程基线 (`--edge-npu-count 2 --dp 2`)

### 2.1 NPUWorker.execute_model — PD 分离的典型路径

参考 [worker.py:550-597](../../vllm-ascend/../vllm-ascend/vllm_ascend/worker/worker.py#L550-L597)：

```python
def execute_model(self, scheduler_output):
    bt = scheduler_output.batch_type

    # FIRST 路径: head forward + send
    if bt in (PREFILL_FIRST, DECODE_FIRST):
        self._wait_pp_send_work(self._hidden_channel_for(scheduler_output))
        return self._execute_model_edge_head(scheduler_output)
    
    # LAST 路径: recv + tail forward + logits
    if bt in (PREFILL_LAST, DECODE_LAST):
        return self._execute_model_edge_tail(scheduler_output)
```

### 2.2 多卡 NPUWorker._execute_model_edge_head

[worker.py:607-657](../../vllm-ascend/../vllm-ascend/vllm_ascend/worker/worker.py#L607-L657)：

1. `model_runner.execute_model(scheduler_output, intermediate_tensors=None)` — 完整预处理 + head forward
2. `edge_cloud_isend_tensor_dict(_gathered, channel=..., dst=...)` — 发送 head 结果到云侧
3. 返回 `DeferredExecutePostprocess`（EMPTY 结果 — 不包含 logits）

### 2.3 多卡 model_runner_v1 的 _fast_path 缓存

[model_runner_v1.py:2530-2568](../../vllm-ascend/../vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2530-L2568)：

```python
# Head 阶段缓存
self._edge_prepare_cache = (attn_metadata, logits_indices, ...)

# Tail 阶段 (_fast_path):
if self._edge_prepare_cache is not None:
    # 跳过 _update_states
    # 跳过 attention metadata 重建
    attn_metadata, logits_indices, ... = self._edge_prepare_cache
    self._edge_prepare_cache = None
```

### 2.4 多卡 _hidden_channel_for 通道区分

[worker.py:599-608](../../vllm-ascend/../vllm-ascend/vllm_ascend/worker/worker.py#L599-L608)：

```python
def _hidden_channel_for(self, scheduler_output) -> int:
    bt = scheduler_output.batch_type
    return _PREFILL_1 if bt in (PREFILL_FIRST, PREFILL_LAST) else _DECODE
```

允许 prefill 和 decode 各有一个 in-flight send（双通道流水线）。

---

## 3. 单卡多 DP 当前状态

### 3.1 SharedModelEdgeWorker.execute_model_batched_pre

[shared_model_edge_worker.py:803-853](../../vllm-ascend/../vllm-ascend/vllm_ascend/worker/edge_cloud/shared_model_edge_worker.py#L803-L853)：

- 不分 batch_type，FIRST 和 LAST 走同一个方法
- 返回 `_BatchedExecuteMarker(bundle=result, worker=self)`

### 3.2 worker_busy_loop end-of-round 流程

[shared_model_multiproc_executor.py:462-571](../../vllm/v1/executor/shared_model_multiproc_executor.py#L462-L571)：

```
Phase A (1 次):  run_batched_head        → 1× merged head forward
Phase A (N 次):  drive_batched_round      → per-dp_rank isend + recv closure
Phase B/C (1 次): drain_batched_round     → recv + 1× merged tail + logits + handle_output
```

### 3.3 worker_busy_loop 全部流程分析

**流程 A**: `execute_model`（非空 batch）→ `_BatchedExecuteMarker` → pause → end-of-round drain  
**流程 B**: `execute_model`（空 batch）→ 直接 `handle_output`，不 pause  
**流程 C**: `initialize_from_config` → pause + `init_kv_cache = True` → 等 `all(paused)` 后 continue  
**流程 D**: `sample_tokens` → 非 SYNC，立即 `handle_output`  
**流程 E**: 其他非 SYNC 方法 → 立即 `handle_output`  
**流程 F**: Legacy `DeferredExecutePostprocess` → `legacy_pending` → `deferred()`  
**流程 G**: sleep → 无消息时 yield

### 3.4 关键差异：单卡多 DP vs 多卡

| 维度 | 多卡 (`--edge-npu-count 2`) | 单卡多 DP (`--edge-npu-count 1`) |
|------|---------------------------|----------------------------------|
| Worker 数 | 2 个 `NPUWorker` 进程 | 1 个 `SharedModelWorkerProc` 进程 |
| MQ 数 | 1 per Worker | N per WorkerProc |
| Forward 次数/dp | 2 次 (head + tail, 分段) | 1 次 (head + tail 合并) |
| PP 通信 | 2 次 isend/recv per dp | 2 次 isend/recv per dp (含合并) |
| `_fast_path` | 有 (`_edge_prepare_cache`) | 无 (一次走完) |
| `batch_type` | 用于路由 FIRST/LAST | **未使用**（不分首尾） |

### 3.5 非MoE round barrier 行为

```python
if all(paused) or not self.is_moe:  # 非MoE: 不等齐
```

| 模型 | Round 触发 | 合并 bundle 数 |
|------|-----------|:---:|
| MoE | `all(paused)` — 等齐 | 保证 = dp_size |
| 非MoE | 每次 for 循环结束**立即触发** | 可能 = 1, 2, ..., dp_size |

---

## 4. PD 分离适配修改点概览

详细修改方案见：[pd_separation_modification_plan.md](pd_separation_modification_plan.md)

| # | 文件 | 改动内容 |
|---|------|---------|
| 1 | `shared_model_multiproc_executor.py` — `_dispatch` | batch_type 感知的三路路由 |
| 2 | `shared_model_multiproc_executor.py` — `worker_busy_loop` end-of-round | 首/尾/非PD 三路分组 + 分流处理 |
| 3 | `shared_model_multiproc_executor.py` — 类属性 | 跨 round 状态缓存 |
| 4 | `shared_model_edge_worker.py` | 新增 marker 类 + head/tail pre 方法 |
| 5 | `batched_model_runner.py` | 新增 `execute_model_tail_pre` |
| 6 | 非 PD 分离路径 | 完全不变 |

---

## 5. 风险与注意事项

### 5.1 batch_type 路由正确性

`_dispatch` 需要 import `BatchType`，建议通过 duck-typing 或延迟导入避免对 vllm core 的硬依赖。

### 5.2 空 batch 不变

`total_num_scheduled_tokens == 0` 时返回 None/EMPTY，不产生 marker，不受影响。

### 5.3 `_fast_path` 缓存正确性

LAST round 需要复用 FIRST round 的 `_merged_attn_ctx_cache`。如果在 FIRST 和 LAST 之间同一 dp_rank 又收到新的 FIRST，缓存会被覆盖。需要确保 EngineCore 不会在 FIRST→LAST 之间发送新的 FIRST。

### 5.4 `not self.is_moe` = True 下 batch 粒度不确定

非MoE 不等齐，每次 end-of-round 可能处理 1 个或 2 个 dp_rank。FIRST 和 LAST 可能在同一个 pending 中。通过三路分组处理，互不干扰。

### 5.5 `layer_slice_info` / `num_tokens_across_dp_merged`

当前 batched path 不传递 `layer_slice_info`。PD 分离适配不改变这个行为。非MoE 模型通常不依赖这些字段。

### 5.6 `_hidden_channel_for` 通道区分

当前 batched path 使用无 channel 版本的 isend。如果需要 prefill/decode 双通道并发，需要在 `drive_head_send` 中添加 channel 参数。**简单方案：保持无 channel，等优化时再加。**

### 5.7 云侧中间层计算的异步特性

FIRST round 完成 isend 后，云侧开始计算中间层。LAST round 的 recv 会阻塞直到云侧完成。如果 EngineCore 在 FIRST 和 LAST 之间发送了其他预处理 RPC（如 `initialize_from_config`），`recv` 不会受影响，因为云侧的 send 在 LAST 的 recv 之前已经排队。

---

## 6. 相关文件索引

| 文件 | 关键内容 |
|------|---------|
| [shared_model_multiproc_executor.py](../../vllm/v1/executor/shared_model_multiproc_executor.py#L319) | `worker_busy_loop` 入口 + end-of-round 逻辑 |
| [shared_model_multiproc_executor.py](../../vllm/v1/executor/shared_model_multiproc_executor.py#L608) | `_dispatch` 方法 |
| [shared_model_edge_worker.py](../../vllm-ascend/../vllm-ascend/vllm_ascend/worker/edge_cloud/shared_model_edge_worker.py#L163) | `_BatchedExecuteMarker` |
| [shared_model_edge_worker.py](../../vllm-ascend/../vllm-ascend/vllm_ascend/worker/edge_cloud/shared_model_edge_worker.py#L804) | `execute_model_batched_pre` |
| [batched_model_runner.py](../../vllm-ascend/../vllm-ascend/vllm_ascend/worker/edge_cloud/batched_model_runner.py#L566) | `BatchedModelRunner.execute_model_pre` |
| [worker.py](../../vllm-ascend/../vllm-ascend/vllm_ascend/worker/worker.py#L550) | 多卡 `NPUWorker.execute_model`（PD 分离基线） |
| [model_runner_v1.py](../../vllm-ascend/../vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2530) | 多卡 `_fast_path` 缓存 |
| [platform.py](../../vllm-ascend/../vllm-ascend/vllm_ascend/platform.py#L510) | `SharedModelEdgeWorker` 判断条件 |
