# `execute_model_batched_pre` 接口调用流程分析

> 基于 `SharedModelEdgeWorker.execute_model_batched_pre` 入口，分析单卡多 DP (`--edge-npu-count 1 --dp 2`) 非MoE 模型从 EngineCore 到最终输出的完整调用链。

---

## 1. 接口概览

### 1.1 定义

```python
# SharedModelEdgeWorker (vllm-ascend)
def execute_model_batched_pre(
    self,
    scheduler_output: "SchedulerOutput",
) -> "_BatchedExecuteMarker | ModelRunnerOutput | None":
```

| 属性 | 说明 |
|------|------|
| **所在文件** | [`vllm-ascend/vllm_ascend/worker/edge_cloud/shared_model_edge_worker.py:804`](../../vllm-ascend/vllm_ascend/worker/edge_cloud/shared_model_edge_worker.py#L804) |
| **继承关系** | `SharedModelEdgeWorker` → `NPUWorker` |
| **对应原接口** | `NPUWorker.execute_model` (多卡流程) |
| **调用方** | `SharedModelWorkerProc._dispatch` |
| **返回值** | `_BatchedExecuteMarker` (正常) / `None` (空 batch 等) |
| **是否区分 batch_type** | **否** — 所有 `execute_model` RPC 统一路由到这里 |

### 1.2 触发条件

在 `_dispatch` 中（[shared_model_multiproc_executor.py:634-637](../../vllm/vllm/v1/executor/shared_model_multiproc_executor.py#L634-L637)）：

```python
if method == "execute_model" and hasattr(
        virtual_worker, "execute_model_batched_pre"):
    output = virtual_worker.execute_model_batched_pre(
        args[0] if args else None)
```

需要同时满足：
1. RPC 方法是 `"execute_model"`（EngineCore 发送）
2. `virtual_worker` 是 `SharedModelEdgeWorker` 类型（`--edge-npu-count 1` + `is_edge_node`）

---

## 2. 完整调用链

### 2.1 总览

```
 EngineCore.send(execute_model, scheduler_output)
                         │
    ┌────────────────────┼────────────────────┐
    │                    ▼                    │
    │  MultiprocExecutor.collective_rpc(      │
    │      "execute_model",                   │
    │      unique_reply_rank=0)               │
    │                    │                    │
    │         rpc_broadcast_mqs[k]            │
    │         .enqueue("execute_model", ...)  │
    └────────────────────┼────────────────────┘
                         │
           ┌─────────────▼─────────────┐
           │  SharedModelWorkerProc    │
           │  .worker_busy_loop()      │
           │                           │
           │  [1] mq.dequeue()         │
           │       ↓                   │
           │  [2] _dispatch(k, ...)    │
           │       ↓                   │
           │  [3] execute_model_       │
           │      batched_pre(...)     │
           │       ↓                   │
           │  [4] execute_model_pre()  │
           │       ↓                   │
           │  返回 _BatchedExecute     │
           │       Marker              │
           │       ↓                   │
           │  存入 _pending_deferred   │
           │       ↓                   │
           │  ── end-of-round ──       │
           │       ↓                   │
           │  [5] run_batched_head()   │
           │       ↓                   │
           │  [6] drive_batched_       │
           │      round() per dp_rank  │
           │       ↓                   │
           │  [7] drain_batched_       │
           │      round()              │
           │       ↓                   │
           │  handle_output(k, None)   │
           └─────────────┬─────────────┘
                         │
              response_mqs[k].enqueue(...)
                         │
                EngineCore 收到结果
```

### 2.2 详细步骤

#### 步骤 1: EngineCore 发起

```python
# vllm/v1/engine/core.py
scheduler_output = self.scheduler.schedule()
future = self.model_executor.execute_model(scheduler_output, non_block=True)
```

每个 dp_rank 的 EngineCore 独立调度并发送 RPC。

#### 步骤 2: MultiprocExecutor 广播

```python
# vllm/v1/executor/multiproc_executor.py:375-383
def execute_model(self, scheduler_output):
    local_only = (enable_edge_cloud and is_edge_node)  # → True
    return self.collective_rpc("execute_model",
        args=(scheduler_output,),
        unique_reply_rank=self.output_rank,  # → 0
        local_only=local_only)
```

`local_only=True` 意味着跳过跨节点 TCP 发送，只投递给本地边侧的 `SharedModelWorkerProc`。

#### 步骤 3: `_dispatch` 路由

```python
# shared_model_multiproc_executor.py:634-637
if method == "execute_model" and hasattr(
        virtual_worker, "execute_model_batched_pre"):
    output = virtual_worker.execute_model_batched_pre(
        args[0] if args else None)
```

关键决策点：`hasattr` 检测到 `SharedModelEdgeWorker` → 路由到 batched path，**跳过** 原来的 `NPUWorker.execute_model` 中的 `batch_type` 分发逻辑。

#### 步骤 4: `execute_model_batched_pre` 内部

```python
# shared_model_edge_worker.py:803-854
def execute_model_batched_pre(self, scheduler_output):
    # 4a. 等待上一次的 PP send 完成
    if self._pp_send_work:
        for handle in self._pp_send_work:
            handle.wait()
        self._pp_send_work = []

    # 4b. profiler step
    if self.profiler is not None:
        self.profiler.step()

    # 4c. 核心: 调用 BatchedModelRunner 做预处理
    result = self.model_runner.execute_model_pre(scheduler_output)

    # 4d. 区分返回类型
    if isinstance(result, _ExecuteModelBundle):
        return _BatchedExecuteMarker(bundle=result, worker=self)
    return result  # None / EMPTY (空 batch 等短路径)
```

#### 步骤 4c 展开: `BatchedModelRunner.execute_model_pre`

```python
# batched_model_runner.py:566-867
def execute_model_pre(self, scheduler_output):
    # I. 更新 input_batch 内部状态
    self._update_states(scheduler_output)

    # II. 构建 per-dp_rank attention metadata
    #     (_build_attn_metadata → _may_reorder_batch → _build_metadata)
    self._run_input_preparation()

    # III. 生成 input_ids, positions, inputs_embeds
    self._preprocess()

    # IV. 打包返回
    return _ExecuteModelBundle(
        cm_base=...,
        per_gid_cm=...,
        per_gid_extra=...,
        input_ids=...,
        positions=...,
        inputs_embeds=...,
        scheduler_output=scheduler_output,
    )
```

#### 步骤 5: 存储 marker（回到 `_dispatch`）

```python
# shared_model_multiproc_executor.py:656-669
if output_rank is None or self.rank == output_rank:  # Shared Model 下永远 True
    if (method == "execute_model"
            and getattr(output, "bundle", None) is not None):
        self._round_bundles[dp_rank] = output.bundle   # 存 bundle
        self._pending_deferred[dp_rank] = output        # 存 marker

# method in SYNC_METHODS → paused[k] = True
```

marker 存入但**不立即处理**，等待 end-of-round。

#### 步骤 6: End-of-round 触发

```python
# shared_model_multiproc_executor.py:462
if all(paused) or not self.is_moe:
```

| 模型 | 触发时机 |
|------|---------|
| **非MoE** | 每次 for 循环结束后立即触发 (`not self.is_moe = True`) |
| **MoE** | 等所有 dp_rank paused (`all(paused)`) |

#### 步骤 7: 三路分流 + Phase A (Batched Head)

```python
# shared_model_multiproc_executor.py:469-607

# 7a. 从 _pending_deferred 取出并按类型分组
pending, self._pending_deferred = (self._pending_deferred, {})

batched_dp_ranks_list = sorted(               # ← 关键: 筛选出需要合并的 dp_rank
    k for k, d in pending.items()
    if _is_batched_execute_marker(d))          #   bundle & worker 都存在
batched_pending: dict[int, Any] = {
    k: pending[k] for k in batched_dp_ranks_list
}
legacy_pending: dict[int, Any] = {
    k: d for k, d in pending.items()
    if k not in batched_pending
}

# 7b. Phase A — 1 次 merged head forward
if batched_pending:
    marker_cls = type(batched_pending[batched_dp_ranks_list[0]])
    marker_cls.run_batched_head(
        batched_dp_ranks_list,
        self._round_bundles,
    )
```

**`batched_dp_ranks_list` 的来源**（[line 500-502](../../vllm/v1/executor/shared_model_multiproc_executor.py#L500-L502)）：

```python
batched_dp_ranks_list = sorted(
    k for k, d in pending.items()
    if _is_batched_execute_marker(d))
```

它是从 `_pending_deferred` 中筛选出所有 `_BatchedExecuteMarker` 类型的 dp_rank 列表，并 **排序** 保证确定性。这个列表可能不等于 `{0, 1, ..., dp_size-1}`，取决于：

| 场景 | `batched_dp_ranks_list` | 说明 |
|------|--------------------------|------|
| 非MoE，两个 dp_rank 都到达 | `[0, 1]` | 正常情况 |
| 非MoE，只有 dp_rank 0 到达 | `[0]` | `not self.is_moe=True`，立即 end-of-round |
| MoE，`all(paused)` | `[0, 1]` | 等齐后永远是全集 |
| 有空 batch 的 dp_rank | 不包含该 rank | 空 batch 不入 `_pending_deferred` |

这个列表作为参数传入 `run_batched_head` → `execute_model_batched_head`，决定了哪些 bundle 被合并到 merged forward 中。然后用 `batched_dp_ranks_list[0]` 取任意一个 marker 来确定 `marker_cls`（所有 batched marker 类型相同）。

#### 步骤 7b 展开: `run_batched_head` → `execute_model_batched_head`

```python
# _BatchedExecuteMarker.run_batched_head (shared_model_edge_worker.py:228-279)
@classmethod
def run_batched_head(cls, batched_dp_ranks, round_bundles):
    leader = find_leader_worker()
    leader_runner = leader.model_runner
    bundles = [round_bundles[k] for k in batched_dp_ranks]
    per_dp_hidden_list = leader_runner.execute_model_batched_head(
        bundles, batched_dp_ranks=batched_dp_ranks)
    cls._per_dp_hidden = dict(zip(batched_dp_ranks, per_dp_hidden_list))
```

```python
# BatchedModelRunner.execute_model_batched_head (batched_model_runner.py:870-1022)
def execute_model_batched_head(self, bundles, batched_dp_ranks):
    # 1. 构建 merged attention context
    merged_ctx = self._get_or_build_merged_attn_ctx(bundles)
    #    → 合并 query_start_loc, seq_lens, block_tables, slot_mapping
    #    → 合并 input_ids, positions, inputs_embeds
    #    → decode-first reorder (如果需要)
    #    → 缓存 merged_attn_ctx_cache

    # 2. 1 次 _model_forward (segment_a)
    hidden_states = self._model_forward(
        num_tokens_padded, merged_input_ids,
        merged_positions, None, merged_inputs_embeds)

    # 3. un-reorder + per-dp_rank 切片
    hidden_cat_order = hidden_states[inv_merged_token_perm]
    per_dp_hidden = [
        hidden_cat_order[offset:offset + n_actual]
        for offset, n_actual in per_dp_token_ranges
    ]
    return per_dp_hidden
```

#### 步骤 8: Phase A — Per-dp_rank PP isend

```python
# shared_model_multiproc_executor.py:537-554
for dp_rank, marker in batched_pending.items():
    marker.drive_batched_round(batched_pending)
```

```python
# _BatchedExecuteMarker.drive_batched_round (shared_model_edge_worker.py:284-330)
def drive_batched_round(self, pending_deferred):
    dp_rank = self.worker.local_rank
    hidden_k = _per_dp_hidden[dp_rank]          # 取 per-dp_rank 的 head 输出

    # all_gather (如有 SP)
    if enable_sp():
        _gathered = self.worker._all_gather_tensor_dict(hidden_k.tensors)
    else:
        _gathered = hidden_k.tensors

    # PP isend 到云侧
    self.worker._pp_send_work = edge_cloud_isend_tensor_dict(
        _gathered, dst=dp_rank + 1, num_tokens=num_tokens)

    # 注册 recv closure (放入 pending_deferred)
    pending_deferred[dp_rank] = (
        self.worker.make_batched_recv_closure(
            src=dp_rank + 1, num_tokens=num_tokens))
```

#### 步骤 9: Phase B/C — Batched Tail

```python
# shared_model_multiproc_executor.py:557-565
marker_cls.drain_batched_round(
    self._round_bundles,
    self._round_intermediates,
    batched_pending,
    on_dp_rank_output=self.handle_output,
)
```

```python
# _BatchedExecuteMarker.drain_batched_round (shared_model_edge_worker.py:335-485)
@classmethod
def drain_batched_round(cls, round_bundles, round_intermediates,
                         pending_deferred, on_dp_rank_output):
    # 1. 逐个调用 recv closure → 接收云侧中间层输出
    for dp_rank, closure in pending_deferred.items():
        intermediate = closure()
        round_intermediates[dp_rank] = intermediate

    # 2. 1 次 batched tail forward
    leader_runner.execute_model_batched_tail(
        bundles, batched_dp_ranks, intermediates)
    #    → concat cloud hidden_states/residual
    #    → reorder (复用 cached merged_attn_ctx)
    #    → _model_forward(tail layers)
    #    → compute_logits
    #    → per-dp_rank 切片

    # 3. per-dp_rank post_batched (写入 execute_model_state)
    for dp_rank, bundle in bundles_in_order:
        worker_runner.execute_model_post_batched(...)

    # 4. 通过回调发回 EngineCore
    for dp_rank in batched_dp_ranks:
        on_dp_rank_output(dp_rank, None)   # → handle_output → response_mq

    # 5. 清理
    cls._per_dp_hidden = None
    round_bundles.clear()
    round_intermediates.clear()
```

---

## 3. `_BatchedExecuteMarker` 详解

### 3.1 定义与作用

```python
# shared_model_edge_worker.py:163-222
class _BatchedExecuteMarker(DeferredExecutePostprocess):
    __slots__ = ("bundle", "worker")
    _per_dp_hidden: dict[int, Any] | None = None  # 类变量，跨实例共享

    def __init__(self, bundle, worker):
        self.bundle = bundle      # _ExecuteModelBundle
        self.worker = worker      # SharedModelEdgeWorker
```

| 属性 | 说明 |
|------|------|
| `bundle` | 预处理结果：attn_metadata + input_ids/pos/embeds + scheduler_output |
| `worker` | 持有者 worker 引用，用于 `local_rank`、`model_runner`、PP 通信 |
| `_per_dp_hidden` (类变量) | Phase A 输出 → Phase A isend 输入 → Phase B/C 时清理 |

### 3.2 方法一览

| 方法 | 类型 | 调用次数/round | 作用 |
|------|------|:---:|------|
| `run_batched_head` | `@classmethod` | 1 | 1次 merged head forward (segment_a) |
| `drive_batched_round` | 实例方法 | N (dp_rank 数) | per-dp_rank PP isend + recv closure 注册 |
| `drain_batched_round` | `@classmethod` | 1 | recv cloud + 1次 merged tail + logits + post + handle_output |

### 3.3 `@classmethod` vs 实例方法

- `run_batched_head` 和 `drain_batched_round` 是类方法：操作不依赖具体 dp_rank，每次 round 只执行 1 次
- `drive_batched_round` 是实例方法：需要 `self.worker.local_rank` 确定 dst 和 src
- 跨实例状态通过**类变量** `_per_dp_hidden` 传递

---

## 4. 关键判断点 `_is_batched_execute_marker`

```python
# shared_model_multiproc_executor.py:89-91
def _is_batched_execute_marker(obj: Any) -> bool:
    return (getattr(obj, "bundle", None) is not None
            and getattr(obj, "worker", None) is not None)
```

Duck-type 检测，避免导入 vllm_ascend。同时存在 `bundle` 和 `worker` 属性的就是 batched marker。

---

## 5. 非MoE 特殊行为

### 5.1 Round barrier 失效

```python
# shared_model_multiproc_executor.py:462
if all(paused) or not self.is_moe:  # not self.is_moe → True → 永远进入
```

非MoE 下，end-of-round 在每次 for 循环结束后立即触发，**不等** 所有 dp_rank 完成。

### 5.2 后果：批处理粒度不确定

```
Round 1:  MQ[0] 有 execute_model，MQ[1] 空
  → end-of-round 立即触发
  → batched_pending = {0: marker0}   ← 只有 1 个 bundle
  → 1× merged forward (1 个 bundle)
  → handle_output(0, None)

Round 2:  MQ[1] 的 execute_model 到达
  → 独立处理
```

MoE 的话会等到 `all(paused)`，保证合并的 bundle 数量 = dp_size。

### 5.3 PD 分离未适配的风险

`execute_model_batched_pre` 不区分 `batch_type`，`DECODE_FIRST` 也走完整 head+tail+logits。适配 PD 分离需要在 `_dispatch` 层按 `batch_type` 分流到不同的预处理入口。

---

## 6. `dp=2` 非MoE 完整时序

```
Round 1:
  时间线                                dp_rank=0              dp_rank=1
  ─────────────────────────────────────────────────────────────────────
  EngineCore 调度                  scheduler_output_0      scheduler_output_1
  enqueue                          → MQ[0]                 → MQ[1]
  dequeue + _dispatch              execute_model_          execute_model_
                                     batched_pre()           batched_pre()
                                   → _BatchedExecuteMarker → _BatchedExecuteMarker
  paused                           paused[0]=True           paused[1]=True
  ─── end-of-round (not self.is_moe=True) ───
  pending                          {0: marker0, 1: marker1}
  batched_pending                  {0: marker0, 1: marker1}
  run_batched_head                 ──── 1× merged head forward ────
                                      (合并 dp_0 + dp_1 的 input)
  drive_batched_round              isend(dst=1)            isend(dst=2)
                                     注册 recv_0             注册 recv_1
  drain_batched_round              recv_0 → interm_0       recv_1 → interm_1
                                     收到云侧输出            收到云侧输出
                                   ──── 1× merged tail forward ────
                                      logits_0              logits_1
                                   post_batched_0          post_batched_1
  handle_output                    response_mq[0] ←        response_mq[1] ←
  ─────────────────────────────────────────────────────────────────────
  EngineCore 收到                  触发 sample_tokens      触发 sample_tokens
```

---

## 7. 与多卡 `NPUWorker.execute_model` 对照

| 维度 | 多卡 (`NPUWorker`) | 单卡 batched (`SharedModelEdgeWorker`) |
|------|-------------------|----------------------------------------|
| **入口** | `execute_model` | `execute_model_batched_pre` |
| **batch_type 路由** | 是 (FIRST/LAST 分流) | **否** (统一入口) |
| **Head forward** | per-worker 独立执行 | 1 次 merged forward (合并所有 dp_rank) |
| **Tail forward** | per-worker 独立执行 | 1 次 merged forward |
| **forward 总次数** | `dp_size * 2` (head + tail) | **2 次** (1 head + 1 tail) |
| **PP 通信模式** | per-worker 独立 isend/recv | per-worker isend + 统一 recv closure |
| **返回时机** | HEAD 返回 None, TAIL 返回 logits | 一次 round 返回全部 |
| **Round barrier** | 不需要 (独立 worker 进程) | MoE: 必须; 非MoE: 不需要 |

---

## 8. 关键文件索引

| 文件 | 关键行 | 内容 |
|------|--------|------|
| `vllm/v1/engine/core.py` | ~421-422 | EngineCore 调度 + execute_model 发送 |
| `vllm/v1/executor/multiproc_executor.py` | 360-383 | MultiprocExecutor.execute_model → collective_rpc |
| `vllm/v1/executor/shared_model_multiproc_executor.py` | 319-607 | `worker_busy_loop` 主循环 + end-of-round |
| `vllm/v1/executor/shared_model_multiproc_executor.py` | 608-678 | `_dispatch` 路由 + marker 存储 |
| `vllm-ascend/vllm_ascend/worker/edge_cloud/shared_model_edge_worker.py` | 163-222 | `_BatchedExecuteMarker` 定义 |
| `vllm-ascend/vllm_ascend/worker/edge_cloud/shared_model_edge_worker.py` | 228-279 | `run_batched_head` (Phase A) |
| `vllm-ascend/vllm_ascend/worker/edge_cloud/shared_model_edge_worker.py` | 284-330 | `drive_batched_round` (Phase A isend) |
| `vllm-ascend/vllm_ascend/worker/edge_cloud/shared_model_edge_worker.py` | 335-485 | `drain_batched_round` (Phase B/C) |
| `vllm-ascend/vllm_ascend/worker/edge_cloud/shared_model_edge_worker.py` | 803-854 | `execute_model_batched_pre` 入口 |
| `vllm-ascend/vllm_ascend/worker/edge_cloud/batched_model_runner.py` | 566-867 | `execute_model_pre` (预处理) |
| `vllm-ascend/vllm_ascend/worker/edge_cloud/batched_model_runner.py` | 870-1022 | `execute_model_batched_head` (合并 head) |
| `vllm-ascend/vllm_ascend/worker/edge_cloud/batched_model_runner.py` | 1025-1243 | `execute_model_batched_tail` (合并 tail) |
| `vllm-ascend/vllm_ascend/worker/edge_cloud/batched_model_runner.py` | 1244-1290 | `execute_model_post_batched` (per-dp_rank 后处理) |
| `vllm-ascend/vllm_ascend/worker/edge_cloud/batched_model_runner.py` | 1542-1776 | `_get_or_build_merged_attn_ctx` (合并 attention 上下文) |
