# 单卡多 DP 非MoE 适配 PD 分离调度框架 — 修改分析

> 基于 [`SharedModelWorkerProc.worker_busy_loop`](../../vllm/v1/executor/shared_model_multiproc_executor.py#L319-L606) 入口，
> 分析非MoE 模型在 `--edge-npu-count 1 --dp 2` 下适配 PD 分离调度需要修改的点。

---

## 1. 问题定义

### 1.1 当前状态：不分首尾，一个 round 走完

```python
# _dispatch (line 634-635):
if method == "execute_model" and hasattr(vw, "execute_model_batched_pre"):
    output = vw.execute_model_batched_pre(args[0])
    # DECODE_FIRST → 也是同一个 execute_model_batched_pre！
    # DECODE_LAST  → 也是同一个 execute_model_batched_pre！
```

end-of-round 流程：

```
run_batched_head        ← head forward + send → 1 次 _model_forward
drive_batched_round     ← isend to cloud + recv closure
drain_batched_round     ← recv + tail forward + logits + handle_output
```

**问题**：DECODE_FIRST 就该只做 head+send 返回空结果，但当前把 tail+logits 也做了。DECODE_LAST 又来一遍完整的 head+send+recv+tail，重复执行。

### 1.2 目标状态：首尾分离，两轮完成

```
Round N (FIRST):
  run_batched_head        ← head forward + isend to cloud
  handle_output(EMPTY)    ← 返回空结果，告诉 EngineCore "head 完成"

Round M (LAST):
  recv from cloud         ← 接收云侧中间层结果
  drain_batched_round     ← tail forward + logits + handle_output(完整结果)
```

### 1.3 核心约束

| # | 约束 | 说明 |
|---|------|------|
| 1 | 首尾不可合并 | FIRST 和 LAST 的处理逻辑完全不同，不能放在同一个 `batched_pending` 中混合处理 |
| 2 | 同类型可合并 | 所有 FIRST marker 拼在一起做 1× merged head forward；所有 LAST marker 拼在一起做 1× merged tail forward |
| 3 | FIRST 不调 `drain_batched_round` | FIRST 只做 head+send，不涉及 recv+tail+logits |
| 4 | LAST 不调 `run_batched_head` + `drive_batched_round` | LAST 只做 recv+tail，不重复 head+send |

---

## 2. 修改概览

### 2.1 修改点列表

| # | 文件 | 位置/方法 | 改动内容 |
|---|------|----------|---------|
| 1 | `shared_model_multiproc_executor.py` | `_dispatch` | batch_type 感知的三路路由 |
| 2 | `shared_model_multiproc_executor.py` | `worker_busy_loop` end-of-round | 首/尾/非PD 三路分组 + 分流处理 |
| 3 | `shared_model_multiproc_executor.py` | 类属性 | 跨 round 状态缓存 `_pd_head_bundles` |
| 4 | `shared_model_edge_worker.py` | 类定义 | 新增 `_FirstRoundMarker`、`_LastRoundMarker` |
| 5 | `shared_model_edge_worker.py` | `SharedModelEdgeWorker` | 新增 `execute_model_head_pre`、`execute_model_tail_pre` |
| 6 | `batched_model_runner.py` | `BatchedModelRunner` | 新增 `execute_model_tail_pre`（轻量预处理） |

### 2.2 改动范围限定

只改**非MoE + PD 分离**路径。非 PD 分离（`batch_type` 不是 FIRST/LAST）走现有 `execute_model_batched_pre` → `_BatchedExecuteMarker` → 完整 Phase A+B/C 流程，**完全不变**。

---

## 3. 修改点 1: `_dispatch` — batch_type 感知路由

**文件**: [`shared_model_multiproc_executor.py`](../../vllm/v1/executor/shared_model_multiproc_executor.py)

**位置**: line 634-675

### 3.1 当前代码

```python
if method == "execute_model" and hasattr(
        virtual_worker, "execute_model_batched_pre"):
    output = virtual_worker.execute_model_batched_pre(
        args[0] if args else None)
```

### 3.2 改为

```python
if method == "execute_model":
    so = args[0] if args else None
    bt = getattr(so, "batch_type", None)

    # PD 分离: FIRST
    if bt in _PD_FIRST_TYPES and hasattr(vw, "execute_model_head_pre"):
        output = vw.execute_model_head_pre(so)   # → _FirstRoundMarker

    # PD 分离: LAST
    elif bt in _PD_LAST_TYPES and hasattr(vw, "execute_model_tail_pre"):
        output = vw.execute_model_tail_pre(so)   # → _LastRoundMarker

    # 非 PD 分离 (原路径，不变)
    elif hasattr(vw, "execute_model_batched_pre"):
        output = vw.execute_model_batched_pre(so)  # → _BatchedExecuteMarker

    # Legacy fallback
    else:
        func = getattr(vw, method)
        output = func(*args, **kwargs)
```

**关键设计**：三种 marker 都设置了 `self.bundle` 和 `self.worker`，后续的 `_round_bundles[k]` 和 `_pending_deferred[k]` 存储逻辑不需要改动。

### 3.3 PD 分离的判断条件（`_PD_FIRST_TYPES` / `_PD_LAST_TYPES`）

参考 [`NPUWorker.execute_model`](../../vllm-ascend/../vllm-ascend/vllm_ascend/worker/worker.py#L578-L592) 中的判断逻辑：

```python
_PD_FIRST_TYPES = frozenset({
    BatchType.DECODE_FIRST,
    BatchType.PREFILL_FIRST,
})
_PD_LAST_TYPES = frozenset({
    BatchType.DECODE_LAST,
    BatchType.PREFILL_LAST,
})
```

非 PD 分离的 `batch_type`（如 `ALL_DECODE`、`PURE_PREFILL`）不走首尾路由，走原有 `execute_model_batched_pre` 路径。

---

## 4. 修改点 2: end-of-round — 首/尾/非PD 三路分流

**文件**: [`shared_model_multiproc_executor.py`](../../vllm/v1/executor/shared_model_multiproc_executor.py)

**位置**: line 500-571（`if batched_pending:` 块及后续）

### 4.1 分组逻辑

```python
pending, self._pending_deferred = (self._pending_deferred, {})

# 收集所有 batched marker (满足 bundle & worker 都存在)
batched_dp_ranks = sorted(
    k for k, d in pending.items()
    if _is_batched_execute_marker(d))

# 按类型分组
first_dp_ranks  = [k for k in batched_dp_ranks
                   if isinstance(pending[k], _FirstRoundMarker)]
last_dp_ranks   = [k for k in batched_dp_ranks
                   if isinstance(pending[k], _LastRoundMarker)]
nonpd_dp_ranks  = [k for k in batched_dp_ranks
                   if k not in first_dp_ranks and k not in last_dp_ranks]
```

### 4.2 三路处理逻辑

#### 分支 A: FIRST marker（只做 head + send, 不做 tail）

```python
if first_dp_ranks:
    marker_cls = type(pending[first_dp_ranks[0]])

    # ─── head forward (合并 1 次) ───
    marker_cls.run_batched_head(first_dp_ranks, self._round_bundles)

    # ─── per-dp_rank isend to cloud ───
    for k in first_dp_ranks:
        pending[k].drive_head_send(self._pd_recv_closures)
        # drive_head_send: isend + 将 recv closure 存入 _pd_recv_closures[k]
        # 注意: 不是 drive_batched_round！不注册立即执行的 recv

    # ─── 返回空结果 ───
    for k in first_dp_ranks:
        self.handle_output(k, EMPTY_MODEL_RUNNER_OUTPUT)

    # ⚠️ 不清除 _round_bundles — LAST round 需要复用
    # ⚠️ 不调 drain_batched_round
```

#### 分支 B: LAST marker（只做 recv + tail, 不做 head + send）

```python
if last_dp_ranks:
    marker_cls = type(pending[last_dp_ranks[0]])

    # ─── recv from cloud (执行 FIRST 时注册的 recv closure) ───
    for k in last_dp_ranks:
        intermediate = self._pd_recv_closures.pop(k)()
        self._round_intermediates[k] = intermediate

    # ─── tail forward + post + handle_output ───
    # 复用 drain_batched_round，但 pending_deferred 里的 closure 已经执行过了
    marker_cls.drain_batched_round(
        self._round_bundles,
        self._round_intermediates,
        pending,                           # 已经被 pop 掉 closure 的 dict
        on_dp_rank_output=self.handle_output,
    )

    # ✅ drain_batched_round 末尾会清除 _round_bundles / _round_intermediates
```

**注意**: `drain_batched_round` 的第一步是调用 `pending_deferred` 中的 recv closure。对于 LAST 路径，我们已经提前执行了 recv 并将 intermediate 存入 `_round_intermediates`，`pending_deferred[k]` 已经被 pop 掉（或变为不可调用），所以 `drain_batched_round` 的 recv 循环会跳过或 no-op。

#### 分支 C: 非PD marker（完整流程，完全不变）

```python
if nonpd_dp_ranks:
    marker_cls = type(pending[nonpd_dp_ranks[0]])
    # 以下 3 步与当前代码完全一致
    marker_cls.run_batched_head(nonpd_dp_ranks, self._round_bundles)
    for k in nonpd_dp_ranks:
        pending[k].drive_batched_round(pending)
    marker_cls.drain_batched_round(
        self._round_bundles,
        self._round_intermediates,
        pending,
        on_dp_rank_output=self.handle_output,
    )
```

### 4.3 为什么 FIRST 和 LAST 会在同一个 pending 里

非MoE 下 `not self.is_moe = True`，end-of-round 不等齐，立即触发：

```
Round N:
  dp_0: DECODE_FIRST 到 MQ → 处理 → handle_output(EMPTY)
  EngineCore[0] 立即发 DECODE_LAST

Round N+1:
  dp_0: DECODE_LAST 到 MQ
  dp_1: DECODE_FIRST 到 MQ
  → pending = {0: _LastRoundMarker, 1: _FirstRoundMarker}
  → 两种 marker 在同一个 pending 中！
```

如果不分组，会把 LAST[0] + FIRST[1] 合并做 head forward — 错误。

### 4.4 顺序随意？

是的。在一个循环迭代中，FIRST 和 LAST 的处理顺序不重要：

```
先处理 FIRST 再处理 LAST ✓
先处理 LAST 再处理 FIRST ✓
```

两者操作的是不同的 `_round_bundles[k]` 和 `_round_intermediates[k]`，互不干扰。

---

## 5. 修改点 3: 跨 Round 状态缓存

**文件**: [`shared_model_multiproc_executor.py`](../../vllm/v1/executor/shared_model_multiproc_executor.py)

### 5.1 为什么需要

FIRST round 产出 head hidden states（isend 到云侧），LAST round 需要：
1. **bundle** — 做 tail forward 时需要的 input_ids、positions、scheduler_output
2. **recv closure** — 接收云侧中间层结果

这两个值在 FIRST round 产生，但 LAST round 才消费。中间隔了一个 round 边界。

### 5.2 需要新增的类属性

```python
class SharedModelWorkerProc:
    # 现有
    _round_bundles: dict[int, _ExecuteModelBundle]
    _round_intermediates: dict[int, Any]
    _pending_deferred: dict[int, Any]

    # 新增: PD 分离跨 round 缓存
    _pd_head_bundles: dict[int, _ExecuteModelBundle] = {}
        # FIRST round 存入, LAST round 消费后 pop
    _pd_recv_closures: dict[int, Callable[[], Any]] = {}
        # FIRST round 注册 recv closure, LAST round 执行后 pop
```

### 5.3 生命周期

```
FIRST round 处理时:
  run_batched_head 产生 per_dp_hidden
  drive_head_send:
    _pd_head_bundles[k] = bundle          ← 保存 bundle
    _pd_recv_closures[k] = recv_closure   ← 保存 recv closure

LAST round 处理时:
  intermediate = _pd_recv_closures.pop(k)()   ← 执行 recv + 清理
  bundle = _pd_head_bundles.pop(k)            ← 取出 bundle + 清理
  drain_batched_round(...)                   ← 用 bundle 做 tail forward
```

### 5.4 为什么不把 bundle 放在 `_round_bundles` 里跨 round？

`_round_bundles` 的语义是**当前 round 的 bundle**，非PD 路径的 `drain_batched_round` 末尾会 `round_bundles.clear()`。如果 FIRST 把 bundle 放在 `_round_bundles` 里不清除，非PD 路径会误伤。独立词典语义更清晰。

---

## 6. 修改点 4: `SharedModelEdgeWorker` — 新增方法 + Marker 类

**文件**: `vllm-ascend/vllm_ascend/worker/edge_cloud/shared_model_edge_worker.py`

### 6.1 `execute_model_head_pre`

```python
def execute_model_head_pre(self, scheduler_output):
    """FIRST 阶段: 完整预处理 → _FirstRoundMarker"""
    self._wait_pp_send_work(
        self._hidden_channel_for(scheduler_output))
    self.profiler.step()
    result = self.model_runner.execute_model_pre(scheduler_output)
    if isinstance(result, _ExecuteModelBundle):
        return _FirstRoundMarker(bundle=result, worker=self)
    return result
```

与 `execute_model_batched_pre` 逻辑几乎完全相同，区别只在于返回的 marker 类型。

### 6.2 `execute_model_tail_pre`

```python
def execute_model_tail_pre(self, scheduler_output):
    """LAST 阶段: 轻量预处理 → _LastRoundMarker"""
    self._wait_pp_send_work(
        self._hidden_channel_for(scheduler_output))
    # 不调 profiler.step() — 这是同一轮的第二步
    result = self.model_runner.execute_model_tail_pre(scheduler_output)
    if isinstance(result, _ExecuteModelBundle):
        return _LastRoundMarker(bundle=result, worker=self)
    return result
```

### 6.3 `_FirstRoundMarker`

```python
class _FirstRoundMarker(DeferredExecutePostprocess):
    """PD 分离 FIRST round marker: 只参与 Phase A (head+send)，不参与 Phase B/C。"""

    _per_dp_hidden: dict[int, Any] | None = None

    __slots__ = ("bundle", "worker")

    def __init__(self, bundle, worker):
        self.bundle = bundle
        self.worker = worker
        super().__init__(postprocess=lambda: None)

    # ─── Phase A: merged head forward (class method, 1 次/round) ───
    @classmethod
    def run_batched_head(cls, batched_dp_ranks, round_bundles):
        """同 _BatchedExecuteMarker.run_batched_head"""
        leader = _find_leader()
        bundles = [round_bundles[k] for k in batched_dp_ranks]
        per_dp_hidden_list = leader.model_runner.execute_model_batched_head(
            bundles, batched_dp_ranks=batched_dp_ranks)
        cls._per_dp_hidden = dict(zip(batched_dp_ranks, per_dp_hidden_list))

    # ─── Phase A: per-dp_rank isend (instance method, N 次/round) ───
    def drive_head_send(self, pd_recv_closures: dict):
        """isend to cloud + 将 recv closure 存入跨 round 缓存"""
        dp_rank = self.worker.local_rank
        hidden_k = self._per_dp_hidden[dp_rank]
        # ... all_gather / merge payload ...
        self.worker._pp_send_work = edge_cloud_isend_tensor_dict(
            hidden_k, dst=dp_rank + 1, num_tokens=...)

        # 注册 recv closure 到外部缓存（不立即执行）
        pd_recv_closures[dp_rank] = self.worker.make_batched_recv_closure(
            src=dp_rank + 1, num_tokens=..., sp_chunk=...)
```

### 6.4 `_LastRoundMarker`

```python
class _LastRoundMarker(DeferredExecutePostprocess):
    """PD 分离 LAST round marker: 只参与 Phase B/C (recv+tail)，不参与 Phase A。"""

    __slots__ = ("bundle", "worker")

    def __init__(self, bundle, worker):
        self.bundle = bundle
        self.worker = worker
        super().__init__(postprocess=lambda: None)

    # 不定义 run_batched_head / drive_batched_round
    # LAST round 的处理全部在 worker_busy_loop 的 end-of-round 中直接写:
    #   1. 执行 _pd_recv_closures[k]()
    #   2. 调 drain_batched_round
```

---

## 7. 修改点 5: `BatchedModelRunner.execute_model_tail_pre`

**文件**: `vllm-ascend/vllm_ascend/worker/edge_cloud/batched_model_runner.py`

### 7.1 为什么需要轻量预处理

LAST round 的 `execute_model` RPC 必须返回非空的 marker（才能走 end-of-round drain），但 tail forward 不需要重建 attention metadata（已在 FIRST round 构建并缓存在 `_merged_attn_ctx_cache` 中）。

参考多卡 `NPUWorker` 的 fast path: 在 [`model_runner_v1.py:2530-2568`](../../vllm-ascend/../vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2530-L2568)，`_fast_path` 缓存 `attn_metadata`、`logits_indices`、`cudagraph_mode` 等。LAST 阶段跳过 `_update_states` 和 attention metadata 重建。

### 7.2 实现

```python
def execute_model_tail_pre(self, scheduler_output):
    """LAST 轻量预处理:
    - 跳过 _update_states（input_batch 不需要重新初始化）
    - 跳过 attention metadata 重建（复用 FIRST round 缓存的 merged attn ctx）
    - 只做必要的同步
    """
    # 1. 轻量同步: 更新 num_computed_tokens
    self.input_batch.num_computed_tokens_cpu[...] = \
        scheduler_output.num_computed_tokens_cpu_tensor

    # 2. 验证一致性
    assert self.input_batch.req_ids == scheduler_output.req_ids

    # 3. 返回最小 bundle（input_ids/positions 从 FIRST 缓存取）
    return _ExecuteModelBundle(
        cm_base=None,                  # 不用 per-dp_rank attn metadata
        per_gid_cm=None,               # tail 用缓存的 merged attn ctx
        per_gid_extra=None,
        input_ids=self._cached_input_ids,
        positions=self._cached_positions,
        inputs_embeds=self._cached_inputs_embeds,
        scheduler_output=scheduler_output,
        ...
    )
```

---

## 8. 现有路径不变 (Backward Compatible)

### 8.1 非 PD 分离路径

`batch_type` 既不是 FIRST 也不是 LAST 的 `execute_model` 走原有的 `execute_model_batched_pre` → `_BatchedExecuteMarker` → 完整 Phase A+B+C 流程。所有现有行为保持不变。

### 8.2 MoE 路径

`not self.is_moe` 为 False，end-of-round 触发条件仍为 `all(paused)`，FIRST/LAST 路由不影响 MoE 路径。

### 8.3 Legacy `DeferredExecutePostprocess`

`_is_batched_execute_marker` 返回 False 的非 marker deferred 继续走 `legacy_pending` → `deferred()` 路径（[line 587-601](../../vllm/v1/executor/shared_model_multiproc_executor.py#L587-L601)），不受影响。

### 8.4 其他 SYNC/NON-SYNC 方法

`sample_tokens`、`add_lora`、`initialize_from_config` 等方法的处理完全不变。

---

## 9. 完整数据流 (FIRST → LAST)

```
┌── Round N (FIRST) ──────────────────────────────────────┐
│                                                           │
│  _dispatch[0]: DECODE_FIRST → execute_model_head_pre     │
│    → _FirstRoundMarker(bundle=b0, worker=w0)             │
│  _dispatch[1]: DECODE_FIRST → execute_model_head_pre     │
│    → _FirstRoundMarker(bundle=b1, worker=w1)             │
│                                                           │
│  End-of-round:                                            │
│    first_dp_ranks = [0, 1]                               │
│                                                           │
│    ┌─ run_batched_head([0, 1], bundles)                 │
│    │   1× _model_forward(merged head)                    │
│    │   → per_dp_hidden = {0: h0, 1: h1}                 │
│    │                                                      │
│    ├─ drive_head_send (per dp_rank):                     │
│    │   k=0: edge_cloud_isend(h0, dst=1)                  │
│    │   k=1: edge_cloud_isend(h1, dst=2)                  │
│    │   _pd_head_bundles = {0: b0, 1: b1}                │
│    │   _pd_recv_closures = {0: recv0, 1: recv1}         │
│    │                                                      │
│    └─ handle_output(0, EMPTY)  → EngineCore[0]          │
│       handle_output(1, EMPTY)  → EngineCore[1]          │
│                                                           │
│    — 此时云侧开始异步计算中间层 —                          │
└───────────────────────────────────────────────────────────┘

┌── Round M (LAST) ───────────────────────────────────────┐
│                                                           │
│  _dispatch[0]: DECODE_LAST → execute_model_tail_pre      │
│    → _LastRoundMarker(bundle=b0', worker=w0)             │
│  _dispatch[1]: DECODE_LAST → execute_model_tail_pre      │
│    → _LastRoundMarker(bundle=b1', worker=w1)             │
│                                                           │
│  End-of-round:                                            │
│    last_dp_ranks = [0, 1]                                │
│                                                           │
│    ┌─ recv (per dp_rank):                                │
│    │   k=0: intermediate0 = _pd_recv_closures.pop(0)()   │
│    │   k=1: intermediate1 = _pd_recv_closures.pop(1)()   │
│    │                                                      │
│    ├─ drain_batched_round(...)                            │
│    │   ├─ 1× execute_model_batched_tail(                 │
│    │   │     [b0, b1], [intermediate0, intermediate1])   │
│    │   │     → merged logits/hidden                      │
│    │   ├─ per-dp_rank 切片                                │
│    │   ├─ execute_model_post_batched (per dp_rank)       │
│    │   └─ handle_output(k, None)                         │
│    │                                                      │
│    └─ 清理: _round_bundles.clear()                       │
│              _round_intermediates.clear()                 │
│              _merged_attn_ctx_cache = None               │
└───────────────────────────────────────────────────────────┘
```

---

## 10. 边界情况

### 10.1 空 batch

`total_num_scheduled_tokens == 0`：不产生 marker，直接 `handle_output`，不 pause，与当前行为一致。

### 10.2 只有部分 dp_rank 是 FIRST

```
Round: pending = {0: _FirstRoundMarker, 1: _LastRoundMarker}
  first_dp_ranks = [0] → head+send for dp_0
  last_dp_ranks  = [1] → recv+tail for dp_1
  nonpd_dp_ranks = []
```

两个分支各自独立执行，互不干扰。

### 10.3 FIRST round 异常

FIRST marker 的 head 失败 → `handle_output(k, e)` 给对应 dp_rank → EngineCore 不会发 DECODE_LAST。`_pd_head_bundles[k]` 和 `_pd_recv_closures[k]` 需要清理（在 exception 处理中 pop）。

### 10.4 LAST round 的 bundle 缺失

如果 FIRST round 失败但 EngineCore 仍发了 DECODE_LAST（不应该发生，但做防御），`_pd_head_bundles` 中缺少对应 k 的 bundle → 在 LAST 处理块中检查 key 存在性，缺失时 `handle_output(k, RuntimeError(...))`。

---

## 11. 与多卡 NPUWorker 的对照

| 维度 | 多卡 NPUWorker | 单卡 Shared Model (本方案) |
|------|---------------|---------------------------|
| FIRST head forward | `model_runner.execute_model(so)` 1 次/dp | `run_batched_head` 1 次/round (合并) |
| FIRST isend | `edge_cloud_isend_tensor_dict(channel=...)` | `drive_head_send` (无 channel) |
| FIRST 返回 | `DeferredExecutePostprocess(EMPTY)` | `handle_output(EMPTY)` |
| LAST 预处理 | `_fast_path`: 跳过 `_update_states`, 复用 `_edge_prepare_cache` | `execute_model_tail_pre`: 跳过 `_update_states`, 复用 `_merged_attn_ctx_cache` |
| LAST recv | `edge_cloud_recv_tensor_dict(src=...)` | `_pd_recv_closures[k]()` |
| LAST tail forward | `model_runner.execute_model(so, intermediate)` 1 次/dp | `drain_batched_round` 1 次/round (合并) |
| LAST 返回 | `model_runner.execute_model` 返回完整结果 | `handle_output(None)` (结果由 `post_batched` 写入 state) |
