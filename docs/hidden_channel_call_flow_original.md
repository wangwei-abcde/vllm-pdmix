# Hidden Channel 完整调用流程（基于原始代码）

> 覆盖从进程启动到请求级通道选择的完整调用链，所有行号指向 vllm-ascend 源码。

---

## 总览：三大阶段

```
┌──────────────────────────────────────────────────────────────────────────┐
│  阶段一: 架构初始化              阶段二: 逻辑通道池       阶段三: 请求级   │
│  (import torch.distributed 后)   (Scheduler 构造时)      (每个 step)       │
│                                                                          │
│  create_alternate_groups()        HiddenChannelManager()  allocate_prefill│
│  create_hidden_channel_groups()   _free_prefills =        allocate_decode │
│                                   [PREFILL_1, PREFILL_2]  release_prefill │
│                                                           release_decode  │
│  isend(tensor, group=...)         ────────────────────────→              │
│  irecv(tensor, group=...)         _hidden_channel_for()   _record/wait    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 阶段一：架构初始化 — 底层通信组创建

### 1.1 入口

```
vllm serve / EngineArgs
  └─ LLMEngine.__init__()
       └─ init_distributed_environment()
            └─ init_model_parallel_group()

文件: vllm_ascend/distributed/parallel_state.py
```

### 1.2 `init_model_parallel_group` 中的通道创建

```python
# parallel_state.py:362-369
# Phase6 hidden data-plane channels are still required in edge-cloud
# mode.  The default PP group is PREFILL_1, the alternate PP group is
# DECODE, and the extra hidden-channel group is PREFILL_2.
pp_group = get_pp_group()
if pp_group.world_size > 1:
    pp_group.create_alternate_groups(backend)          # → L367
    if hasattr(pp_group, "create_hidden_channel_groups"):
        pp_group.create_hidden_channel_groups(backend)  # → L369
```

### 1.3 `create_alternate_groups` — 创建 DECODE 通道组

**文件**: [patch_distributed.py:263-305](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/patch/worker/patch_distributed.py#L263-L305)

```
create_alternate_groups(backend)
│
├─ hccl_pg_options = create_hccl_pg_options("pp_alt")     # L279
│
├─ for ranks in self._all_group_ranks:                    # L285
│   ├─ torch.distributed.new_group(ranks, backend, pg_options=hccl_pg_options)  # L286-289
│   └─ torch.distributed.new_group(ranks, backend="gloo")                       # L291-293
│
├─ self.alt_device_group = self_alt_device_group          # L299
├─ self.alt_cpu_group = self_alt_cpu_group                # L300
│
└─ 日志: "[PP Group] DECODE alt_device_group"             # L301-304
```

**产出**: `alt_device_group`, `alt_cpu_group` → 后续 DECODE 通道复用此组。

### 1.4 `create_hidden_channel_groups` — 创建 PREFILL_2 通道组

**文件**: [patch_distributed.py:307-341](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/patch/worker/patch_distributed.py#L307-L341)

```
create_hidden_channel_groups(backend)
│
├─ hccl_pg_options = create_hccl_pg_options("pp_prefill2")  # L320
│
├─ for ranks in self._all_group_ranks:                      # L323
│   ├─ torch.distributed.new_group(ranks, backend, pg_options=hccl_pg_options)  # L324-328
│   └─ torch.distributed.new_group(ranks, backend="gloo")                       # L329
│
├─ self.prefill2_device_group = prefill2_device_group       # L335
├─ self.prefill2_cpu_group = prefill2_cpu_group             # L336
│
└─ 日志: "[PP Group] PREFILL_2"                             # L337-340
```

**产出**: `prefill2_device_group`, `prefill2_cpu_group` → PREFILL_2 专用通信组。

### 1.5 三个通信组总览

```
GroupCoordinator (patch_distributed.py)
│
├── self.device_group / cpu_group       → PREFILL_1 (默认组)
├── self.alt_device_group / cpu_group   → DECODE (L299-300)
└── self.prefill2_device_group / cpu_group → PREFILL_2 (L335-336)
```

**流隔离原理**:
- `create_hccl_pg_options("pp_alt")` → 独立 HCCL 流 A
- `create_hccl_pg_options("pp_prefill2")` → 独立 HCCL 流 B
- PREFILL_1 用默认流
- 不同名称 → 不同流 → 不同请求间的通信不互相阻塞

---

## 阶段二：逻辑通道池 — Scheduler 初始化和通道分配

### 2.1 `HiddenChannelManager` 定义和初始化

**文件**: [pd_separated_scheduler.py:35-95](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/core/pd_separated_scheduler.py#L35-L95)

```python
# L35-51
class HiddenChannelManager:
    def __init__(self) -> None:
        self._free_prefills: deque[HiddenChannelType] = deque([
            HiddenChannelType.PREFILL_1,   # L45
            HiddenChannelType.PREFILL_2,   # L46
        ])
        self._head_token_to_channel: dict[str, HiddenChannelType] = {}  # L51
```

池结构:
```
_free_prefills (deque)
┌──────────────┐
│  PREFILL_1   │  ← popleft() 取
│  PREFILL_2   │
└──────────────┘
  append() 归还 →

_decode: 固定 DECODE，无空闲池
  → decode_channel() 静态方法 (L82-84)
```

### 2.2 `PDSeparatedScheduler` 实例化

**文件**: [pd_separated_scheduler.py:137](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/core/pd_separated_scheduler.py#L137)

```python
# L135-137
# Phase6 data-plane channel manager.
self.hidden_channel_manager = HiddenChannelManager()
```

### 2.3 通道分配 / 释放方法

| 方法 | 行号 | 作用 |
|------|------|------|
| `allocate_prefill(head_token)` | [L56-65](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/core/pd_separated_scheduler.py#L56-L65) | `popleft()` 取通道, 记录到 `_head_token_to_channel` |
| `release_prefill(head_token)` | [L67-74](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/core/pd_separated_scheduler.py#L67-L74) | `pop` 映射, `append` 归还 |
| `has_free_prefill()` | [L76-77](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/core/pd_separated_scheduler.py#L76-L77) | 判断池非空 |
| `decode_channel()` (静态) | [L82-84](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/core/pd_separated_scheduler.py#L82-L84) | 固定返回 `DECODE` |
| `get_channel(head_token)` | [L89-90](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/core/pd_separated_scheduler.py#L89-L90) | 查询映射 |

---

## 阶段三：请求级通道选择（每个调度 step）

### 3.1 Scheduler 调度入口

```
schedule()                                         # L158
  └─ _schedule_pd_separated()                     # L167
       └─ state = _prefill_state()                # L168
            ├─ IDLE: prefill_inflight_count == 0  # L30
            ├─ LOW:  prefill_inflight_count == 1  # L31
            └─ HIGH: prefill_inflight_count >= limit # L32
       └─ _pick_by_state(state)                   # L179
```

### 3.2 PREFILL_FIRST 调度 → 分配通道

**文件**: [pd_separated_scheduler.py:400-416](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/core/pd_separated_scheduler.py#L400-L416)

```
_pick_prefill_first_batch()
│
├─ 前置检查: _can_schedule_prefill_first()
│   ├─ prefill_inflight_count < prefill_inflight_limit    # L268
│   └─ hidden_channel_manager.has_free_prefill()          # L269
│
├─ scheduler_output.batch_type = PREFILL_FIRST            # L409
├─ scheduler_output.head_token = uuid4().hex              # L410
├─ scheduler_output.hidden_channel =
│       hidden_channel_manager.allocate_prefill(head_token)  # L411-414
│       ├─ channel = _free_prefills.popleft()             # L63
│       └─ _head_token_to_channel[head_token] = channel   # L64
│
└─ prefill_inflight_count += 1                            # L416
```

### 3.3 DECODE_FIRST 调度 → 固定 DECODE 通道

**文件**: [pd_separated_scheduler.py:569-574](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/core/pd_separated_scheduler.py#L569-L574)

```
_pick_decode_first_batch()
│
├─ scheduler_output.batch_type = DECODE_FIRST             # L570
├─ scheduler_output.head_token = uuid4().hex              # L571
├─ scheduler_output.hidden_channel =
│       hidden_channel_manager.decode_channel()            # L572-573
│       └─ return HiddenChannelType.DECODE                 # L84
│
└─ decode_inflight_count += 1                             # L576
```

### 3.4 PREFILL_LAST — 云端返回，校验通道

```
_prefill_last_batch()  (从 prefills_last_ready 队列取)
│
├─ _validate_prefill_tail_channel(scheduler_output)       # L470
│   ├─ channel 必须 ∈ {PREFILL_1, PREFILL_2}              # L478
│   └─ get_channel(head_token) == channel                 # L482-483
│
└─ update_from_output()                                   # L632
    ├─ prefill_inflight_count -= 1                        # L639
    └─ hidden_channel_manager.release_prefill(head_token)  # L641-643
        ├─ _head_token_to_channel.pop(head_token)          # L70
        └─ _free_prefills.append(channel)                  # L73
```

### 3.5 DECODE_LAST — 云端返回，校验通道

```
_pick_decode_last_batch()  (从 decodes_last_ready 队列取)
│
├─ _validate_decode_tail_channel(scheduler_output)        # L503
│   └─ channel 必须 == DECODE                              # L490
│
└─ update_from_output()  (decode_inflight 在 DF 时已释放)
```

---

## Worker 端：通道解析 → 通信执行

### 3.6 `execute_model` 入口

**文件**: [worker.py:560-596](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/worker/worker.py#L560-L596)

```
NPUWorker.execute_model(scheduler_output)
│
├─ [1] 等待通道上之前的 send
│   if self.model_runner._edge_cloud_enabled:
│       bt = scheduler_output.batch_type
│       if bt in (PREFILL_FIRST, DECODE_FIRST, PREFILL_LAST, DECODE_LAST):
│           self._wait_pp_send_work(
│               self._hidden_channel_for(scheduler_output)  # L571
│           )
│       else:
│           self._wait_pp_send_work()   # fallback: 等所有
│
├─ [2] 分派
│   if bt in (PREFILL_FIRST, DECODE_FIRST):
│       _execute_model_edge_head()                          # L584-586
│   if bt in (PREFILL_LAST, DECODE_LAST):
│       _execute_model_edge_tail()                          # L588-591
```

### 3.7 `_hidden_channel_for` — 通道解析

**文件**: [worker.py:598-607](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/worker/worker.py#L598-L607)

```python
def _hidden_channel_for(self, scheduler_output):
    channel = scheduler_output.hidden_channel      # L599: 优先用 scheduler 设的值
    if channel is not None:
        return channel                              # L601
    # 以下为 fallback (调度器未设 hidden_channel 时的兜底):
    bt = scheduler_output.batch_type
    if bt in (PREFILL_FIRST, PREFILL_LAST):
        return HiddenChannelType.PREFILL_1          # L604
    if bt in (DECODE_FIRST, DECODE_LAST):
        return HiddenChannelType.DECODE             # L606
```

### 3.8 HEAD (isend) 路径

**文件**: [worker.py:609-654](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/worker/worker.py#L609-L654)

```
_execute_model_edge_head(scheduler_output)
│
├─ output = model_runner.execute_model(...)           # L616
│   (模型前向 → segment_a → IntermediateTensors)
│
├─ 获得 channel = _hidden_channel_for(scheduler_output)  # L640
│
├─ 调用 _record_pp_send_work(
│       edge_cloud_send_tensor_dict(_gathered, channel=channel, ...),
│       channel=channel                                # L641-645
│   )
│
└─ edge_cloud_send_tensor_dict() → edge_cloud_isend_tensor_dict()
    └─ parallel_state.py:739
        ├─ group = _get_edge_cloud_hidden_channel_device_group(pp_group, channel)
        │   └─ parallel_state.py:709-712
        │       └─ pp_group._hidden_channel_groups(channel)
        │           └─ patch_distributed.py:343-354
        │               ├─ "prefill_1" → device_group         # L345-346
        │               ├─ "decode"   → alt_device_group      # L347-350
        │               └─ "prefill_2"→ prefill2_device_group # L351-354
        │
        └─ torch.distributed.isend(tensor, dst, group=group)  # L865 / L891
```

### 3.9 TAIL (recv) 路径

**文件**: [worker.py:656-697](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/worker/worker.py#L656-L697)

```
_execute_model_edge_tail(scheduler_output)
│
├─ channel = _hidden_channel_for(scheduler_output)    # L665
│
├─ tensor_dict, comm_handles, comm_postprocess =
│       edge_cloud_broadcast_recv(                     # L666-670
│           num_tokens=...,
│           channel=channel,
│           sp_chunk=...
│       )
│
└─ edge_cloud_broadcast_recv()
    └─ parallel_state.py:1169-1279
        ├─ edge_cloud_irecv_tensor_dict_on_hidden_channel(channel=channel, ...)
        │   └─ _get_edge_cloud_hidden_channel_device_group(pp_group, channel)
        │       → 选通信组
        │   └─ torch.distributed.irecv(buf, src, group=group)
        │
        └─ TP 组内广播:
            └─ torch.distributed.broadcast(tensor, src=tp_rank_0, group=tp_group)
```

### 3.10 `_record_pp_send_work` 和 `_wait_pp_send_work`

**文件**: [worker.py:495-516](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/worker/worker.py#L495-L516)

```python
# L495-501
def _record_pp_send_work(self, handles, channel=None):
    if channel is None:
        self._pp_send_work.append(handles)
    else:
        self._pp_send_work_by_channel[channel.value] = handles  # L501

# L503-516
def _wait_pp_send_work(self, channel=None):
    if channel is None:
        # 等所有
        for handles in self._pp_send_work:
            for h in handles: h.wait()
        self._pp_send_work.clear()
        for handles in self._pp_send_work_by_channel.values():
            for h in handles: h.wait()
        self._pp_send_work_by_channel.clear()
    else:
        handles = self._pp_send_work_by_channel.pop(channel.value, [])  # L514
        for h in handles:
            h.wait()                                                      # L515
```

---

## SharedModel 路径

### 3.11 `_FirstRoundMarker.drive_head_send`

**文件**: [shared_model_edge_worker.py:586-605](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/worker/edge_cloud/shared_model_edge_worker.py#L586-L605)

```
drive_head_send()
├─ channel = self.worker._hidden_channel_for(scheduler_output)  # L595-596
├─ self.worker._wait_pp_send_work(channel)                      # L598
├─ handles = edge_cloud_isend_tensor_dict(
│       _gathered, dst=dp_rank + 1, num_tokens, channel=channel # L599-604
│   )
└─ self.worker._record_pp_send_work(handles, channel)           # L605
```

### 3.12 `_LastRoundMarker.do_direct_recv`

**文件**: [shared_model_edge_worker.py:622-653](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/worker/edge_cloud/shared_model_edge_worker.py#L622-L653)

```
do_direct_recv()
├─ channel = self.worker._hidden_channel_for(scheduler_output)  # L642-643
├─ self.worker._wait_pp_send_work(channel)                      # L645
└─ edge_cloud_broadcast_recv(num_tokens, src, channel)          # L647-652
```

---

## 完整调用链总图

```
进程启动
│
├─────────────────────────────────────────────────── 阶段一: 底层通信组 ───
│
├─ init_distributed_environment()
│   └─ init_model_parallel_group()
│       ├─ create_alternate_groups(backend)
│       │   └─ patch_distributed.py:263  → alt_device_group  (DECODE)
│       └─ create_hidden_channel_groups(backend)
│           └─ patch_distributed.py:307  → prefill2_device_group (PREFILL_2)
│                                          (PREFILL_1 复用 device_group)
│
├─────────────────────────────────────────────────── 阶段二: 逻辑通道池 ───
│
├─ PDSeparatedScheduler.__init__()                  # pd_separated_scheduler.py:137
│   └─ HiddenChannelManager()                       # L43
│       ├─ _free_prefills = [PREFILL_1, PREFILL_2]  # L44-47
│       ├─ _head_token_to_channel = {}              # L51
│       └─ decode_channel() → 固定 DECODE           # L82-84
│
├─────────────────────────────────────────────────── 阶段三: 请求级 ───
│  每个 step:
│
├─ schedule() → _pick_by_state(state)
│   │
│   ├─ PREFILL_FIRST:
│   │   └─ allocate_prefill(head_token)             # L411-414
│   │       └─ popleft() → PREFILL_1 或 PREFILL_2   # L63
│   │
│   ├─ DECODE_FIRST:
│   │   └─ decode_channel() → DECODE                # L572-573
│   │
│   ├─ scheduler_output.hidden_channel = channel
│   │
│   ├─ PREFILL_LAST / DECODE_LAST:
│   │   ├─ _validate_*_tail_channel()               # L470 / L503
│   │   └─ update_from_output()
│   │       └─ release_prefill(head_token)           # L641-643
│   │           └─ append() 归还                    # L73
│   │
│   └─ Worker.execute_model(scheduler_output)
│       ├─ _hidden_channel_for(scheduler_output)     # worker.py:598
│       │   └─ scheduler_output.hidden_channel        # L599
│       │
│       ├─ _wait_pp_send_work(channel)               # L571
│       │
│       ├─ HEAD: _execute_model_edge_head()
│       │   ├─ edge_cloud_send_tensor_dict(channel=channel)
│       │   │   └─ _get_edge_cloud_hidden_channel_device_group()
│       │   │       └─ _hidden_channel_groups(channel)
│       │   │           ├─ "prefill_1" → device_group
│       │   │           ├─ "decode"   → alt_device_group
│       │   │           └─ "prefill_2" → prefill2_device_group
│       │   │   └─ torch.distributed.isend(..., group=group)
│       │   └─ _record_pp_send_work(handles, channel)
│       │
│       ├─ TAIL: _execute_model_edge_tail()
│       │   └─ edge_cloud_broadcast_recv(channel=channel)
│       │       └─ edge_cloud_irecv_tensor_dict_on_hidden_channel(channel)
│       │           └─ _get_edge_cloud_hidden_channel_device_group()
│       │           └─ torch.distributed.irecv(..., group=group)
│       │       └─ torch.distributed.broadcast(..., group=tp_group)
│       │
│       └─ CLOUD: _execute_model_cloud()             # worker.py:720-805
│           ├─ edge_cloud_broadcast_recv(channel, src=0)  # L747-752
│           ├─ 模型前向
│           └─ edge_cloud_send_tensor_dict(channel)  # 回发 hidden states
│               └─ _record_pp_send_work(handles, channel)
│
└─────────────────────────────────────────────────── SharedModel 路径 ───
│
    worker_busy_loop()                                # shared_model_multiproc_executor.py:334
    │
    ├─ _process_first_markers()
    │   └─ _FirstRoundMarker.drive_head_send()        # shared_model_edge_worker.py:586
    │       ├─ _wait_pp_send_work(channel)            # L598
    │       ├─ edge_cloud_isend_tensor_dict(channel)  # L599-604
    │       └─ _record_pp_send_work(handles, channel) # L605
    │
    └─ _process_last_markers()
        └─ _LastRoundMarker.do_direct_recv()           # shared_model_edge_worker.py:622
            ├─ _wait_pp_send_work(channel)             # L645
            └─ edge_cloud_broadcast_recv(channel, src) # L647-652
```
