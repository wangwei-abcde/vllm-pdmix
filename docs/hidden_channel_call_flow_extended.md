# Hidden Channel DP 扩展后调用流程

> 基于 `hidden_channel_dp_extension.md` 方案，dp>1 且 edge_npu_count==1 时的完整调用链。

---

## 与原始代码的核心差异

| 维度 | 原始 (dp=1) | 扩展后 (dp=2 为例) |
|------|-------------|---------------------|
| 通信组数量 | 3 (默认+alt+prefill2) | 6 (默认+alt+prefill2+prefill3+prefill4+decode2) |
| prefill 通道总数 | 2 | dp×2 = 4 (全局) |
| decode 通道总数 | 1 | dp×1 = 2 (全局) |
| **每 dp_rank 的 prefill** | 2 (PREFILL_1, PREFILL_2) | 2 (dp0: PREFILL_1/2, dp1: PREFILL_3/4) |
| **每 dp_rank 的 decode** | 1 (DECODE 固定) | 1 (dp0: DECODE_1, dp1: DECODE_2) |
| 组存储方式 | 逐个命名属性 | 数组索引 |
| decode 分配 | 固定 `decode_channel()` | `allocate_decode()` / `release_decode()` 队列 |
| `prefill_inflight_limit` | 2 | 2 (不变，每 dp_rank 独立) |

---

## 总览：三大阶段

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  阶段一: 架构初始化              阶段二: 逻辑通道池         阶段三: 请求级     │
│  (每个进程独立)                 (每个 EngineCore_DP)       (每个 step)         │
│                                                                              │
│  create_alternate_groups()       HiddenChannelManager(dp_rank)                │
│  create_hidden_channel_groups()  dp0 → _free_prefills=[PREFILL_1, PREFILL_2]  │
│    num_prefill=dp×2              dp1 → _free_prefills=[PREFILL_3, PREFILL_4]  │
│    num_decode=dp×1               dp0 → _free_decodes =[DECODE_1]              │
│                                  dp1 → _free_decodes =[DECODE_2]              │
│  isend(group=_prefill_device_                                                  │
│    groups[idx])                  allocate_prefill / allocate_decode           │
│  irecv(group=_decode_device_     release_prefill / release_decode             │
│    groups[idx])                                                                │
│                                  _hidden_channel_for()                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 阶段一：架构初始化 — 通信组数组化创建

### 1.1 入口

```
vllm serve / EngineArgs
  └─ LLMEngine.__init__()
       └─ init_distributed_environment()
            └─ init_model_parallel_group()

文件: vllm_ascend/distributed/parallel_state.py
```

### 1.2 `init_model_parallel_group` 中的通道创建 (扩展后)

```python
# parallel_state.py (扩展后)

if dp_size > 1 and edge_npu_count == 1:
    num_prefill = dp_size * 2   # e.g. dp=2 → 4
    num_decode  = dp_size       # e.g. dp=2 → 2
else:
    num_prefill = 2             # legacy
    num_decode  = 1             # legacy

pp_group = get_pp_group()
if pp_group.world_size > 1:
    pp_group.create_alternate_groups(backend)
    if hasattr(pp_group, "create_hidden_channel_groups"):
        pp_group.create_hidden_channel_groups(backend, num_prefill, num_decode)
```

### 1.3 `create_hidden_channel_groups` (扩展后)

**文件**: [patch_distributed.py:307-341](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/patch/worker/patch_distributed.py#L307-L341) (需修改)

```
create_hidden_channel_groups(backend, num_prefill=4, num_decode=2)
│
├─ [Prefill 组]
│   ├─ _prefill_device_groups = [device_group]            # PREFILL_1 (复用默认)
│   ├─ _prefill_device_groups.append(prefill2_device_group) # PREFILL_2 (复用旧)
│   ├─ _create_one_hidden_group("pp_prefill3", ...)       # PREFILL_3 (新建)
│   └─ _create_one_hidden_group("pp_prefill4", ...)       # PREFILL_4 (新建)
│
└─ [Decode 组]
    ├─ _decode_device_groups = [alt_device_group]          # DECODE_1 (复用)
    └─ _create_one_hidden_group("pp_decode2", ...)        # DECODE_2 (新建)

_create_one_hidden_group(pg_name, dev_list, cpu_list, backend):
    hccl_pg_options = create_hccl_pg_options(pg_name)
    for ranks in self._all_group_ranks:
        torch.distributed.new_group(ranks, backend, pg_options=hccl_pg_options)
        torch.distributed.new_group(ranks, backend="gloo")
```

### 1.4 扩展后通信组结构

```
GroupCoordinator (patch_distributed.py)
│
├── device_group / cpu_group              → PREFILL_1 (默认)
├── alt_device_group / cpu_group          → DECODE_1
│
├── _prefill_device_groups = [
│       device_group,                     # idx 0 → PREFILL_1
│       prefill2_device_group,            # idx 1 → PREFILL_2
│       prefill3_device_group,            # idx 2 → PREFILL_3
│       prefill4_device_group,            # idx 3 → PREFILL_4
│   ]
│
└── _decode_device_groups = [
        alt_device_group,                 # idx 0 → DECODE_1
        decode2_device_group,             # idx 1 → DECODE_2
    ]
```

### 1.5 流隔离原理 (扩展后)

```
create_hccl_pg_options("pp_alt")       → HCCL stream A (DECODE_1)
create_hccl_pg_options("pp_prefill2")  → HCCL stream B (PREFILL_2)
create_hccl_pg_options("pp_prefill3")  → HCCL stream C (PREFILL_3)
create_hccl_pg_options("pp_prefill4")  → HCCL stream D (PREFILL_4)
create_hccl_pg_options("pp_decode2")   → HCCL stream E (DECODE_2)
PREFILL_1 使用默认 HCCL stream
```

---

## 阶段二：逻辑通道池 — 扩展后的 HiddenChannelManager

### 2.1 `HiddenChannelManager` (扩展后) — 按 dp_rank 切片

**核心设计**：每个 `EngineCore_DP` 独立创建自己的 `HiddenChannelManager`，只管理属于该 dp_rank 的通道切片，不包含其他 dp_rank 的通道。

**文件**: [pd_separated_scheduler.py:35-95](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/core/pd_separated_scheduler.py#L35-L95) (需修改)

```python
# 可配置常量，未来可扩展
_PREFILL_CHANNELS_PER_DP = 2
_DECODE_CHANNELS_PER_DP = 1

class HiddenChannelManager:
    def __init__(
        self,
        dp_rank: int = 0,
        prefill_per_dp: int = _PREFILL_CHANNELS_PER_DP,
        decode_per_dp: int = _DECODE_CHANNELS_PER_DP,
    ):
        # 根据 dp_rank 计算通道起始索引
        prefill_start = dp_rank * prefill_per_dp + 1
        decode_start = dp_rank * decode_per_dp + 1

        self._free_prefills: deque[HiddenChannelType] = deque(
            HiddenChannelType.prefill(i)
            for i in range(prefill_start, prefill_start + prefill_per_dp)
        )
        self._free_decodes: deque[HiddenChannelType] = deque(
            HiddenChannelType.decode(i)
            for i in range(decode_start, decode_start + decode_per_dp)
        )
        self._head_token_to_channel: dict[str, HiddenChannelType] = {}

    # allocate_prefill / release_prefill / allocate_decode / release_decode 不变

    @staticmethod
    def prefill_inflight_limit(prefill_per_dp: int = _PREFILL_CHANNELS_PER_DP) -> int:
        return prefill_per_dp

    @staticmethod
    def required_prefill_groups(dp_size: int) -> int:
        return dp_size * _PREFILL_CHANNELS_PER_DP

    @staticmethod
    def required_decode_groups(dp_size: int) -> int:
        return dp_size * _DECODE_CHANNELS_PER_DP
```

### 2.2 每 dp_rank 的池结构 (dp=2)

```
EngineCore_DP0                              EngineCore_DP1
HiddenChannelManager(dp_rank=0)             HiddenChannelManager(dp_rank=1)
                                           
_free_prefills:                            _free_prefills:
┌──────────────┐                           ┌──────────────┐
│  PREFILL_1   │  ← popleft()              │  PREFILL_3   │  ← popleft()
│  PREFILL_2   │                           │  PREFILL_4   │
└──────────────┘                           └──────────────┘
                                           
_free_decodes:                             _free_decodes:
┌──────────────┐                           ┌──────────────┐
│  DECODE_1    │  ← popleft()              │  DECODE_2    │  ← popleft()
└──────────────┘                           └──────────────┘
      
prefill_inflight_limit = 2                prefill_inflight_limit = 2
```

### 2.3 `PDSeparatedScheduler` 实例化 (扩展后)

```python
# pd_separated_scheduler.py (扩展后)
dp_rank = self.vllm_config.parallel_config.data_parallel_rank
self.prefill_inflight_limit = HiddenChannelManager.prefill_inflight_limit()
self.hidden_channel_manager = HiddenChannelManager(dp_rank=dp_rank)
```

---

## 阶段三：请求级通道选择 (扩展后)

### 3.1 PREFILL_FIRST 调度 (无变化，池独立但大小不变)

```
_pick_prefill_first_batch()
│
├─ 前置检查:
│   ├─ prefill_inflight_count < prefill_inflight_limit  # 2, 不变
│   └─ hidden_channel_manager.has_free_prefill()
│
├─ scheduler_output.hidden_channel =
│       hidden_channel_manager.allocate_prefill(head_token)
│       └─ dp0: popleft() → PREFILL_1 → PREFILL_2
│          dp1: popleft() → PREFILL_3 → PREFILL_4
│
└─ prefill_inflight_count += 1
```

### 3.2 DECODE_FIRST 调度 (改为 allocate_decode)

```
_pick_decode_first_batch()
│
├─ scheduler_output.batch_type = DECODE_FIRST
├─ scheduler_output.head_token = uuid4().hex
├─ scheduler_output.hidden_channel =
│       hidden_channel_manager.allocate_decode(head_token)  # 新增!
│       └─ dp0: popleft() → DECODE_1
│          dp1: popleft() → DECODE_2
│
└─ decode_inflight_count += 1
```

### 3.3 通道分配时序示例 (dp=2, 两个 EngineCore_DP 各自独立调度)

```
EngineCore_DP0:                              EngineCore_DP1:

Step 1: 调度 P首 req_A                      Step 1: 调度 P首 req_X
  allocate_prefill("A") → PREFILL_1           allocate_prefill("X") → PREFILL_3
  _free_prefills: [PREFILL_2]                 _free_prefills: [PREFILL_4]

Step 2: 调度 D首 req_B                      Step 2: 调度 D首 req_Y
  allocate_decode("B")  → DECODE_1            allocate_decode("Y")  → DECODE_2
  _free_decodes: []                           _free_decodes: []

Step 3: 调度 P首 req_C (无空闲 prefill!)     Step 3: 调度 P首 req_Z (无空闲 prefill!)
  → blocked                                   → blocked

Step 4: 完成 req_A P尾                       Step 4: 完成 req_X P尾
  release_prefill("A") → 归还 PREFILL_1       release_prefill("X") → 归还 PREFILL_3
  _free_prefills: [PREFILL_1]                 _free_prefills: [PREFILL_4]

Step 5: 完成 req_B D尾
  release_decode("B") → 归还 DECODE_1
  _free_decodes: [DECODE_1]
```

### 3.4 PREFILL_LAST (校验使用本地池，非全局集合)

```
_prefill_last_batch()
│
├─ _validate_prefill_tail_channel(scheduler_output)
│   └─ channel 必须 ∈ {本 dp_rank 的 prefill 通道集合}
│       dp0: {PREFILL_1, PREFILL_2}
│       dp1: {PREFILL_3, PREFILL_4}
│
└─ update_from_output()
    ├─ prefill_inflight_count -= 1
    └─ release_prefill(head_token)
        └─ append() 归还到右侧
```

### 3.5 DECODE_LAST (校验使用本地池)

```
_pick_decode_last_batch()
│
├─ _validate_decode_tail_channel(scheduler_output)
│   └─ channel 必须 ∈ {本 dp_rank 的 decode 通道集合}
│       dp0: {DECODE_1}
│       dp1: {DECODE_2}
│
└─ update_from_output()
    └─ release_decode(head_token)  # 新增!
        └─ append() 归还 DECODE_i
```

---

## Worker 端：通道解析 → 通信执行 (扩展后)

### 3.6 `_hidden_channel_for` (无变化)

```python
def _hidden_channel_for(self, scheduler_output):
    channel = scheduler_output.hidden_channel  # 已有 scheduler 设的值
    if channel is not None:
        return channel                         # PREFILL_3, DECODE_2 等新通道直接走这里
    # fallback (legacy)
    bt = scheduler_output.batch_type
    ...
```

由于 scheduler 端已经设好了 `hidden_channel`（包括新的 PREFILL_3, DECODE_2 等），Worker 端无需改动。

### 3.7 `_hidden_channel_groups` — Group 映射 (数组化)

**文件**: [patch_distributed.py:343-355](file:///Users/wangwei/one_card_dense/vllm-ascend/vllm_ascend/patch/worker/patch_distributed.py#L343-L355) (需修改)

```
_hidden_channel_groups(channel)
│
├─ value = channel.value
├─ if value.startswith("prefill_"):
│   └─ idx = int(value.split("_")[1]) - 1
│       → _prefill_device_groups[idx], _prefill_cpu_groups[idx]
│         "prefill_1" → idx 0 → device_group
│         "prefill_2" → idx 1 → prefill2_device_group
│         "prefill_3" → idx 2 → prefill3_device_group
│         "prefill_4" → idx 3 → prefill4_device_group
│
└─ if value.startswith("decode_"):
    └─ idx = int(value.split("_")[1]) - 1
        → _decode_device_groups[idx], _decode_cpu_groups[idx]
          "decode_1" → idx 0 → alt_device_group
          "decode_2" → idx 1 → decode2_device_group
```

### 3.8 HEAD (isend) 路径 (下游无变化)

```
_execute_model_edge_head(scheduler_output)
│
├─ channel = _hidden_channel_for(scheduler_output)    # 返回 e.g. PREFILL_3
│
├─ edge_cloud_send_tensor_dict(..., channel=channel)
│   └─ _get_edge_cloud_hidden_channel_device_group(pp_group, channel)
│       └─ _hidden_channel_groups(channel)
│           └─ "prefill_3" → _prefill_device_groups[2]
│               → torch.distributed.isend(..., group=prefill3_device_group)
│
└─ _record_pp_send_work(handles, channel)
```

### 3.9 TAIL (recv) 路径 (下游无变化)

```
_execute_model_edge_tail(scheduler_output)
│
├─ channel = _hidden_channel_for(scheduler_output)    # 返回 e.g. DECODE_2
│
├─ edge_cloud_broadcast_recv(..., channel=channel)
│   └─ _get_edge_cloud_hidden_channel_device_group(pp_group, channel)
│       └─ _hidden_channel_groups(channel)
│           └─ "decode_2" → _decode_device_groups[1]
│               → torch.distributed.irecv(..., group=decode2_device_group)
│
└─ TP broadcast 不变
```

---

## SharedModel 路径 (扩展后)

### 3.10 `_FirstRoundMarker.drive_head_send` (无变化)

```
drive_head_send()
├─ channel = self.worker._hidden_channel_for(scheduler_output)  # e.g. PREFILL_3
├─ self.worker._wait_pp_send_work(channel)
├─ edge_cloud_isend_tensor_dict(..., channel=channel)
│   └─ group = _prefill_device_groups[2]  (via _hidden_channel_groups)
└─ self.worker._record_pp_send_work(handles, channel)
```

### 3.11 `_LastRoundMarker.do_direct_recv` (无变化)

```
do_direct_recv()
├─ channel = self.worker._hidden_channel_for(scheduler_output)  # e.g. DECODE_2
├─ self.worker._wait_pp_send_work(channel)
└─ edge_cloud_broadcast_recv(..., channel=channel)
    └─ group = _decode_device_groups[1]  (via _hidden_channel_groups)
```

---

## 改动文件汇总

| 文件 | 改动 | 影响范围 |
|------|------|----------|
| `vllm/v1/core/sched/output.py` | 枚举扩展 + `prefill(i)` / `decode(i)` 工厂方法 | 通道定义 |
| `vllm_ascend/patch/worker/patch_distributed.py` | 数组化属性 + `create_hidden_channel_groups` 参数化 + `_hidden_channel_groups` 索引化 | 通信组创建 |
| `vllm_ascend/distributed/parallel_state.py` | `init_model_parallel_group` 传参 `num_prefill`/`num_decode` | 初始化入口 |
| `vllm_ascend/core/pd_separated_scheduler.py` | `HiddenChannelManager(dp_rank)` 按切片 + `allocate_decode`/`release_decode` | 逻辑通道池 |
| `vllm_ascend/scheduler_conflicts.py` | 更新 `required_channels` 范围 | 兼容性校验 |

**Worker 端无需改动** — `_hidden_channel_for`、`_execute_model_edge_head`、`_execute_model_edge_tail`、`drive_head_send`、`do_direct_recv` 等接口完全不变，因为 `channel` 参数和 `_hidden_channel_groups(channel)` 自动适配了新通道。

---

## 完整调用链总图 (dp=2, edge_npu_count=1)

```
每个 EngineCore_DP 进程独立执行以下流程:

进程启动
│
├─────────────────────────────────────── 阶段一: 底层通信组 (数组化，所有进程一起) ───
│
├─ init_distributed_environment()
│   └─ init_model_parallel_group(dp_size=2)
│       ├─ create_alternate_groups(backend)
│       │   └─ → alt_device_group  (DECODE_1, 所有 dp 共用)
│       └─ create_hidden_channel_groups(backend, num_prefill=4, num_decode=2)
│           ├─ _prefill_device_groups[0] = device_group            (PREFILL_1)
│           ├─ _prefill_device_groups[1] = prefill2_device_group   (PREFILL_2)
│           ├─ _prefill_device_groups[2] = 新建 pg_options="pp_prefill3"
│           ├─ _prefill_device_groups[3] = 新建 pg_options="pp_prefill4"
│           ├─ _decode_device_groups[0]  = alt_device_group         (DECODE_1)
│           └─ _decode_device_groups[1]  = 新建 pg_options="pp_decode2"
│           ↑ 所有 EngineCore_DP 进程都建了全量 group
│             (因为 new_group 是 collective，必须所有进程参与)
│
├─────────────────────────────────────── 阶段二: 逻辑通道池 (各进程独立切片) ───
│
├─ EngineCore_DP0:
│   └─ HiddenChannelManager(dp_rank=0)
│       ├─ _free_prefills = [PREFILL_1, PREFILL_2]  ← 只取前 2 个
│       └─ _free_decodes  = [DECODE_1]
│
├─ EngineCore_DP1:
│   └─ HiddenChannelManager(dp_rank=1)
│       ├─ _free_prefills = [PREFILL_3, PREFILL_4]  ← 只取后 2 个
│       └─ _free_decodes  = [DECODE_2]
│
├─────────────────────────────────────── 阶段三: 请求级 (各进程独立调度) ───
│  每个 step:
│
├─ schedule() → _pick_by_state(state)
│   │
│   ├─ PREFILL_FIRST: allocate_prefill(head_token) → PREFILL_i (本 dp 切片内)
│   ├─ DECODE_FIRST:  allocate_decode(head_token)  → DECODE_j  (本 dp 切片内)
│   │
│   ├─ scheduler_output.hidden_channel = channel
│   │
│   ├─ PREFILL_LAST → release_prefill → append 归还
│   ├─ DECODE_LAST  → release_decode  → append 归还
│   │
│   └─ Worker.execute_model(scheduler_output)        ← 无需改动
│       ├─ _hidden_channel_for() → channel
│       ├─ HEAD: edge_cloud_send_tensor_dict(channel)
│       │   └─ _hidden_channel_groups("prefill_3")
│       │       → _prefill_device_groups[2] → isend (dp1 用 PREFILL_3)
│       │
│       └─ TAIL: edge_cloud_broadcast_recv(channel)
│           └─ _hidden_channel_groups("decode_2")
│               → _decode_device_groups[1] → irecv (dp1 用 DECODE_2)
│
└─────────────────────────────────────── SharedModel 路径 ───
│
    worker_busy_loop()
    │
    ├─ _process_first_markers()
    │   └─ drive_head_send()
    │       ├─ channel = _hidden_channel_for()          # 自动适配
    │       ├─ _wait_pp_send_work(channel)
    │       ├─ edge_cloud_isend_tensor_dict(channel)    # 自动选组
    │       └─ _record_pp_send_work(handles, channel)
    │
    └─ _process_last_markers()
        └─ do_direct_recv()
            ├─ channel = _hidden_channel_for()          # 自动适配
            ├─ _wait_pp_send_work(channel)
            └─ edge_cloud_broadcast_recv(channel, src)
```

---

## 回退兼容

- **dp_rank=0 时**: `prefill_start=1, decode_start=1`，行为和原始代码完全一致（`[PREFILL_1, PREFILL_2]`, `[DECODE_1]`）
- **dp_size=1 时**: `num_prefill=2, num_decode=1`，只创建原始需要的 group，不创建 PREFILL_3+ / DECODE_2+
- `DECODE` 别名保留为 `DECODE_1`，所有旧代码无需修改
- `_hidden_channel_groups` 的 `"prefill_1"` / `"prefill_2"` / `"decode"` 映射保留，回退路径不变
- `prefill_inflight_limit` 保持 2，每 dp_rank 的调度行为不变
