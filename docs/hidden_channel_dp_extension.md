# Hidden Channel DP 扩展方案

## 背景

在 edge-cloud PD 分离模式下，存在三种类型的 `HiddenChannelType` 通道用于 hidden states 的边云传输：

| 通道 | 通信组 | 用途 |
|------|--------|------|
| `PREFILL_1` | `device_group` (默认) | 第 1 路 prefill |
| `DECODE` (= `DECODE_1`) | `alt_device_group` | decode |
| `PREFILL_2` | `prefill2_device_group` | 第 2 路 prefill |

每个通道通过独立的 HCCL `new_group` + `pg_options` 实现流隔离，避免不同请求间的 HCCL 流串行化。

当前设计仅支持 **dp=1** 场景 (prefill 2P, decode 1D)。当 `--data-parallel-size > 1` 且 `--edge-npu-count == 1` 时，需要扩展通道数量：

- **prefill 通道**: `dp × 2` 个
- **decode 通道**: `dp × 1` 个 (新增)

## 通道与 `prefill_inflight_count` 的关系

通道分配和 `prefill_inflight_count` 是一一对应的原子操作：

```
调度 P首 → prefill_inflight_count += 1
         → HiddenChannelManager.allocate_prefill(head_token)

完成 P尾 → prefill_inflight_count -= 1
         → HiddenChannelManager.release_prefill(head_token)
```

因此 `prefill_inflight_count` 即"在用的通道数"，扩展后 `prefill_inflight_limit = dp × 2`。

---

## 方案设计

### 核心思路：数组化

PREFILL_2 的新增模式是逐个命名的 (`self.prefill2_device_group`)，扩展到 N 个时会爆炸。改用**数组索引**：

```
PREFILL_1 → prefill_device_groups[0]  (复用 device_group)
PREFILL_2 → prefill_device_groups[1]  (复用 prefill2_device_group)
PREFILL_3 → prefill_device_groups[2]  (新建 prefill3_device_group)
PREFILL_N → prefill_device_groups[N-1]

DECODE_1 → decode_device_groups[0]   (复用 alt_device_group)
DECODE_2 → decode_device_groups[1]   (新建 decode2_device_group)
DECODE_M → decode_device_groups[M-1]
```

### 改动总览

| 文件 | 改动内容 |
|------|----------|
| `vllm/v1/core/sched/output.py` | 枚举扩展到 PREFILL_1..16 + DECODE_1..8 + 工厂方法 |
| `patch_distributed.py` | 数组化 group 属性 + `create_hidden_channel_groups` 参数化 + `_hidden_channel_groups` 索引化 |
| `parallel_state.py` | 初始化传参 dp_size/edge_npu_count + `_get_edge_cloud_hidden_channel_device_group` 更新 |
| `pd_separated_scheduler.py` | `HiddenChannelManager(dp_rank)` 按切片 + `allocate_decode`/`release_decode` |
| `scheduler_conflicts.py` | 更新 `required_channels` 检查 |

---

## 详细改动

### 第 1 层：枚举扩展

**文件**: `vllm/v1/core/sched/output.py`

```python
class HiddenChannelType(enum.Enum):
    """Data-plane hidden tensor channel for edge-cloud PD separation."""
    # Prefill channels (dp × 2)
    PREFILL_1  = "prefill_1"
    PREFILL_2  = "prefill_2"
    PREFILL_3  = "prefill_3"
    PREFILL_4  = "prefill_4"
    PREFILL_5  = "prefill_5"
    PREFILL_6  = "prefill_6"
    PREFILL_7  = "prefill_7"
    PREFILL_8  = "prefill_8"
    PREFILL_9  = "prefill_9"
    PREFILL_10 = "prefill_10"
    PREFILL_11 = "prefill_11"
    PREFILL_12 = "prefill_12"
    PREFILL_13 = "prefill_13"
    PREFILL_14 = "prefill_14"
    PREFILL_15 = "prefill_15"
    PREFILL_16 = "prefill_16"
    # Decode channels (dp × 1)
    DECODE_1 = "decode_1"
    DECODE_2 = "decode_2"
    DECODE_3 = "decode_3"
    DECODE_4 = "decode_4"
    DECODE_5 = "decode_5"
    DECODE_6 = "decode_6"
    DECODE_7 = "decode_7"
    DECODE_8 = "decode_8"
    # Legacy alias
    DECODE = DECODE_1

    @staticmethod
    def prefill(i: int) -> "HiddenChannelType":
        """Returns PREFILL_i, 1-indexed."""
        return HiddenChannelType[f"PREFILL_{i}"]

    @staticmethod
    def decode(i: int) -> "HiddenChannelType":
        """Returns DECODE_i, 1-indexed."""
        return HiddenChannelType[f"DECODE_{i}"]
```

---

### 第 2 层：Group 属性数组化

**文件**: `vllm_ascend/patch/worker/patch_distributed.py`

将原先的单个属性改为数组：

```python
# --- 旧 ---
self.prefill2_device_group: torch.distributed.ProcessGroup | None = None
self.prefill2_cpu_group: torch.distributed.ProcessGroup | None = None

# --- 新 ---
# idx 0 = PREFILL_1 (默认 device_group), idx 1 = PREFILL_2, ...
self._prefill_device_groups: list[torch.distributed.ProcessGroup] = []
self._prefill_cpu_groups: list[torch.distributed.ProcessGroup] = []

# idx 0 = DECODE_1 (alt_device_group), idx 1 = DECODE_2, ...
self._decode_device_groups: list[torch.distributed.ProcessGroup] = []
self._decode_cpu_groups: list[torch.distributed.ProcessGroup] = []
```

---

### 第 3 层：Group 创建参数化

**文件**: `vllm_ascend/patch/worker/patch_distributed.py`

将 `create_hidden_channel_groups` 改为接收 `num_prefill` 和 `num_decode`：

```python
def create_hidden_channel_groups(
    self,
    torch_distributed_backend: str | Backend,
    num_prefill: int,   # dp × 2
    num_decode: int,     # dp × 1
) -> None:
    """Create hidden data-plane channel groups.

    PREFILL_1 reuses default device_group.
    PREFILL_2 reuses legacy prefill2_device_group.
    PREFILL_3+ are created by iterating _all_group_ranks.
    DECODE_1 reuses legacy alt_device_group.
    DECODE_2+ are created analogously.
    """
    # --- Prefill ---
    self._prefill_device_groups = [self.device_group]   # PREFILL_1
    self._prefill_cpu_groups = [self.cpu_group]

    if num_prefill >= 2:
        if self.prefill2_device_group is not None:
            self._prefill_device_groups.append(self.prefill2_device_group)
            self._prefill_cpu_groups.append(self.prefill2_cpu_group)
        else:
            self._create_one_hidden_group(
                "pp_prefill2", self._prefill_device_groups,
                self._prefill_cpu_groups, torch_distributed_backend,
            )

    for i in range(3, num_prefill + 1):
        self._create_one_hidden_group(
            f"pp_prefill{i}", self._prefill_device_groups,
            self._prefill_cpu_groups, torch_distributed_backend,
        )

    # --- Decode ---
    assert self.alt_device_group is not None
    self._decode_device_groups = [self.alt_device_group]   # DECODE_1
    self._decode_cpu_groups = [self.alt_cpu_group]

    for i in range(2, num_decode + 1):
        self._create_one_hidden_group(
            f"pp_decode{i}", self._decode_device_groups,
            self._decode_cpu_groups, torch_distributed_backend,
        )

    logger.info(
        "[PP Group] Hidden channels created: prefill=%d, decode=%d",
        num_prefill, num_decode,
    )


def _create_one_hidden_group(
    self,
    pg_name: str,
    dev_list: list,
    cpu_list: list,
    backend: str | Backend,
) -> None:
    """Create one device/cpu group pair identified by pg_name."""
    hccl_pg_options = create_hccl_pg_options(pg_name)
    device_group = None
    cpu_group = None
    for ranks in self._all_group_ranks:
        dev_g = torch.distributed.new_group(
            ranks, backend=backend, pg_options=hccl_pg_options,
        )
        cpu_g = torch.distributed.new_group(ranks, backend="gloo")
        if self.rank in ranks:
            device_group = dev_g
            cpu_group = cpu_g
    assert device_group is not None
    assert cpu_group is not None
    dev_list.append(device_group)
    cpu_list.append(cpu_group)
```

**关键**：每个 `new_group` 传不同的 `pg_options("pp_prefill3")`、`"pp_prefill4"` … 和 `"pp_decode2"`、`"pp_decode3"` … 确保 HCCL 流隔离。

---

### 第 4 层：Group 映射索引化

**文件**: `vllm_ascend/patch/worker/patch_distributed.py`

```python
def _hidden_channel_groups(self, channel: Any):
    value = getattr(channel, "value", channel)
    if value.startswith("prefill_"):
        idx = int(value.split("_")[1]) - 1   # "prefill_3" → 2
        return self._prefill_device_groups[idx], self._prefill_cpu_groups[idx]
    if value.startswith("decode_"):
        idx = int(value.split("_")[1]) - 1   # "decode_2" → 1
        return self._decode_device_groups[idx], self._decode_cpu_groups[idx]
    raise ValueError(f"Unknown hidden channel: {channel}")
```

---

### 第 5 层：Group 解析更新

**文件**: `vllm_ascend/distributed/parallel_state.py`

`_get_edge_cloud_hidden_channel_device_group` 的 fallback 路径扩展：

```python
def _get_edge_cloud_hidden_channel_device_group(
    pp_group: GroupCoordinator,
    channel: HiddenChannelType | None = None,
    use_alt_group: bool = False,
):
    if channel is not None:
        if hasattr(pp_group, "_hidden_channel_groups"):
            device_group, _ = pp_group._hidden_channel_groups(channel)
            return device_group
        # Fallback: legacy single-group path (dp=1)
        if channel in (HiddenChannelType.PREFILL_1,):
            return pp_group.device_group
        if channel in (
            HiddenChannelType.DECODE_1,
            HiddenChannelType.DECODE,
        ):
            assert pp_group.alt_device_group is not None
            return pp_group.alt_device_group
        raise RuntimeError(
            f"Channel {channel} requires create_hidden_channel_groups(). "
            f"Call with num_prefill and num_decode parameters."
        )
    # ...
```

初始化入口传参：

```python
# parallel_state.py init_model_parallel_group

if dp_size > 1 and edge_npu_count == 1:
    num_prefill = dp_size * 2
    num_decode = dp_size
else:
    num_prefill = 2   # legacy: PREFILL_1, PREFILL_2
    num_decode = 1    # legacy: DECODE

if hasattr(pp_group, "create_hidden_channel_groups"):
    pp_group.create_hidden_channel_groups(backend, num_prefill, num_decode)
```

---

### 第 6 层：调度管理扩展

**文件**: `vllm_ascend/core/pd_separated_scheduler.py`

#### 6a. HiddenChannelManager 扩展 — 按 dp_rank 切片

**核心设计**: 每个 `EngineCore_DP` 独立创建 `HiddenChannelManager`，只管理该 dp_rank 的通道切片。

```python
# 可配置常量
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

    # --- Prefill (已有) ---
    def allocate_prefill(self, head_token: str) -> HiddenChannelType:
        channel = self._free_prefills.popleft()
        self._head_token_to_channel[head_token] = channel
        return channel

    def release_prefill(self, head_token: str) -> HiddenChannelType | None:
        channel = self._head_token_to_channel.pop(head_token, None)
        if channel is not None:
            self._free_prefills.append(channel)
        return channel

    # --- Decode (新增) ---
    def allocate_decode(self, head_token: str) -> HiddenChannelType:
        channel = self._free_decodes.popleft()
        self._head_token_to_channel[head_token] = channel
        return channel

    def release_decode(self, head_token: str) -> HiddenChannelType | None:
        channel = self._head_token_to_channel.pop(head_token, None)
        if channel is not None:
            self._free_decodes.append(channel)
        return channel

    # --- 工具方法 ---
    @staticmethod
    def prefill_inflight_limit() -> int:
        return _PREFILL_CHANNELS_PER_DP

    @staticmethod
    def required_prefill_groups(dp_size: int) -> int:
        return dp_size * _PREFILL_CHANNELS_PER_DP

    @staticmethod
    def required_decode_groups(dp_size: int) -> int:
        return dp_size * _DECODE_CHANNELS_PER_DP
```

**切片示例 (dp_size=2)**:

```
EngineCore_DP0:  prefill_start=0×2+1=1 → [PREFILL_1, PREFILL_2]
                 decode_start =0×1+1=1 → [DECODE_1]

EngineCore_DP1:  prefill_start=1×2+1=3 → [PREFILL_3, PREFILL_4]
                 decode_start =1×1+1=2 → [DECODE_2]
```

#### 6b. 调度处调用

- **DECODE_FIRST 调度**: `_pick_by_state` 中，DECODE_FIRST batch 完成后调用 `allocate_decode(head_token)`，设置 `scheduler_output.hidden_channel`
- **DECODE_LAST 完成**: `update_from_output` 中，DECODE_LAST 时调用 `release_decode(head_token)`

#### 6c. `_hidden_channel_for` 更新

```python
def _hidden_channel_for(self, scheduler_output):
    channel = scheduler_output.hidden_channel
    if channel is not None:
        return channel
    # Fallback (legacy)
    bt = scheduler_output.batch_type
    if bt in (PREFILL_FIRST, PREFILL_LAST):
        return HiddenChannelType.PREFILL_1
    if bt in (DECODE_FIRST, DECODE_LAST):
        return HiddenChannelType.DECODE
```

---

### 第 7 层：兼容性检查

**文件**: `vllm_ascend/scheduler_conflicts.py`

```python
required_channels = tuple(
    HiddenChannelType.prefill(i)
    for i in range(1, dp_size * 2 + 1)
) + tuple(
    HiddenChannelType.decode(i)
    for i in range(1, dp_size + 1)
)
```

---

## 生命周期流程

```
[初始化]
init_distributed_environment(dp_size=2, edge_npu_count=1)
  ├─ create_alternate_groups(backend)              → alt_device_group (DECODE_1)
  └─ create_hidden_channel_groups(backend, 4, 2)
       ├─ PREFILL_1  → device_group (默认)
       ├─ PREFILL_2  → prefill2_device_group (pg_options="pp_prefill2")
       ├─ PREFILL_3  → prefill3_device_group (pg_options="pp_prefill3")
       ├─ PREFILL_4  → prefill4_device_group (pg_options="pp_prefill4")
       ├─ DECODE_1   → alt_device_group (复用)
       └─ DECODE_2   → decode2_device_group (pg_options="pp_decode2")

[Scheduler]
self.hidden_channel_manager = HiddenChannelManager(dp_rank=dp_rank)
  dp_rank=0 → _free_prefills=[PREFILL_1, PREFILL_2], _free_decodes=[DECODE_1]
  dp_rank=1 → _free_prefills=[PREFILL_3, PREFILL_4], _free_decodes=[DECODE_2]

[调度 P首 (dp_rank=0)]
allocate_prefill(head_token) → PREFILL_1 (popleft)
scheduler_output.hidden_channel = PREFILL_1

[调度 D首 (dp_rank=0)]
allocate_decode(head_token) → DECODE_1 (popleft)
scheduler_output.hidden_channel = DECODE_1

[完成 P尾/ D尾]
release_prefill / release_decode → channel 归还到右侧

[Worker]
_hidden_channel_for(scheduler_output) → channel
  └─ 用于 isend/irecv/broadcast 的 group 选择
```

---

## 回退兼容

- `dp_size == 1`: `num_prefill=2, num_decode=1`，和现有行为完全一致
- `DECODE` 别名保留为 `DECODE_1`，所有旧代码无需修改
- `_hidden_channel_for` 的 fallback 逻辑保留，旧 scheduler_output (无 hidden_channel) 仍可用
