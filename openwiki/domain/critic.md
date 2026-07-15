# SceneBenchmark Critic 系统

> **核心包**：`scenesmith/scenebenchmark_critic/` (~20 个模块)  
> **内嵌 README**：`scenesmith/scenebenchmark_critic/README.md`（中文，含详细用法）

SceneSmith 内嵌了 [SceneBenchmark](https://github.com/) 的规则评估器，为生成中的场景提供**零 VLM、确定性几何反馈**。Critic 可在每个 Pipeline 阶段后运行，结果反馈给 Agent 的下一轮 Planner 用于自我修正。

## 整体架构

```
RoomScene
    │
    ▼
┌────────────────────────────────────┐
│  annotate_room_scene()             │ ← 可选 VLM 资产标注
│   → 写入 functional_hints 元数据    │
└────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────┐
│  room_scene_to_case_pack()         │ ← Adapter: SceneSmith → SceneBenchmark 格式
│  适配器: 推断类别/功能/支撑面/关系   │
└────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────┐
│  stabilize_orientation_contracts() │ ← 正面/访问方向标准化
└────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  ┌──────────────────────┐  ┌──────────────────────────────┐ │
│  │ Vendored Rules       │  │ SceneSmith 自有扩展检查        │ │
│  │ (vendor/scenebenchmark/) │                              │ │
│  │  • spatial_accessibility │  • media_support_alignment   │ │
│  │  • functional_dependency│  • room_center_alignment     │ │
│  │  • interaction_clearance│  • dining_seat_distribution  │ │
│  │                      │  • manipuland_completeness    │ │
│  │                      │  • dining_place_setting_aln   │ │
│  └──────────────────────┘  └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────┐
│  build_evaluation_payload()        │ → JSON + Markdown 报告
│  write_report()                    │
└────────────────────────────────────┘
```

## 评估指标

### 1. spatial_accessibility（空间可达性）

- **基于网格的可达区域评估**：计算可站立区域、障碍物掩码
- **可达距离**：测量物体到最近可达点的距离
- **功能区连通性**：检查主要功能区是否连通

### 2. functional_dependency（功能依赖）

基于模板提议器和规则评分的依赖关系检查：

| 依赖类型 | 检查内容 |
|----------|----------|
| 物体-支撑物 | 物体是否位于正确支撑面上 |
| 灯-表面 | 台灯应在桌/柜上，吊灯在天花板 |
| 座椅-工作面 | 椅子应面向桌子/工作台 |
| 座椅-媒体 | 客厅座椅应面向电视/壁炉 |
| 床-床头柜 | 床两侧应有床头柜 |
| 餐桌-座椅 | 座椅围绕餐桌均匀分布 |
| 壁挂 TV-电视柜 | TV 应在电视柜上方（2026-07-14 新增） |
| 房间中心 | prompt 要求"中心"的家具应在几何中心（2026-07-15 新增） |

### 3. interaction_clearance（交互净空）— 默认关闭

基于预计算人体锚定净空标注的确定性检查：

- **静态净空**：6092 件 HSSD 资产的 keep-clear 区域（5 种类型：落座/接近/操作/上方站立/通行）
- **动态净空**：2120 件铰接资产的扫掠体开合包络
- 按 `asset_id` 查表，将局部系净空区投影到世界系
- 检测其他物体是否侵入操作空间

启用方式：
```yaml
scenebenchmark_critic:
  enabled: true
  metrics: [spatial_accessibility, functional_dependency, interaction_clearance]
```

## SceneSmith 扩展检查

这些检查在 vendored 规则之上附加执行，弥补模板规则的盲区：

### media_support_alignment
**文件**: `media_support_alignment.py` (14KB) | Commit: `ff9dc38`

- 确保壁挂 TV 与电视柜保持垂直对齐
- 防止 clearance 修复将 TV 从 TV stand 上方推离
- 支持多表面匹配

### room_center_alignment
**文件**: `room_center_alignment.py` (12KB) | Commit: `ff9dc38`

- 检查 prompt 明确要求"房间中心"的家具是否在几何中心附近
- 忽略"桌子中心"、"沙发和电视之间"等相对中心语义
- 使用正则匹配 prompt 中的 `center`/`centre`/`middle` 等关键词

### dining_seat_distribution
**文件**: `dining_seat_distribution.py` (8KB) | Commit: `87b79b9`

- 检查餐桌旁座椅在桌子局部坐标系中的分布
- 检测同边多椅挤在一侧或单椅偏离桌边中心
- 按桌局部坐标计算通用座椅分布

### dining_place_setting_alignment
**文件**: `dining_place_setting_alignment.py` (25KB)

| Commit | 变更内容 |
|--------|----------|
| 87b79b9 | 初始化：餐位与座椅一对一对应 |
| 6c69c15 | 增强对齐逻辑 + 新测试 |
| **30da02f** | **多表面支持 + 对齐工具** |

最新功能（30da02f）：
- **多表面支持**：餐桌可能有多个 support surface，算法可匹配多个 surface 上的物品
- **对齐工具**：`_recommended_anchor_center()` 计算推荐位置，`_seat_lane_assignment()` 做车道分配
- **纵向槽容忍**：`_anchor_longitudinal_tolerance()` 沿座椅前后方向
- **横向容忍**：`_anchor_centerline_tolerance()` 沿座椅左右方向
- 支持对 companion（配套餐具）做成组验证

基本逻辑：
1. 只有离散座位（非长凳）且餐位锚点数≥2 且与座位数一致时才激活
2. 每把椅子分配一条"前方车道"
3. 检查每个餐盘的落点是否在对应车道的容忍范围内
4. 报告具体的横向/纵向偏移诊断信息

### manipuland_completeness
**文件**: `manipuland_completeness.py` (13KB) | Commit: `81a68b7`

- 检查餐桌等成组 tabletop manipulands 是否在物理后处理中被误删
- 使用正则匹配 prompt 中的用餐物品关键词（plate, bowl, drinkware, fork, knife 等）
- 支持计数词解析（"two", "three", "four" 等）
- 通用餐具语义（cutlery, flatware, silverware 等）自动展开
- 2026-07-13 更新：稳定通用餐具语义

### orientation_contracts（方向契约）
**文件**: `orientation_contracts.py` (24KB)

关键演进：

| Commit | 变更内容 |
|--------|----------|
| b73aabe | Stabilize wall seating contracts（墙边座椅契约稳定化） |
| 16b9481 | Scale-relative wall seating（按资产尺寸缩放墙边判定） |
| 6d64828 | Stabilize standalone chair critic |

- `WALL_ANCHOR_GAP_RATIO = 0.45`：墙边独立座椅判定随资产尺寸缩放
- 检测 `seating_to_media`、`seating_to_work_surface`、`back_against_wall` 等关系
- 自动注入 `_scenebenchmark_orientation_contracts` 元数据到 scene objects

## 适配器（Adapter）

**文件**: `adapter.py` (~58KB)

`room_scene_to_case_pack()` 是核心转换函数：

- 从 `name`、`description`、`metadata`、`placement_info` 推导类别
- 映射 `support_region_summary`（支撑面摘要）
- 传递 `front_face`、`access_direction`、`operation_space` 等标注字段
- `benchmark_relevance` 字段抑制非功能物体检查
- 支持多源功能识别：`affordances`、`functional_categories`、`candidate_affordances`
- 2026-07-07 更新：将 BlenderServer 前向传递至资产标注，支持渲染证据图

## Vendored 规则

**目录**: `vendor/scenebenchmark/`

SceneBenchmark 核心规则代码 vendored 进本仓库，避免运行时依赖：

```
vendor/scenebenchmark/metrics/
├── functional_dependency/
│   ├── relations.py          # FD 模板匹配与评分
│   ├── semantics.py          # 语义推断
│   ├── support.py            # 支撑关系评分
│   └── profiles.py           # 功能画像
├── interaction_clearance/
│   └── ...                   # 净空检查实现
└── spatial_accessibility/
    ├── core.py               # 核心可达性算法
    ├── obstacles.py          # 障碍物处理
    └── zones.py              # 功能区定义
```

### FD 提议器模式

| 模式 | 说明 | 使用场景 |
|------|------|----------|
| `template` | 确定性模板匹配（默认） | 内存安全、无 VLM 依赖 |
| `vlm` | VLM 驱动的功能关系提议 | 需要 SceneBenchmark VLM 栈 |
| `hybrid` | 混合模式 | 自动选择最优 |
| `auto` | 自动检测可用后端 | 渐进式回退 |

## 报告输出

**文件**: `reports.py` (5KB)、`prompt_context.py` (32KB)

评估结果以两种格式输出：

- **JSON**（`scenebenchmark_critic.json`）：完整结构化评估数据，适合下游自动化
- **Markdown**（`scenebenchmark_critic.md`）：人类可读摘要

报告输出到：
```
room_*/scene_states/scene_after_furniture/scenebenchmark_critic.{json,md}
room_*/scene_states/final_scene/scenebenchmark_critic.{json,md}
```

Agent 反馈上下文通过 `format_prompt_context()` 生成，限制最多 top-N 个问题（默认 8 个）。
使用 `format_agent_prompt_context()` 时，会按 `agent_prompt_context_filter_enabled` 过滤，只保留当前 agent 可执行的局部问题。

## 配置

**文件**: `config.py` (5KB)

```python
@dataclass
class CriticConfig:
    enabled: bool = False
    metrics: list[str] = field(default_factory=lambda: [
        "spatial_accessibility", "functional_dependency"
    ])
    room_stage_hooks: list[str] = field(default_factory=lambda: [
        "scene_after_furniture", "final_scene"
    ])
    house_stage_hooks: list[str] = field(default_factory=list)
    fd_relation_proposer_mode: str = "template"
    max_issues: int = 8
    agent_prompt_context_filter_enabled: bool = True
    agent_prompt_context_debug_write: bool = False
```

## Git 历史（critic 核心演进）

| Commit | 变更 |
|--------|------|
| 30da02f | 餐桌设置对齐工具 + 多表面支持 |
| ff9dc38 | 保留 media 和 room anchor 检查 |
| eaa43bf | 窗口 resize + 餐椅朝向检查 |
| e0571d3 | 支持 shared_base 多批次并发 probe |
| b77a3f5 | 窗口约束保护墙面空间 |
| 87b79b9 | 餐桌座椅对齐与分布初始化 |
| 16b9481 | Scale-relative wall seating + 通用餐具语义 |
| b73aabe | Stabilize wall seating contracts |
| 4b2edd8 | 床边对齐验证 |
| a0780a5 | 统一 front 轴为 yaw=0 |

## 关键源文件

| 文件 | 大小 | 职责 |
|------|------|------|
| `api.py` | 7KB | 公开 API：evaluate_room/house_scene |
| `adapter.py` | 58KB | SceneSmith → SceneBenchmark 适配器 |
| `checks.py` | 40KB | 检查执行逻辑 |
| `clearance_source.py` | 41KB | 净空标注数据源 |
| `asset_annotation.py` | 59KB | 资产 VLM 标注 |
| `orientation_contracts.py` | 24KB | 正面/访问方向标准化 |
| `prompt_context.py` | 32KB | Agent 反馈格式化 |
| `dining_place_setting_alignment.py` | 25KB | 餐位对齐 |
| `dining_seat_distribution.py` | 8KB | 座椅分布 |
| `media_support_alignment.py` | 14KB | TV-TV stand 对齐 |
| `room_center_alignment.py` | 12KB | 房间中心对齐 |
| `manipuland_completeness.py` | 13KB | 桌面物品完整性 |
| `config.py` | 5KB | 配置定义 |
| `reports.py` | 5KB | 报告输出 |
| `vendor/rules.py` | - | Vendored SceneBenchmark 规则入口 |
