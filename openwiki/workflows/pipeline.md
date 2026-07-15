# Pipeline 工作流

> **核心文件**：`scenesmith/experiments/indoor_scene_generation.py`、`main.py`

## Pipeline 阶段执行

`IndoorSceneGenerationExperiment` 按 `AgentType` 枚举顺序执行 5 个阶段：

```
AgentType 枚举 (scenesmith/agent_utils/room.py):
  floor_plan (0) → furniture (1) → wall_mounted (2) → ceiling_mounted (3) → manipuland (4)
```

### 阶段详情

#### 1. Floor Plan Agent
- 根据 prompt 设计房间形状、尺寸、门窗
- 使用 `floor_plan_tools.py` 和 `door_window_mixin.py` 创建房间多边形
- 模式：`house`（多房间）或 `room`（单房间）
- 输出：带墙壁、地板、天花板的 `RoomScene`

#### 2. Furniture Agent
- 放置主要家具（床、沙发、桌子、椅子、柜子）
- 集成 `furniture_accessibility_guard.py`（柜体正面可访问）
- 集成 `seating_orientation_guard.py`（座椅朝向合理）
- 物理检查：碰撞检测、地面穿透、支撑面验证

#### 3. Wall Agent
- 放置画、镜子、壁挂 TV、置物架
- `window_tools.py` 处理窗口净空约束
- `prompt_constraints.py` 定义墙面放置规则

#### 4. Ceiling Agent
- 放置灯、吊扇等天花板附着物体

#### 5. Manipuland Agent
- 放置小物体：杯子、书籍、餐具、装饰品
- 支持堆叠（`thin_covering_generator.py`）和容器填充
- Scale-aware：`manipuland_scale.py` 确保比例合理

### Pipeline 控制

```bash
# 在指定阶段后停止
python main.py +name=partial experiment.pipeline.stop_stage=furniture

# 从指定阶段恢复（需前一阶段的 checkpoint）
python main.py +name=resume experiment.pipeline.start_stage=wall_mounted

# A/B 分支：从已有实验继续
python main.py +name=branch \
  experiment.pipeline.start_stage=manipuland \
  experiment.pipeline.resume_from_path=outputs/2025-12-21/10-30-45
```

### Checkpoint 依赖

| start_stage | 需要的前一阶段 checkpoint |
|-------------|--------------------------|
| floor_plan | 无 |
| furniture | 无 |
| wall_mounted | `scene_after_furniture` |
| ceiling_mounted | `scene_after_wall_objects` |
| manipuland | `scene_after_ceiling_objects` |

Checkpoint 保存路径：`scene_states/{stage_name}/room_scene.json`

---

## 实验任务系统

**文件**: `scenesmith/experiments/base_experiment.py`

实验支持多种任务类型，在 `experiment.tasks` 列表中指定：

| 任务 | 说明 |
|------|------|
| `generate_scenes` | 执行完整场景生成管线 |
| `evaluate_scenes` | 对已有场景运行 critic 评估（覆盖写入报告） |
| `export_scenes` | 导出场景为 SDF/MuJoCo/USD 格式 |

```bash
python main.py +name=demo "experiment.tasks=[generate_scenes,evaluate_scenes]"
```

---

## 物理后处理

**文件**: `scenesmith/agent_utils/physical_feasibility.py` (~77KB)

每个 agent 完成设计后运行两阶段物理一致性修复：

1. **Projection** — IK 基碰撞解析，可配置 DOF 约束
2. **Simulation** — 物理沉降到静态平衡（全程 6DOF）

来源：改编自 [steerable-scene-generation](https://github.com/nepfaff/steerable-scene-generation) 仓库。

后处理管线：
- **穿透检测**：python-fcl AABB 碰撞检测
- **穿透解析**：分离穿透物体，优先移动小/轻物体
- **地面穿透**：`physics_validation.py` 检测物体插入地板
- **支撑面验证**：确保物体位于支撑面中心附近
- **椅子对齐**：`seating_orientation_guard.py` 将椅子朝向最近桌面
- **柜体正面访问**：`furniture_accessibility_guard.py` 确保柜门可打开

---

## 并发与并行

### 多房间并行

```yaml
experiment:
  num_workers: 4              # 并行场景数
  pipeline:
    parallel_rooms: true      # 房间间并行
    max_parallel_rooms: 2     # 最大并行房间数
```

### 多 GPU 资产生成

几何生成服务器自动检测 `CUDA_VISIBLE_DEVICES` 并生成对应数量的 worker 进程：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py +name=parallel_demo
```

### GPU 分区

| GPU 编号 | 用途 |
|----------|------|
| GPU 0 到 n-2 | BlenderServer + 几何生成（高占用） |
| GPU n-1 | 检索服务（低占用），`_get_retrieval_gpu_device()` 分配 |

bubblewrap 用于隔离每个 BlenderServer 实例到单一 GPU。

---

## WorkflowTools Invalidation

**文件**: `scenesmith/agent_utils/workflow_tools.py` | Commit: `26dc381`

当 checkpoint reset 发生时，Designer 已创建但未完成的 TODO 项会被自动标记为 `invalidated`，并记录失效原因。这防止了以下场景：

1. Designer 创建了基于绝对坐标的 TODO（如 "将沙发移到 X=2.5"）
2. 场景回滚到前一 checkpoint
3. 原有 TODO 中的绝对坐标不再有效
4. Invalidated TODO 不再显示在 `get_next` / `list_all` 中

---

## Stage Replay 调试

**文件**: `scripts/debug_replay_scene_stage.sh` | Commit: `624a0ed`

用于重放指定场景阶段的执行过程，不需要重新运行整个 pipeline。用法：

```bash
bash scripts/debug_replay_scene_stage.sh <output_dir> <stage_name>
```

---

## 工作流相关源文件

| 文件 | 作用 |
|------|------|
| `scenesmith/experiments/indoor_scene_generation.py` | 主编排器，5 agent 调度 |
| `scenesmith/experiments/base_experiment.py` | 实验基类 |
| `scenesmith/agent_utils/base_stateful_agent.py` | Agent 循环实现 |
| `scenesmith/agent_utils/workflow_tools.py` | TODO 管理 + invalidation |
| `scenesmith/agent_utils/scoring.py` | 评分引擎 |
| `scenesmith/agent_utils/turn_trimming_session.py` | 对话裁剪 |
| `scenesmith/agent_utils/physical_feasibility.py` | 物理后处理 |
| `scenesmith/agent_utils/physics_validation.py` | 物理验证 |
| `scenesmith/agent_utils/physics_tools.py` | 物理工具 |
| `scripts/debug_replay_scene_stage.sh` | 阶段重放调试脚本 |
| `scripts/run_single_room_critic_probe.sh` | 单房间 critic 探针 |
