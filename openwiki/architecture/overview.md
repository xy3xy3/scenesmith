# 系统架构总览

> **相关源码**：`/scenesmith/experiments/`、`/scenesmith/agent_utils/`、`/configurations/`

## 分层 Pipeline 架构

SceneSmith 采用**多 Agent 分层流水线架构**。由 `IndoorSceneGenerationExperiment` 编排 5 个领域 Agent 顺序执行，每个 Agent 在共享的 `RoomScene` 状态对象上构建场景。

```
User Prompt
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  IndoorSceneGenerationExperiment (实验编排器)             │
│  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌────────┐  ┌──────────┐ │
│  │Floor Plan│→│Furniture │→│  Wall  │→│Ceiling │→│Manipuland│ │
│  │  Agent   │ │  Agent   │ │  Agent │ │ Agent  │ │  Agent   │ │
│  └──────────┘ └──────────┘ └────────┘ └────────┘ └──────────┘ │
│       │            │           │          │           │        │
│       ▼            ▼           ▼          ▼           ▼        │
│       └────────────┴───────────┴──────────┴───────────┘        │
│                          │                                      │
│                    RoomScene (共享状态)                         │
│                          │                                      │
│                    ┌─────┴─────┐                                │
│                    │ SceneBench│ (每阶段可选)                     │
│                    │ mark Critic│                                │
│                    └───────────┘                                │
└─────────────────────────────────────────────────────────┘
                │
                ▼
       Blender 渲染 + 物理导出 (SDF / MuJoCo / USD)
```

## 核心抽象

### BaseStatefulAgent

**文件**: `scenesmith/agent_utils/base_stateful_agent.py` (~60KB)

所有 domain agent 共享的基类，提供：

- **Planner / Designer / Critic 三阶段循环**：使用 OpenAI Agents SDK 的 `Runner.run()`
- **对话状态持久化**：通过 `SQLiteSession` 保存对话历史
- **Checkpoint / Rollback**：每轮后保存快照，critic 评分下降时回滚
- **Scoring 引擎**：多维度评分（功能、物理、语义），对比评分决定接受/拒绝
- **TurnTrimming**：长对话自动裁剪历史，防止 token 溢出
- **IntraTurnImageFilter**：回合内图片去重，减少 API 费用
- **Token 追踪**：按 agent 角色记录 input/output/reasoning tokens
- **Hard Stop 机制**：`_planner_hard_stop()` 在 workflow 预算耗尽时强制终止循环
- **No-Progress 检测**：`_configured_no_progress_limit()` 检测连续无进展轮次

### RoomScene

**文件**: `scenesmith/agent_utils/room.py` (~78KB)

跨 Agent 共享的场景表示，包含：

- 房间几何（多边形、墙壁、门窗）
- 物体列表（`SceneObject`，含位姿、尺寸、类型、元数据、物理属性）
- 支撑面拓扑（support surfaces）
- 房间级标注（类型、边界、语义标签）

### HouseScene

**文件**: `scenesmith/agent_utils/house.py` (~54KB)

多房间聚合表示，包含多个 `RoomScene` 和房间间连接（门、走廊）。

---

## 5 个 Domain Agent

每个 Agent 继承 `BaseStatefulAgent` 并封装领域特有的 tools 和 prompts：

| Agent | 目录 | 职责 | 关键工具文件 |
|-------|------|------|----------|
| **Floor Plan** | `floor_plan_agents/` | 房间布局、墙壁、门窗 | `floor_plan_tools.py`, `door_window_mixin.py` |
| **Furniture** | `furniture_agents/` | 家具摆放、排列、朝向 | `scene_tools.py`, `furniture_accessibility_guard.py` |
| **Wall** | `wall_agents/` | 墙面物体（画、镜子、TV） | `wall_tools.py`, `window_tools.py`, `prompt_constraints.py` |
| **Ceiling** | `ceiling_agents/` | 天花板物体（灯、吊扇） | `stateful_ceiling_agent.py` |
| **Manipuland** | `manipuland_agents/` | 桌面小物体、餐具、装饰 | `manipuland_tools.py`, `simple_manipuland_primitives.py` |

---

## Hydra 配置系统

**根配置**: `configurations/config.yaml`

采用 Hydra 组合配置模式，支持层级覆盖和 CLI 快捷参数：

```yaml
defaults:
  - experiment: indoor_scene_generation
  - floor_plan_agent: stateful_floor_plan_agent
  - furniture_agent: stateful_furniture_agent
  - wall_agent: stateful_wall_agent
  - ceiling_agent: stateful_ceiling_agent
  - manipuland_agent: stateful_manipuland_agent
```

关键配置项：

| 配置路径 | 说明 |
|----------|------|
| `experiment.tasks` | 任务列表：`generate_scenes`, `evaluate_scenes`, `export_scenes` |
| `pipeline.start_stage` | 起始管线阶段 |
| `pipeline.stop_stage` | 终止管线阶段 |
| `pipeline.parallel_rooms` | 房间级并行开关 |
| `experiment.num_workers` | 并行场景数 |
| `openai.service_tier` | API 服务等级：`default`, `flex`, `priority` |

---

## 后端服务器架构

SceneSmith 启动多个后台服务进程，通过 HTTP 协作：

| 服务 | 位置 | 用途 | GPU |
|------|------|------|-----|
| BlenderServer | `agent_utils/blender/` | 渲染 + 几何处理 | 推荐专用 GPU |
| GeometryGenerationServer | `agent_utils/geometry_generation_server/` | 文生 3D 资产 | 多 GPU 并行 |
| HssdRetrievalServer | `agent_utils/hssd_retrieval_server/` | HSSD 资产检索 | 可选 |
| ObjaverseRetrievalServer | `agent_utils/objaverse_retrieval_server/` | Objaverse 检索 | 可选 |
| ArticulatedRetrievalServer | `agent_utils/articulated_retrieval_server/` | 铰接物体检索 | 可选 |
| MaterialsRetrievalServer | `agent_utils/materials_retrieval_server/` | PBR 材质检索 | 可选 |
| ConvexDecompositionServer | `agent_utils/convex_decomposition_server/` | V-HACD/CoACD 分解 | 可选 |

GPU 分区策略：检索服务分配到最后一块逻辑 GPU，避免与 Blender/几何生成竞争。

---

## 关键源文件地图

| 关注点 | 源文件 |
|--------|--------|
| Agent 基类 | `scenesmith/agent_utils/base_stateful_agent.py` |
| 实验编排 | `scenesmith/experiments/indoor_scene_generation.py` |
| 实验基类 | `scenesmith/experiments/base_experiment.py` |
| 场景表示 | `scenesmith/agent_utils/room.py` |
| 房屋表示 | `scenesmith/agent_utils/house.py` |
| 物理后处理 | `scenesmith/agent_utils/physical_feasibility.py` |
| 物理验证 | `scenesmith/agent_utils/physics_validation.py` |
| 资产管理 | `scenesmith/agent_utils/asset_manager.py` (101KB) |
| 渲染 | `scenesmith/agent_utils/rendering.py` |
| 支撑面提取 | `scenesmith/agent_utils/support_surface_extraction.py` |
| 配置根 | `configurations/config.yaml` |
| 入口 | `main.py` |
| 机器人评估 | `scenesmith/robot_eval/` |
