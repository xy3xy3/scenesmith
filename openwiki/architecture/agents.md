# Agent 系统

> **核心文件**：`scenesmith/agent_utils/base_stateful_agent.py` (~60KB)

## 架构模式

所有 domain agent 采用统一的 **Planner → Designer → Critic** 三阶段循环。该模式定义在 `BaseStatefulAgent` 抽象类中，每个 domain agent 通过继承实现领域特有行为。

```
                  ┌──────────────┐
                  │   Planner    │   接收当前场景 + Critic 反馈，制定修改计划
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐
                  │  Designer    │   调用领域 Function Tools 修改场景（直接操作 RoomScene）
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐
                  │   Critic     │   LLM-as-Judge 评分 + 修正建议
                  └──────┬───────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
         评分改善              评分未改善
        继续下一轮            回滚 + 重试
              │                     │
              └──────────┬──────────┘
                         ▼
                  达到 max_turns
                  输出最佳结果
```

### 关键特性

- **max_turns**：每 Agent 最大循环轮数，由 Hydra 配置
- **回滚机制**：评分下降时自动恢复到上一 checkpoint（保存 `previous_checkpoint_scores`）
- **Hard Stop**：`_planner_hard_stop()` 在 budget 耗尽时强制终止，防止本地模型（如 Qwen）忽略停止条件反复调用工具
- **No-Progress 限制**：`_configured_no_progress_limit()` 检测连续无进展轮次

## Scoring 引擎

**文件**: `scenesmith/agent_utils/scoring.py`

多维评分系统，由 `CritiqueWithScores` 数据类承载：

- **功能评分**：物体是否可用（椅子面向桌子、床两侧有床头柜）
- **物理评分**：无穿透、正确支撑、不悬浮
- **语义评分**：是否符合 prompt 要求
- 支持 `align_scores_for_comparison` 做公平对比
- `compute_total_score` 加权汇总
- `format_score_deltas_for_planner` 生成差异化反馈

## 对话 Token 管理

- **TurnTrimmingSession**：`turn_trimming_session.py` — 长对话自动裁剪，防止 token 溢出
- **IntraTurnImageFilter**：`intra_turn_image_filter.py` — 同一轮内去除重复 base64 图片
- **详细 Token 日志**：`log_agent_usage()` 按角色统计 input/output/reasoning/cached tokens

## WorkflowTools

**文件**: `scenesmith/agent_utils/workflow_tools.py` (~7KB)

提供领域无关的任务管理工具，供 Designer Agent 追踪设计进度：

- `designer_todo_manager`：增删查改 TODO 项
- 状态机：`pending` → `completed` 或 `invalidated`
- **Invalidation 机制**（2026-07-10 新增）：被 checkpoint reset 的 TODO 会自动标记为 `invalidated`，防止 Designer 使用过期计划
- 每个 domain agent 通过 `self.workflow_tools = WorkflowTools()` 关联实例

## Agent 创建模式

`BaseStatefulAgent` 提供标准化的子 agent 创建方法：

| 方法 | 作用 |
|------|------|
| `_create_designer_agent(tools, prompt_enum)` | 创建 Designer Agent，注入领域 tools |
| `_create_critic_agent(tools, prompt_enum, output_type)` | 创建 Critic Agent，绑定评分输出类型 |
| `_create_planner_agent(tools, prompt_enum)` | 创建 Planner Agent |
| `_get_model_settings(settings_key, tool_choice)` | 配置 reasoning effort、verbosity、timeout、service_tier |

## 物理验证集成

每个 agent 的设计阶段完成后，自动运行物理后处理：

1. **碰撞检测**：python-fcl 进行 AABB/网格碰撞检测
2. **穿透解析**：优先移动小物体分离穿透
3. **地面穿透检测**：`check_physics_violations()` 检查物体插入地板
4. **支撑面验证**：确保物体正确放置于支撑面
5. **椅子朝向**：`seating_orientation_guard.py` 将椅子朝向最近的桌子边
6. **柜体前端访问**：`furniture_accessibility_guard.py` 确保柜门可打开

## Domain Agent 注册

**文件**: `scenesmith/experiments/base_experiment.py`

实验通过 `compatible_*_agents` 字典注册可用的 domain agent：

```python
class BaseExperiment(ABC):
    compatible_floor_plan_agents: dict[str, type] = {}
    compatible_furniture_agents: dict[str, type] = {}
    # ... 其他 agent 类型
```

每个 key 对应 `configurations/{agent_type}/` 下的 YAML 文件名。

## Git 历史（Agent 系统相关）

| Commit | 变更 |
|--------|------|
| 26dc381 | 在 checkpoint reset 时失效 stale TODO，防止使用过期计划 |
| a55f182 | 更新 furniture/manipuland agent 的 critique 和 progress 限制 |
| 81a68b7 | Manipuland inventory 保护，防止空库存 |
| 01f7a66 | Refine manipuland validation |
