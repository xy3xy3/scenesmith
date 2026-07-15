# SceneSmith 代码 Wiki

> **SceneSmith**: Agentic Generation of Simulation-Ready Indoor Scenes  
> 论文：[arXiv 2602.09153](https://arxiv.org/abs/2602.09153) | 项目主页：[scenesmith.github.io](https://scenesmith.github.io/)  
> MIT License

## 概述

SceneSmith 是一个**多 Agent 协作系统**，能够从自然语言描述**全自动生成仿真就绪的室内 3D 场景**。用户输入一句话（如 "一个带沙发、地毯和咖啡桌的现代客厅"），SceneSmith 依次调用多个专业 Agent 完成房间布局、家具摆放、墙面物体、天花板附着物和小物体配置。

每个 Agent 均采用 **Planner → Designer → Critic** 循环架构，借助 [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) 实现多轮自我修正。3D 资产通过预定义库检索（HSSD/Objaverse）或文生 3D 模型（SAM3D / Hunyuan3D-2）获取，全部物体包含碰撞几何与物理属性，可直接在 Drake、MuJoCo 等仿真器中运行。

### 核心能力

| 能力 | 说明 |
|------|------|
| **文本到场景** | 单句 prompt → 完整室内场景（含物理属性） |
| **多 Agent 协作** | 5 个领域 Agent 顺序执行，共享 `RoomScene` 状态 |
| **自我修正** | Planner/Designer/Critic 循环 + SceneBenchmark 规则评分 |
| **开放集资产** | 支持 HSSD 检索 + SAM3D/Hunyuan3D 生成 |
| **仿真就绪** | Drake SDF 输出，支持导出到 MuJoCo/USD |
| **机器人评估** | 场景内任务生成 + 策略验证管线 |

---

## 快速导航

| 页面 | 内容 |
|------|------|
| [architecture/overview.md](architecture/overview.md) | 系统架构总览：5 个 domain agent、Hydra 配置、后端服务器 |
| [architecture/agents.md](architecture/agents.md) | Agent 系统：BaseStatefulAgent、Planner/Designer/Critic 循环、Scoring |
| [workflows/pipeline.md](workflows/pipeline.md) | Pipeline 工作流：阶段编排、checkpoint/resume、物理后处理、并发 |
| [domain/critic.md](domain/critic.md) | SceneBenchmark Critic：规则评估指标、新增 alignment 检查、适配器 |
| [domain/assets.md](domain/assets.md) | 资产管线：HSSD 检索、VLM 筛选、SAM3D/Hunyuan3D 生成、铰接物体 |
| [operations/runbook.md](operations/runbook.md) | 运行手册：安装、配置、Docker、多 GPU、调试脚本、输出结构 |
| [testing/guidance.md](testing/guidance.md) | 测试指南：测试结构、关键测试模式、mock、最佳实践 |

---

## 技术栈

| 组件 | 技术 |
|------|------|
| Agent 框架 | [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) (v0.6.4+) |
| 3D 引擎 | Blender 4.5.4 (`bpy`) |
| 仿真/导出 | Drake v1.47+、MuJoCo (实验性导出) |
| 配置 | Hydra (OmegaConf) |
| 生成模型 | SAM3D、Hunyuan3D-2、Google Gemini |
| 检索 | OpenCLIP、HSSD、Objaverse、ArtVIP、PartNet-Mobility |
| 几何处理 | Trimesh、Manifold3D、CoACD、V-HACD、python-fcl |
| 测试 | Pytest (asyncio mode)、pytest-testmon |
| 包管理 | `uv` |

---

## 快速开始

### 安装

```bash
# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 进入仓库并安装依赖
cd scenesmith
uv sync

# 3. 激活虚拟环境
source .venv/bin/activate

# 4. 安装 pre-commit
pre-commit install
```

### 环境变量

```bash
export OPENAI_API_KEY="your-openai-key"
# 可选：自定义 API 端点（支持代理/国内镜像）
export OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"
```

### 运行场景生成

```bash
# 单 prompt 直接生成
python main.py +name=my_scene +prompt="A modern living room with a sofa, rug, coffee table, and two floor lamps."

# 启用 SceneBenchmark critic 评估
python main.py +name=my_scene_critic \
  "experiment.tasks=[generate_scenes,evaluate_scenes]" \
  experiment.scenebenchmark_critic.enabled=true \
  +prompt="A small bedroom with a bed and a nightstand beside it."
```

### 运行测试

```bash
# 需要在虚拟环境中（需要 bpy）
pytest tests/unit/ -x           # 单元测试（首个失败停止）
pytest tests/unit/ -k critic    # 只运行 critic 相关
pytest tests/integration/ -x    # 集成测试
pytest tests/ --testmon         # 增量测试
```

---

## 仓库顶层结构

```
scenesmith/
├── main.py                          # 入口：Hydra 驱动的实验启动器
├── pyproject.toml                   # 项目配置与依赖管理
├── configurations/                  # Hydra 配置（YAML）
│   └── config.yaml                  #   根配置，聚合所有子 config
├── scenesmith/
│   ├── agent_utils/                 # 共享框架（~60 个模块）
│   ├── experiments/                 # 实验编排器
│   ├── floor_plan_agents/           # Floor Plan Agent
│   ├── furniture_agents/            # Furniture Agent
│   ├── wall_agents/                 # Wall Agent
│   ├── ceiling_agents/              # Ceiling Agent
│   ├── manipuland_agents/           # Manipuland Agent
│   ├── scenebenchmark_critic/       # 内嵌规则评估器（~20 个模块）
│   ├── prompts/                     # Agent prompt 注册与管理（YAML）
│   ├── robot_eval/                  # 机器人任务评估管线
│   └── utils/                       # 通用工具函数
├── scripts/                         # 安装、数据、评估、调试脚本
├── data/                            # 外部数据集挂载点
├── external/                        # 外部依赖（SAM3D、Hunyuan3D）
├── outputs/                         # 生成场景输出
└── tests/
    ├── unit/                        # 单元测试（~95 个文件）
    └── integration/                 # 集成测试（12 个文件）
```

---

## 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Agent 通信 | 共享 `RoomScene` 状态对象 | 避免 Agent 间直接通信耦合 |
| 资产生成 | SAM3D 优先，HSSD 检索可选 | 开放集生成灵活度高 |
| 评估 | 内嵌 SceneBenchmark 规则 | 零 VLM 开销的确定性几何检查 |
| 并行 | 多进程隔离 + GPU 分区 | Blender 渲染需进程级隔离 |

---

## 开发者指引

- **`AGENTS.md`**：共享云盘服务器规则、critic 模块优先级、测试/模型运行约定、代码修改注释规范
- **`CLAUDE.md`**：OpenWiki 使用说明

修改代码时请遵循 AGENTS.md 中的注释规范（日期 + 修改原因）和提交规则。

---

## Backlog

| 领域 | 原因 |
|------|------|
| robot_eval 管线详情 | 使用率较低，当前文档优先 critic/资产管线 |
| external/ 子模块集成细节 | 依赖外部仓库，变动频率低 |
| BlenderServer 内部架构 | 较深的技术细节，在架构 overview 中已概括 |

---

*最后更新: 2026-07-15 | HEAD: 30da02f*
