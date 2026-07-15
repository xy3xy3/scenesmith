# 运行与运维手册

## 环境要求

### 硬件

| 组件 | 最低要求 | 推荐 |
|------|----------|------|
| GPU 内存 | 24GB (Hunyuan3D) | 32GB+ (SAM3D) |
| 总 GPU 内存 | 45GB+ (完整管线) | L40S (论文使用) |
| 多 GPU | 推荐 | AWS g6e.48xlarge |
| 系统 | Linux (Ubuntu) | Ubuntu 22.04+ |

### 软件

- Python 3.11（严格 `<3.12`）
- Blender 4.5.4
- NVIDIA Container Toolkit（Docker 模式）
- bubblewrap（可选，多 GPU 隔离）

## 安装流程

### 1. 基础安装

```bash
uv sync
source .venv/bin/activate
pre-commit install
```

### 2. 3D 资产生成后端

#### SAM3D（推荐，高质量，需 32GB）

```bash
# HuggingFace 认证
huggingface-cli login

# 安装
bash scripts/install_sam3d.sh
```

配置启用：
```yaml
asset_manager:
  backend: "sam3d"
```

#### Hunyuan3D-2（轻量，24GB 可用）

```bash
git submodule update --init --recursive
bash scripts/install_hunyuan3d.sh
```

### 3. 数据准备

#### 必需数据

```bash
# ArtVIP 铰接物体（推荐）
huggingface-cli download nepfaff/scenesmith-preprocessed-data \
    artvip/artvip_vhacd.tar.gz --repo-type dataset --local-dir .
mkdir -p data/artvip_sdf
tar xzf artvip/artvip_vhacd.tar.gz -C data/artvip_sdf

# AmbientCG PBR 材质
python scripts/download_ambientcg.py --output data/materials
```

#### 可选数据

```bash
# HSSD 数据集（检索模式）
cd data
git lfs install
git clone git@hf.co:datasets/hssd/hssd-models
bash ../scripts/download_hssd_data.sh

# Objaverse（ObjectThor 子集）
bash scripts/download_objaverse_data.sh
python scripts/prepare_objaverse.py
```

## 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `OPENAI_API_KEY` | **是** | OpenAI API 密钥 |
| `OPENAI_BASE_URL` | 否 | 自定义 API 端点（代理/国内镜像） |
| `OPENAI_USE_RESPONSES` | 否 | 是否使用 Responses API（默认 `true`） |
| `OPENAI_TRACING_KEY` | 否 | 独立 tracing API key |
| `GOOGLE_API_KEY` | 否 | Gemini 图像生成后端 |
| `LOGLEVEL` | 否 | 日志级别（默认 `INFO`） |

## Docker 部署

```bash
# 构建
docker build -t scenesmith .

# 交互式
docker compose run --rm scenesmith bash

# 单次命令
docker compose run --rm scenesmith python main.py +name=my_experiment

# 冒烟测试
docker compose run --rm scenesmith \
    python -c "import torch; print(torch.cuda.is_available()); import scenesmith"

# 单元测试
docker compose run --rm scenesmith pytest tests/unit/ -x
```

### 卷挂载

| 宿主机路径 | 容器路径 | 内容 |
|-----------|----------|------|
| `./data/` | `/app/data/` | HSSD、Objaverse、材质、索引 |
| `./external/checkpoints/` | `/app/external/checkpoints/` | SAM3D 模型权重 |
| `./outputs/` | `/app/outputs/` | 生成场景 |

## 运行场景生成

### 基本用法

```bash
python main.py +name=my_scene +prompt="A modern living room with a sofa, rug, coffee table, and two floor lamps."
```

### 多场景 (CSV)

在 `experiment.csv_path` 中指定 prompts CSV 文件路径。

### Pipeline 控制

```bash
# 只生成到家具阶段
python main.py +name=partial experiment.pipeline.stop_stage=furniture

# 从墙面阶段恢复
python main.py +name=resume experiment.pipeline.start_stage=wall_mounted

# A/B 分支（从已有输出继续）
python main.py +name=branch \
  experiment.pipeline.start_stage=manipuland \
  experiment.pipeline.resume_from_path=outputs/2025-12-21/10-30-45
```

### SceneBenchmark Critic 启用

```bash
# 生成 + 评估
python main.py +name=with_critic \
  "experiment.tasks=[generate_scenes,evaluate_scenes]" \
  experiment.scenebenchmark_critic.enabled=true \
  +prompt="A small bedroom with a bed and a nightstand"

# 仅重新评估已有输出
python main.py +name=critic_eval \
  "experiment.tasks=[evaluate_scenes]" \
  experiment.scenebenchmark_critic.enabled=true \
  hydra.run.dir=/path/to/existing/output_dir
```

## 多 GPU 最佳实践

```bash
# 使用 4 个 GPU 生成资产
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py +name=multi_gpu
```

检索服务自动分配到最后一块逻辑 GPU，避免与 Blender/几何生成竞争。

多房间并行：
```yaml
experiment:
  num_workers: 4
  pipeline:
    parallel_rooms: true
    max_parallel_rooms: 2
```

## 调试脚本

| 脚本 | 用途 |
|------|------|
| `scripts/debug_replay_scene_stage.sh` | 重放指定场景阶段（无需重跑整个 pipeline） |
| `scripts/run_single_room_critic_probe.sh` | 单房间 critic 探针（支持 shared_base 多批次并发） |
| `scripts/clean_scene_output.py` | 清理输出文件，只保留最终场景 |
| `scripts/run_final_scene_critic.py` | 对最终场景运行完整 critic |
| `scripts/run_final_scene_critic_ablation.py` | Critic 消融实验 |
| `scripts/run_single_scene_original_critic.py` | 单场景原始 critic 对比 |
| `scripts/test_local_qwen_final_scene.py` | 使用本地 Qwen 模型评估场景 |
| `scripts/probe_hssd_vs_openclip_retrieval.py` | HSSD vs OpenCLIP 检索对比 |
| `scripts/test_asset_retrieval.py` | 资产检索测试（含多视角渲染可视化） |
| `scripts/test_material_retrieval.py` | PBR 材质检索测试 |

## 输出目录结构

```
outputs/{run_name}/
├── resolved_config.yaml           # 解析后完整配置
├── experiment.log                 # 实验日志
├── house_000/                     # 房屋级输出
│   ├── room_000/                  # 房间级输出
│   │   ├── scene_states/
│   │   │   ├── scene_after_furniture/
│   │   │   │   ├── room_scene.json
│   │   │   │   ├── scenebenchmark_critic.json
│   │   │   │   └── scenebenchmark_critic.md
│   │   │   ├── scene_after_wall_objects/
│   │   │   ├── scene_after_ceiling_objects/
│   │   │   └── final_scene/
│   │   └── ...
│   └── combined_house/            # 房屋级报告
├── latest-run -> ...              # 最新运行符号链接
└── ...
```

## 性能优化

- **OMP_NUM_THREADS=4**、**MKL_NUM_THREADS=4**、**MALLOC_ARENA_MAX=2**：控制 CPU 线程数
- **内存安全模式**：`template` FD 提议器 + 串行生成可显著降低内存占用
- **Tracemalloc**：`maybe_start_tracemalloc()` 在内存紧张时自动启动内存追踪

## 已知注意事项

1. **bpy 导入顺序**：必须最先导入 bpy，否则 OpenGL 上下文冲突导致 segfault（`conftest.py` 和 `main.py` 已处理）
2. **LLM 兼容性**：若后端不支持 Responses API，设置 `OPENAI_USE_RESPONSES=false`
3. **多 GPU 隔离**：无 bubblewrap 时所有 Blender 实例共享 GPU 0，易 OOM
4. **PartNet-Mobility 质量低**：铰接物体推荐使用 ArtVIP
5. **凭证安全**：使用 `config.yml` 文件 + 内联 Python 脚本避免 API key 暴露在 shell 历史中
