# SceneSmith 内嵌 SceneBenchmark 评测器

本包将 SceneBenchmark 的 `spatial_accessibility` 和 `functional_dependency` 规则评测器直接内嵌到 SceneSmith 中。专为已拥有 `RoomScene` 对象的 SceneSmith 开发者设计，无需引入完整的 SceneBenchmark 运行时或 VLM 依赖即可获得几何反馈。组合房屋辅助功能仍可使用，但推荐的 SceneSmith 路径以房间为优先。

默认的 SceneSmith 集成是单房间模式：评估房间检查点，除非显式设置 `house_stage_hooks`，否则不生成组合房屋报告。

## 检查内容

- `spatial_accessibility`：基于网格/可达性的功能访问区域、障碍物掩码、连通站立区域和可达距离评估。
- `functional_dependency`：基于模板提议和规则评分的依赖关系，包括物体-支撑物、灯-表面、座椅-工作面、座椅-媒体、床-床头柜、餐桌/工作台，以及相关的靠近/朝向关系。
- `interaction_clearance`（默认关闭，需在 `metrics` 中显式启用）：功能净空。这是 SceneBenchmark vendored spec 中**早已保留、却未实现**的 `interaction_clearance` 指标（见 `vendor/.../models.py` 的 `MetricName`，adapter 也已为它计算 `metric_relevance` 权重）——本模块即是它的实现，落在保留槽内而非另起 off-spec 名。基于预计算的人体锚定净空标注（非铰链 6092 件）与铰链扫掠体开合包络（2120 件），按 HSSD `asset_id` 查表，将物体局部系的 keep-clear 区按位姿投影到世界系，检测其它物体是否侵入。零 VLM、纯几何确定性评分；详见下文「功能净空（interaction_clearance）」。

适配器从 SceneSmith 对象名称、描述、元数据、支撑表面和 `placement_info` 中推导类别、功能提示、对象功能画像、支撑区域、房间多边形和放置关系。资产标注字段（如 `front_face`、`access_direction`、`operation_space`、`target_relation` 和 `explicit_target_relation`）会被保留；标注的正面/访问方向也会映射为 SceneBenchmark 风格的交互面，使空间可达性检查使用标注的操作侧。`benchmark_relevance` 设为非功能值的资产标注会抑制功能可供性检查，与 SceneBenchmark 转换器行为一致。`affordances`、`functional_categories` 和 `candidate_affordances` 均可作为可供性输入，元数据中的功能依赖以及 `support_regions`/`support_region` 可位于对象元数据根目录或 `metadata.functional_hints` 内。当标注缺失时，适配器镜像 SceneBenchmark 演示转换器的轻量级默认值：归一化带房间前缀的实例名（如 `bedroom_nightstand_1_f0_c`），推断 `category_keywords`、`front_hint`、`target_relation` 和 `metric_relevance`，并为检测到的或 SceneSmith 提供的支撑区域写入 `support_region_summary`。SceneSmith 还会在单个房间中存在相应对象时，实例化 SceneBenchmark 的分组功能依赖检查（如餐桌组、工作台、多床头柜床边对等）。

vendored 规则代码从 `~/proj/SceneBenchmark/src` 复制到 `vendor/scenebenchmark/`。SceneSmith 默认使用 SceneBenchmark 的确定性模板 FD 提议器。若将 `fd_relation_proposer_mode` 显式设为 `vlm`、`hybrid` 或 `auto`，则使用 vendored FD 提议器入口；当可选的 SceneBenchmark VLM 栈不可用时，回退到模板提议器。默认的 SceneSmith 集成不使用完整的 SceneBenchmark 渲染/请求流水线。详见 `vendor/README.md` 了解源码清单和有意的 vendoring 差异。

## 资产 VLM 标注

SceneSmith 可选地在每个房间评测报告前运行 SceneBenchmark 风格的资产语义标注器。标注器将有效的资产提示写入 `SceneObject.metadata.functional_hints`，使现有适配器和 vendored 规则检查无需额外的转换步骤即可消费标注。当 `write_files` 启用时，还会在 stage 目录下存储每个对象的 YAML/请求产物。

默认标注器禁用。用于确定性本地测试或转换器风格的 dry run：

```bash
uv run python main.py \
  +name=critic_asset_annotation_mock \
  "experiment.tasks=[evaluate_scenes]" \
  experiment.scenebenchmark_critic.enabled=true \
  experiment.scenebenchmark_critic.asset_annotation.enabled=true \
  experiment.scenebenchmark_critic.asset_annotation.backend=mock \
  hydra.run.dir=/path/to/existing/output_dir
```

调用配置的 OpenAI 兼容 VLM 时，切换 backend：

```bash
uv run python main.py \
  +name=critic_asset_annotation_vlm \
  "experiment.tasks=[evaluate_scenes]" \
  experiment.scenebenchmark_critic.enabled=true \
  experiment.scenebenchmark_critic.asset_annotation.enabled=true \
  experiment.scenebenchmark_critic.asset_annotation.backend=vlm \
  hydra.run.dir=/path/to/existing/output_dir
```

当向 `annotate_room_scene` 提供 `BlenderServer` 时，使用网格多视角渲染作为视觉证据。在标准报告路径中，标注器优先使用每个对象保存的 `image_path`（若可用），否则使用对象元数据和启发式先验。

## 生成时启用

评测器默认禁用。通过 Hydra 覆盖启用。默认仅刷新房间报告：

```bash
uv run python main.py \
  +name=critic_demo \
  "experiment.tasks=[generate_scenes,evaluate_scenes]" \
  experiment.scenebenchmark_critic.enabled=true \
  "experiment.prompts=['A bedroom with a nightstand and a mug.','A classroom with desks and chairs.','A living room with a sofa, rug, plants, and a mug.']"
```

在线生成需要与 SceneSmith 其余部分相同的 LLM/API 凭证。报告写入生成的 SceneSmith 检查点旁边：

- `room_*/scene_states/scene_after_furniture/scenebenchmark_critic.json|md`
- `room_*/scene_states/final_scene/scenebenchmark_critic.json|md`

## VLM API 配置

critic 默认使用环境变量配置 VLM（当 `asset_annotation.backend=vlm` 时）：

| 参数 | 环境变量 | hydra 配置 | 优先级 |
|------|----------|------------|--------|
| API Key | `OPENAI_API_KEY` | - | OpenAI SDK 自动读取 |
| Base URL | `OPENAI_BASE_URL` | `openai.base_url` | hydra 优先 |
| Use Responses | `OPENAI_USE_RESPONSES` | `openai.use_responses` | hydra 优先 |

推荐方式：设置环境变量或在 `config.yml` 中配置：

```yaml
# config.yml
openai_api_key: "sk-xxx"
openai_base_url: "https://your-api.com/v1"
openai_use_responses: false  # 可选，默认 false
```

组合房屋报告需显式启用：

- `combined_house_after_furniture/scenebenchmark_critic.json|md`
- `combined_house/scenebenchmark_critic.json|md`

内存安全的单房间 smoke test：保持串行生成，使用模板 FD 提议器，家具放置后停止：

```bash
uv run python main.py \
  +name=scenebenchmark_critic_memsafe_smoke \
  "experiment.tasks=[generate_scenes]" \
  experiment.scenebenchmark_critic.enabled=true \
  "experiment.scenebenchmark_critic.room_stage_hooks=[scene_after_furniture]" \
  experiment.scenebenchmark_critic.fd_relation_proposer_mode=template \
  experiment.num_workers=1 \
  experiment.pipeline.parallel_rooms=false \
  experiment.pipeline.max_parallel_rooms=1 \
  experiment.pipeline.stop_stage=furniture \
  furniture_agent.asset_manager.router.parallel_workers=1 \
  "experiment.prompts=['A small bedroom with a bed and a nightstand beside it.']"
```

完整单房间测试（含全流程生成 + 评测）：

```bash
uv run python main.py \
  +name=single_room_full_critic \
  "experiment.tasks=[generate_scenes,evaluate_scenes]" \
  experiment.scenebenchmark_critic.enabled=true \
  experiment.num_workers=1 \
  experiment.pipeline.parallel_rooms=false \
  experiment.pipeline.max_parallel_rooms=1 \
  experiment.materials_retrieval_server.enabled=false \
  floor_plan_agent.materials.use_retrieval_server=false \
  furniture_agent.asset_manager.general_asset_source=hssd \
  manipuland_agent.asset_manager.general_asset_source=hssd \
  wall_agent.asset_manager.general_asset_source=hssd \
  ceiling_agent.asset_manager.general_asset_source=hssd \
  "experiment.prompts=['A cozy bedroom with a bed, two nightstands, a wardrobe, a desk with 4 chairs, and a mug on the desk.']"
```

若凭证存储在本地 `config.yml` 中（包含 `openai_api_key` 和 `openai_base_url`），使用 wrapper 脚本避免凭证泄露到 shell 历史：

```bash
uv run python - <<'PY'
import os
import runpy
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path("config.yml").read_text(encoding="utf-8")) or {}
os.environ["OPENAI_API_KEY"] = str(config["openai_api_key"])
os.environ["OPENAI_BASE_URL"] = str(config["openai_base_url"])
os.environ["OPENAI_USE_RESPONSES"] = str(config.get("openai_use_responses", False)).lower()
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

sys.argv = [
    "main.py",
    "+name=scenebenchmark_critic_memsafe_smoke",
    "experiment.tasks=[generate_scenes]",
    "experiment.scenebenchmark_critic.enabled=true",
    "experiment.scenebenchmark_critic.room_stage_hooks=[scene_after_furniture]",
    "experiment.scenebenchmark_critic.fd_relation_proposer_mode=template",
    "experiment.num_workers=1",
    "experiment.pipeline.parallel_rooms=false",
    "experiment.pipeline.max_parallel_rooms=1",
    "experiment.pipeline.stop_stage=furniture",
    "furniture_agent.asset_manager.router.parallel_workers=1",
    "openai.use_responses=false",
    "experiment.prompts=['A small bedroom with a bed and a nightstand beside it.']",
]
runpy.run_path("main.py", run_name="__main__")
PY
```

仅刷新 smoke test 报告而不重新生成场景，使用 `experiment.tasks=[evaluate_scenes]` 重新运行同一输出目录。

Smoke test 后，验证至少存在房间级报告：

```bash
find /path/to/output_dir \
  -path '*/scene_states/scene_after_furniture/scenebenchmark_critic.json' \
  -o -path '*/scene_states/scene_after_furniture/scenebenchmark_critic.md'
```

无需 `jq` 即可汇总 JSON 报告：

```bash
uv run python - <<'PY'
import json
from pathlib import Path

root = Path("/path/to/output_dir")
for report_path in sorted(root.glob("scene_*/room_*/scene_states/*/scenebenchmark_critic.json")):
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    checks = payload.get("case_pack", {}).get("checks", [])
    proposer_checks = [
        check
        for check in checks
        if check.get("check_source") == "fd_relation_proposer"
    ]
    print(report_path)
    print("  scope:", payload.get("scope"), "stage:", payload.get("stage"))
    print("  metrics:", sorted((payload.get("summary", {}).get("metric_summary") or {})))
    print("  total_checks:", payload.get("summary", {}).get("scene_summary", {}).get("total_checks"))
    print("  fd_proposer_checks:", len(proposer_checks))
PY
```

上述单房间 smoke 配置的报告是主要的手动验收产物。

## 重新评估已有输出

刷新已有输出目录的报告：

```bash
uv run python main.py \
  +name=critic_eval \
  "experiment.tasks=[evaluate_scenes]" \
  experiment.scenebenchmark_critic.enabled=true \
  hydra.run.dir=/path/to/existing/output_dir
```

若凭证在 `config.yml` 中，使用相同的 wrapper 模式：

```bash
uv run python - <<'PY'
import os
import runpy
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path("config.yml").read_text(encoding="utf-8")) or {}
os.environ["OPENAI_API_KEY"] = str(config["openai_api_key"])
os.environ["OPENAI_BASE_URL"] = str(config["openai_base_url"])
os.environ["OPENAI_USE_RESPONSES"] = str(config.get("openai_use_responses", False)).lower()

sys.argv = [
    "main.py",
    "+name=critic_eval_config",
    "experiment.tasks=[evaluate_scenes]",
    "experiment.scenebenchmark_critic.enabled=true",
    "openai.use_responses=false",
    "hydra.run.dir=/path/to/existing/output_dir",
]
runpy.run_path("main.py", run_name="__main__")
PY
```

`evaluate_scenes` 扫描已保存的房间 `scene_state.json` 文件并就地覆盖房间评测报告。仅当 `house_stage_hooks` 设为非空列表时，才会刷新组合房屋的 `house_state.json` 报告。

## Python API

```python
from scenesmith.scenebenchmark_critic import (
    CriticConfig,
    evaluate_room_scene,
    format_prompt_context,
    room_scene_to_case_pack,
    write_room_stage_report,
)

config = CriticConfig(
    enabled=True,
    metrics=("spatial_accessibility", "functional_dependency"),
)

case_pack = room_scene_to_case_pack(room_scene, stage="final_scene")
payload = evaluate_room_scene(room_scene, config=config, stage="final_scene")
context = format_prompt_context(payload, max_issues=8)
write_room_stage_report(room_scene, stage_dir, config=config, stage="final_scene")
```

`evaluate_room_scene()` 立即运行，用于临时或脚本化检查。`write_*_stage_report()` 辅助函数是 stage hook 入口点，遵守 `enabled` 和配置的 stage hook 列表。

`room_scene_to_case_pack()` 暴露适配后的 SceneBenchmark 风格几何数据，用于调试或下游自动化。评估载荷包含该 `case_pack`、规则 `results`、stage 元数据和汇总计数。Markdown 报告用于快速人工检查；JSON 报告足够稳定，可供下游自动化使用。

## 功能净空（interaction_clearance）

净空评测把我们（yz）预计算的净空标注接进评测器，实现 SceneBenchmark spec 保留但缺失的 `interaction_clearance` 指标（第三个 metric）。

**数据**（vendored 进 `clearance_data/`，按 40 位 HSSD `asset_id` 索引）：

- `nonartic_clearance_index.json`：6092 个非铰链物体的人体锚定净空（落座/接近/操作/上方站立/通行 5 型；depth/height 锚人体测量学常数，width 取物体面）。
- `artic_clearance_index.json`：2120 个铰链物体的扫掠体开合包络（`expand` = swept/static 逐轴外扩比）。

**核心模块** `clearance_source.py`（纯 Python，无 Drake/Blender 依赖，可独立单测）：

- `get_clearance(asset_id)` → 统一净空记录（这是「净空服务」内核，可再包一层 HTTP 当在线服务）。
- `project_keep_clear(record, bbox_world, yaw_deg)` → 物体局部净空按位姿投影到**世界系 keep-clear AABB**（原始索引 meta 记录 HSSD mesh 帧 `front=-Y`，但 SceneSmith 摆放位姿的消费方正面是局部 `+Y`；因此有向净空按局部 `+Y` 经 yaw 旋转后吸附到最近世界轴；「四周」对称件生成环形四面；「上方站立」生成头顶垂直净空；继承门控小件返回空，不占独立盒）。
- `build_clearance_checks` / `evaluate_clearance` → 逐物体把 keep-clear 区与其它物体 `bbox_world` 做 AABB 侵入检测，`pass`/`fail` + 侵入物清单 + 置信度（high/med/low→0.9/0.6/0.3）。

**接入点**：`asset_annotation.py` 给 `AssetAnnotation` 加可选 `clearance` 字段并镜像进 `metadata.clearance`；`checks.py::build_checks` 在 `interaction_clearance ∈ metrics` 时生成净空检查；`vendor/rules.py` 按 `metric=="interaction_clearance"` 分派评分。（模块/字段名仍叫 `clearance`，只有 metric dispatch key 用 spec 名 `interaction_clearance`。）

**启用**（默认关闭，向后兼容）：

```yaml
scenebenchmark_critic:
  enabled: true
  metrics: [spatial_accessibility, functional_dependency, interaction_clearance]
```

世界系约定与适配器 `bbox_world` 一致：X/Y=地面、Z=向上。`asset_id` 缺失或不在索引中的物体自动跳过。

## 集成说明

- 评测器默认仅生成报告。`hard_gate` 元数据会被记录，但 v1 不会回滚或重写 SceneSmith 场景。
- LLM 评测器提示注入仅对家具和 manipuland 代理运行。报告始终保留全量规则结果；注入到 LLM critic 的摘要默认经过 `agent_prompt_context_filter_enabled` 过滤，只保留当前代理可执行的局部问题。
- `agent_prompt_context_debug_write` 可写出原始/过滤后的 issue id 摘要，用于排查 critic-on 生成差异；默认关闭。
- 组合房屋报告仍可通过 `write_house_stage_report` 或非空 `house_stage_hooks` 使用，但默认集成仅限房间。
- 组合房屋家具 stage 报告会过滤掉 manipuland，使 stage 级检查仅包含实际已放置的对象。
- `vendor/rules.py` 是 vendored SceneBenchmark 模块的桥接，有意避免在运行时导入外部 SceneBenchmark 仓库。
