#!/usr/bin/env bash
# 从已有 SceneSmith 批次 checkpoint 恢复任意 pipeline 阶段的单批次调试脚本。
# 设计目标：只验证当前阶段的 critic/tool 行为，不重新生成前置 floor plan/furniture。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
START_STAGE="${START_STAGE:-}"
STOP_STAGE="${STOP_STAGE:-$START_STAGE}"
RUN_ID="${RUN_ID:-debug_replay_$(date +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/critic_probe/$RUN_ID}"
PORT_BASE="${PORT_BASE:-23000}"
MODEL_NAME="${MODEL_NAME:-Qwen3.6-27B-Q8_0}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8002/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-sk-123}"
OPENAI_USE_RESPONSES="${OPENAI_USE_RESPONSES:-false}"
HF_HOME="${HF_HOME:-/data/task3_2/L202500266_hrk/.cache/huggingface}"

if [[ -z "$RESUME_FROM_PATH" || -z "$START_STAGE" ]]; then
    echo "用法："
    echo "  RESUME_FROM_PATH=/path/to/batch_004 START_STAGE=manipuland \\"
    echo "  STOP_STAGE=manipuland bash scripts/debug_replay_scene_stage.sh"
    exit 2
fi

case "$START_STAGE" in
    floor_plan|furniture|wall_mounted|ceiling_mounted|manipuland) ;;
    *) echo "错误：START_STAGE=$START_STAGE 不是有效 pipeline 阶段"; exit 2 ;;
esac
case "$STOP_STAGE" in
    furniture|wall_mounted|ceiling_mounted|manipuland) ;;
    *) echo "错误：STOP_STAGE=$STOP_STAGE 不是有效停止阶段"; exit 2 ;;
esac
if [[ ! -d "$RESUME_FROM_PATH" ]]; then
    echo "错误：RESUME_FROM_PATH 不存在：$RESUME_FROM_PATH"
    exit 2
fi

source "$PROJECT_ROOT/.venv/bin/activate"
mkdir -p "$OUTPUT_ROOT"

# 2026-07-14 修改原因：调试回放需要自己的服务端口，避免与用户正在运行的
# 8002/8014 模型服务及其他 Blender/检索服务互相覆盖。
if (( PORT_BASE + 374 > 65535 )); then
    echo "错误：PORT_BASE=$PORT_BASE 导致端口块超出 65535"
    exit 2
fi

PORT_ARGS=(
    "experiment.geometry_generation_server.port=$((PORT_BASE + 5))"
    "experiment.hssd_retrieval_server.port=$((PORT_BASE + 6))"
    "experiment.articulated_retrieval_server.port=$((PORT_BASE + 7))"
    "experiment.materials_retrieval_server.port=$((PORT_BASE + 8))"
    "experiment.objaverse_retrieval_server.port=$((PORT_BASE + 9))"
    "floor_plan_agent.rendering.blender_server_port_range=[$((PORT_BASE + 100)),$((PORT_BASE + 124))]"
    "furniture_agent.rendering.blender_server_port_range=[$((PORT_BASE + 125)),$((PORT_BASE + 199))]"
    "wall_agent.rendering.blender_server_port_range=[$((PORT_BASE + 200)),$((PORT_BASE + 224))]"
    "ceiling_agent.rendering.blender_server_port_range=[$((PORT_BASE + 225)),$((PORT_BASE + 249))]"
    "manipuland_agent.rendering.blender_server_port_range=[$((PORT_BASE + 200)),$((PORT_BASE + 249))]"
    "furniture_agent.collision_geometry.server_port_range=[$((PORT_BASE + 250)),$((PORT_BASE + 324))]"
    "wall_agent.collision_geometry.server_port_range=[$((PORT_BASE + 325)),$((PORT_BASE + 349))]"
    "ceiling_agent.collision_geometry.server_port_range=[$((PORT_BASE + 350)),$((PORT_BASE + 374))]"
    "manipuland_agent.collision_geometry.server_port_range=[$((PORT_BASE + 325)),$((PORT_BASE + 374))]"
)

CSV_SOURCE="$RESUME_FROM_PATH/batch_cases.csv"
CSV_OUTPUT="$OUTPUT_ROOT/batch_cases.csv"
if [[ -f "$CSV_SOURCE" ]]; then
    cp "$CSV_SOURCE" "$CSV_OUTPUT"
else
    # 2026-07-14 修改原因：某些旧 checkpoint 没有批次清单；main.py 仍需要
    # 一个合法 csv 路径才能保存本次调试输出，因此生成最小占位清单。
    printf 'scene_index,prompt,case_id,critic_goal\n' > "$CSV_OUTPUT"
fi

export OPENAI_API_KEY OPENAI_BASE_URL OPENAI_USE_RESPONSES HF_HOME
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

cd "$PROJECT_ROOT"
exec python main.py \
    "+name=debug_replay_${START_STAGE}_${STOP_STAGE}" \
    "experiment.num_workers=1" \
    "experiment.pipeline.parallel_rooms=false" \
    "experiment.pipeline.max_parallel_rooms=1" \
    "experiment.pipeline.skip_wall_mounted=false" \
    "experiment.pipeline.skip_ceiling_mounted=false" \
    "experiment.materials_retrieval_server.enabled=${ENABLE_MATERIALS_RETRIEVAL:-true}" \
    "experiment.scenebenchmark_critic.room_stage_hooks=[scene_after_furniture,scene_after_wall_objects,scene_after_ceiling_objects,final_scene]" \
    "experiment.scenebenchmark_critic.house_stage_hooks=[]" \
    "experiment.scenebenchmark_critic.enabled=true" \
    "experiment.scenebenchmark_critic.asset_annotation.enabled=false" \
    "experiment.scenebenchmark_critic.fd_relation_proposer_mode=${CRITIC_FD_RELATION_PROPOSER_MODE:-template}" \
    "experiment.scenebenchmark_critic.max_fd_relation_proposals=${CRITIC_MAX_FD_RELATION_PROPOSALS:-8}" \
    "floor_plan_agent.openai.model=$MODEL_NAME" \
    "furniture_agent.openai.model=$MODEL_NAME" \
    "wall_agent.openai.model=$MODEL_NAME" \
    "ceiling_agent.openai.model=$MODEL_NAME" \
    "manipuland_agent.openai.model=$MODEL_NAME" \
    "openai.use_responses=$OPENAI_USE_RESPONSES" \
    "experiment.tasks=[generate_scenes,evaluate_scenes]" \
    "experiment.pipeline.start_stage=$START_STAGE" \
    "experiment.pipeline.stop_stage=$STOP_STAGE" \
    "experiment.pipeline.resume_from_path=$RESUME_FROM_PATH" \
    "hydra.run.dir=$OUTPUT_ROOT" \
    "experiment.csv_path=$CSV_OUTPUT" \
    "furniture_agent.asset_manager.general_asset_source=hssd" \
    "wall_agent.asset_manager.general_asset_source=hssd" \
    "ceiling_agent.asset_manager.general_asset_source=hssd" \
    "manipuland_agent.asset_manager.general_asset_source=hssd" \
    "furniture_agent.asset_manager.hssd_front_axis.source=${CRITIC_HSSD_FRONT_AXIS_SOURCE:-scenebenchmark_critic}" \
    "wall_agent.asset_manager.hssd_front_axis.source=${CRITIC_HSSD_FRONT_AXIS_SOURCE:-scenebenchmark_critic}" \
    "ceiling_agent.asset_manager.hssd_front_axis.source=${CRITIC_HSSD_FRONT_AXIS_SOURCE:-scenebenchmark_critic}" \
    "manipuland_agent.asset_manager.hssd_front_axis.source=${CRITIC_HSSD_FRONT_AXIS_SOURCE:-scenebenchmark_critic}" \
    "furniture_agent.asset_manager.router.strategies.articulated.enabled=false" \
    "wall_agent.asset_manager.router.strategies.articulated.enabled=false" \
    "ceiling_agent.asset_manager.router.strategies.articulated.enabled=false" \
    "manipuland_agent.asset_manager.router.strategies.articulated.enabled=false" \
    "${PORT_ARGS[@]}"
