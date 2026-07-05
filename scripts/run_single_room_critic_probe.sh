#!/usr/bin/env bash
# 批量运行单房间场景，对比开启/关闭 SceneBenchmark critic 的效果。
# 设计目标：
# 1. 复用当前项目里常用的单房间、串行、HSSD、禁用 articulated 资源的配置。
# 2. 内置一组四个典型单房间场景，更容易触发 spatial_accessibility / functional_dependency。
# 3. 支持 shared_base -> critic_off -> critic_on 分叉，减少前缀随机性。
# 4. 支持按阶段停止，或跳过 wall / ceiling 后继续测试 manipulands。
# 5. 默认分别跑 critic=off 与 critic=on，方便直接对照生成结果和评测报告。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 运行模式：
#   both: 先跑 critic_off，再跑 critic_on（默认）
#   off : 只跑关闭 critic
#   on  : 只跑开启 critic
MODE="${1:-both}"
REQUESTED_MODE="$MODE"

case "$MODE" in
    both|off|on)
        ;;
    *)
        echo "用法: $0 [both|off|on]"
        echo "示例:"
        echo "  $0"
        echo "  $0 off"
        echo "  MAX_CASES=2 $0 on"
        echo "  PIPELINE_STOP_STAGE=wall_mounted $0 both"
        exit 1
        ;;
esac

cd "$PROJECT_ROOT"

# 尽量沿用你现在的环境；如果外部没设，再补默认值。
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-123}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8002/v1}"
export OPENAI_USE_RESPONSES="${OPENAI_USE_RESPONSES:-false}"
export HF_HOME="${HF_HOME:-/data/task3_2/L202500266_hrk/.cache/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

MODEL_NAME="${MODEL_NAME:-Qwen3.6-27B-Q8_0}"
EXPERIMENT_NAME_PREFIX="${EXPERIMENT_NAME_PREFIX:-single_room_critic_probe}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/critic_probe/$RUN_ID}"
MAX_CASES="${MAX_CASES:-0}"
SCENE_BATCH_SIZE="${SCENE_BATCH_SIZE:-1}"
PIPELINE_STOP_STAGE="${PIPELINE_STOP_STAGE:-furniture}"
SKIP_WALL_MOUNTED="${SKIP_WALL_MOUNTED:-false}"
SKIP_CEILING_MOUNTED="${SKIP_CEILING_MOUNTED:-false}"
BRANCH_FROM_SHARED_BASE="${BRANCH_FROM_SHARED_BASE:-false}"
SHARED_BASE_STOP_STAGE="${SHARED_BASE_STOP_STAGE:-floor_plan}"
SHARED_BASE_ROOT="${SHARED_BASE_ROOT:-}"
CRITIC_ASSET_ANNOTATION="${CRITIC_ASSET_ANNOTATION:-true}"
CRITIC_ROOM_STAGE_HOOKS="${CRITIC_ROOM_STAGE_HOOKS:-}"
CRITIC_FD_RELATION_PROPOSER_MODE="${CRITIC_FD_RELATION_PROPOSER_MODE:-vlm}"
CRITIC_MAX_FD_RELATION_PROPOSALS="${CRITIC_MAX_FD_RELATION_PROPOSALS:-8}"
CRITIC_PROBE_PARALLEL="${CRITIC_PROBE_PARALLEL:-false}"
CRITIC_REPORT_STAGE_LABEL=""
BRANCH_START_STAGE=""
PORT_ARGS=()

normalize_bool() {
    case "${1,,}" in
        1|true|yes|y|on)
            printf 'true'
            ;;
        0|false|no|n|off|'')
            printf 'false'
            ;;
        *)
            return 1
            ;;
    esac
}

next_stage_after() {
    case "$1" in
        floor_plan)
            printf 'furniture'
            ;;
        furniture)
            printf 'wall_mounted'
            ;;
        wall_mounted)
            printf 'ceiling_mounted'
            ;;
        ceiling_mounted)
            printf 'manipuland'
            ;;
        *)
            return 1
            ;;
    esac
}

mkdir -p "$OUTPUT_ROOT"

case "$SCENE_BATCH_SIZE" in
    ''|*[!0-9]*)
        echo "错误：SCENE_BATCH_SIZE 必须是正整数，当前为 '$SCENE_BATCH_SIZE'"
        exit 1
        ;;
esac

if [ "$SCENE_BATCH_SIZE" -lt 1 ]; then
    echo "错误：SCENE_BATCH_SIZE 至少为 1"
    exit 1
fi

if ! SKIP_WALL_MOUNTED="$(normalize_bool "$SKIP_WALL_MOUNTED")"; then
    echo "错误：SKIP_WALL_MOUNTED 必须是 true/false"
    exit 1
fi

if ! SKIP_CEILING_MOUNTED="$(normalize_bool "$SKIP_CEILING_MOUNTED")"; then
    echo "错误：SKIP_CEILING_MOUNTED 必须是 true/false"
    exit 1
fi

if ! BRANCH_FROM_SHARED_BASE="$(normalize_bool "$BRANCH_FROM_SHARED_BASE")"; then
    echo "错误：BRANCH_FROM_SHARED_BASE 必须是 true/false"
    exit 1
fi

if ! CRITIC_ASSET_ANNOTATION="$(normalize_bool "$CRITIC_ASSET_ANNOTATION")"; then
    echo "错误：CRITIC_ASSET_ANNOTATION 必须是 true/false"
    exit 1
fi

case "${CRITIC_FD_RELATION_PROPOSER_MODE,,}" in
    template|vlm|hybrid|auto)
        CRITIC_FD_RELATION_PROPOSER_MODE="${CRITIC_FD_RELATION_PROPOSER_MODE,,}"
        ;;
    *)
        echo "错误：CRITIC_FD_RELATION_PROPOSER_MODE 必须是 template / vlm / hybrid / auto"
        exit 1
        ;;
esac

case "$CRITIC_MAX_FD_RELATION_PROPOSALS" in
    ''|*[!0-9]*)
        echo "错误：CRITIC_MAX_FD_RELATION_PROPOSALS 必须是正整数，当前为 '$CRITIC_MAX_FD_RELATION_PROPOSALS'"
        exit 1
        ;;
esac

if [ "$CRITIC_MAX_FD_RELATION_PROPOSALS" -lt 1 ]; then
    echo "错误：CRITIC_MAX_FD_RELATION_PROPOSALS 至少为 1"
    exit 1
fi

if ! CRITIC_PROBE_PARALLEL="$(normalize_bool "$CRITIC_PROBE_PARALLEL")"; then
    echo "错误：CRITIC_PROBE_PARALLEL 必须是 true/false"
    exit 1
fi

case "$PIPELINE_STOP_STAGE" in
    furniture)
        CRITIC_REPORT_STAGE_LABEL="scene_after_furniture"
        ;;
    wall_mounted)
        CRITIC_REPORT_STAGE_LABEL="scene_after_wall_objects"
        ;;
    ceiling_mounted)
        CRITIC_REPORT_STAGE_LABEL="scene_after_ceiling_objects"
        ;;
    manipuland)
        CRITIC_REPORT_STAGE_LABEL="final_scene"
        ;;
    *)
        echo "错误：PIPELINE_STOP_STAGE 必须是 furniture / wall_mounted / ceiling_mounted / manipuland"
        exit 1
        ;;
esac

if [ "$BRANCH_FROM_SHARED_BASE" = "true" ]; then
    case "$SHARED_BASE_STOP_STAGE" in
        floor_plan|furniture|wall_mounted|ceiling_mounted)
            ;;
        *)
            echo "错误：SHARED_BASE_STOP_STAGE 必须是 floor_plan / furniture / wall_mounted / ceiling_mounted"
            exit 1
            ;;
    esac

    if ! BRANCH_START_STAGE="$(next_stage_after "$SHARED_BASE_STOP_STAGE")"; then
        echo "错误：无法从 SHARED_BASE_STOP_STAGE=$SHARED_BASE_STOP_STAGE 推导分叉起点"
        exit 1
    fi

    if [ -n "$SHARED_BASE_ROOT" ] && [ ! -d "$SHARED_BASE_ROOT" ]; then
        echo "错误：SHARED_BASE_ROOT 不存在或不是目录: $SHARED_BASE_ROOT"
        exit 1
    fi
fi

if [ -z "$CRITIC_ROOM_STAGE_HOOKS" ]; then
    CRITIC_ROOM_STAGE_HOOKS="[$CRITIC_REPORT_STAGE_LABEL]"
fi

if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "错误：未找到虚拟环境 $PROJECT_ROOT/.venv"
    exit 1
fi

# shellcheck disable=SC1091
source "$PROJECT_ROOT/.venv/bin/activate"

echo "检查 bpy 安装..."
python - <<'PY'
from pathlib import Path
import importlib.metadata as metadata
import sys

try:
    dist = metadata.distribution("bpy")
except metadata.PackageNotFoundError:
    print("错误：当前虚拟环境未安装 bpy。请在 .venv 中重新安装项目依赖。", file=sys.stderr)
    raise SystemExit(1)

required = [
    "bpy/4.5/datafiles/colormanagement/config.ocio",
    "bpy/4.5/datafiles/fonts/Inter.woff2",
    "bpy/4.5/datafiles/fonts/DejaVuSansMono.woff2",
    "bpy/4.5/scripts/modules/bpy_types.py",
]
missing = [path for path in required if not Path(dist.locate_file(path)).exists()]
if missing:
    print("错误：bpy wheel 数据文件不完整，缺失：", file=sys.stderr)
    for path in missing:
        print(f"  - {path}", file=sys.stderr)
    print("建议修复：source .venv/bin/activate && uv pip install --reinstall bpy==4.5.4", file=sys.stderr)
    raise SystemExit(1)

try:
    import bpy  # noqa: F401
except Exception as exc:
    print(f"错误：bpy 导入失败：{exc}", file=sys.stderr)
    print("建议修复：source .venv/bin/activate && uv pip install --reinstall bpy==4.5.4", file=sys.stderr)
    raise SystemExit(1)
PY
echo "bpy 安装正常。"
echo

echo "=========================================="
echo "单房间 critic 对照批跑"
echo "项目目录: $PROJECT_ROOT"
echo "输出根目录: $OUTPUT_ROOT"
echo "请求运行模式: $REQUESTED_MODE"
echo "实际运行模式: $MODE"
echo "模型名: $MODEL_NAME"
echo "MAX_CASES: $MAX_CASES (0 表示不限制)"
echo "SCENE_BATCH_SIZE: $SCENE_BATCH_SIZE"
echo "PIPELINE_STOP_STAGE: $PIPELINE_STOP_STAGE"
echo "SKIP_WALL_MOUNTED: $SKIP_WALL_MOUNTED"
echo "SKIP_CEILING_MOUNTED: $SKIP_CEILING_MOUNTED"
echo "BRANCH_FROM_SHARED_BASE: $BRANCH_FROM_SHARED_BASE"
echo "SHARED_BASE_STOP_STAGE: $SHARED_BASE_STOP_STAGE"
echo "SHARED_BASE_ROOT: ${SHARED_BASE_ROOT:-<auto-generate under OUTPUT_ROOT>}"
echo "BRANCH_START_STAGE: ${BRANCH_START_STAGE:-<none>}"
echo "CRITIC_ASSET_ANNOTATION: $CRITIC_ASSET_ANNOTATION"
echo "CRITIC_ROOM_STAGE_HOOKS: $CRITIC_ROOM_STAGE_HOOKS"
echo "CRITIC_FD_RELATION_PROPOSER_MODE: $CRITIC_FD_RELATION_PROPOSER_MODE"
echo "CRITIC_MAX_FD_RELATION_PROPOSALS: $CRITIC_MAX_FD_RELATION_PROPOSALS"
echo "CRITIC_PROBE_PARALLEL: $CRITIC_PROBE_PARALLEL"
echo "OPENAI_BASE_URL: $OPENAI_BASE_URL"
echo "=========================================="
echo

echo "等待本地 llama.cpp / OpenAI 兼容服务就绪..."
bash /data/task3_2/L202500266_hrk/code/check_llama_cpp.sh
echo "模型服务已就绪。"
echo

# 每一项格式：
#   case_id|主要想刺激的 critic 类型|英文 prompt
CASES=(
    "living_room_media_bottleneck|沙发-茶几-TV 功能关系 + 客厅通行瓶颈|A living room with a sofa against the back wall facing a TV stand and television on the opposite wall, a coffee table centered between the sofa and TV stand, two armchairs flanking the coffee table near each end of the sofa, and a floor lamp beside one armchair. A remote control and a few magazines lie on the coffee table, and a small rug lies between the coffee table and TV stand."
    "study_desk_access_crunch|书桌-办公椅-显示器功能关系 + 书房接近空间|A study with a desk centered against the back wall, an office chair tucked under the desk, a computer monitor on the desk, two guest chairs against the side wall facing the desk, and a bookshelf on the adjacent wall. A desk lamp and a notebook sit on the desk, a pen holder next to the monitor, and a small trash can beside the desk."
    "bedroom_bedside_blockage|床-床头柜-台灯功能关系 + 床侧与衣柜可达性|A bedroom with a bed centered on the main wall, a nightstand with a table lamp on each side of the bed, a dresser against the opposite wall directly facing the bed, and a wardrobe placed next to the dresser. An alarm clock sits on one nightstand, a book on the other, and a small wastebasket near the dresser."
    "dining_room_service_squeeze|餐桌-餐椅-餐具功能关系 + 用餐区与餐边柜可达性|A dining room with a dining table in the center, four dining chairs arranged around it with one on each side, a sideboard against the wall behind the chairs on one side, and table settings for four including plates, cutlery, and glasses. A centerpiece vase with flowers sits in the middle of the table, and a set of coasters sits on the sideboard."
)

COMMON_ARGS=(
    "experiment.num_workers=${SCENE_BATCH_SIZE}"
    "experiment.pipeline.parallel_rooms=false"
    "experiment.pipeline.max_parallel_rooms=1"
    "experiment.pipeline.skip_wall_mounted=${SKIP_WALL_MOUNTED}"
    "experiment.pipeline.skip_ceiling_mounted=${SKIP_CEILING_MOUNTED}"
    "experiment.materials_retrieval_server.enabled=false"
    "experiment.scenebenchmark_critic.room_stage_hooks=${CRITIC_ROOM_STAGE_HOOKS}"
    "experiment.scenebenchmark_critic.house_stage_hooks=[]"
    "floor_plan_agent.materials.use_retrieval_server=false"
    "furniture_agent.asset_manager.general_asset_source=hssd"
    "furniture_agent.asset_manager.router.strategies.articulated.enabled=false"
    "manipuland_agent.asset_manager.general_asset_source=hssd"
    "manipuland_agent.asset_manager.router.strategies.articulated.enabled=false"
    "wall_agent.asset_manager.general_asset_source=hssd"
    "wall_agent.asset_manager.router.strategies.articulated.enabled=false"
    "ceiling_agent.asset_manager.general_asset_source=hssd"
    "ceiling_agent.asset_manager.router.strategies.articulated.enabled=false"
    "floor_plan_agent.openai.model=${MODEL_NAME}"
    "furniture_agent.openai.model=${MODEL_NAME}"
    "wall_agent.openai.model=${MODEL_NAME}"
    "ceiling_agent.openai.model=${MODEL_NAME}"
    "manipuland_agent.openai.model=${MODEL_NAME}"
    "openai.use_responses=false"
    "experiment.scenebenchmark_critic.fd_relation_proposer_mode=${CRITIC_FD_RELATION_PROPOSER_MODE}"
    "experiment.scenebenchmark_critic.max_fd_relation_proposals=${CRITIC_MAX_FD_RELATION_PROPOSALS}"
)

build_port_args() {
    local run_kind="$1"

    case "$run_kind" in
        critic_off)
            PORT_ARGS=(
                "experiment.geometry_generation_server.port=7005"
                "experiment.hssd_retrieval_server.port=7006"
                "experiment.articulated_retrieval_server.port=7007"
                "experiment.materials_retrieval_server.port=7008"
                "experiment.objaverse_retrieval_server.port=7009"
                "floor_plan_agent.rendering.blender_server_port_range=[8000,8025]"
                "furniture_agent.rendering.blender_server_port_range=[8000,8175]"
                "wall_agent.rendering.blender_server_port_range=[8000,8125]"
                "ceiling_agent.rendering.blender_server_port_range=[8000,8125]"
                "manipuland_agent.rendering.blender_server_port_range=[8000,8125]"
                "furniture_agent.collision_geometry.server_port_range=[7100,7275]"
                "wall_agent.collision_geometry.server_port_range=[7100,7225]"
                "ceiling_agent.collision_geometry.server_port_range=[7100,7225]"
                "manipuland_agent.collision_geometry.server_port_range=[7100,7225]"
            )
            ;;
        critic_on)
            PORT_ARGS=(
                "experiment.geometry_generation_server.port=7015"
                "experiment.hssd_retrieval_server.port=7016"
                "experiment.articulated_retrieval_server.port=7017"
                "experiment.materials_retrieval_server.port=7018"
                "experiment.objaverse_retrieval_server.port=7019"
                "floor_plan_agent.rendering.blender_server_port_range=[8026,8050]"
                "furniture_agent.rendering.blender_server_port_range=[8176,8350]"
                "wall_agent.rendering.blender_server_port_range=[8126,8250]"
                "ceiling_agent.rendering.blender_server_port_range=[8126,8250]"
                "manipuland_agent.rendering.blender_server_port_range=[8126,8250]"
                "furniture_agent.collision_geometry.server_port_range=[7276,7450]"
                "wall_agent.collision_geometry.server_port_range=[7226,7350]"
                "ceiling_agent.collision_geometry.server_port_range=[7226,7350]"
                "manipuland_agent.collision_geometry.server_port_range=[7226,7350]"
            )
            ;;
        shared_base|*)
            PORT_ARGS=()
            ;;
    esac
}

csv_quote() {
    local value="$1"
    value=${value//\"/\"\"}
    printf '"%s"' "$value"
}

run_batch() {
    local run_kind="$1"
    local batch_index="$2"
    shift 2
    local batch_entries=("$@")

    local critic_enabled="false"
    local tasks_override="[generate_scenes,evaluate_scenes]"
    local stop_stage_override="$PIPELINE_STOP_STAGE"
    local start_stage_override=""
    local resume_from_path=""

    case "$run_kind" in
        shared_base)
            tasks_override="[generate_scenes]"
            stop_stage_override="$SHARED_BASE_STOP_STAGE"
            ;;
        critic_off)
            critic_enabled="false"
            ;;
        critic_on)
            critic_enabled="true"
            ;;
        *)
            echo "错误：未知 run_kind=$run_kind"
            exit 1
            ;;
    esac

    if [ "$run_kind" != "shared_base" ] && [ "$BRANCH_FROM_SHARED_BASE" = "true" ]; then
        start_stage_override="$BRANCH_START_STAGE"
        if [ -n "$SHARED_BASE_ROOT" ]; then
            resume_from_path="$SHARED_BASE_ROOT/$(printf "batch_%03d" "$batch_index")"
        else
            resume_from_path="$OUTPUT_ROOT/shared_base/$(printf "batch_%03d" "$batch_index")"
        fi

        if [ ! -d "$resume_from_path" ]; then
            echo "错误：未找到可复用的 shared_base 批次目录: $resume_from_path"
            echo "请确认 SHARED_BASE_ROOT、SCENE_BATCH_SIZE、MAX_CASES 与 shared_base 产物一致。"
            exit 1
        fi
    fi

    local batch_label
    batch_label=$(printf "batch_%03d" "$batch_index")
    local hydra_dir="$OUTPUT_ROOT/$run_kind/$batch_label"
    local exp_name="${EXPERIMENT_NAME_PREFIX}_${run_kind}_${batch_label}"
    local batch_csv="$hydra_dir/batch_cases.csv"
    local case_summary=()

    build_port_args "$run_kind"

    mkdir -p "$hydra_dir"

    printf 'scene_index,prompt,case_id,critic_goal\n' > "$batch_csv"

    local entry
    for entry in "${batch_entries[@]}"; do
        IFS='|' read -r scene_index case_id critic_goal prompt <<< "$entry"
        printf '%s,%s,%s,%s\n' \
            "$scene_index" \
            "$(csv_quote "$prompt")" \
            "$(csv_quote "$case_id")" \
            "$(csv_quote "$critic_goal")" >> "$batch_csv"
        case_summary+=("scene_${scene_index}:$case_id")
    done

    echo "------------------------------------------"
    echo "开始运行批次: $run_kind / $batch_label"
    echo "批次场景数: ${#batch_entries[@]}"
    echo "批次映射: ${case_summary[*]}"
    echo "输出目录: $hydra_dir"
    echo "批次清单: $batch_csv"
    echo "critic 开关: $critic_enabled"
    echo "tasks: $tasks_override"
    echo "stop_stage: $stop_stage_override"
    if [ "${#PORT_ARGS[@]}" -gt 0 ]; then
        echo "port overrides: ${PORT_ARGS[*]}"
    else
        echo "port overrides: <default config>"
    fi
    if [ -n "$start_stage_override" ]; then
        echo "start_stage: $start_stage_override"
        echo "resume_from_path: $resume_from_path"
    fi
    echo "------------------------------------------"

    local cmd=(
        python main.py
        "+name=${exp_name}"
        "${COMMON_ARGS[@]}"
        "${PORT_ARGS[@]}"
        "experiment.tasks=${tasks_override}"
        "experiment.pipeline.stop_stage=${stop_stage_override}"
        "experiment.scenebenchmark_critic.enabled=${critic_enabled}"
        "hydra.run.dir=${hydra_dir}"
        "experiment.csv_path=${batch_csv}"
    )

    if [ "$CRITIC_ASSET_ANNOTATION" = "true" ]; then
        cmd+=(
            "experiment.scenebenchmark_critic.asset_annotation.enabled=true"
            "experiment.scenebenchmark_critic.asset_annotation.backend=vlm"
            "experiment.scenebenchmark_critic.asset_annotation.model=${MODEL_NAME}"
        )
    fi

    if [ -n "$start_stage_override" ]; then
        cmd+=(
            "experiment.pipeline.start_stage=${start_stage_override}"
            "experiment.pipeline.resume_from_path=${resume_from_path}"
        )
    fi

    "${cmd[@]}"

    echo "批次运行完成: $run_kind / $batch_label"
    echo
}

run_mode() {
    local run_kind="$1"
    local count=0
    local batch_index=0
    local batch_entries=()

    for case_entry in "${CASES[@]}"; do
        IFS="|" read -r case_id critic_goal prompt <<< "$case_entry"
        count=$((count + 1))

        if [ "$MAX_CASES" -gt 0 ] && [ "$count" -gt "$MAX_CASES" ]; then
            echo "已达到 MAX_CASES=$MAX_CASES，停止继续提交新场景。"
            echo
            break
        fi

        batch_entries+=("${count}|${case_id}|${critic_goal}|${prompt}")

        if [ "${#batch_entries[@]}" -ge "$SCENE_BATCH_SIZE" ]; then
            batch_index=$((batch_index + 1))
            run_batch "$run_kind" "$batch_index" "${batch_entries[@]}"
            batch_entries=()
        fi
    done

    if [ "${#batch_entries[@]}" -gt 0 ]; then
        batch_index=$((batch_index + 1))
        run_batch "$run_kind" "$batch_index" "${batch_entries[@]}"
    fi
}

echo "本次内置场景列表："
for case_entry in "${CASES[@]}"; do
    IFS="|" read -r case_id critic_goal prompt <<< "$case_entry"
    echo "- $case_id"
    echo "  关注点: $critic_goal"
    echo "  Prompt: $prompt"
done
echo

if [ "$BRANCH_FROM_SHARED_BASE" = "true" ]; then
    if [ -n "$SHARED_BASE_ROOT" ]; then
        echo "========== 第零部分：复用已有 shared_base =========="
        echo "shared_base 来源: $SHARED_BASE_ROOT"
        echo
    else
        echo "========== 第零部分：生成 shared_base =========="
        echo
        run_mode "shared_base"
    fi
fi

run_both_parallel() {
    local off_log="$OUTPUT_ROOT/critic_off.log"
    local on_log="$OUTPUT_ROOT/critic_on.log"

    echo "========== 并行运行 critic_off / critic_on =========="
    echo "critic_off 日志: $off_log"
    echo "critic_on  日志: $on_log"
    echo

    (
        echo "========== 第一部分：关闭 critic =========="
        echo
        run_mode "critic_off"
    ) > "$off_log" 2>&1 &
    local pid_off=$!

    (
        echo "========== 第二部分：开启 critic =========="
        echo
        run_mode "critic_on"
    ) > "$on_log" 2>&1 &
    local pid_on=$!

    local rc_off=0
    local rc_on=0
    wait $pid_off || rc_off=$?
    wait $pid_on  || rc_on=$?

    echo
    echo "critic_off 返回码: $rc_off"
    echo "critic_on  返回码: $rc_on"

    if [ "$rc_off" -ne 0 ] || [ "$rc_on" -ne 0 ]; then
        echo "错误：并行运行 critic_off / critic_on 时至少一个失败"
        exit 1
    fi
}

if [ "$MODE" = "both" ]; then
    if [ "$CRITIC_PROBE_PARALLEL" = "true" ]; then
        run_both_parallel
    else
        echo "========== 第一部分：关闭 critic =========="
        echo
        run_mode "critic_off"
        echo "========== 第二部分：开启 critic =========="
        echo
        run_mode "critic_on"
    fi
elif [ "$MODE" = "off" ]; then
    echo "========== 第一部分：关闭 critic =========="
    echo
    run_mode "critic_off"
elif [ "$MODE" = "on" ]; then
    echo "========== 第二部分：开启 critic =========="
    echo
    run_mode "critic_on"
fi

echo "=========================================="
echo "全部批跑完成。"
echo "输出根目录: $OUTPUT_ROOT"
echo
echo "建议重点对比："
if [ "$BRANCH_FROM_SHARED_BASE" = "true" ]; then
    echo "1. shared_base 与 critic_on 下，同一 case 的 ${CRITIC_REPORT_STAGE_LABEL} 场景差异。"
    echo "2. 先看各批次目录里的 batch_cases.csv，确认 scene_XXX 对应哪个 case_id。"
    echo "3. 确认 shared_base / critic_on 的 batch_XXX 一一对应。"
    echo "4. 重点查看 critic_on 下的 scenebenchmark_critic.md 和 scenebenchmark_critic.json。"
else
    echo "1. critic_off 与 critic_on 下，同一 case 的 ${CRITIC_REPORT_STAGE_LABEL} 场景差异。"
    echo "2. 先看各批次目录里的 batch_cases.csv，确认 scene_XXX 对应哪个 case_id。"
    echo "3. 重点查看 scenebenchmark_critic.md 和 scenebenchmark_critic.json。"
fi
echo
echo "可用命令示例："
echo "find \"$OUTPUT_ROOT\" -name 'scenebenchmark_critic.md' | sort"
echo "find \"$OUTPUT_ROOT\" -name 'scenebenchmark_critic.json' | sort"
echo "find \"$OUTPUT_ROOT\" -name 'batch_cases.csv' | sort"
echo "=========================================="
