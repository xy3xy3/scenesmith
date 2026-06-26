#!/usr/bin/env bash
# 批量运行单房间场景，对比开启/关闭 SceneBenchmark critic 的效果。
# 设计目标：
# 1. 复用当前项目里常用的单房间、串行、HSSD、禁用 articulated 资源的配置。
# 2. 内置一组更容易触发 spatial_accessibility / functional_dependency 提示的英文 prompt。
# 3. 默认分别跑 critic=off 与 critic=on，方便直接对照生成结果和评测报告。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 运行模式：
#   both: 先跑 critic_off，再跑 critic_on（默认）
#   off : 只跑关闭 critic
#   on  : 只跑开启 critic
MODE="${1:-both}"

case "$MODE" in
    both|off|on)
        ;;
    *)
        echo "用法: $0 [both|off|on]"
        echo "示例:"
        echo "  $0"
        echo "  $0 off"
        echo "  MAX_CASES=2 $0 on"
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

MODEL_NAME="${MODEL_NAME:-Qwen3.6-27B-Q4_K_M}"
EXPERIMENT_NAME_PREFIX="${EXPERIMENT_NAME_PREFIX:-single_room_critic_probe}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/critic_probe/$RUN_ID}"
MAX_CASES="${MAX_CASES:-0}"
SCENE_BATCH_SIZE="${SCENE_BATCH_SIZE:-1}"

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

if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "错误：未找到虚拟环境 $PROJECT_ROOT/.venv"
    exit 1
fi

# shellcheck disable=SC1091
source "$PROJECT_ROOT/.venv/bin/activate"

echo "=========================================="
echo "单房间 critic 对照批跑"
echo "项目目录: $PROJECT_ROOT"
echo "输出根目录: $OUTPUT_ROOT"
echo "运行模式: $MODE"
echo "模型名: $MODEL_NAME"
echo "MAX_CASES: $MAX_CASES (0 表示不限制)"
echo "SCENE_BATCH_SIZE: $SCENE_BATCH_SIZE"
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
    "tight_bedroom_bedside|床-床头柜关系 + 过度拥挤可达性|A very small bedroom with a bed squeezed into a corner, only one nightstand, a large wardrobe crowding the bed, a narrow desk with 4 chairs, and a mug on the desk."
    "cramped_dining_room|餐桌-座椅关系 + 餐区可达性|A tiny dining room with a small dining table, 6 dining chairs packed tightly around it, a sideboard, a table lamp on the sideboard, and a bowl on the table."
    "living_room_bad_media_view|座位-电视关系 + 客厅拥挤|A cramped living room with a sofa turned away from the television, an armchair, a coffee table, a side table, a table lamp, a remote, and a mug, with the seating packed very tightly around the coffee table."
    "narrow_home_office|座椅-工作台关系 + 工作区通行|A narrow home office with a desk, an office chair, 3 extra chairs, a bookshelf, a table lamp, a laptop, a monitor, a keyboard, a mug, and a stack of books, with almost no free walking space between the furniture."
    "studio_with_floor_objects|小物体支撑关系 + 混合功能拥挤|A cramped studio room with a bed, a desk, a chair, a sofa, a television, a coffee table, a mug on the floor, a book on the floor, and a remote on the floor."
    "packed_bedroom_many_objects|床边可达性 + 多个小物体支撑关系|A tiny bedroom with a bed, two nightstands, a dresser, a desk with 3 chairs, a desk lamp, 2 mugs, 2 books, and 2 plants, with the furniture packed tightly together and barely any walking clearance around the bed."
)

BASE_ARGS=(
    "experiment.tasks=[generate_scenes,evaluate_scenes]"
    "experiment.num_workers=${SCENE_BATCH_SIZE}"
    "experiment.pipeline.parallel_rooms=false"
    "experiment.pipeline.max_parallel_rooms=1"
    "experiment.materials_retrieval_server.enabled=false"
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
    "experiment.scenebenchmark_critic.fd_relation_proposer_mode=template"
)

csv_quote() {
    local value="$1"
    value=${value//\"/\"\"}
    printf '"%s"' "$value"
}

run_batch() {
    local critic_mode="$1"
    local batch_index="$2"
    shift 2
    local batch_entries=("$@")

    local critic_enabled="false"
    if [ "$critic_mode" = "critic_on" ]; then
        critic_enabled="true"
    fi

    local batch_label
    batch_label=$(printf "batch_%03d" "$batch_index")
    local hydra_dir="$OUTPUT_ROOT/$critic_mode/$batch_label"
    local exp_name="${EXPERIMENT_NAME_PREFIX}_${critic_mode}_${batch_label}"
    local batch_csv="$hydra_dir/batch_cases.csv"
    local case_summary=()

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
    echo "开始运行批次: $critic_mode / $batch_label"
    echo "批次场景数: ${#batch_entries[@]}"
    echo "批次映射: ${case_summary[*]}"
    echo "输出目录: $hydra_dir"
    echo "批次清单: $batch_csv"
    echo "critic 开关: $critic_enabled"
    echo "------------------------------------------"

    python main.py \
        "+name=${exp_name}" \
        "${BASE_ARGS[@]}" \
        "experiment.scenebenchmark_critic.enabled=${critic_enabled}" \
        "hydra.run.dir=${hydra_dir}" \
        "experiment.csv_path=${batch_csv}"

    echo "批次运行完成: $critic_mode / $batch_label"
    echo
}

run_mode() {
    local critic_mode="$1"
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
            run_batch "$critic_mode" "$batch_index" "${batch_entries[@]}"
            batch_entries=()
        fi
    done

    if [ "${#batch_entries[@]}" -gt 0 ]; then
        batch_index=$((batch_index + 1))
        run_batch "$critic_mode" "$batch_index" "${batch_entries[@]}"
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

if [ "$MODE" = "both" ] || [ "$MODE" = "off" ]; then
    echo "========== 第一部分：关闭 critic =========="
    echo
    run_mode "critic_off"
fi

if [ "$MODE" = "both" ] || [ "$MODE" = "on" ]; then
    echo "========== 第二部分：开启 critic =========="
    echo
    run_mode "critic_on"
fi

echo "=========================================="
echo "全部批跑完成。"
echo "输出根目录: $OUTPUT_ROOT"
echo
echo "建议重点对比："
echo "1. critic_off 与 critic_on 下，同一 case 的最终场景差异。"
echo "2. 先看各批次目录里的 batch_cases.csv，确认 scene_XXX 对应哪个 case_id。"
echo "3. critic_on 下各房间的 scene_after_furniture / final_scene 报告。"
echo "4. 重点查看 scenebenchmark_critic.md 和 scenebenchmark_critic.json。"
echo
echo "可用命令示例："
echo "find \"$OUTPUT_ROOT\" -name 'scenebenchmark_critic.md' | sort"
echo "find \"$OUTPUT_ROOT\" -name 'scenebenchmark_critic.json' | sort"
echo "find \"$OUTPUT_ROOT\" -name 'batch_cases.csv' | sort"
echo "=========================================="
