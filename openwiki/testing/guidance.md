# 测试指南

## 测试结构

```
tests/
├── __init__.py
├── conftest.py              # 共享 fixture（bpy 导入顺序处理）
├── unit/                    # 单元测试（~95 个文件）
│   ├── __init__.py
│   ├── mock_utils.py        # Mock 工具
│   ├── test_*.py            # 按模块命名的测试文件
│   └── ...
└── integration/             # 集成测试（12 个文件）
    ├── __init__.py
    ├── common.py            # 集成测试共享代码
    └── test_*.py
```

## 运行测试

### 基础命令

```bash
# 需要激活虚拟环境（bpy 依赖）
source .venv/bin/activate

# 运行所有单元测试
pytest tests/unit/

# 遇到第一个失败停止
pytest tests/unit/ -x

# 详细输出
pytest tests/unit/ -v

# 运行特定测试文件
pytest tests/unit/test_scenebenchmark_critic.py -x -v

# 按名称匹配
pytest tests/unit/ -k "critic"

# 集成测试
pytest tests/integration/ -x

# 增量测试（testmon）
pytest tests/ --testmon
```

### Docker 中运行

```bash
docker compose run --rm scenesmith pytest tests/unit/ -x
```

## Conftest 注意事项

**文件**: `tests/conftest.py` (1.3KB)

```python
# bpy 必须在所有测试之前导入
# 否则 Drake OpenGL 初始化会冲突导致 segfault
import bpy  # noqa: F401
```

`conftest.py` 同时处理：
- **每测试后 GC**：`pytest_runtest_teardown` 强制垃圾回收，清理 Drake C++ 对象
- **会话结束 GC**：`pytest_sessionfinish` 在全部测试完成后再次 GC，防止 Drake leak detector 挂起

## Mock 工具

**文件**: `tests/unit/mock_utils.py` (<1KB)

提供轻量 mock 辅助，主要用于隔离 LLM 调用和文件系统依赖。

## 关键测试文件分组

### Critic 测试

| 文件 | 大小 | 测试内容 |
|------|------|----------|
| `test_scenebenchmark_critic.py` | **262KB** | 核心 critic 套件 — 最大测试文件 |
| `test_manipuland_completeness.py` | 23KB | Manipuland 完整性检查 |
| `test_dining_seat_distribution.py` | 3KB | 餐桌座椅分布 |
| `test_room_center_alignment.py` | 3KB | 房间中心对齐 |
| `test_window_critic.py` | 16KB | 窗口约束验证 |
| `test_clearance.py` | 15KB | 净空检查 |

### Agent 测试

| 文件 | 大小 | 测试内容 |
|------|------|----------|
| `test_furniture_agent_tools.py` | 56KB | Furniture agent 工具 |
| `test_floor_plan_tools.py` | 42KB | Floor plan 工具 |
| `test_manipuland_tools.py` | 27KB | Manipuland 工具 |
| `test_furniture_checkpoint.py` | 16KB | 家具 checkpoint |
| `test_stateful_workflow_limits.py` | 8KB | 工作流限制 (2026-07-10 新增) |
| `test_workflow_tools.py` | 1KB | 工作流 TODO 工具 (2026-07-10 新增) |

### 物理验证测试

| 文件 | 大小 | 测试内容 |
|------|------|----------|
| `test_physics_validation.py` | 59KB | 物理验证全套 |
| `test_physical_feasibility.py` | 62KB | 物理后处理 |
| `test_pile.py` | 12KB | 物体堆叠 |
| `test_penetration_resolution.py` | 26KB | 穿透解析 |

### 资产生成与检索

| 文件 | 大小 | 测试内容 |
|------|------|----------|
| `test_asset_manager.py` | 49KB | 资产管理器 |
| `test_asset_generation.py` | 18KB | 资产生成管线 |
| `test_asset_router.py` | 29KB | 资产路由 |
| `test_hssd_retrieval.py` | 22KB | HSSD 检索 |
| `test_sam3d_generation.py` | 9KB | SAM3D 生成 |

### 场景与渲染

| 文件 | 大小 | 测试内容 |
|------|------|----------|
| `test_scene.py` | 43KB | 场景核心逻辑 |
| `test_blender.py` | 40KB | Blender 操作 |
| `test_room_placement.py` | 32KB | 房间物体放置 |
| `test_scene_tools.py` | 21KB | 场景工具 |

### Prompts 测试

| 文件 | 大小 | 测试内容 |
|------|------|----------|
| `test_prompts.py` | 21KB | Prompt 注册与加载 |
| `test_wall_prompt_requirements.py` | 4KB | 墙面 prompt 约束 |

## 测试模式与最佳实践

### 1. 确定性测试优先

由于 SceneSmith 涉及大量 LLM 调用，单元测试应避免实际 API 调用。Mock LLM 响应 + 确定性规则检查是最佳实践。

### 2. Critic 测试模式

```python
def test_dining_seat_distribution():
    # 构造特定场景的 CasePack
    case_pack = create_test_case_pack(...)
    # 调用评估函数
    results = evaluate_dining_seat_distribution(case_pack)
    # 断言规则触发/不触发
    assert any(r["rule"] == "dining_seat_distribution" for r in results)
```

### 3. 物理测试模式

使用已知几何体（立方体、球体）进行碰撞检测验证：

```python
def test_floor_penetration():
    scene = create_test_scene()
    violations = check_physics_violations(scene)
    assert len(violations) == 0
```

### 4. 回归测试

每次修复 critic 规则时，应同时添加或更新对应测试用例。Git 历史显示 critic 规则修复和测试更新通常在同一 commit 中。

### 5. 测试 critic 优先

根据 AGENTS.md，优先在 `scenebenchmark_critic/` 模块内修改规则。仅在规则不满足时才修改其他 agents 的 tool/prompt。

## 编写新测试的指南

1. **定位**：`tests/unit/test_{module_name}.py`
2. **引入 bpy**：确保测试文件中 bpy 在 Drake 之前导入
3. **使用 conftest**：共享 fixture 放在 `conftest.py` 中
4. **避免 API 调用**：使用 mock 替代真实 LLM 调用
5. **测试 critic 规则**：构造最小场景验证特定规则
6. **命名**：`test_<功能>_<条件>()`
7. **断言**：使用标准 `assert`，辅以清晰错误消息

## 持续集成

- `.github/workflows/openwiki-update.yml`：OpenWiki 文档自动更新工作流
- `pre-commit`：代码格式化（black, isort, autoflake）

## 测试配置

**文件**: `pyproject.toml`

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
```
