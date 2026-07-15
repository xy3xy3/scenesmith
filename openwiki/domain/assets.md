# 资产获取管线

## 概览

SceneSmith 的资产获取子系统负责将自然语言描述的物体请求转化为仿真就绪的 3D 资产。核心包含两条路径：

- **检索**：从 HSSD/Objaverse/ArtVIP 等预建库中检索现成网格
- **生成**：文生图 (DALL-E/Gemini) → 文生 3D (Hunyuan3D/SAM3D)

所有资产最后都经过规范化（朝向、物理属性）并生成 Drake SDF，供下游仿真使用。

```
Asset Request
    │
    ├─ AssetRouter (VLM 请求分析)
    │   └─ 拆分组合物体 / 过滤无效项 / 选择策略
    │
    ├─ Strategy: "hssd" ──→ HSSD 检索
    │         ├─ 语义搜索: CLIP 或 Zvec 后端
    │         ├─ 尺寸排序 (L1 bbox 距离)
    │         ├─ [可选] VLM 渲染图重排序
    │         └─ VLM 物理分析 (材质、质量、正面)
    │
    ├─ Strategy: "generated" ──→ 文生图 → 文生3D (Hunyuan3D/SAM3D)
    │                              └─ VLM 验证 + 重试
    │
    ├─ Strategy: "articulated" ──→ PartNet-Mobility / ArtVIP 检索
    ├─ Strategy: "objaverse" ──→ Objaverse (ObjectThor) 检索
    └─ Strategy: "thin_covering" ──→ 薄层几何生成 (地毯等)
              │
              ↓
    VLM 验证 (多视角渲染 → GPT-5 检查)
    ↓
    网格规范化 (Z-up, Y-forward 朝向)
    ↓
    碰撞几何 (CoACD / V-HACD 凸分解)
    ↓
    SDF 生成 (Drake 格式 + 物理属性)
    ↓
    SceneObject 注册到 AssetRegistry
```

## HSSD 检索

**首选路径**。两阶段检索：

1. **CLIP 语义搜索**：用物体描述文本 embedding 检索 HSSD 数据集索引
2. **尺寸排序**：按 L1 bbox 距离排序，选最匹配尺寸的资产

VLM 在此流程中参与多个筛选环节：
- 渲染图视觉排序（多视角渲染 → GPT-5 评分）
- 资产验证（VLM 检查资产是否符合语义）
- 物理属性分析（材质、质量、朝向）

## SAM3D / Hunyuan3D-2 生成

| 后端 | GPU 内存 | 质量 | 推荐场景 |
|------|---------|------|----------|
| SAM3D | 32GB+ | 高 | 论文使用，生产级 |
| Hunyuan3D-2 | 24GB | 低 | 验证/原型 |

配置方式：
```yaml
asset_manager:
  backend: "sam3d"        # 或 "hunyuan3d"
```

## 铰接物体检索

支持两个来源：
- **ArtVIP**（推荐）：预处理完成，含 VHACD 和 CoACD 两种碰撞几何变体
- **PartNet-Mobility**（质量低）：需自行转换 SDF

使用 CLIP embedding 进行语义匹配：
```yaml
asset_manager:
  articulated:
    use_top_k: 5
    sources:
      artvip:
        enabled: true
        data_path: data/artvip_sdf
      partnet_mobility:
        enabled: false
```

## 材质系统

PBR 材质来源：[AmbientCG](https://ambientcg.com/)（CC0 协议）

- 下载：`python scripts/download_ambientcg.py --output data/materials`
- CLIP embedding 预计算：用于材质检索匹配
- 通过 `MaterialsRetrievalServer` 在运行时检索

## AssetRouter

**文件**: `scenesmith/agent_utils/asset_router/`

VLM 驱动的智能资产路由：
- 分析物体请求，拆分组合物体（如 "带抽屉的书桌"）
- 过滤无效/不合理的物体请求
- 选择最佳策略（检索 vs 生成 vs 简单几何）

## 资产规范化

所有资产最终经过：
1. **朝向对齐**：Z-up, Y-forward 标准朝向
2. **碰撞几何**：CoACD / V-HACD 凸分解
3. **SDF 生成**：Drake SDFormat + 物理属性（质量、惯性）
4. **注册**：`AssetRegistry` 管理资产生命周期

## 关键源文件

| 文件 | 大小 | 职责 |
|------|------|------|
| `agent_utils/asset_manager.py` | 101KB | 资产管理核心 |
| `agent_utils/asset_registry.py` | 9KB | 资产注册表 |
| `agent_utils/asset_router/` | - | VLM 路由分析 |
| `agent_utils/mesh_utils.py` | 30KB | 网格处理工具 |
| `agent_utils/mesh_canonicalization.py` | 3KB | 网格朝向标准化 |
| `agent_utils/sdf_generator.py` | 19KB | SDF 生成 |
| `agent_utils/mesh_physics_analyzer.py` | 14KB | 网格物理分析 |
| `agent_utils/convex_decomposition_server/` | - | 凸分解服务 |
| `agent_utils/hssd_retrieval_server/` | - | HSSD 检索服务 |
| `agent_utils/objaverse_retrieval_server/` | - | Objaverse 检索服务 |
| `agent_utils/articulated_retrieval_server/` | - | 铰接物体检索 |
| `agent_utils/materials_retrieval_server/` | - | 材质检索 |
| `agent_utils/support_surface_extraction.py` | 54KB | 支撑面提取 |
| `agent_utils/thin_covering_generator.py` | 29KB | 薄层几何生成 |
