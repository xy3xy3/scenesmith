"""Build functional_partners_index.json from the UD4/funeval source layer.

Source: /data/.../ud4_funeval_library/asset_annotations (funeval@1.0), per-asset
ud4_extension.functional_dependencies (relations used_with + requires, conf>=0.5),
keyed by HSSD hash. Output: {hash: {cat, partners[]}} consumed by clearance_source
for partner-exclusion. Re-run only when the source annotation layer changes.
"""
import json, glob, re, collections
base="/data/250010098/ud4_funeval_library/asset_annotations"
idx=base+"/_index.jsonl"
OUT="/data/250010098/scenesmith_fork_yz/scenesmith/scenebenchmark_critic/metrics/interaction_clearance/data/functional_partners_index.json"

def strip_synset(s):  # "dining_table.n.01" -> "dining_table"
    return re.sub(r"\.n\.\d+$","",str(s or "")).lower()

REL_KEEP={"used_with","requires"}  # functional adjacency (not placed_on=vertical support)
CONF_MIN=0.5

def main() -> None:
    # 2026-07-16 修改原因：迁移后的数据构建脚本会被 registry 的模块导入
    # 扫描发现；必须只在显式执行脚本时访问外部标注目录。
    items = {}
    with open(idx) as f:
        for line in f:
            r = json.loads(line)
            if r.get("source") != "hssd":
                continue
            p = base + "/" + r["path"].split("asset_annotations/")[-1]
            try:
                d = json.load(open(p))
            except Exception:
                continue
            fd = d.get("ud4_extension", {}).get("functional_dependencies") or []
            partners = sorted(
                {
                    strip_synset(x.get("target_category"))
                    for x in fd
                    if x.get("relation") in REL_KEEP
                    and (x.get("confidence") or 0) >= CONF_MIN
                }
            )
            items[r["source_id"]] = {
                "cat": strip_synset(r.get("category")),
                "partners": partners,
            }
    meta = {
        "src": "funeval@1.0 functional_dependencies",
        "rel": sorted(REL_KEEP),
        "conf_min": CONF_MIN,
        "count": len(items),
    }
    json.dump(
        {"meta": meta, "items": items},
        open(OUT, "w"),
        ensure_ascii=False,
    )
    print("wrote", OUT, "items=", len(items))

    na = json.load(
        open(
            "/data/250010098/scenesmith_fork_yz/scenesmith/"
            "scenebenchmark_critic/metrics/interaction_clearance/data/"
            "nonartic_clearance_index.json"
        )
    )["items"]
    ar = json.load(
        open(
            "/data/250010098/scenesmith_fork_yz/scenesmith/"
            "scenebenchmark_critic/metrics/interaction_clearance/data/"
            "artic_clearance_index.json"
        )
    )["items"]
    clearance_hashes = set(na) | set(ar)
    covered = len(clearance_hashes & set(items))
    print(
        f"clearance hashes={len(clearance_hashes)}, "
        f"covered by partner index={covered} "
        f"({100 * covered / len(clearance_hashes):.0f}%)"
    )

    for cat in ["chair", "armchair", "sofa", "table", "desk", "nightstand", "bed"]:
        examples = [value for value in items.values() if value["cat"] == cat]
        if examples:
            print(f"  {cat}: e.g. partners={examples[0]['partners'][:6]}")


if __name__ == "__main__":
    main()
