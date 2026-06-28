"""Build functional_partners_index.json from the UD4/funeval source layer.

Source: /data/.../ud4_funeval_library/asset_annotations (funeval@1.0), per-asset
ud4_extension.functional_dependencies (relations used_with + requires, conf>=0.5),
keyed by HSSD hash. Output: {hash: {cat, partners[]}} consumed by clearance_source
for partner-exclusion. Re-run only when the source annotation layer changes.
"""
import json, glob, re, collections
base="/data/250010098/ud4_funeval_library/asset_annotations"
idx=base+"/_index.jsonl"
OUT="/data/250010098/scenesmith_fork_yz/scenesmith/scenebenchmark_critic/clearance_data/functional_partners_index.json"

def strip_synset(s):  # "dining_table.n.01" -> "dining_table"
    return re.sub(r"\.n\.\d+$","",str(s or "")).lower()

REL_KEEP={"used_with","requires"}  # functional adjacency (not placed_on=vertical support)
CONF_MIN=0.5

items={}
n=0
with open(idx) as f:
    for line in f:
        r=json.loads(line)
        if r.get("source")!="hssd": continue
        p=base+"/"+r["path"].split("asset_annotations/")[-1]
        try: d=json.load(open(p))
        except: continue
        fd=d.get("ud4_extension",{}).get("functional_dependencies") or []
        partners=sorted({strip_synset(x.get("target_category")) for x in fd
                         if x.get("relation") in REL_KEEP and (x.get("confidence") or 0)>=CONF_MIN})
        items[r["source_id"]]={"cat": strip_synset(r.get("category")), "partners": partners}
        n+=1
meta={"src":"funeval@1.0 functional_dependencies","rel":sorted(REL_KEEP),"conf_min":CONF_MIN,"count":len(items)}
json.dump({"meta":meta,"items":items}, open(OUT,"w"), ensure_ascii=False)
print("wrote",OUT,"items=",len(items))

# coverage vs clearance indices
na=json.load(open("/data/250010098/scenesmith_fork_yz/scenesmith/scenebenchmark_critic/clearance_data/nonartic_clearance_index.json"))["items"]
ar=json.load(open("/data/250010098/scenesmith_fork_yz/scenesmith/scenebenchmark_critic/clearance_data/artic_clearance_index.json"))["items"]
cl=set(na)|set(ar)
cov=len(cl & set(items))
print(f"clearance hashes={len(cl)}, covered by partner index={cov} ({100*cov/len(cl):.0f}%)")

# sanity: partners for a few categories
for cat in ["chair","armchair","sofa","table","desk","nightstand","bed"]:
    ex=[v for k,v in items.items() if v["cat"]==cat]
    if ex:
        print(f"  {cat}: e.g. partners={ex[0]['partners'][:6]}")
