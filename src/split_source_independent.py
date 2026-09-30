from collections import Counter,defaultdict
from pathlib import Path
import shutil
SOURCE_DIR=Path("data/processed/dataset")
OUTPUT_DIR=Path("data/processed/source_independent")
SPLIT_IDS={"train":{"183","253","469","481","585","599"},"validation":{"672","720"},"test":{"866","878"}}

def get_source_id(filename): return Path(filename).stem.split("_")[0]
def main():
    images=[p for p in SOURCE_DIR.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
    grouped=defaultdict(list)
    for p in images: grouped[get_source_id(p.name)].append(p)
    found=set(grouped); assigned=set().union(*SPLIT_IDS.values())
    if found-assigned: raise ValueError(f"Unassigned source IDs: {sorted(found-assigned)}")
    if assigned-found: raise ValueError(f"Configured source IDs missing from dataset: {sorted(assigned-found)}")
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    counts=Counter()
    for sid,items in grouped.items():
        split=next(k for k,v in SPLIT_IDS.items() if sid in v)
        for p in items:
            cls=p.parent.name
            if cls not in {"deepfake","real"}: raise ValueError(f"Unexpected class directory: {cls}")
            dst=OUTPUT_DIR/split/cls; dst.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dst/p.name); counts[(split,cls)]+=1
    print("\nSOURCE-INDEPENDENT DATASET")
    for split in ("train","validation","test"): print(f"{split}: deepfake={counts[(split,'deepfake')]} real={counts[(split,'real')]} total={counts[(split,'deepfake')]+counts[(split,'real')]}")
    print("Total:",sum(counts.values())); print("Source assignment:", {k:sorted(v,key=int) for k,v in SPLIT_IDS.items()}); print("Original dataset was not modified.")
if __name__=="__main__": main()
