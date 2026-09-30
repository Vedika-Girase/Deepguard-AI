from pathlib import Path
IMAGE_EXTENSIONS={".jpg",".jpeg",".png"}

def get_source_id(filename): return Path(filename).stem.split("_")[0]
def collect_ids(root):
    root=Path(root); result={}
    for split in ("train","validation","test"):
        ids=set(); d=root/split
        if d.exists():
            for f in d.rglob("*"):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS: ids.add(get_source_id(f.name))
        result[split]=ids
    return result

def audit(root="data/processed/source_independent"):
    ids=collect_ids(root); pairs=(("train","validation"),("train","test"),("validation","test")); passed=True
    print("\n==========================================\nDEEPGUARD - SOURCE LEAKAGE AUDIT\n==========================================")
    for s,v in ids.items(): print(f"{s:12}: {len(v)} source IDs -> {sorted(v,key=lambda x:int(x) if x.isdigit() else x)}")
    for a,b in pairs:
        overlap=ids[a]&ids[b]; print(f"{a} ∩ {b}: {len(overlap)} -> {sorted(overlap)}"); passed &= not overlap
    print("\nSTATUS:","PASS" if passed else "FAIL")
    return passed
if __name__=="__main__": audit()
