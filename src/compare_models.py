import csv, json
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_ROOT=Path("results")
RUNS_DIR=RESULTS_ROOT/"runs"
OUTPUT_DIR=RESULTS_ROOT/"comparison"
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)


def discover_records():
    records=[]
    for p in RUNS_DIR.rglob("experiment_record.json") if RUNS_DIR.exists() else []:
        try: records.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception: pass
    # Also read legacy standardized records so old research results remain comparable.
    for p in RESULTS_ROOT.rglob("*.json"):
        if OUTPUT_DIR in p.parents or "runs" in p.parts: continue
        try:
            r=json.loads(p.read_text(encoding="utf-8"))
            if isinstance(r,dict) and "run_id" in r: records.append(r)
        except Exception: pass
    by={r["run_id"]:r for r in records}
    return list(by.values())


def save_registry(records):
    json_path=OUTPUT_DIR/"experiment_records.json"; csv_path=OUTPUT_DIR/"experiment_records.csv"
    json_path.write_text(json.dumps(records,indent=4),encoding="utf-8")
    fields=["run_id","model","tag","epochs","epochs_completed","best_epoch","best_validation_accuracy","test_accuracy","precision_macro","recall_macro","f1_macro","training_time_seconds","inference_time_seconds","learning_rate","weight_decay","dropout","optimizer","pretrained"]
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(records)


def bar(records,key,title,ylabel,percent=False,filename="chart.png"):
    rs=[r for r in records if r.get(key) is not None]
    if not rs:return
    labels=[r["run_id"] for r in rs]; vals=[float(r[key])*(100 if percent else 1) for r in rs]
    plt.figure(figsize=(max(10,len(labels)*1.3),6)); plt.bar(range(len(labels)),vals); plt.xticks(range(len(labels)),labels,rotation=45,ha="right"); plt.ylabel(ylabel); plt.title(title); plt.grid(axis="y",alpha=.25); plt.tight_layout(); plt.savefig(OUTPUT_DIR/filename,dpi=180); plt.close()


def grouped_metrics(records):
    rs=[r for r in records if r.get("test_accuracy") is not None]
    if not rs:return
    labels=[r["run_id"] for r in rs]; x=range(len(labels)); width=.2
    plt.figure(figsize=(max(10,len(labels)*1.5),6))
    for i,(key,label) in enumerate([("test_accuracy","Accuracy"),("precision_macro","Precision"),("recall_macro","Recall"),("f1_macro","F1")]):
        vals=[float(r.get(key,0))*100 for r in rs]; plt.bar([v+(i-1.5)*width for v in x],vals,width,label=label)
    plt.xticks(list(x),labels,rotation=45,ha="right"); plt.ylabel("Score (%)"); plt.ylim(0,100); plt.title("Model Test Metrics Comparison"); plt.legend(); plt.grid(axis="y",alpha=.25); plt.tight_layout(); plt.savefig(OUTPUT_DIR/"test_metrics_comparison.png",dpi=180); plt.close()


def scatter_val_test(records):
    rs=[r for r in records if r.get("best_validation_accuracy") is not None and r.get("test_accuracy") is not None]
    if not rs:return
    plt.figure(figsize=(8,6)); x=[r["best_validation_accuracy"]*100 for r in rs]; y=[r["test_accuracy"]*100 for r in rs]
    plt.scatter(x,y)
    for xi,yi,r in zip(x,y,rs): plt.annotate(r["run_id"],(xi,yi),fontsize=8,xytext=(4,4),textcoords="offset points")
    plt.xlabel("Best validation accuracy (%)"); plt.ylabel("Test accuracy (%)"); plt.title("Validation vs Test Generalization"); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(OUTPUT_DIR/"validation_vs_test_accuracy.png",dpi=180); plt.close()


def main():
    records=discover_records()
    if not records: print("No experiment records found."); return
    records.sort(key=lambda r:r.get("run_id","")); save_registry(records)
    bar(records,"best_validation_accuracy","Best Validation Accuracy Comparison","Validation accuracy (%)",True,"validation_accuracy_comparison.png")
    bar(records,"test_accuracy","Test Accuracy Comparison","Test accuracy (%)",True,"test_accuracy_comparison.png")
    grouped_metrics(records)
    bar(records,"training_time_seconds","Training Time Comparison","Training time (seconds)",False,"training_time_comparison.png")
    bar(records,"inference_time_seconds","Inference Time Comparison","Inference time (seconds)",False,"inference_time_comparison.png")
    scatter_val_test(records)
    print(f"\nCompared {len(records)} experiment records.")
    print("Saved comparison registry and graphs in:",OUTPUT_DIR)

if __name__=="__main__": main()
