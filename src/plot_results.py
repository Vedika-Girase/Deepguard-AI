import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def _history(record):
    return record.get("training_history", [])


def plot_training_history(record, output_dir):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); hist=_history(record)
    if not hist: return
    e=[r["epoch"] for r in hist]
    train_loss=[r["train_loss"] for r in hist]; val_loss=[r["validation_loss"] for r in hist]
    train_acc=[r["train_accuracy"]*100 for r in hist]; val_acc=[r["validation_accuracy"]*100 for r in hist]
    lr=[r.get("learning_rate") for r in hist]
    gap=[r.get("generalization_gap") for r in hist]

    plt.figure(figsize=(9,5)); plt.plot(e,train_loss,label="Train Loss"); plt.plot(e,val_loss,label="Validation Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{record['run_id']} — Loss"); plt.legend(); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out/"loss_curve.png",dpi=180); plt.close()
    plt.figure(figsize=(9,5)); plt.plot(e,train_acc,label="Train Accuracy"); plt.plot(e,val_acc,label="Validation Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title(f"{record['run_id']} — Accuracy"); plt.legend(); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out/"accuracy_curve.png",dpi=180); plt.close()
    if any(v is not None for v in lr):
        plt.figure(figsize=(9,5)); plt.plot(e,lr); plt.xlabel("Epoch"); plt.ylabel("Learning rate"); plt.title(f"{record['run_id']} — Learning Rate"); plt.yscale("log"); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out/"learning_rate_curve.png",dpi=180); plt.close()
    plt.figure(figsize=(9,5)); plt.plot(e,[v*100 for v in gap]); plt.axhline(0,linewidth=1); plt.xlabel("Epoch"); plt.ylabel("Train accuracy − validation accuracy (percentage points)"); plt.title(f"{record['run_id']} — Generalization Gap"); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out/"generalization_gap.png",dpi=180); plt.close()


def plot_confusion_matrix(record, output_dir):
    cm=np.array(record.get("confusion_matrix",[]))
    if cm.size==0:return
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    names=list(record.get("class_mapping",{}).keys()) or ["Class 0","Class 1"]
    plt.figure(figsize=(6,5)); plt.imshow(cm, interpolation="nearest"); plt.title(f"{record['run_id']} — Confusion Matrix"); plt.colorbar(); ticks=np.arange(len(names)); plt.xticks(ticks,names,rotation=20); plt.yticks(ticks,names); plt.xlabel("Predicted"); plt.ylabel("Actual")
    threshold=cm.max()/2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): plt.text(j,i,str(cm[i,j]),ha="center",va="center")
    plt.tight_layout(); plt.savefig(out/"confusion_matrix.png",dpi=180); plt.close()


def plot_metric_summary(record, output_dir):
    keys=[("test_accuracy","Accuracy"),("precision_macro","Macro Precision"),("recall_macro","Macro Recall"),("f1_macro","Macro F1")]
    vals=[record.get(k,0)*100 for k,_ in keys]
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(8,5)); plt.bar([x[1] for x in keys],vals); plt.ylim(0,100); plt.ylabel("Score (%)"); plt.title(f"{record['run_id']} — Test Metrics"); plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(out/"test_metrics.png",dpi=180); plt.close()


def generate_run_plots(record, output_dir):
    plot_training_history(record,output_dir); plot_confusion_matrix(record,output_dir); plot_metric_summary(record,output_dir)


def generate_from_json(path):
    p=Path(path); record=json.loads(p.read_text(encoding="utf-8")); generate_run_plots(record,p.parent)
