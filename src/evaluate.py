import json, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_model(model, loader, device, class_names, output_dir=None):
    model.eval(); y_true=[]; y_pred=[]
    start=time.perf_counter()
    with torch.no_grad():
        for images, labels in loader:
            outputs=model(images.to(device))
            y_pred.extend(outputs.argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())
    inference_time=time.perf_counter()-start
    cm=confusion_matrix(y_true,y_pred,labels=list(range(len(class_names))))
    report=classification_report(y_true,y_pred,target_names=class_names,zero_division=0)
    metrics={
        "test_accuracy":float(accuracy_score(y_true,y_pred)),
        "precision_macro":float(precision_score(y_true,y_pred,average="macro",zero_division=0)),
        "recall_macro":float(recall_score(y_true,y_pred,average="macro",zero_division=0)),
        "f1_macro":float(f1_score(y_true,y_pred,average="macro",zero_division=0)),
        "inference_time_seconds":float(inference_time),
        "confusion_matrix":cm.tolist(),
    }
    print("\n==========================================\nFINAL TEST RESULTS\n==========================================")
    for key,label in [("test_accuracy","Accuracy"),("precision_macro","Macro Precision"),("recall_macro","Macro Recall"),("f1_macro","Macro F1")]:
        print(f"{label:16}: {metrics[key]*100:.2f}%")
    print(f"{'Inference Time':16}: {inference_time:.2f} seconds")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    if output_dir:
        out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
        (out/"test_metrics.json").write_text(json.dumps(metrics,indent=4),encoding="utf-8")
        (out/"classification_report.txt").write_text(report,encoding="utf-8")
    return metrics, report
