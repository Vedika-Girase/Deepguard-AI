import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# DEEPGUARD - MODEL COMPARISON
# ============================================================

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/comparison")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# RESULT FILES
# ============================================================

MODEL_FILES = {
    "Custom CNN": "cnn_faceforensics_results.json",
    "ResNet18": "resnet18_faceforensics_results.json",
    "ResNet18 Fine-Tuned": "resnet18_finetuned_faceforensics_results.json"
}


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():

    results = []

    for model_name, filename in MODEL_FILES.items():

        filepath = RESULTS_DIR / filename

        if not filepath.exists():
            print(f"WARNING: Result file not found: {filepath}")
            continue

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            results.append({
                "Model": model_name,
                "Validation Accuracy": data.get(
                    "validation_accuracy",
                    data.get(
                        "best_validation_accuracy",
                        data.get(
                            "best_val_accuracy",
                            data.get("val_accuracy", 0)
        )
    )
),
                
                "Test Accuracy": data.get("test_accuracy", 0),
                "Precision": data.get("precision", 0),
                "Recall": data.get("recall", 0),
                "F1 Score": data.get("f1_score", 0),
                "Training Time (s)": data.get(
                    "training_time_seconds",
                    data.get("training_time", 0)
                ),
                "Inference Time (s)": data.get(
                    "inference_time_seconds",
                    data.get("inference_time", 0)
                )
            })

        except Exception as e:
            print(f"ERROR reading {filename}: {e}")

    return results


# ============================================================
# CREATE DATAFRAME
# ============================================================

results = load_results()

if not results:
    print("\nNo model results found.")
    print("Check the files inside the results/ folder.")
    exit()


df = pd.DataFrame(results)


# ============================================================
# CONVERT ACCURACY METRICS TO PERCENTAGE
# ============================================================

metric_columns = [
    "Validation Accuracy",
    "Test Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

for column in metric_columns:
    df[column] = df[column] * 100


# ============================================================
# DISPLAY COMPARISON
# ============================================================

print("\n")
print("=" * 90)
print("DEEPGUARD - MODEL COMPARISON")
print("=" * 90)

print(
    df.to_string(
        index=False,
        formatters={
            "Validation Accuracy": "{:.2f}%".format,
            "Test Accuracy": "{:.2f}%".format,
            "Precision": "{:.2f}%".format,
            "Recall": "{:.2f}%".format,
            "F1 Score": "{:.2f}%".format,
            "Training Time (s)": "{:.2f}".format,
            "Inference Time (s)": "{:.2f}".format
        }
    )
)


# ============================================================
# FIND BEST MODEL
# ============================================================

best_index = df["Test Accuracy"].idxmax()
best_model = df.loc[best_index, "Model"]
best_accuracy = df.loc[best_index, "Test Accuracy"]

print("\n" + "=" * 90)
print("BEST MODEL")
print("=" * 90)

print(f"Model          : {best_model}")
print(f"Test Accuracy  : {best_accuracy:.2f}%")
print(f"F1 Score       : {df.loc[best_index, 'F1 Score']:.2f}%")
print(f"Precision      : {df.loc[best_index, 'Precision']:.2f}%")
print(f"Recall         : {df.loc[best_index, 'Recall']:.2f}%")


# ============================================================
# SAVE CSV
# ============================================================

csv_path = OUTPUT_DIR / "model_comparison.csv"

df.to_csv(csv_path, index=False)

print(f"\nComparison CSV saved:")
print(csv_path)


# ============================================================
# SAVE JSON
# ============================================================

json_path = OUTPUT_DIR / "model_comparison.json"

json_data = df.to_dict(orient="records")

with open(json_path, "w") as f:
    json.dump(json_data, f, indent=4)

print(f"Comparison JSON saved:")
print(json_path)


# ============================================================
# PERFORMANCE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

x = range(len(df))
width = 0.15

metrics_to_plot = [
    "Test Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

for i, metric in enumerate(metrics_to_plot):

    values = df[metric].values

    positions = [
        value + (i - 1.5) * width
        for value in x
    ]

    plt.bar(
        positions,
        values,
        width=width,
        label=metric
    )


plt.xticks(
    list(x),
    df["Model"],
    rotation=15
)

plt.ylabel("Performance (%)")
plt.xlabel("Model")
plt.title("DeepGuard Model Performance Comparison")

plt.ylim(0, 110)
plt.legend()

plt.tight_layout()

graph_path = OUTPUT_DIR / "model_performance_comparison.png"

plt.savefig(graph_path, dpi=300)

plt.close()

print(f"Performance graph saved:")
print(graph_path)


# ============================================================
# TRAINING TIME GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    df["Model"],
    df["Training Time (s)"]
)

plt.ylabel("Training Time (seconds)")
plt.xlabel("Model")
plt.title("Training Time Comparison")

plt.xticks(rotation=15)

plt.tight_layout()

training_graph = OUTPUT_DIR / "training_time_comparison.png"

plt.savefig(training_graph, dpi=300)

plt.close()

print(f"Training time graph saved:")
print(training_graph)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print("COMPARISON COMPLETED")
print("=" * 90)

print("\nGenerated files:")

print("1. results/comparison/model_comparison.csv")
print("2. results/comparison/model_comparison.json")
print("3. results/comparison/model_performance_comparison.png")
print("4. results/comparison/training_time_comparison.png")

print("\nNext research step:")
print("External Dataset Generalization Testing")
print("=========================================")
print("The best model will be tested on an unseen dataset.")