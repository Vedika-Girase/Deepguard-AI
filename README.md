# DeepGuard-AI

DeepGuard-AI is a computer-vision research project for classifying real and
synthetically manipulated face images and video frames. The project currently
contains dataset preparation utilities, CNN and ResNet18 training scripts,
evaluation reports, and model-comparison results.

> **Project status:** Work in progress. The repository is being developed
> incrementally, so training configurations, datasets, and application
> interfaces may change.

## Contents

- [`src/`](src/) - dataset preparation, training, evaluation, and analysis scripts
- [`results/`](results/) - generated metrics, reports, and comparison plots
- [`data/`](data/) - local datasets and processed images (not committed)
- [`models/`](models/) - trained model checkpoints (not committed)
- [`notebooks/`](notebooks/) - exploratory notebooks
- [`api/`](api/) - API work area
- [`dashboard/`](dashboard/) - dashboard work area

## Current results

The checked-in comparison artifacts include the following experiment results:

| Model | Validation accuracy | Test accuracy | F1 score |
| --- | ---: | ---: | ---: |
| Custom CNN | 63.91% | 0.00% | 0.00% |
| ResNet18 | 65.75% | 60.81% | 60.27% |
| ResNet18 fine-tuned | 100.00% | 100.00% | 100.00% |

These values are historical experiment outputs, not a production benchmark.
They should be revalidated with a fixed dataset split and an independent test
set before drawing conclusions about generalization.

## Requirements

- Python 3.10 or newer
- PyTorch and torchvision
- scikit-learn
- NumPy
- Pillow
- Matplotlib

The scripts are designed to use CUDA when it is available and otherwise fall
back to the CPU.

## Setup

Create and activate a virtual environment, then install the dependencies used
by the project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision scikit-learn numpy pillow matplotlib
```

If you use a CUDA-enabled PyTorch installation, install the matching PyTorch
build from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Dataset layout

The training scripts use `torchvision.datasets.ImageFolder`. Prepare the local
dataset with this structure:

```text
data/
└── processed/
    └── dataset/
        ├── train/
        │   ├── real/
        │   └── fake/
        ├── validation/
        │   ├── real/
        │   └── fake/
        └── test/
            ├── real/
            └── fake/
```

The exact class directory names must match the labels used by the dataset
preparation scripts. Large datasets are intentionally excluded from Git by
`.gitignore`.

## Typical workflow

Run commands from the repository root:

```powershell
# Inspect or validate the dataset
python src\check_data_leakage.py
python src\check_video_dataset.py

# Prepare frames, faces, and dataset splits as needed
python src\extract_frames.py
python src\extract_faces.py
python src\process_faces.py
python src\split_dataset.py

# Train models
python src\train_cnn.py
python src\train_resnet.py
python src\train_resnet_finetuned.py

# Compare generated results
python src\compare_models.py
```

Some scripts target specific experiments or source-independent/video-level
splits. Review the configuration section at the top of each script before
running it, especially the input paths, output paths, number of epochs, and
batch size.

## Outputs

Training scripts write checkpoints to `models/` and evaluation artifacts to
`results/`. The repository includes lightweight result summaries and plots,
but generated datasets and model weights remain local by default.

## Reproducibility and limitations

- The custom CNN and ResNet scripts use fixed seeds in their experiments, but
  results can still vary by hardware, library versions, and data order.
- Dataset quality, source leakage, class balance, and the selected split can
  materially affect reported metrics.
- A high score on one split does not establish robustness to unseen
  manipulation methods or real-world video compression.
- Use the data-leakage and source-independent checks before comparing models.

## Development workflow

The project is developed incrementally. Start new work from `develop`, create a
focused branch, and open a pull request for review:

```powershell
git switch develop
git pull --ff-only
git switch -c feature/your-change
```

Do not commit datasets, virtual environments, secrets, or generated model
checkpoints unless a future change explicitly introduces a versioned artifact
policy.

## License

No license has been selected for this project yet. Until a license is added,
please obtain permission before redistributing or reusing the code or results.
