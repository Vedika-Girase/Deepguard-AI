# DeepGuard experiment log

The project uses source-video-independent evaluation as the main research protocol.

| Run | Epochs | Best validation accuracy | Test accuracy | Macro F1 | Notes |
|---|---:|---:|---:|---:|---|
| CNN baseline | 10 | 69.20% | 71.60% | 61.20% (legacy binary metric) | Strong deepfake bias |
| CNN | 20 | 74.55% | 70.40% | 67.56% | Strong deepfake bias |
| CNN | 100 | 78.12% | 72.80% | 71.57% | Overfitting visible |
| CNN 2D | 20 | 79.91% | 66.40% | 62.35% | Validation did not transfer to test |
| CNN 2E | 25 | ~73.21% | 50.00% | 49.57% | Severe generalization failure |

Earlier frame-level experiments are retained as a leakage-affected baseline and must not be presented as source-independent generalization performance.
