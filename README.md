# Wood Identification — Sal vs Deodar (CNN Classifier)

A convolutional neural network (CNN) built with TensorFlow/Keras that classifies wood images into two timber species: **Sal** (*Shorea robusta*) and **Deodar** (*Cedrus deodara*). The project is implemented as a single Google Colab notebook.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Known Issues & Notes](#known-issues--notes)
- [Future Improvements](#future-improvements)

---

## Overview

Manual wood species identification is slow and needs expert knowledge. This project shows how a lightweight CNN can tell two commercially important Indian timbers apart from images alone.

The notebook covers the full pipeline:

1. Loading the image dataset from Google Drive
2. Building an 80/20 train/validation split
3. Training a 3-block CNN from scratch
4. Evaluating with accuracy, confusion matrix, classification report and ROC curve
5. Saving the trained model in `.keras` format

## Dataset

| Property | Value |
|---|---|
| Location | `/content/drive/MyDrive/UCOSTProject` |
| Total images | 220 |
| Classes | 2 — `Sal images`, `deodar images` |
| Training split | 176 images (80%) |
| Validation split | 44 images (20%) — 25 Sal, 19 Deodar |
| Image size | 224 × 224 × 3 (resized on load) |
| Batch size | 32 |
| Split seed | 123 |

Expected folder layout (one sub-folder per class; folder names become class labels):

```
UCOSTProject/
├── Sal images/
│   ├── img_001.jpg
│   └── ...
└── deodar images/
    ├── img_001.jpg
    └── ...
```

> The dataset itself is not included in this repository. Place your own images in the structure above.

## Model Architecture

A sequential CNN with ~11.17 million trainable parameters (~42.6 MB):

| Layer | Details |
|---|---|
| Rescaling | Normalizes pixels to [0, 1] |
| Conv2D + MaxPool | 32 filters, 3×3, ReLU → 2×2 pool |
| Conv2D + MaxPool | 64 filters, 3×3, ReLU → 2×2 pool |
| Conv2D + MaxPool | 128 filters, 3×3, ReLU → 2×2 pool |
| Flatten | — |
| Dense | 128 units, ReLU |
| Dense (output) | 2 units, Softmax |

## Training Configuration

- **Optimizer:** Adam
- **Loss:** Sparse categorical cross-entropy
- **Metric:** Accuracy
- **Epochs:** 10
- **Data pipeline:** `tf.data` with `prefetch(AUTOTUNE)`

## Results

Evaluated on the 44-image validation set after 10 epochs:

| Metric | Value |
|---|---|
| Training accuracy | 100% |
| Validation accuracy | **95.45%** |
| Best validation accuracy | 97.73% (epochs 8 and 9) |

**Classification report**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Sal | 1.00 | 0.92 | 0.96 | 25 |
| Deodar | 0.90 | 1.00 | 0.95 | 19 |
| **Accuracy** | | | **0.95** | 44 |

The notebook also produces: accuracy/loss curves, a confusion matrix, and an ROC curve with AUC.

## Project Structure

```
.
├── Copy_of_Copy_of_wood_identification.ipynb   # Full pipeline notebook
└── README.md
```

Saved outputs (written to Google Drive by the notebook):

- `wood_model_deodar_sal.keras`
- `wood_model_quercus_teak.keras` (see [Known Issues](#known-issues--notes))

## Getting Started

### Prerequisites

- Python 3.10+
- Google account (the notebook mounts Google Drive)
- GPU runtime recommended (Colab: *Runtime → Change runtime type → GPU*)

### Dependencies

```bash
pip install tensorflow matplotlib scikit-learn numpy pandas seaborn
```

### Run on Google Colab

1. Upload the dataset to Google Drive as `MyDrive/UCOSTProject/` (structure shown above).
2. Open the notebook in Colab.
3. Run all cells (*Runtime → Run all*) and authorize Drive access when prompted.

To run outside Colab, remove the `google.colab` import and `drive.mount(...)` lines and change `dataset_path` to a local folder.

## Usage

Load the saved model and classify a new image:

```python
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("wood_model_deodar_sal.keras")
class_names = ["Sal images", "deodar images"]  # alphabetical, as loaded by Keras

img = tf.keras.utils.load_img("sample.jpg", target_size=(224, 224))
x = tf.keras.utils.img_to_array(img)[np.newaxis, ...]

probs = model.predict(x)[0]
print(class_names[np.argmax(probs)], f"({probs.max():.2%})")
```

## Known Issues & Notes

- **Small dataset:** 220 images total and only 44 for validation, so each image is worth ~2.3 percentage points. Reported accuracy is a rough estimate, not a robust benchmark.
- **Overfitting signs:** training accuracy hits 100% while validation loss fluctuates. Data augmentation, dropout and early stopping would help.
- **"Test accuracy" is actually validation accuracy:** there is no separate held-out test set; the validation split is used for all evaluation.
- **t-SNE cell uses dummy data:** the t-SNE section generates random numbers as placeholder features, so the plot does **not** reflect the real model or dataset. Replace it with features extracted from the CNN's penultimate layer before using it in a report.
- **Stale labels:** a code comment says "Quercus, Teak" and the last cell saves a model as `wood_model_quercus_teak.keras`. These are left over from an earlier version; the classes are Sal and Deodar. That last cell re-saves the same model under a misleading name and can be removed.
- **Deprecation warnings:** `input_shape` in `Rescaling` should be replaced by an `Input(...)` layer, and `n_iter` in `TSNE` is now `max_iter`.
- **Class-imbalance:** 25 vs 19 validation images — mild, but worth monitoring.

## Future Improvements

- Add data augmentation (flip, rotation, zoom, brightness)
- Add dropout / L2 regularization and early stopping with `ModelCheckpoint`
- Use transfer learning (MobileNetV2, EfficientNet, ResNet50) for better generalization
- Create a proper train / validation / test split, or use k-fold cross-validation
- Collect more images across varied lighting, cuts (cross / radial / tangential) and surface finishes
- Extend to more species and deploy as a web or mobile app (TFLite)
- Add Grad-CAM to see which grain features drive predictions

## Acknowledgements

Built with [TensorFlow/Keras](https://www.tensorflow.org/), [scikit-learn](https://scikit-learn.org/), [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/), running on Google Colab.

## License

Add a license of your choice (e.g., MIT) before publishing.
