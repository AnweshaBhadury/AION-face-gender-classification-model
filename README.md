# AION Face-Gender Classification Model

This repository contains a deep learning pipeline that detects human faces and classifies their gender using an EfficientNet-based model. Face embeddings are optionally generated via a pretrained FaceNet (InceptionResNetV1) model for downstream tasks.

## 📌 Key Features

* 🔍 **Face Detection** (retinaface)
* 👦👧 **Gender Classification** using **EfficientNetB0**
* 🧠 **Face Embedding** using **FaceNet** (`facenet_triplet.pth`)
* 📦 **Pretrained Weights** available for instant testing
* 🧪 **Test Script** for both image and video inference
* 📈 Training Metrics Visualization
* 📁 Supports image, video, and live webcam feed

---

## 📂 Directory Structure

```
├── main.py                           # Main pipeline script
├── train.py                          # Gender classifier training script
├── testscript.py                     # Run saved model on images/videos
├── facenet_triplet.pth               # FaceNet pretrained weights (PyTorch)
├── gender_classification_efficientnet_final.keras  # Trained EfficientNet model
├── gender_classification_efficientnet_best.h5      # Best model (val_accuracy)
├── requirements.txt                  # Dependencies
├── training_history.png              # Training metric plots
├── evaluation_results.json           # Accuracy, Precision, F1, etc.
├── data/                             # Sample test files (images/videos)
├── Comys_Hackathon5/Task_A/train     # Training dataset
├── Comys_Hackathon5/Task_A/val       # Validation dataset
```

---

## 📊 Model Performance

| Metric        | Value  |
| ------------- | ------ |
| **Accuracy**  | 75.11% |
| **Precision** | 56.42% |
| **Recall**    | 75.11% |
| **F1 Score**  | 64.44% |

📁 Found in: `evaluation_results.json`

---

## ⚙️ Installation

```bash
# (Optional) Create and activate a virtual environment
python -m venv hack
.\hack\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### ▶️ 1. Main Gender Classification (Live/Webcam)

```bash
python main.py
```

This will:

* Load the trained EfficientNet model
* Use Haarcascade to detect faces
* Predict gender and display results

---

### 🧪 2. Testing with Images or Videos

Edit the `testscript.py` to point to your image/video path:

```python
input_path = "data/samplevideo.mp4"  # or .jpg, .jpeg
```

Then run:

```bash
python testscript.py
```

---

### 🧠 3. Retrain Model (Optional)

Make sure your dataset is arranged like:

```
Comys_Hackathon5/
└── Task_A/
    ├── train/
    │   ├── male/
    │   └── female/
    └── val/
        ├── male/
        └── female/
```

Then train:

```bash
python train.py
```

Trained weights will be saved as:

* `gender_classification_efficientnet_best.h5`
* `gender_classification_efficientnet_final.keras`

---

## 🧠 Diagram – Architecture Overview

```text
          +--------------------+
          |  Input Image/Video |
          +--------+-----------+
                   |
                   v
          +--------+-----------+
          |  Face Detection    |  <-- (Haarcascade or MTCNN)
          +--------+-----------+
                   |
                   v
          +--------+-----------+
          |  Resize + Preprocess |
          +--------+-----------+
                   |
                   v
       +-----------+-------------+
       | EfficientNetB0 Model    |  <-- (Pretrained on ImageNet)
       | + Dense(128) + Dropout |
       | + Dense(2, softmax)    |
       +-----------+-------------+
                   |
                   v
         +---------+---------+
         |   Gender Output   |
         +-------------------+
```

---

## 📥 Pretrained Weights

* `gender_classification_efficientnet_final.keras` — Final trained model
* `gender_classification_efficientnet_best.h5` — Best model based on val accuracy
* `facenet_triplet.pth` — PyTorch model for generating 128D face embeddings

> These weights are loaded automatically by `main.py` and `testscript.py`. No manual setup needed.

---

## 🙋‍♀️ Author

👤 **Anwesha Bhadury**
💡 3rd year
🔗 GitHub: [@AnweshaBhadury](https://github.com/AnweshaBhadury)

---

## 🙏 Acknowledgements

* [Keras Applications](https://keras.io/api/applications/)
* [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
* [FaceNet PyTorch](https://github.com/timesler/facenet-pytorch)

---
