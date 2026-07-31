# 🐾 Image Classification using Convolutional Neural Networks (CNN)

**Author:** Yash  
**Registration Number:** 23BCE10362  
**Assignment:** AI-ML Assignment – 9  

---

## 📌 Objective

An animal welfare organization aims to automate the classification of pet images into **Cats** and **Dogs**. This project builds, trains, and evaluates a **Convolutional Neural Network (CNN)** model capable of accurately distinguishing between the two classes with high confidence.

---

## 🔗 Dataset

| Property | Details |
|:---|:---|
| **Dataset Name** | Kaggle Dog and Cat Classification Dataset (`PetImages`) |
| **Kaggle URL** | [Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset) |
| **Cat Images** | 12,499 |
| **Dog Images** | 12,499 |
| **Total Images** | 24,998 |
| **Classes** | 2 (`Cat`, `Dog`) |

---

## 🛠️ Libraries & Tools

| Category | Libraries |
|:---|:---|
| **Deep Learning** | `TensorFlow`, `Keras` (`Sequential`, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `ImageDataGenerator`) |
| **Data Processing** | `NumPy`, `Pillow (PIL)` |
| **Model Evaluation** | `scikit-learn` (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `classification_report`) |
| **Visualization** | `Matplotlib`, `Seaborn` |

---

## 🔬 Methodology

### Step 1 — Data Understanding & Inspection
- Inspected the folder structure (`PetImages/Cat` and `PetImages/Dog`).
- Visualized 5 sample images per class with their labels.
- Identified 2 target classes, variable raw image dimensions, and 24,998 total samples.

### Step 2 — Data Preprocessing
- Resized all images to a uniform resolution of **128 × 128 pixels**.
- Normalized pixel values from `[0, 255]` to `[0, 1]` via float division (`1/255.0`).
- Split dataset: **80% Training** (20,000 samples) | **20% Testing** (4,998 samples).
- Applied TensorFlow/Keras `ImageDataGenerator` for batch processing.

### Step 3 — Model Development & Architecture
- Designed a 9-layer CNN using the Keras `Sequential` API.
- Compiled with **Adam optimizer**, **Binary Crossentropy** loss, and **Accuracy** metric.
- Trained for **10 epochs** with real-time validation.

### Step 4 — Evaluation & Analysis
- Computed test accuracy, precision, recall, and F1-score.
- Plotted a **Confusion Matrix Heatmap**.
- Generated **Accuracy vs. Epoch** and **Loss vs. Epoch** learning curves.

---

## 🏗️ CNN Architecture

| Layer | Type | Output Shape | Filter / Neurons | Activation | Parameters |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Layer 1** | Conv2D | `(None, 126, 126, 32)` | 32 filters (3×3) | ReLU | 896 |
| **Layer 2** | MaxPooling2D | `(None, 63, 63, 32)` | Pool Size (2×2) | — | 0 |
| **Layer 3** | Conv2D | `(None, 61, 61, 64)` | 64 filters (3×3) | ReLU | 18,496 |
| **Layer 4** | MaxPooling2D | `(None, 30, 30, 64)` | Pool Size (2×2) | — | 0 |
| **Layer 5** | Conv2D | `(None, 28, 28, 128)` | 128 filters (3×3) | ReLU | 73,856 |
| **Layer 6** | MaxPooling2D | `(None, 14, 14, 128)` | Pool Size (2×2) | — | 0 |
| **Layer 7** | Flatten | `(None, 25088)` | — | — | 0 |
| **Layer 8** | Dense | `(None, 128)` | 128 Neurons | ReLU | 3,211,392 |
| **Layer 9** | Output (Dense) | `(None, 1)` | 1 Neuron | Sigmoid | 129 |

> 🔢 **Total Trainable Parameters:** 3,304,769

---

## 📊 Results & Performance

| Metric | Score |
|:---|:---|
| ✅ **Test Accuracy** | **80.47%** |
| 🎯 **Precision** | **0.81** |
| 🔁 **Recall** | **0.80** |
| 📈 **F1-Score** | **0.80** |

### 🔍 Performance Observations

1. **Steady Metric Convergence:** Training accuracy increased steadily across all 10 epochs, while binary crossentropy loss declined consistently.
2. **Balanced Classification:** High and balanced precision (0.81) and recall (0.80) across both Cat and Dog classes indicate no class bias.
3. **Generalization Capabilities:** Validation accuracy tracked training accuracy closely, suggesting minimal overfitting.
4. **Hierarchical Feature Extraction:** Stacked convolution and max-pooling layers progressively extracted edges, textures, and abstract visual patterns.

---

## 🎯 Conclusion

This project successfully built and evaluated a 9-layer **Convolutional Neural Network (CNN)** for binary image classification of Cats vs. Dogs.

By standardizing input images to **128 × 128 pixels** and normalizing pixel intensities to **[0, 1]**, the model achieved an impressive **80.47% accuracy** on unseen test data.

### Key Takeaways

| Aspect | Insight |
|:---|:---|
| **Convolution + Pooling** | Automatically detect local features (edges, textures, shapes) while reducing spatial dimensions |
| **CNN vs. ANN** | CNNs preserve 2D spatial structure and leverage parameter sharing — far more efficient than flattening images for ANN |
| **Limitations** | CNNs require large labeled datasets and GPU acceleration to avoid overfitting on complex, high-resolution real-world images |

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/yash23bce10362/Image-Classification-using-Convolutional-Neural-Networks.git
cd Image-Classification-using-Convolutional-Neural-Networks

# Install dependencies
pip install tensorflow numpy pillow scikit-learn matplotlib seaborn

# Launch the notebook
jupyter notebook Assignment-9.ipynb
```

---

*Built with ❤️ using TensorFlow & Keras*
