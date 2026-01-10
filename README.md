# Iris Classification Using Neural Network from Scratch in Rust 🦀🧠

This project implements a **fully connected neural network from scratch in Rust** to classify the **Iris dataset**, without using any machine learning libraries.  
The objective is to deeply understand the **core mechanics of neural networks**—from data preprocessing and forward propagation to backpropagation and optimization—by implementing everything manually.

---

## 🚀 Project Overview

- **Dataset:** Iris dataset (150 samples, 3 classes)
- **Language:** Rust
- **Machine Learning Libraries:** ❌ None (from scratch)
- **Optimizer:** Gradient Descent
- **Evaluation:** 30 independent randomized runs

This project focuses on **learning by implementation**, emphasizing fundamentals over abstraction.

---

## 🧠 Neural Network Architecture

| Layer | Description |
|------|------------|
| Input Layer | 4 neurons (Sepal length, Sepal width, Petal length, Petal width) |
| Hidden Layer | 1 layer with 8 neurons (ReLU activation) |
| Output Layer | 3 neurons (Softmax activation) |

### Why this architecture?
- 4 inputs directly map to the Iris features  
- 8 hidden neurons capture non-linear relationships without overfitting  
- ReLU is efficient and widely used  
- Softmax outputs class probabilities  

---

## 📊 Performance Summary (Mean ± Std over 30 Runs)

| Metric | Mean Accuracy | Std Dev |
|------|--------------|--------|
| Train | **98.81%** | ± 0.77% |
| Test | **96.56%** | ± 3.28% |

---

## 🗂 Project Structure

```text
.
├── data/
│   └── iris.csv
├── src/
│   ├── main.rs           # Entry point
│   ├── data.rs           # Data loading, normalization, encoding
│   ├── model.rs          # Neural network implementation
│   ├── training.rs       # Accuracy and confusion matrix
│   ├── experiment.rs    # Training loop and experiments
│   └── utils.rs          # Metrics and CSV exports
├── plots.py              # Visualization script
├── plots/                # Generated plots
├── summary.csv           # Mean ± Std accuracy table
├── confusion_matrix.csv
└── README.md

```
## 🔬 Implementation Details

### 📥 Data Processing (`data.rs`)
- CSV loading using `csv::ReaderBuilder`
- Label mapping:
  - `Iris-setosa → 0`
  - `Iris-versicolor → 1`
  - `Iris-virginica → 2`
- Train/Test split (80/20) with random shuffling
- Feature normalization using mean and standard deviation
- One-hot encoding for labels

---

### 🧮 Neural Network (`model.rs`)
Implemented fully from scratch:
- Weight initialization
- Forward propagation
- ReLU activation and derivative
- Softmax output
- Cross-entropy loss
- Backpropagation
- Gradient descent updates
- Model serialization to JSON

---

### 🎯 Training & Evaluation (`training.rs`)
- Accuracy computation
- Confusion matrix generation

---

### 🧪 Experiment Pipeline (`experiment.rs`)
- End-to-end training loop
- Tracks:
  - Training loss
  - Train accuracy
  - Test accuracy
- Saves:
  - Loss curve
  - Accuracy curves
  - Confusion matrix
  - Trained model (`trained_model.json`)

---

### 📈 Multiple Runs & Statistics (`main.rs`)
- Executes 30 independent runs
- Computes mean and standard deviation of accuracy
- Saves summary and accuracy distributions

---

## 📊 Visualization

The `plots.py` script generates:
- Confusion Matrix (heatmap)
- Training Loss vs Epoch
- Accuracy vs Epoch
- Test Accuracy Histogram
- Test Accuracy Boxplot
- Mean ± Std summary table

> Python is used **only for visualization**, not for training or modeling.

---

## ▶️ How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/LalithKanna/Iris-Classification-Using-Rust---1
cd Iris-Classification-Using-Rust---1
```
### 2️⃣ Run Training
```bash
cargo run --release
```
### 3️⃣ Generate Plots
```bash
python plots.py
```
---
### 🧠 Key Learnings

-Deep understanding of neural network internals

-Confidence in architectural decision-making

-Practical experience with Rust for numerical computing

-Insight into why Rust is suitable for memory-safe, high-performance ML systems
---
### 🦀 Why Rust for Machine Learning?

-Memory safety without garbage collection

-High performance and fine-grained control

-Suitable for edge AI and production systems

-Growing ML ecosystem
---

### 🚀 Future Improvements

-Mini-batch shuffling per epoch

-Adam optimizer

-Multiple hidden layers

-Configurable architectures

-Benchmarking against ML libraries
---
###⭐ Acknowledgments

-UCI Iris Dataset

-Rust community

-Learning by doing 🚀
---
