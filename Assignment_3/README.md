# Assignment 3: Mammals Image Classification

A comprehensive comparison study between traditional Machine Learning with engineered features and modern Deep Learning approaches using Transfer Learning for multi-class image classification.

[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/datasets/asaniczka/mammals-image-classification-dataset-45-animals)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Results](#results)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)
- [Contributors](#contributors)

---

## 🎯 Overview

This project implements and evaluates a comprehensive "bake-off" comparison between:
- **Traditional ML** approaches using hand-crafted features (HOG + PCA)
- **Deep Learning** methods leveraging pre-trained models (ResNet-50, VGG-16, EfficientNet-B0)
- **Hybrid approaches** combining deep feature extraction with classical ML classifiers

**Total Models Evaluated:** 13 different pipeline combinations

---

## 📊 Dataset

**Source:** [Mammals Image Classification Dataset](https://www.kaggle.com/datasets/asaniczka/mammals-image-classification-dataset-45-animals)

| Metric | Value |
|--------|-------|
| Total Images | 13,751 |
| Number of Classes | 45 |
| Image Format | RGB |
| Class Distribution | Mildly imbalanced |
| Smallest Class | Vicuna (215 images) |
| Largest Class | Polar Bear (356 images) |

### Data Split Strategy
Stratified split to preserve class distributions:
- **Training Set:** 60% (8,250 images)
- **Validation Set:** 20% (2,750 images)
- **Test Set:** 20% (2,751 images)

---

## 📂 Project Structure

```
Assignment_3/
│
├── features/                    # Feature extraction modules
│
├── notebook/                    # Jupyter notebooks for experiments
│
├── models/                      # Saved model weights
│
├── results/                     # Experiment results and figures
│
├── requirements.txt             # Python dependencies
│
└── README.md                    # Project documentation
```

---

## 🏆 Results

### Performance Comparison

![Model Performance Comparison](results/model_performance_comparison.png)

The comprehensive evaluation reveals distinct performance patterns across all 13 model configurations, presented in four key perspectives:

1. **Test Accuracy Comparison** - Overall classification accuracy on test set
2. **F1 Score Comparison** - Balanced metric accounting for precision and recall
3. **Training Time Comparison** - Computational efficiency during training
4. **Accuracy by Feature Method** - Performance grouped by feature extraction approach

---

### Complete Test Results

| Model Configuration | Test Accuracy | F1-Score | Training Time (s) |
|---------------------|---------------|----------|-------------------|
| **EfficientNet + Logistic Regression** | **93.4%** | **93.4%** | **~5** |
| EfficientNet + SVM | 93.2% | 93.2% | ~3 |
| EfficientNet + Random Forest | 89.5% | 89.4% | ~85 |
| ResNet-50 + Logistic Regression | 91.5% | 91.5% | ~10 |
| ResNet-50 + SVM | 90.8% | 90.8% | ~4 |
| ResNet-50 (Fine-tuned) | 88.7% | 88.7% | ~130 |
| ResNet-50 + Random Forest | 88.5% | 88.5% | ~60 |
| VGG-16 + Logistic Regression | 88.3% | 88.3% | ~12 |
| VGG-16 + SVM | 87.0% | 87.0% | ~4 |
| VGG-16 + Random Forest | 82.1% | 82.1% | ~20 |
| **HOG + Logistic Regression** | **14.0%** | **13.6%** | **~15** |
| HOG + SVM | 10.9% | 10.8% | ~105 |
| HOG + Random Forest | 8.4% | 6.5% | ~203 |

---

## 💡 Key Findings

### 🥇 Champion: EfficientNet-B0 + Logistic Regression

Achieved the highest performance with **93.4% accuracy and F1-score** on the test set, while maintaining the fastest training time (~5 seconds) among high-performing models.

---

### 🔍 Critical Insights

#### 1. **Feature Quality Dominates Algorithm Choice**

Once high-quality features are extracted from deep models, the choice of classifier has minimal impact on performance:
- EfficientNet features: 89-93% accuracy across all three classifiers
- ResNet-50 features: 88-91% accuracy range
- VGG-16 features: 82-88% accuracy range

**Key Takeaway:** The battle is won at the **feature extraction stage**, not the classifier stage.

---

#### 2. **Hand-Crafted Features Fail Catastrophically**

The HOG+PCA pipeline achieved only **8-14% accuracy** across all classifiers, performing worse than random guessing (which would yield ~2.2% on 45 classes).

**The Paradox:** Despite being the fastest feature extraction method, HOG features are practically useless for this complex visual recognition task, proving that **speed without accuracy is meaningless**.

---

#### 3. **EfficientNet-B0 Outperforms Larger Architectures**

Surprisingly, the compact EfficientNet-B0 (1,280-dim features) consistently outperformed both VGG-16 (512-dim) and ResNet-50 (2,048-dim) features.

**Performance Hierarchy:**
- 🥇 EfficientNet-B0 features: 89-93% accuracy range
- 🥈 ResNet-50 features: 88-91% accuracy range  
- 🥉 VGG-16 features: 82-88% accuracy range
- ❌ HOG features: 8-14% accuracy range

This validates EfficientNet's compound scaling approach and demonstrates that **architectural efficiency matters more than raw model size**.

---

#### 4. **Fine-Tuning vs Feature Extraction Trade-off**

Interestingly, **fine-tuning ResNet-50 (88.7%)** performed worse than simply using ResNet-50 as a feature extractor with Logistic Regression (91.5%).

**Possible Reasons:**
- Limited training data (8,250 images) may be insufficient for fine-tuning
- Frozen pre-trained features already capture relevant patterns
- Fine-tuning risks overfitting on this relatively small dataset

**Recommendation:** For similar dataset sizes, feature extraction + simple classifier is more effective and efficient.

---

#### 5. **Training Efficiency Analysis**

| Approach | Accuracy | Training Time | Efficiency Score |
|----------|----------|---------------|------------------|
| EfficientNet + LR | 93.4% | ~5s | ⭐⭐⭐⭐⭐ |
| EfficientNet + SVM | 93.2% | ~3s | ⭐⭐⭐⭐⭐ |
| ResNet-50 (Fine-tuned) | 88.7% | ~130s | ⭐⭐ |
| HOG + Random Forest | 8.4% | ~203s | ❌ |

**Winner:** EfficientNet + Logistic Regression achieves the best accuracy-to-speed ratio, making it ideal for both research and production deployment.

---

### 📊 Summary Insights

✅ **Deep learning features are essential** for complex image classification  
✅ **EfficientNet-B0** provides the best feature representations  
✅ **Simple classifiers** (LR, SVM) work excellently with good features  
✅ **Fine-tuning** may not always be better than feature extraction  
❌ **Hand-crafted features** (HOG) are obsolete for modern vision tasks

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 16GB RAM minimum

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Assignment_3.git
cd Assignment_3

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the dataset
# Visit: https://www.kaggle.com/datasets/asaniczka/mammals-image-classification-dataset-45-animals
# Extract to ./data/raw/
```

### Dependencies

```
torch>=1.9.0
torchvision>=0.10.0
scikit-learn>=0.24.0
scikit-image>=0.18.0
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
seaborn>=0.11.0
jupyter>=1.0.0
```

---

## 🚀 Usage

### Running Experiments

```bash
# Run all experiments via Jupyter notebooks
jupyter notebook
```

---

## 📚 References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (ResNet)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (VGG)
- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- [Histograms of Oriented Gradients for Human Detection](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) (HOG)

---

## 👥 Contributors

- **Cao Nhat Lam** 
- **Course:** Machine Learning
- **Semester:** 2025

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Kaggle for providing the Mammals dataset
- Course instructors and TAs for guidance
- PyTorch and scikit-learn communities for excellent documentation

---

## 📧 Contact

For questions or feedback, please reach out via:
- Email: lam.caonhat0404@hcmut.edu.vn

---

<p align="center">
  Made with ❤️ for Machine Learning Course - Assignment 3
</p>