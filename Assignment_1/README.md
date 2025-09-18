
# ML-251 Assignment 1: Melbourne Housing Price Analysis & Prediction

This repository contains the code and resources for Assignment 1 of the ML-251 course, focused on analyzing and predicting house prices in Melbourne using machine learning techniques. The project covers data exploration, preprocessing, feature engineering, model training, and interactive experimentation.

## Quick Start

- **Google Colab:** Run the notebook online: [Open in Colab](https://colab.research.google.com/drive/1GAUbgeEsvjU-iEGN_TrHmg8Ruim0jAfb?usp=sharing)
- **Local:** See instructions below for running locally.

## Setup & Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

For full notebook interactivity, enable Jupyter widgets:

```bash
jupyter nbextension enable --py widgetsnbextension
```

## Usage

### Local
1. Clone or download this repository.
2. Open a terminal in the `Assignment_1` directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Enable Jupyter widgets:
   ```bash
   jupyter nbextension enable --py widgetsnbextension
   ```
5. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
6. Open and run `notebook/251_ML_Assignment1.ipynb`.

### Google Colab
1. Click the Colab link above.
2. Run all cells for a cloud-based, interactive experience.

### Model Experimentation
- Use widgets in the notebook to select scaler, PCA variance, and model type.
- Run experiments and compare results interactively.

### Outputs
- Model results, feature importance plots, and processed features are saved in the respective folders.

## Model Execution Note

By default, the notebook runs only one model for demonstration purposes. If you wish to run all models, simply uncomment the relevant code sections in the notebook to enable full model training and comparison.

## Features

- **Data Loading:** Melbourne housing dataset is loaded and preprocessed automatically.
- **Exploratory Data Analysis:**
  - Descriptive statistics, missing value analysis, outlier detection
  - Price distribution, correlation, regional and geospatial analysis
- **Preprocessing:**
  - Imputation, one-hot encoding, feature scaling, PCA
- **Model Training & Evaluation:**
  - Linear Regression, SVR, Random Forest, MLPRegressor
  - Hyperparameter tuning (GridSearchCV), R² and MSE metrics
- **Interactive Widgets:**
  - Select scaler, PCA variance, and model type
  - Run and compare models interactively in the notebook
- **Results Comparison:**
  - Tabular and visual comparison of model results and feature importance

## Folder Structure

```
Assignment_1/
├── features/      # train.npy, test.npy (feature data)
├── models/        # preprocess.py, IOmanip.py (scripts)
├── notebook/      # 251_ML_Assignment1.ipynb (Jupyter notebook)
├── requirements.txt
└── README.md
```

## Key Insights

- The MLP (Deep Learning) model achieved the highest prediction accuracy (R²), outperforming all traditional machine learning models when properly tuned.
- Among traditional models, Support Vector Regression (SVR) ranked best, consistently delivering high R² scores (around 0.85), followed by Random Forest.
- Random Forest also performed well, especially when combined with data scaling and PCA, but was slightly less accurate than SVR and MLP.
- Linear Regression served as a baseline and showed significantly lower performance compared to more advanced models.
- The most influential factors for house price prediction are property type, physical size (number of rooms, land size), and geographic location.
- Distance from the Central Business District (CBD) is a strong negative predictor: properties farther from the CBD tend to be less expensive.
- The dataset is high quality, with minimal missing values, enabling effective model training and evaluation.

## Results

Model performance is compared using R² and MSE. Feature importance is visualized for Random Forest. Deep learning results are included for benchmarking.

