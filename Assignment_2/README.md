# ML-251 Assignment 2: AG News Classification
This repository contains the code and resources for Assignment 2 of the CSEML04 course, focused on text classification. The project aims to categorize news articles from the AG News dataset into one of four classes (World, Sports, Business, Sci/Tech) and compares the performance of traditional machine learning models against a modern deep learning approach.

## Quick Start
**Google Colab**: Run the notebook online: [Open in Colab](https://colab.research.google.com/drive/11dA7fUtUfbSD7a7KDgNgB7Gk50PzcSqk)

**Local**: See instructions below for running locally.

## Setup & Requirements
Before running the notebook, install the required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage
### Local Environment
Clone or download this repository.

Open a terminal in the project directory.

Install the dependencies:
```bash
pip install -r requirements.txt
```
Launch Jupyter Notebook:
```bash
jupyter notebook
```
Open and run `notebook/251_ML_Assignment2.ipynb`.

### Google Colab
Click the Colab link above.

Ensure the runtime is set to use a GPU for training the deep learning model (Runtime > Change runtime type > T4 GPU).

Run all cells to execute the project.

## Project Workflow
The notebook follows a standard machine learning pipeline for text data:

### Data Loading & EDA:

Loads a subset of the AG News dataset.

Performs Exploratory Data Analysis (EDA) to visualize class distribution, text length, and word frequencies.

### Traditional Model Pipeline:

Feature Extraction: Converts text into numerical features using TfidfVectorizer and saves them as .npy files.

Model Training: Loads the .npy files and trains multiple traditional models (Naive Bayes, Logistic Regression, SVC, Random Forest) using GridSearchCV to find the best hyperparameters.

### Deep Learning Pipeline:

Tokenization: Prepares raw text data for the deep learning model using a BERT Tokenizer.

Model Training: Fine-tunes a pre-trained bert-base-uncased model for the classification task.

### Results Comparison:

Compares the performance of all models on the test set using accuracy and F1-score.

Presents a comprehensive summary table to identify the best-performing model.

## Folder Structure

```
Assignment_2/
├── notebook/      # Contains 251_ML_Assignment2.ipynb
├── features/      # Stores .npy files for features and labels
├── requirements.txt
└── README.md
```
## Key Insights
The fine-tuned BERT (Deep Learning) model achieved the highest test accuracy (91.3%), significantly outperforming all traditional methods.

Among the traditional models, Support Vector Classifier (SVC) was the top performer, providing a strong baseline with 87.1% accuracy.

The results highlight the superior ability of transformer-based models like BERT to understand language context, which is crucial for high-performance text classification.

While computationally more intensive, the deep learning approach offers a clear advantage in accuracy for this task.

## Results
Model performance is compared using a detailed table that includes Accuracy and Macro F1-Score for all tested configurations. The best traditional model and the fine-tuned BERT model are highlighted for a clear comparison.