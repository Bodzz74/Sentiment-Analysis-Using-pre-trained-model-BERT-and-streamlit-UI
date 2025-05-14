# 🎬 Sentiment Analysis For Movie Review using BERT

This project performs sentiment analysis on IMDB movie reviews using the DistilBERT model from HuggingFace's Transformers library. It includes a training pipeline built with TensorFlow and an interactive web interface built with Streamlit for real-time sentiment predictions.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributors](#contributors)
- [License](#license)

---

## 🧠 Overview

Sentiment analysis is the process of determining whether a piece of text expresses positive or negative emotions. In this project, we use **DistilBERT**, a lightweight version of BERT, fine-tuned on the IMDB movie review dataset to classify reviews as positive or negative.

The project is divided into two main parts:

1. **Model Training**: Fine-tuning DistilBERT using TensorFlow and HuggingFace.
2. **Web Interface**: Streamlit application that allows users to input custom reviews and view sentiment predictions.

---

## 📂 Dataset

- Dataset: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Format: CSV
- Fields:
  - `review`: Text of the movie review
  - `sentiment`: Label (`positive` or `negative`)

---

## 🧱 Model Architecture

- **Pre-trained Model**: `distilbert-base-uncased`
- **Tokenizer**: `DistilBertTokenizerFast`
- **Classification Head**: Added a dense layer for binary classification
- **Loss Function**: `SparseCategoricalCrossentropy(from_logits=True)`
- **Optimizer**: Adam with learning rate `5e-5`
- **Metrics**: Accuracy

---

## ⚙️ Installation

### Requirements

- Python 3.8+
- TensorFlow 2.11+
- Transformers
- Pandas
- NumPy
- Streamlit
- scikit-learn

### Install Dependencies

```bash
pip install -r requirements.txt
