# Email Spam Classification: Custom Naive Bayes and k-Nearest Neighbors

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.2-green)
![Pandas](https://img.shields.io/badge/Pandas-2.0.3-red)

## 📧 Overview

This project implements **custom Naive Bayes and k-Nearest Neighbors (k-NN) classifiers** to identify email spam. These classifiers provide a foundational understanding of supervised machine learning techniques for text classification tasks.

## 🚀 Key Features

### **Naive Bayes Classifier**
- **Multinomial Likelihood:** Suited for discrete data (e.g., word counts in emails).
- **Class Priors and Likelihoods:** Calculates the probability of spam/ham emails based on training data.
- **Laplace Smoothing:** Avoids zero probabilities by adding smoothing to the likelihoods.
- **Logarithmic Optimization:** Uses log probabilities for numerical stability.

### **k-Nearest Neighbors Classifier**
- **Custom Distance Metric:** Computes distances for feature-rich datasets.
- **k Selection:** Allows flexibility in the number of neighbors for classification.
- **Majority Voting:** Predicts class based on the most frequent label among k nearest neighbors.

## 🔍 Implementation Details

### **Naive Bayes Implementation**
- **File:** `naive_bayes.py`
- **Methods:**
  - `train(data, y)`: Trains the model on feature counts and class labels.
  - `predict(data)`: Predicts class labels for new data.
  - `accuracy(y, y_pred)`: Computes classification accuracy.
- **Approach:**
  - Computes class priors and likelihoods.
  - Predicts classes using the maximum a posteriori probability.

### **k-Nearest Neighbors Implementation**
- **File:** `knn.py`
- **Methods:**
  - `fit(X, y)`: Stores training data and labels.
  - `predict(X_test)`: Predicts labels based on the nearest neighbors.
  - `accuracy(y, y_pred)`: Computes classification accuracy.
- **Approach:**
  - Calculates distances between test samples and training samples.
  - Assigns the most frequent label among k nearest neighbors.

## 🎨 Visual Examples

coming soon

## 🛠️ Technologies & Skills

- **Programming Languages:** 
  - [Python 3.x](https://www.python.org/)

- **Libraries:** 
  - [NumPy](https://numpy.org/) for data manipulation and computation.
  - [Pandas](https://pandas.pydata.org/) for data analysis and cleaning.
  - [Matplotlib](https://matplotlib.org/) for data visualization.
- **Machine Learning Techniques:** 
  - **Naive Bayes:** Probabilistic modeling for classification tasks.
  - **k-NN:** Instance-based learning with distance computations.

## 🚀 Getting Started

coming soon

## 📈 Example Project: Email Classifier Model Comparison

coming soon
