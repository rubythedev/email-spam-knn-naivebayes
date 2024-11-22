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

### **Spiral Dataset 1: Train and Validation Data**
This visual represents the training and validation sets for the spiral 1 dataset. The train data is shown as one color, while the validation data is represented by a different color. It provides insight into how well the classifiers can generalize the data.

<img src="https://github.com/rubythedev/email-spam-knn-naivebayes/blob/main/images/spiral_1.png" width="400" />

### **Spiral Dataset 2: Train and Validation Data**
Similar to the first spiral dataset, this visual depicts the train and validation sets for spiral dataset 2, highlighting the structure of the dataset and its complexity for classification.

<img src="https://github.com/rubythedev/email-spam-knn-naivebayes/blob/main/images/spiral_2.png" width="400" />

### **Spiral Dataset 1: Class Boundaries**
This image shows the decision boundaries for the spiral 1 dataset when classified using either the Naive Bayes or k-NN algorithm. The color-coded boundaries give a visual representation of how well each model differentiates between the classes.

<img src="https://github.com/rubythedev/email-spam-knn-naivebayes/blob/main/images/spiral_1_class_boundaries.png" width="400" />

### **Spiral Dataset 2: Class Boundaries**
Like the first, this visual shows the decision boundaries for spiral dataset 2. It demonstrates how the algorithms classify the different regions based on the training data.

<img src="https://github.com/rubythedev/email-spam-knn-naivebayes/blob/main/images/spiral_2_class_boundaries.png" width="400" />

### **Spiral Dataset 1: Accuracy vs. k**
This plot visualizes the accuracy of the k-NN model on spiral dataset 1, as the number of neighbors (k) varies. It shows how performance changes based on the choice of k.

<img src="https://github.com/rubythedev/email-spam-knn-naivebayes/blob/main/images/spiral_dataset_1_accuracy_vs_k.png" width="400" />

### **Spiral Dataset 2: Accuracy vs. k**
This visual shows the same accuracy vs. k analysis, but for spiral dataset 2. It helps in selecting the optimal k value for better model performance.

<img src="https://github.com/rubythedev/email-spam-knn-naivebayes/blob/main/images/spiral_dataset_2_accuracy_vs_k.png" width="400" />

### **Naive Bayes vs k-NN Accuracy**
A comparison of the classification accuracies between the Naive Bayes and k-NN models on both spiral datasets. This helps to evaluate which classifier performs better under different scenarios.

| k-NN Accuracy | Naive Bayes Accuracy |
|---------------|----------------------|
| <img src="https://github.com/rubythedev/email-spam-knn-naivebayes/blob/main/images/knn_accuracy.png" width="400" /> | <img src="https://github.com/rubythedev/email-spam-knn-naivebayes/blob/main/images/naive_bayes_accuracy.png" width="400" /> |

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
