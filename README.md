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
A comparison of the classification accuracies between the Naive Bayes and k-NN models for classifying emails as spam or ham. This demonstrates how each classifier performs under different scenarios.

| Naive Bayes Accuracy | k-NN Accuracy |
|---------------|----------------------|
| 89.59% | 91% |

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

Below is a demonstration of the **Naive Bayes** and **KNN** Classifier on a numerical dataset. You can use it for numeric datasets (i.e., CSV files). Follow the steps below to get started.

### KNN Classifier

Before running the project, make sure you have the following libraries installed in your Python environment:  
- **NumPy**  
- **Pandas**  
- **Matplotlib**  
- **SciPy**
- **Random**
- **os**

Install them using `pip`:

```python
pip install numpy pandas matplotlib scipy
```

### **1. Import Required Libraries**

```python
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

### 2. Load and Visualize Data
Load the dataset and split it into features (X) and labels (y).

```python
import numpy as np
import matplotlib.pyplot as plt

# Set plotting styles and options
plt.style.use(['seaborn-v0_8-colorblind', 'seaborn-v0_8-darkgrid'])
plt.rcParams.update({'font.size': 20})
np.set_printoptions(suppress=True, precision=5)

# Load the dataset
yourdataset_train = np.loadtxt('data/yourdataset.csv', skiprows=1, delimiter=',')
yourdataset_val = np.loadtxt('data/yourdataset.csv', skiprows=1, delimiter=',')

# Split features (X) and labels (y)
train_y = yourdataset_train[:, -1]  # Assuming the target is the last column
val_y = yourdataset_val[:, -1]

train_X = yourdataset_train[:, :-1]  # All columns except the last
val_X = yourdataset_val[:, :-1]

# Create plots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Training set plot
scatter_train = axes[0].scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap='viridis')
axes[0].set_title('YourDataset - Training Set')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# Validation set plot
scatter_val = axes[1].scatter(val_X[:, 0], val_X[:, 1], c=val_y, cmap='viridis')
axes[1].set_title('YourDataset - Validation Set')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')

# Add color bar
cbar = fig.colorbar(scatter_train, ax=axes, location='right', shrink=0.8, pad=0.1)
cbar.set_label('Class')

# Finalize and show the plot
plt.tight_layout()
plt.show()
```

### 3. Train the Classifier
Train the classifier on the dataset.

```python
classifier.train(X, y)
```

### 4. Predict Using KNN
Use the trained classifier to make predictions. Choose a value for k (e.g., k=1 or k=2).

```python
k = 3
y_pred = classifier.predict(X, k)
```

### 5. Evaluate Model Accuracy
Calculate the accuracy of the predictions.

```python
accuracy = classifier.accuracy(y, y_pred)
print(f'Accuracy with K={k}: {accuracy:.2f}')
```
### 6. Find the Best k Value
To find the optimal k for your dataset, compute accuracy for a range of k values and visualize the results.

```python
k_values = list(range(1, 16))
accuracies = []

for k in k_values:
    y_pred = classifier.predict(X, k)
    acc = classifier.accuracy(y, y_pred)
    accuracies.append(acc)

# Plot accuracy vs. k
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. k - YourDataset')
plt.grid(True)
plt.show()
```

### 7. Visualize Decision Boundaries
Visualize how the classifier separates classes in the dataset.

```python
best_k = 3  # Replace with the optimal k value
n_sample_pts = 100  # Number of points to sample for visualization
classifier.plot_predictions(best_k, n_sample_pts)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Class Boundaries - YourDataset')
plt.show()
```

### Naive Bayes Classifier

This project demonstrates how to build a binary classifier using a **Naive Bayes algorithm**. The project uses a custom Naive Bayes implementation, with a dataset named `yourdataset.csv`. The guide will walk you through the steps of data preparation, training, and evaluation using the custom classifier.

---

## Prerequisites

1. Python installed on your system (3.7 or later recommended).
2. Required libraries:
   - `numpy`
   - `pandas`
3. A CSV file named `yourdataset.csv`.

---

## Dataset Format

Your dataset should include:
- **Features:** Numerical values representing the input data for classification.
- **Labels:** A column indicating whether the record belongs to class "0" (e.g., class A) or "1" (e.g., class B).

Example format of `yourdataset.csv`:

| feature1 | feature2 | feature3 | label |
|----------|----------|----------|-------|
| 1.2      | 0.8      | 3.1      | 1     |
| 2.1      | 1.1      | 0.9      | 0     |

- Features: `feature1`, `feature2`, `feature3`
- Label: `label` (0 = class A, 1 = class B)

### Step 1: Import Libraries

Load the necessary libraries to handle data and implement the classifier.

```python
import numpy as np
import pandas as pd
from naive_bayes import NaiveBayes  # Custom Naive Bayes implementation
```

### **1. Import Required Libraries**

```python
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

### 2. Load and Visualize Data
Load the dataset and split it into features (X) and labels (y).

```python
import numpy as np
import matplotlib.pyplot as plt

# Set plotting styles and options
plt.style.use(['seaborn-v0_8-colorblind', 'seaborn-v0_8-darkgrid'])
plt.rcParams.update({'font.size': 20})
np.set_printoptions(suppress=True, precision=5)

# Load the dataset
yourdataset_train = np.loadtxt('data/yourdataset.csv', skiprows=1, delimiter=',')
yourdataset_val = np.loadtxt('data/yourdataset.csv', skiprows=1, delimiter=',')

# Split features (X) and labels (y)
train_y = yourdataset_train[:, -1]  # Assuming the target is the last column
val_y = yourdataset_val[:, -1]

train_X = yourdataset_train[:, :-1]  # All columns except the last
val_X = yourdataset_val[:, :-1]

# Create plots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Training set plot
scatter_train = axes[0].scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap='viridis')
axes[0].set_title('YourDataset - Training Set')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# Validation set plot
scatter_val = axes[1].scatter(val_X[:, 0], val_X[:, 1], c=val_y, cmap='viridis')
axes[1].set_title('YourDataset - Validation Set')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')

# Add color bar
cbar = fig.colorbar(scatter_train, ax=axes, location='right', shrink=0.8, pad=0.1)
cbar.set_label('Class')

# Finalize and show the plot
plt.tight_layout()
plt.show()
```

### 3. Train the Classifier
Train the classifier on the dataset.

```python
classifier.train(X, y)
```

### 4. Predict Using KNN
Use the trained classifier to make predictions. Choose a value for k (e.g., k=1 or k=2).

```python
k = 3
y_pred = classifier.predict(X, k)
```

### 5. Evaluate Model Accuracy
Calculate the accuracy of the predictions.

```python
accuracy = classifier.accuracy(y, y_pred)
print(f'Accuracy with K={k}: {accuracy:.2f}')
```
### 6. Find the Best k Value
To find the optimal k for your dataset, compute accuracy for a range of k values and visualize the results.

```python
k_values = list(range(1, 16))
accuracies = []

for k in k_values:
    y_pred = classifier.predict(X, k)
    acc = classifier.accuracy(y, y_pred)
    accuracies.append(acc)

# Plot accuracy vs. k
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. k - YourDataset')
plt.grid(True)
plt.show()
```

### 7. Visualize Decision Boundaries
Visualize how the classifier separates classes in the dataset.

```python
best_k = 3  # Replace with the optimal k value
n_sample_pts = 100  # Number of points to sample for visualization
classifier.plot_predictions(best_k, n_sample_pts)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Class Boundaries - YourDataset')
plt.show()
```


### 10. Final Thoughts

Once the K-Means clustering process is complete, you can use the results to analyze the data. The visualizations will give you insights into how your data is grouped and whether it corresponds to any meaningful patterns. Keep in mind that K-Means is sensitive to initial centroid placement and may require multiple runs for stable results.

## 📈 Example Project: Email Classifier Model Comparison

coming soon
