# Email Spam Classification: Custom Naive Bayes and k-Nearest Neighbors

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.2-green)
![Pandas](https://img.shields.io/badge/Pandas-2.0.3-red)

## 📧 Overview

This project implements **custom Naive Bayes and k-Nearest Neighbors (k-NN) classifiers** to identify email spam. These classifiers provide a foundational understanding of supervised machine learning techniques for text classification tasks.

## 🚀 **Key Features**

### **Custom Naive Bayes Classifier**
- **Multinomial Likelihood**: Suited for discrete data such as word counts (e.g., email classification).
- **Class Priors and Likelihoods**: Calculates the probability of each class (e.g., spam or ham) based on training data and class priors.
- **Laplace Smoothing**: Prevents zero probabilities for unseen features by applying a smoothing factor to the likelihoods.
- **Logarithmic Optimization**: Uses log probabilities for numerical stability, especially when handling large datasets.

### **Custom k-Nearest Neighbors Classifier**
- **Custom Distance Metric**: Calculates distances between data points based on custom feature-rich datasets.
- **k Selection**: Provides flexibility in choosing the number of neighbors (`k`) for classification, allowing for fine-tuning of the model.
- **Majority Voting**: Makes predictions based on the most frequent label among the `k` nearest neighbors.

### **Email Preprocessor for `.txt` Files**
- **Text Preprocessing**: Tokenizes and cleans raw email text data by removing stopwords, punctuation, and applying lowercasing.
- **Feature Extraction**: Converts the preprocessed text into numerical feature vectors, which can be directly used for Naive Bayes and k-Nearest Neighbors models.
- **Support for `.txt` Files**: Reads emails from `.txt` files, extracting the content for classification tasks.

### **Custom Implementations (No External Libraries)**
- **No External Libraries**: Both Naive Bayes and k-NN classifiers are implemented from scratch without relying on `sklearn`, ensuring a clear understanding of the underlying algorithms.
- **Numeric Data Handling**: Both models are designed to handle continuous numerical features in addition to text data.
- **Model Evaluation**: Implements evaluation metrics such as accuracy and confusion matrix to assess the models’ performance.
- **Core Python Libraries**: Uses core Python libraries like `numpy`, `pandas`, and custom helper functions to preprocess data and build the models.


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

### **Email Preprocessor Implementation**
- **File:** `email_preprocessor.py`
- **Methods:**
  - `tokenize_words(text)`: Tokenizes a given email text into a list of words.
  - `count_words(email_path='data/enron')`: Counts the frequency of each word in the entire dataset.
  - `find_top_words(word_freq, num_features=200)`: Retrieves the top N most frequent words from the dataset.
  - `make_feature_vectors(top_words, num_emails, email_path='data/enron')`: Generates feature vectors for each email based on the top N words.
  - `make_train_test_sets(features, y, test_prop=0.2, shuffle=True)`: Splits the dataset into training and testing sets.
  - `retrieve_emails(inds, email_path='data/enron')`: Retrieves emails based on indices from the dataset.
- **Approach:**
  - **Tokenization:** Tokenizes email text into words, normalizing them to lowercase.
  - **Word Counting:** Counts word frequencies across all emails in the dataset.
  - **Top Words:** Selects the most frequent words as features for machine learning algorithms.
  - **Feature Vector Construction:** Constructs a vector of word counts for each email, representing its feature set.
  - **Train-Test Split:** Divides the dataset into training and testing subsets for model evaluation.

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

#### **1. Import Required Libraries**

```python
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

#### 2. Load and Visualize Data
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

#### 3. Train the Classifier
Train the classifier on the dataset.

```python
classifier.train(X, y)
```

#### 4. Predict Using KNN
Use the trained classifier to make predictions. Choose a value for k (e.g., k=1 or k=2).

```python
k = 3
y_pred = classifier.predict(X, k)
```

#### 5. Evaluate Model Accuracy
Calculate the accuracy of the predictions.

```python
accuracy = classifier.accuracy(y, y_pred)
print(f'Accuracy with K={k}: {accuracy:.2f}')
```
#### 6. Find the Best k Value
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

#### 7. Visualize Decision Boundaries
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

#### 8. Final Thoughts
The K-Nearest Neighbors (KNN) classifier is a simple yet effective algorithm for classification tasks, where the model predicts the class of a data point based on the majority class of its nearest neighbors. The performance of KNN is highly influenced by the choice of k, with smaller values more sensitive to noise and larger values potentially underfitting. By evaluating accuracy across a range of k values and visualizing decision boundaries, we can find the optimal k for better generalization. While KNN is computationally expensive for large datasets and struggles with high-dimensional data, it remains a strong, interpretable choice for smaller datasets with well-defined boundaries.

### Naive Bayes Classifier

The implementation of the Naive Bayes classification algorithm using custom Python code, designed to handle both text and numeric datasets. The model is trained, tested, and evaluated on two distinct datasets: a text dataset (such as emails) and a numeric dataset (such as feature vectors from various data sources). This approach eliminates the need for external machine learning libraries, highlighting a deep understanding of core machine learning concepts and feature engineering.

Before running the project, make sure you have the following libraries installed in your Python environment:  
- **NumPy**  
- **Pandas**  
- **Matplotlib**  
- **Random**
- **os**

Install them using `pip`:

```python
pip install numpy pandas matplotlib
```

#### **For Numeric Data**

##### 1. Import Required Libraries

```python
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

##### 2. Load and Visualize Data
Begin by loading the dataset into your Python environment. You can visualize the first few rows of the dataset to understand its structure.

```python
# Load the dataset
dataset_path = 'path_to_your_dataset.csv'  # Replace with actual file path
data = pd.read_csv(dataset_path)

# Visualize the first few rows of the dataset
print(data.head())

# Separate features (X) and labels (y)
X = data.drop(columns=['label']).values  # Replace 'label' with your target column name
y = data['label'].values  # Replace 'label' with your target column name

# Optionally, visualize data distribution for target labels
print("Label distribution:")
print(pd.Series(y).value_counts())
```

##### 3. Train-Test Split
The data is split into training and testing sets using a custom train_test_split function, ensuring that the dataset is shuffled before splitting.

```python
# Custom train_test_split function
def train_test_split_custom(data, labels, test_size=0.2, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    data = np.array(data)
    labels = np.array(labels)

    # Shuffle data and labels in unison
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    # Split the dataset
    test_size_count = int(test_size * len(data))
    X_test = data[:test_size_count]
    y_test = labels[:test_size_count]
    X_train = data[test_size_count:]
    y_train = labels[test_size_count:]

    return X_train, X_test, y_train, y_test

# Split the dataset using the custom function
X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_seed=42)
```

##### 4. Initialize and Train Naive Bayes Classifier
Now, initialize the Naive Bayes classifier and train it on the training data. The custom classifier will calculate the class priors and feature likelihoods.

```python
from naive_bayes import NaiveBayes  # Import the custom Naive Bayes class

# Initialize and train the Naive Bayes classifier
num_classes = len(np.unique(y))  # Calculate the number of unique classes
nb_classifier = NaiveBayes(num_classes=num_classes)

nb_classifier.train(X_train, y_train)
```

##### 5. Make Predictions and Evaluate the Model
Use the trained Naive Bayes model to make predictions on the test set, then evaluate its accuracy and generate a confusion matrix.

```python
# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Evaluate the classifier
accuracy = nb_classifier.accuracy(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}%")

# Generate and display a confusion matrix
conf_matrix = nb_classifier.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
```

##### 6. Visualize the Confusion Matrix
Visualizing the confusion matrix helps in understanding the performance of the classifier.

```python
# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

#### **For Text Data**

##### 1. Import Required Libraries

```python
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

##### 2. Preprocess the Text Dataset
Assume that the text dataset is a collection of text samples (such as documents or sentences) with corresponding labels. The preprocessing steps remain largely the same, but now you’re working with a general text dataset.

```python
import email_preprocessor as epp
# Count words across the entire dataset (general text data)
word_freq, num_samples = epp.count_words(email_path='data/text')  # Change to your text dataset path

# Find the top `num_features` most frequent words in the dataset
top_words, counts = epp.find_top_words(word_freq, num_features=200)

# Generate feature vectors for each text sample (document/line/sentence)
features, labels = epp.make_feature_vectors(top_words, num_samples, email_path='data/text')  # Update path

# Split the dataset into training and test sets
x_train, y_train, inds_train, x_test, y_test, inds_test = epp.make_train_test_sets(features, labels, test_prop=0.2, shuffle=True)
```

##### 3. Train the Naive Bayes Classifier
The data is split into training and testing sets using a custom train_test_split function, ensuring that the dataset is shuffled before splitting.

```python
# Initialize Naive Bayes with the number of classes (you can adjust this for your dataset)
nb = NaiveBayes(num_classes=2)  # Assuming binary classification for now (e.g., spam vs. non-spam)

# Train the Naive Bayes classifier
nb.train(x_train, y_train)
```

##### 4. Make Predictions and Evaluate the Model
Make predictions using the trained model and evaluate its performance on the test data.

```python
# Make Predictions and Evaluate
# Predict the classes for the test data
y_pred = nb.predict(x_test)

# Calculate the accuracy of the model
accuracy = nb.accuracy(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}%')

# Create and display the confusion matrix
conf_matrix = nb.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Optionally, visualize the confusion matrix
fig, ax = plt.subplots()
cax = ax.matshow(conf_matrix, cmap='Blues')
fig.colorbar(cax)
ax.set_xticklabels([''] + ['Class 0', 'Class 1'])  # Update according to your labels
ax.set_yticklabels([''] + ['Class 0', 'Class 1'])  # Update according to your labels
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

##### Final Thoughts
The Naive Bayes classifier is a fundamental and widely-used machine learning model based on Bayes' theorem, often utilized for both text and numeric data classification tasks. Its simplicity and efficiency make it especially useful for applications like spam detection and document categorization. While Naive Bayes assumes feature independence, which may not always hold in real-world scenarios, it still performs surprisingly well in practice, particularly when features are conditionally independent or nearly so. By leveraging custom code for training, testing, and evaluation, this implementation enhances your understanding of machine learning principles, including probability calculations, data preprocessing, and model evaluation. Although Naive Bayes may struggle with datasets that violate the independence assumption or have highly correlated features, it remains a fast and interpretable tool for classification tasks, especially in text-based domains.

## 📈 Example Project: Email Classifier Model Comparison

### Step 1: Import Required Libraries and Setup
Set up the environment by importing necessary libraries for data processing, visualization, and numerical operations. Update plotting styles and print settings for better readability.

```python
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])
plt.rcParams.update({'font.size': 20})

np.set_printoptions(suppress=True, precision=5)

# Automatically reload external modules
%load_ext autoreload
%autoreload 2
```

### Step 2: Preprocess Emails
Use a custom preprocessing module to extract features and labels for email classification. Test utility functions like `count_words` and `find_top_words` to identify significant terms and generate feature vectors.

```python
import email_preprocessor as epp

#### Test `count_words` and `find_top_words`
top_words, top_counts = epp.find_top_words(word_freq)
features, y = epp.make_feature_vectors(top_words, num_emails)
```

### Step 3: Split Data into Training and Testing Sets
Split the processed data into training and testing sets with a fixed random seed for reproducibility.

```python
np.random.seed(0)
x_train, y_train, inds_train, x_test, y_test, inds_test = epp.make_train_test_sets(features, y)
```

### Step 4: Save Data in Binary Format
Save the training and testing data into `.npy` files for efficient storage and quick access in future runs.

```python
np.save('data/email_train_x.npy', x_train)
np.save('data/email_train_y.npy', y_train)
np.save('data/email_train_inds.npy', inds_train)
np.save('data/email_test_x.npy', x_test)
np.save('data/email_test_y.npy', y_test)
np.save('data/email_test_inds.npy', inds_test)
```

### Step 5: Load Preprocessed Data
Reload the preprocessed data from the saved binary files to use in model training and evaluation.

```python
x_train = np.load('data/email_train_x.npy')
y_train = np.load('data/email_train_y.npy')
inds_train = np.load('data/email_train_inds.npy')
x_test = np.load('data/email_test_x.npy')
y_test = np.load('data/email_test_y.npy')
inds_test = np.load('data/email_test_inds.npy')
```

### Step 6: Train Naive Bayes on Email Data
Train the Naive Bayes classifier on the email dataset and predict the test set labels. Compute the classification accuracy.

```python
num_classes = 2
enron_nbc = NaiveBayes(num_classes=num_classes)

enron_nbc.train(x_train, y_train)

test_y_pred = enron_nbc.predict(x_test)

accuracy = enron_nbc.accuracy(y_test, test_y_pred)

print(f"Accuracy: {accuracy:.2f}%")
```

### Step 7: Compute Confusion Matrix
Generate a confusion matrix to evaluate the model's performance in terms of true positives, false positives, true negatives, and false negatives.

```python
confusion_matrix = enron_nbc.confusion_matrix(y_test, test_y_pred)
print(confusion_matrix)
```

### Step 9: Comparison with KNN
Compare the Naive Bayes results with a k-Nearest Neighbors (KNN) model to understand performance differences in accuracy and confusion matrix.

```python
from knn import KNN

x_train = np.load('data/email_train_x.npy')
y_train = np.load('data/email_train_y.npy')
inds_train = np.load('data/email_train_inds.npy')
x_test = np.load('data/email_test_x.npy')
y_test = np.load('data/email_test_y.npy')
inds_test = np.load('data/email_test_inds.npy')
num_classes = 2

enron_knn = KNN(num_classes)
enron_knn.train(x_train, y_train)

test_y_pred = enron_knn.predict(x_test, k=3)

accuracy = enron_knn.accuracy(y_test, test_y_pred)
print(f"Accuracy: {accuracy:.2f}%")

confusion_matrix = enron_knn.confusion_matrix(y_test, test_y_pred)
print(confusion_matrix)
```

### Final Thoughts
This project showcased the end-to-end process of building and evaluating email classification models using Naive Bayes and k-Nearest Neighbors (KNN). It covered preprocessing data, splitting datasets, training models, and comparing performance through accuracy and confusion matrices. Naive Bayes demonstrated strong results with its probabilistic approach, while KNN offered a non-parametric alternative. Future improvements could include exploring advanced feature engineering, optimizing hyperparameters, and testing on larger datasets to enhance scalability and real-world applicability.
