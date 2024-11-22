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

### Prerequisites

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
from scipy.spatial import distance
import kmeans
import pandas as pd
from matplotlib.image import imread
```

### 2. Load Your Data

#### For Numerical Data:
To start, load your dataset (e.g., a CSV file) using Pandas. Then, convert it into a NumPy array to prepare it for the K-Means algorithm.

```python
# Load a CSV file as a DataFrame
df = pd.read_csv('data/your_dataset.csv')

# Convert the DataFrame to a NumPy array
your_data = df.values

# Preview the data
print(your_data)
```

#### For Image Data:
To start, load your dataset (e.g., a CSV file) using Pandas. Then, convert it into a NumPy array to prepare it for the K-Means algorithm.

```python
# Load an image
image = imread('data/your_image.jpg')

# Flatten the image
def flatten(img):
    '''Flattens an image to N 1D vectors.'''
    num_rows, num_cols, rgb = img.shape    
    flattened_img = img.reshape(num_rows * num_cols, rgb)
    return flattened_img

flattened_image = flatten(image)

# Preview the flattened image shape
print(flattened_image.shape)
```

### 3. Prepare Your Data (For Image or Numerical Data)

#### For Image Data:
If you're working with images, you need to load the image and flatten it into 1D vectors, which K-Means can use as data points for clustering. This is necessary because each pixel in the image is represented by its RGB (Red, Green, Blue) values, and these need to be treated as individual data points in the clustering process.

```python
# Load an image
image = imread('data/your_image.jpg')

# Flatten the image to 1D vectors (each pixel's RGB values)
def flatten(img):
    '''Flattens an image to N 1D vectors.'''
    num_rows, num_cols, rgb = img.shape    
    return img.reshape(num_rows * num_cols, rgb)

# Flatten the image
flattened_image = flatten(image)

# Preview the flattened image shape (total pixels x 3 RGB values)
print(flattened_image.shape)
```

### 4. Initialize K-Means Class

Now that you have your data prepared, it's time to initialize the **K-Means** class, where you'll specify the number of clusters (k) and apply the algorithm to your data.

#### For Numerical Data:
```
# Create an instance of the KMeans class with the input data
cluster = kmeans.KMeans(your_data)

# Specify the number of clusters
k = 3

# Initialize the centroids
init_centroids = cluster.initialize(k)

# Preview the initial centroids
print(init_centroids)
```

#### For Image Data:
```python
# Create an instance of the KMeans class with the flattened image data
image_cluster = kmeans.KMeans(flattened_image)

# Specify the number of clusters
k = 5

# Initialize the centroids
image_init_centroids = image_cluster.initialize(k)

# Preview the initial centroids
print(image_init_centroids)
```

### 5. Assign Data Points to Clusters (Update Labels)

After initializing the centroids, the next step is to assign each data point to its closest centroid, thereby forming clusters.

#### For Numerical Data:
```python
# Assign data points to the nearest centroids, producing cluster labels
new_labels = cluster.update_labels(init_centroids)

# Preview the new labels (which data points belong to which clusters)
print(new_labels)
```

#### For Image Data:
```python
# Assign image pixels (data points) to the nearest centroids
image_new_labels = image_cluster.update_labels(image_init_centroids)

# Preview the new labels for image data
print(image_new_labels)
```

### 6. Update Centroids

The next step is to update the centroids based on the mean of the data points assigned to each cluster. 

#### For Numerical Data:
```python
# Update the centroids and calculate the difference between the new and previous centroids
new_centroids, diff_from_prev_centroids = cluster.update_centroids(k, new_labels, init_centroids)

# Preview the new centroids
print(new_centroids)
```

#### For Image Data:
```python
# Update the centroids and calculate the difference for image data
image_new_centroids, image_diff_from_prev_centroids = image_cluster.update_centroids(k, image_new_labels, image_init_centroids)

# Preview the updated centroids for the image data
print(image_new_centroids)
```

### 7. Perform the Clustering Process

Now, you're ready to run the K-Means clustering process for your data with the chosen number of clusters.

#### For Numerical Data:
```python
# Perform the clustering process
cluster.cluster(k)

# Preview the clustered data
cluster.plot_clusters()

# Display the plot
plt.show()
```

#### For Image Data:
```python
# Perform the clustering process for image data
image_cluster.cluster(k)

# Replace colors in the image with the centroid colors
image_cluster.replace_color_with_centroid()

# Reshape the compressed image back to its original shape
compressed_image = np.reshape(image_cluster.data, image.shape)

# Plot the original and compressed images side by side
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot original image
ax[0].imshow(image)
ax[0].axis('off')
ax[0].set_title('Original Image')

# Plot compressed image
ax[1].imshow(compressed_image)
ax[1].axis('off')
ax[1].set_title('Compressed Image')

plt.show()
```

### 8. Evaluate Your Model: The Elbow Method

The **Elbow Method** is a useful technique to determine the optimal number of clusters (`k`) for your data. This method plots the sum of squared distances (inertia) against various values of `k` to find the point where adding more clusters provides diminishing returns.

#### For Numerical Data:
```python
# Set the maximum number of clusters to evaluate
max_k = 10

# Generate the elbow plot
cluster.elbow_plot(max_k)
plt.title("Elbow Plot (Numerical Data)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()
```

#### For Image Data:
```python
# Set the maximum number of clusters to evaluate
max_k = 10

# Generate the elbow plot for the image
image_cluster.elbow_plot(max_k)
plt.title("Elbow Plot (Image Data)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()
```

### 9. Batch Clustering (Optional)

If you want to experiment with batch clustering, you can perform clustering in batches rather than iterating over the entire dataset all at once. This can be useful for large datasets.

#### For Numerical Data:
```python
# Perform batch clustering
cluster.cluster_batch(k=3, n_iter=10)

# Plot the clustered data
cluster.plot_clusters()

# Display the plot
plt.show()
```

#### For Image Data:
```python
# Perform batch clustering for image data
image_cluster.cluster_batch(k=5, n_iter=20)

# Replace colors in the image with the centroid colors
image_cluster.replace_color_with_centroid()

# Reshape the compressed image back to its original shape
compressed_image_batch = np.reshape(image_cluster.data, image.shape)

# Plot the original and compressed images side by side
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot original image
ax[0].imshow(image)
ax[0].axis('off')
ax[0].set_title('Original Image')

# Plot compressed image
ax[1].imshow(compressed_image_batch)
ax[1].axis('off')
ax[1].set_title('Compressed Image (Batch Clustering)')

plt.show()
```

### 10. Final Thoughts

Once the K-Means clustering process is complete, you can use the results to analyze the data. The visualizations will give you insights into how your data is grouped and whether it corresponds to any meaningful patterns. Keep in mind that K-Means is sensitive to initial centroid placement and may require multiple runs for stable results.

## 📈 Example Project: Email Classifier Model Comparison

coming soon
