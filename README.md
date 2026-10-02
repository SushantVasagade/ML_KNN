# K-Nearest Neighbors (KNN) Classification

## 📌 Overview

This project implements the **K-Nearest Neighbors (KNN)** machine learning algorithm for solving a classification problem.

KNN is a **supervised machine learning algorithm** that classifies a new data point based on the classes of its nearest neighboring data points. The algorithm does not build an explicit mathematical model during training. Instead, it stores the training data and performs classification when a new observation is provided.

This experiment demonstrates the complete workflow of a KNN classification model, including:

- Dataset loading and exploration
- Data preprocessing
- Feature and target separation
- Train-test splitting
- Feature scaling
- KNN model implementation
- Model prediction
- Model evaluation
- Accuracy calculation
- Confusion matrix
- Classification report
- Testing predictions

---

## 🎯 Aim

To implement the **K-Nearest Neighbors (KNN)** classification algorithm and evaluate its performance using appropriate classification metrics.

---

## 🎯 Objectives

The main objectives of this experiment are:

1. To understand the working principle of the KNN algorithm.
2. To load and explore a classification dataset.
3. To preprocess the dataset before model training.
4. To divide the dataset into training and testing sets.
5. To apply feature scaling where required.
6. To implement the KNN classifier using Scikit-learn.
7. To make predictions on unseen test data.
8. To evaluate the model using classification metrics.
9. To understand the effect of the value of `K` on classification.
10. To visualize and interpret the confusion matrix.

---

## 🧠 Theory

### What is K-Nearest Neighbors?

**K-Nearest Neighbors (KNN)** is a supervised machine learning algorithm used mainly for:

- Classification
- Regression

In classification, KNN assigns a class to a new data point based on the classes of its **K nearest neighboring points**.

The basic idea is:

> Similar data points are generally located close to each other in the feature space.

For example, if most of the nearest neighbors of a new data point belong to Class A, KNN will classify the new data point as Class A.

---

## ⚙️ Working of KNN

The KNN algorithm follows these basic steps:

1. Select the value of `K`.
2. Calculate the distance between the new data point and all training data points.
3. Identify the `K` nearest data points.
4. Observe the class labels of these neighbors.
5. Select the class having the majority of neighbors.
6. Assign that class to the new data point.

---

## 📐 Distance Calculation

KNN commonly uses **Euclidean distance**.

The Euclidean distance between two points can be calculated as:

```text
d = √[(x₁-y₁)² + (x₂-y₂)² + ... + (xₙ-yₙ)²]
```

For two-dimensional data:

```text
d = √[(x₂-x₁)² + (y₂-y₁)²]
```

The points with the smallest distances are considered the nearest neighbors.

---

## 🔢 Choosing the Value of K

The value of `K` is an important hyperparameter.

### Small K

A very small value such as:

```text
K = 1
```

can make the model sensitive to noise and individual observations.

This may result in **overfitting**.

### Large K

A very large value of `K` considers many neighboring points and may make the model too generalized.

This can result in **underfitting**.

Therefore, an appropriate value of `K` should be selected based on the dataset and validation performance.

---

## 📊 Dataset

The experiment uses a classification dataset containing numerical features and class labels.

The dataset is explored before model training to understand:

- Number of observations
- Number of features
- Feature values
- Target classes
- Missing values
- Statistical properties

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Data Preprocessing
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
KNN Model
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Results
```

---

## 🧹 Data Preprocessing

Before applying KNN, the data is checked and prepared.

The preprocessing steps include:

- Checking dataset shape
- Checking column information
- Checking missing values
- Checking duplicate records
- Separating features and target
- Splitting data into training and testing sets
- Scaling numerical features

Feature scaling is particularly important for KNN because the algorithm uses distance calculations.

---

## 📏 Feature Scaling

KNN is a distance-based algorithm.

If one feature has a much larger numerical range than another feature, it can dominate the distance calculation.

For example:

```text
Age       → 18–60
Salary    → 20,000–200,000
```

Salary would have a much larger numerical scale.

Therefore, scaling can improve the fairness of distance calculations.

A common technique is **Standardization**:

```text
z = (x - μ) / σ
```

where:

- `x` = original value
- `μ` = mean
- `σ` = standard deviation

---

## 🧪 Model Implementation

The KNN classifier is implemented using:

```python
from sklearn.neighbors import KNeighborsClassifier
```

A KNN model can be created using:

```python
model = KNeighborsClassifier(n_neighbors=5)
```

Here:

```text
n_neighbors = 5
```

means that the model considers the five nearest training observations when making a prediction.

---

## 📈 Model Evaluation

After training, the model is evaluated on unseen test data.

Important evaluation metrics include:

### Accuracy

Accuracy represents the proportion of correctly classified observations.

```text
Accuracy =
Correct Predictions / Total Predictions
```

---

### Confusion Matrix

A confusion matrix shows the relationship between actual and predicted class labels.

For a binary classification problem:

```text
                 Predicted
              Positive Negative
Actual Positive   TP       FN
       Negative   FP       TN
```

Where:

- **TP** = True Positive
- **TN** = True Negative
- **FP** = False Positive
- **FN** = False Negative

---

### Precision

Precision measures how many observations predicted as positive are actually positive.

```text
Precision = TP / (TP + FP)
```

---

### Recall

Recall measures how many actual positive observations were correctly identified.

```text
Recall = TP / (TP + FN)
```

---

### F1-Score

F1-score combines precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Jupyter Notebook | Experiment development |
| NumPy | Numerical operations |
| Pandas | Data manipulation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning implementation |

---

## 📂 Project Structure

```text
ML_KNN/
│
├── KNN.ipynb
├── .gitignore
└── README.md
```

---

## 💻 Requirements

Install the required Python libraries using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

If you are using the Anaconda environment:

```bash
conda activate ml
```

Then launch Jupyter Notebook:

```bash
jupyter notebook
```

---

## ▶️ How to Run

### Step 1: Clone the Repository

```bash
git clone https://github.com/SushantVasagade/ML_KNN.git
```

### Step 2: Open the Project

```bash
cd ML_KNN
```

### Step 3: Activate the Environment

```bash
conda activate ml
```

### Step 4: Start Jupyter Notebook

```bash
jupyter notebook
```

### Step 5: Run the Notebook

Open `KNN.ipynb` and execute the cells sequentially.

---

## 📌 Key Learning Outcomes

After completing this experiment, the following concepts are understood:

- Supervised learning
- Classification
- KNN algorithm
- Distance-based learning
- Euclidean distance
- Selection of `K`
- Feature scaling
- Train-test splitting
- Model prediction
- Confusion matrix
- Accuracy
- Precision
- Recall
- F1-score
- Classification report

---

## ⚠️ Advantages of KNN

- Simple and easy to understand
- Easy to implement
- No complicated training phase
- Can be effective for small datasets
- Can handle multi-class classification

---

## ⚠️ Limitations of KNN

- Prediction can be computationally expensive for large datasets
- Sensitive to feature scaling
- Sensitive to irrelevant features
- Performance depends strongly on the value of `K`
- Can be affected by noisy observations
- Requires storing the training dataset

---

## 🌍 Real-World Applications

KNN can be used in applications such as:

- Recommendation systems
- Pattern recognition
- Image classification
- Customer classification
- Medical diagnosis support
- Document classification
- Handwriting recognition

---

## 📝 Conclusion

The **K-Nearest Neighbors (KNN)** algorithm was successfully implemented for classification.

The experiment demonstrated the complete machine learning workflow, starting from dataset exploration and preprocessing to model training, prediction, and evaluation.

The experiment also demonstrates why **feature scaling and appropriate selection of K** are important for distance-based algorithms.

KNN provides a simple yet effective approach for classification problems, particularly when the dataset is relatively small and similar observations tend to have similar class labels.

---

## 👨‍💻 Author

**Sushant Vasagade**

B.Tech – Information Technology

Government College of Engineering, Karad

---

## 📚 Repository

[![GitHub Repository](https://img.shields.io/badge/GitHub-ML__KNN-black?logo=github)](https://github.com/SushantVasagade/ML_KNN)

**Repository:** [ML_KNN](https://github.com/SushantVasagade/ML_KNN)

---

## 📚 Machine Learning Lab

**Experiment:** K-Nearest Neighbors Classification

**Domain:** Machine Learning

**Algorithm:** K-Nearest Neighbors (KNN)
