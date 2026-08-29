# A Comparative Study of Dimensionality Reduction and Embedding Techniques on MNIST

## Locally Linear Embedding (LLE) Implementation in R

---

## 📌 Project Overview

The MNIST dataset contains 28 × 28 grayscale images of handwritten digits. Each image can be represented using **784 pixel features**, resulting in a high-dimensional dataset that can be difficult to visualize and analyse directly.

This project investigates **dimensionality reduction and manifold learning** on the MNIST handwritten digit dataset, with **Locally Linear Embedding (LLE)** as the primary technique.

The project implements the major stages of LLE in **R**, following a modular programming approach rather than treating the algorithm as a black-box operation.

The main goal is to transform the high-dimensional MNIST representation into a lower-dimensional space while preserving the **local geometric relationships** between data points.

The project will ultimately produce a **2-dimensional embedding** of MNIST that can be visualized and evaluated.

---

# 🎯 Project Objectives

The main objectives of this project are:

- Load and explore the MNIST handwritten digit dataset using R.
- Convert 28 × 28 images into 784-dimensional feature vectors.
- Normalize pixel values.
- Create a computationally manageable subset of the dataset.
- Identify nearest neighbours for each data point.
- Calculate local reconstruction weights.
- Implement the Locally Linear Embedding algorithm.
- Reduce the 784-dimensional representation to 2 dimensions.
- Visualize the resulting embedding.
- Evaluate preservation of local structure.
- Study the effect of parameters such as the number of neighbours.
- Analyse computational time and scalability.
- Document the complete implementation and experimental results.

---

# 🧠 What is Locally Linear Embedding?

**Locally Linear Embedding (LLE)** is a nonlinear dimensionality-reduction technique based on manifold learning.

The central idea behind LLE is that although a dataset may exist in a high-dimensional space, small neighbourhoods around individual data points can often be approximated using local linear relationships.

For every data point, LLE performs three major steps:

### 1. Find Nearest Neighbours

For every image, the algorithm identifies its K nearest neighbours.

In the current implementation:


K = 10

Conceptually:

              Image i
                 │
       ┌─────────┼─────────┐
       ↓         ↓         ↓
   Neighbour  Neighbour  ... Neighbour
       1          2          10
2. Calculate Reconstruction Weights

The original image is reconstructed using a weighted combination of its neighbours.

Conceptually:

Image i ≈ w1 × Neighbour 1
        + w2 × Neighbour 2
        + ...
        + wK × Neighbour K

The weights describe the local geometric structure surrounding the data point.

The weights satisfy:

w1 + w2 + ... + wK = 1
3. Construct the Low-Dimensional Embedding

After calculating the reconstruction weights, LLE finds a lower-dimensional representation that preserves these local relationships.

For this project:

784-dimensional representation
              ↓
             LLE
              ↓
2-dimensional representation
📊 Dataset
MNIST Handwritten Digits

MNIST is a standard dataset used for machine learning, computer vision and pattern recognition.

Each image is a grayscale handwritten digit ranging from 0 to 9.

Property	Value
Total images	70,000
Training images	60,000
Test images	10,000
Image dimensions	28 × 28
Features per image	784
Number of classes	10
Classes	0–9
Pixel range	0–255

The dataset is loaded using the keras3 package in R.

Dataset Source

The MNIST dataset is accessed through the TensorFlow/Keras dataset interface.

🔄 Data Processing

The original MNIST training data has the structure:

60,000 × 28 × 28

Each image contains:

28 × 28 = 784 pixels

Therefore, each image is converted into a 784-dimensional feature vector.

The resulting feature matrix is:

60,000 × 784

Pixel values are normalized from approximately:

0 – 255

to:

0 – 1

For the initial LLE implementation, a controlled subset of 5,000 images is used.

This allows the algorithm to be developed and tested while keeping computational requirements manageable.

🏗️ Overall Project Pipeline

The complete planned pipeline is:

                    MNIST Dataset
                         │
                         ▼
                   Data Loading
                         │
                         ▼
                   Preprocessing
                         │
                  28 × 28 → 784
                         │
                         ▼
                    Normalize
                         │
                         ▼
                 Select 5,000
                    samples
                         │
                         ▼
              Nearest Neighbours
                    K = 10
                         │
                         ▼
            Local Reconstruction Weights
                         │
                         ▼
                   LLE Embedding
                         │
                         ▼
                 2D Representation
                         │
                         ▼
                  Visualization
                         │
                         ▼
                    Evaluation
                         │
                         ▼
               Experimental Analysis
📁 Repository Structure
A-Comparative-Study-of-Dimensionality-Reduction-and-Embedding-Techniques-on-MNIST/
│
├── README.md
├── MNIST-LLE.Rproj
│
├── R/
│   ├── 01_load_data.R
│   ├── 02_preprocessing.R
│   ├── 03_neighbours.R
│   ├── 04_weights.R
│   ├── 05_lle.R
│   ├── 06_visualization.R
│   └── 07_evaluation.R
│
├── data/
│
├── results/
│   ├── plots/
│   └── embeddings/
│
└── documentation/
🧩 Implementation Modules
01 — Data Loading
R/01_load_data.R

This module is responsible for:

Loading the MNIST dataset.
Separating training and testing data.
Inspecting dataset dimensions.
Examining the structure of the dataset.
Displaying sample handwritten digits.
Checking corresponding digit labels.

The training data has the structure:

x_train → 60,000 × 28 × 28
y_train → 60,000 labels
02 — Preprocessing
R/02_preprocessing.R

This module prepares the MNIST data for the LLE algorithm.

The main operations are:

Image Flattening

Each 28 × 28 image is converted into a 784-dimensional vector.

28 × 28 image
      ↓
784 pixel features

For the selected subset:

5,000 × 784
Normalization

Pixel values are divided by 255:

normalized_pixel = pixel / 255

This converts the pixel values to approximately:

0 – 1
03 — Nearest Neighbours
R/03_neighbours.R

LLE depends on the local neighbourhood surrounding each data point.

For every MNIST image, the algorithm identifies its K nearest neighbours.

Current configuration:

Number of samples = 5,000
K = 10

The resulting neighbour matrix has dimensions:

5,000 × 10

Each row represents one image and contains the indices of its 10 nearest neighbours.

The RANN package is used for efficient nearest-neighbour search.

04 — Reconstruction Weights
R/04_weights.R

This module calculates the local reconstruction weights required by LLE.

For a particular image:

Xi

and its K neighbours:

X1, X2, ..., XK

LLE attempts to reconstruct the original point using a weighted combination of its neighbours:

Xi ≈ w1X1 + w2X2 + ... + wKXK

The reconstruction weights describe the local geometric relationship between the point and its neighbours.

The weights are constrained so that:

w1 + w2 + ... + wK = 1

These weights are calculated independently for each data point.

05 — LLE Embedding
R/05_lle.R

After calculating the reconstruction weights, this module constructs the final LLE embedding.

The goal is to find low-dimensional coordinates that preserve the local reconstruction relationships calculated in the previous stage.

The intended transformation is:

5,000 × 784
      ↓
     LLE
      ↓
5,000 × 2

The resulting two-dimensional coordinates will be used for visualization.

06 — Visualization
R/06_visualization.R

This module will generate visualizations of the LLE embedding.

Planned visualizations include:

2D LLE scatter plot.
Digit-labelled embedding.
Distribution of different digit classes.
Cluster structure.
Comparison of different values of K.
Comparison of different dataset sizes.

The labels will be used for visualization and evaluation, not for calculating the LLE neighbourhood relationships.

07 — Evaluation
R/07_evaluation.R

This module will evaluate the quality of the resulting embedding.

The evaluation will focus on:

Neighbour Preservation

How well local neighbourhoods in the original 784-dimensional space are preserved in the lower-dimensional space.

Local Structure Preservation

Whether nearby points remain relatively close after dimensionality reduction.

Digit Cluster Separation

Whether different handwritten digit classes form distinguishable groups in the 2D embedding.

Computational Performance

The project will also investigate:

Execution time.
Memory requirements.
Effect of dataset size.
Effect of the number of neighbours.
