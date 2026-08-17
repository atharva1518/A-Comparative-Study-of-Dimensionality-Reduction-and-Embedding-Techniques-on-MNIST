# A-Comparative-Study-of-Dimensionality-Reduction-and-Embedding-Techniques-on-MNIST
# MNIST Dimensionality Reduction and Embedding



### Project Overview

This project focuses on studying **dimensionality reduction and embedding techniques** using the MNIST handwritten digit dataset.

MNIST consists of 28 × 28 grayscale images, where each image can be represented using **784 pixel features**. The project aims to transform this high-dimensional representation into a lower-dimensional space while preserving meaningful patterns and structures in the data.

The project will be implemented using **R**.

---

## Objectives

* Load and preprocess the MNIST dataset.
* Convert 28 × 28 images into 784-dimensional feature vectors.
* Normalize the dataset for analysis.
* Apply dimensionality reduction and embedding techniques.
* Visualize the resulting lower-dimensional representations.
* Compare the performance and characteristics of different techniques.

### Techniques

* **PCA (Principal Component Analysis)**
* **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
* **UMAP (Uniform Manifold Approximation and Projection)**

---

## Dataset

### MNIST Handwritten Digits

| Property           | Value      |
| ------------------ | ---------- |
| Total Samples      | 70,000     |
| Image Size         | 28 × 28    |
| Features per Image | 784        |
| Classes            | 10         |
| Classes            | Digits 0–9 |

The original images will be transformed into a feature matrix of the form:

[
M \times 784
]

The dimensionality reduction techniques will then generate lower-dimensional representations.

---

## Proposed Approach

```text
MNIST Images
     │
     ▼
28 × 28 Pixels
     │
     ▼
784-Dimensional Representation
     │
     ├──────── PCA
     │
     ├──────── t-SNE
     │
     └──────── UMAP
              │
              ▼
      Lower-Dimensional
         Embeddings
              │
              ▼
       Visualization &
         Comparison
```

---

## Expected Analysis

The techniques will be compared based on:

* Dimensionality reduction
* Information/variance preservation
* Digit cluster separation
* Visualization quality
* Computational efficiency

---

## Technology

**Programming Language:** R

**Expected Libraries:**

* `keras`
* `ggplot2`
* `Rtsne`
* `umap`
* `dplyr`

---

## Repository Structure

```text
MNIST-Dimensionality-Reduction/
│
├── README.md
├── R/
├── data/
├── results/
└── documentation/
```

The repository structure will be expanded as the project progresses.

---

## Project Status

**Status:** Initial Proposal / Development Started

### Current Progress

* [x] Project topic selected
* [x] MNIST dataset selected
* [x] Project objectives defined
* [x] Proposed techniques identified
* [ ] Dataset preprocessing
* [ ] PCA implementation
* [ ] t-SNE implementation
* [ ] UMAP implementation
* [ ] Visualization
* [ ] Comparative analysis
* [ ] Final report

---

---

## Future Work

The project will progressively implement the proposed techniques, generate visualizations, and perform a comparative analysis of the resulting embeddings.

---

**Language:** R
**Dataset:** MNIST
