A Comparative Study of Dimensionality Reduction and Embedding Techniques on MNIST
Project Overview
This project studies dimensionality reduction on the MNIST
handwritten-digit dataset, with Locally Linear Embedding (LLE) as
the primary nonlinear technique. It implements the core LLE mathematics
directly in R and compares classification performance against a
raw-pixel baseline and PCA.

Each MNIST image is a grayscale image of size 28 × 28 pixels,
represented as 784 features. The pipeline uses a stratified subset
for LLE training, a separate validation subset for model selection, and
the official MNIST test set for final evaluation.
Implementation note: This README reflects the implementation
described in the supplied file-by-file study notes. It does not assume
that planned outputs or unimplemented features already exist.

Objectives
- Load and preprocess MNIST images and labels.
- Create stratified training and validation subsets.
- Normalize pixel values from 0--255 to 0--1.
- Implement LLE's neighbor search, reconstruction weights, sparse
  weight matrix, cost matrix, and eigenvector-based embedding in R.
- Project unseen images into a learned LLE representation.
- Compare a raw-pixel 5-nearest-neighbor (5-NN) baseline with
  classifiers trained on LLE and PCA representations.
- Tune LLE neighbor count, embedding dimensions, and classifier family
  using validation data.
- Evaluate selected models on the official test set.
- Generate visualizations, metrics, a confusion matrix, and a Markdown
  summary report.
Dataset
MNIST contains 70,000 grayscale handwritten-digit images across ten
classes (digits 0--9):
  Split                             Number of images Role
  Official training set                       60,000 Source for stratified
                                                     LLE-training and
                                                     validation subsets

  LLE training subset                          4,000 Fit the LLE embedding
                                                     and train models

  Validation subset                            1,000 Select
                                                     hyperparameters and
                                                     classifier

  Official test set                           10,000 Final held-out
                                                 evaluation
Each image has 28 × 28 = 784 pixel features. The code normalizes
pixel intensities by dividing by 255. The 4,000-image training subset
and 1,000-image validation subset are selected stratified by digit
label, with 400 training examples and 100 validation examples per class.
A fixed random seed (42) supports reproducibility.

The implementation described in the study notes loads MNIST with
dslabs::read_mnist(). If the dataset download fails because the
configured HarvardX endpoint has an expired TLS certificate, see
Dataset download troubleshooting.
Repository Structure
The implementation described in the study notes uses the following
files:
A-Comparative-Study-of-Dimensionality-Reduction-and-Embedding-Techniques-on-MNIST/
├── main.R
├── 01_load_and_preprocess.R
├── 02_baseline_model.R
├── 03_lle_core.R
├── 04_lle_visualization.R
├── 05_hyperparameter_tuning.R
├── 06_final_evaluation.R
├── test_lle.R
├── mnist_sample_digits.png       # generated output
├── pca_vs_lle_2d.png             # generated output
├── confusion_matrix.png          # generated output
└── project_summary.md            # generated report (configured path-dependent)
The actual files should be checked against your local repository. The
study notes describe these scripts at the project root; they do not
describe the alternative R/ subfolder structure shown in earlier
planning material.
Pipeline and Modules
main.R --- Orchestration
Coordinates the pipeline by sourcing the modules and calling their
functions in sequence. It sets the random seed, loads and preprocesses
data, runs the baseline, generates PCA-versus-LLE plots, tunes
configurations, and runs final evaluation.
01_load_and_preprocess.R --- Data loading and preparation
- Loads MNIST through dslabs::read_mnist().
- Separates image features and digit labels.
- Checks for missing pixel values.
- Builds stratified subsets of 4,000 training images and 1,000
  validation images.
- Checks that training and validation indices do not overlap.
- Normalizes pixel features to the 0--1 range.
- Produces a plot of 25 sample handwritten digits.
Key returned objects include X_tr_lle, y_tr_lle, X_val, y_val,
X_test, y_test, and the full training features and labels.
02_baseline_model.R --- Raw-pixel baseline
Runs a 5-NN classifier using the original 784-dimensional pixel
features. It reports accuracy, macro precision, macro recall, macro F1,
predictions, and runtime. The KNN classifier is provided by R's class
package; it is not manually implemented.
03_lle_core.R --- Manual LLE implementation
This is the core algorithm module. compute_lle() learns an embedding
from training data; project_lle() maps new observations into the
learned embedding.

The LLE fitting sequence is:
1. Find neighbors: RANN::nn2() identifies the nearest neighbors.
   During training, the code requests k + 1 neighbors and removes the
   first result, assuming it is the point itself.
2. Build local differences: for each image, calculate differences
   between the image and its neighbors.
3. Compute local covariance: form the local matrix (C = Z^T Z).
4. Regularize: add a small diagonal term controlled by reg to
   improve numerical stability.
5. Calculate reconstruction weights: solve the local linear system
   and normalize the weights so they sum to one.
6. Build sparse weight matrix: assemble the local weights into
   sparse matrix (W).
7. Build the cost matrix: calculate (M = (I-W)^T(I-W)).
8. Solve the eigenvalue problem: use RSpectra::eigs_sym() with
   fallback solver configurations to find the smallest relevant
   eigenvalues and eigenvectors.
9. Create the embedding: discard the trivial constant-vector
   solution and retain the requested nontrivial eigenvectors.
The main function's documented defaults are k = 15, d = 30, and
reg = 1e-3; calls can override them. The training subset contains
4,000 images with 784 input features. The tuning pipeline fits up to 50
embedding dimensions.

Out-of-sample projection: project_lle() finds training neighbors
for each new image, computes local reconstruction weights, and uses the
weighted sum of those neighbors' learned coordinates to place the new
image in the existing embedding.
04_lle_visualization.R --- PCA and LLE visualization
Creates two-dimensional PCA and LLE plots using the training subset and
colors points by digit label. The plots are combined and saved as
pca_vs_lle_2d.png.

The study notes identify a visualization issue: the Set1 color palette
supports at most nine distinct colors, while MNIST has ten digit
classes. This can cause a warning and dropped points. Fix the palette
(for example, use a palette supporting ten classes) and regenerate the
plot before relying on it.
05_hyperparameter_tuning.R --- Validation experiments
Evaluates 48 configurations using validation data:
- LLE neighbors: 10, 15, 20, 30
- Embedding dimensions: 10, 20, 30, 50
- Classifiers: 5-NN, RBF SVM, Random Forest
For each neighbor count, the code fits a maximum 50-dimensional LLE
embedding and reuses its leading coordinates for smaller dimensions. The
classifier settings are fixed in this experiment: 5-NN uses k = 5; RBF
SVM uses cost = 10 and gamma = 0.05; Random Forest uses
ntree = 100. The search does not comprehensively tune all
classifier-specific parameters.

Configurations are ranked by validation accuracy, with F1-score used as
a tie-breaker. The study notes record a best validation configuration of
15 LLE neighbors, 50 dimensions, RBF SVM, with 94.30% validation
accuracy for the reported run. This is a recorded result, not a
guarantee that every run will reproduce it.
06_final_evaluation.R --- Held-out test evaluation
Refits LLE with the selected configuration on the 4,000-image training
subset, projects the 10,000 official test images into the learned space,
trains the selected classifier, and calculates test metrics. It also
fits PCA on the training features, transforms the test features using
the training PCA mean and rotation, and compares PCA-based
classification with LLE and the raw-pixel baseline.

The module generates a confusion-matrix heatmap (confusion_matrix.png)
and reports frequently confused digit pairs.
test_lle.R --- Diagnostic checks
A separate diagnostic script, not part of the normal main.R sequence.
The study notes describe checks for:
- Embedding dimensions
- Reconstruction-weight row sums
- Sparsity pattern of (W)
- Symmetry of the cost matrix (M)
- Non-negligible embedding-coordinate variance
- Out-of-sample projection dimensions and finite values
- Label length and missing values
The checks are defined in the script, but the actual diagnostic output
should be reviewed before claiming all checks passed.
Installation and Setup
Use a recent R installation and open the project in RStudio. Install the
packages used by the scripts:
install.packages(c(
  "dslabs",
  "ggplot2",
  "gridExtra",
  "RANN",
  "Matrix",
  "RSpectra",
  "class",
  "e1071",
  "randomForest"
))
The modules use namespaced calls for many dependencies (for example,
RANN::nn2(), RSpectra::eigs_sym(), e1071::svm(), and
randomForest::randomForest()). Install any additional package required
by your local script if it is not listed above.

Set the working directory to the folder containing main.R and the
module scripts. The study notes include an example path (d:/MN LLE)
that is specific to one machine; replace it with your own project
location or open the .Rproj file in RStudio. Relative paths only work
when the working directory is correct.
Running the Project
Run the pipeline from the project root:
source("main.R")
To run diagnostic checks separately:
source("test_lle.R")
The diagnostic script may perform a separate, smaller run. Running
main.R can be computationally expensive because LLE includes
nearest-neighbor searches and an eigenvalue problem.
Reported Results
The supplied study notes record the following results for one run:

  Model                       Test accuracy   Macro F1
  Raw-pixel 5-NN baseline            92.86%     0.9282
  LLE + RBF SVM                      92.93%     0.9291
  PCA + RBF SVM                      94.79%     0.9481

The recorded best LLE configuration had 15 neighbors and 50 embedding
dimensions, with an RBF SVM classifier. The reported validation accuracy
for that configuration was 94.30%.

Interpretation: In the recorded experiment, PCA achieved the highest
test accuracy, LLE was slightly above the raw-pixel 5-NN baseline, and
the results do not show that one dimensionality-reduction technique is
universally superior. These are results documented in the study notes;
rerun the project to verify them in your environment.

Runtime comparisons should be interpreted cautiously because the timing
boundaries differ between the baseline, LLE, and PCA pipelines.
Generated Outputs
The implementation described in the study notes explicitly produces:
  Output                              Description
  mnist_sample_digits.png           Grid of 25 sample training images

  pca_vs_lle_2d.png                 Side-by-side PCA and LLE 2D plots

  confusion_matrix.png              Heatmap of test-set classification
                                      errors

  project_summary.md                Markdown report containing selected
                                      settings and metrics, subject to
                                  its configured output path
The code described in the notes does not explicitly export embedding
coordinates or the complete hyperparameter-tuning table as CSV files.
Those exports would need to be added if required.
Dataset Download Troubleshooting
The supplied implementation calls dslabs::read_mnist(). In the
environment described during setup, this attempted to download from:
https://www2.harvardx.harvard.edu/courses/IDS_08_v2_03/train-images-idx3-ubyte.gz
PowerShell's curl.exe -Iv reported SEC_E_CERT_EXPIRED, meaning the
TLS certificate presented by that endpoint was expired at the time of
testing. This is not a missing-R-package error. Do not disable
certificate verification to work around it.

If the download still fails:
1. Check that the computer's date, time, and time zone are correct.
2. Test the endpoint from a browser or PowerShell to confirm whether
   the certificate issue persists.
3. Use a trusted MNIST mirror if the original endpoint remains
   unavailable.
4. Update the actual data-loading code or configure it to read
   downloaded files locally. Downloading files separately will not help
   if the script continues to call the old endpoint.
5. Verify that all four MNIST gzip files are present and valid before
   running preprocessing.
A mirror URL should be tested in your environment before treating it as
a working replacement. Keep the original data-loading logic and file
format consistent when changing the source.
Implementation Limitations and Checks
- Visualization palette: Set1 has too few colors for ten digit
  classes in the current plot code; fix and regenerate the plot.
- Neighbor self-match assumption: training neighbor search removes
  the first returned neighbor, assuming it is the query point itself.
  Check duplicate points and tied distances if needed.
- Runtime fairness: timings use different boundaries and are not a
  controlled benchmark.
- CSV exports: embeddings and the complete tuning table are not
  explicitly exported by the code described in the notes.
- Hardcoded report path: the report writer uses
  d:/MN LLE/project_summary.md; update this to a portable
  project-relative path such as
  file.path("results", "project_summary.md") and ensure the
  directory exists.
- Diagnostics: confirm actual test_lle.R output before reporting
  that all checks passed.
- Reproducibility: the seed is fixed at 42, but package versions,
  numerical libraries, and platform differences can still affect
  runtime and numerical results.
Main Contribution
The main contribution is a modular, manual implementation of the LLE
pipeline in R rather than relying on a single black-box LLE function.
The project computes local neighborhoods and reconstruction weights,
constructs sparse (W) and the cost matrix (M = (I-W)^T(I-W)), obtains
the embedding from eigenvectors, and implements a procedure for
projecting unseen images. It then compares LLE with PCA and an
original-pixel baseline, selects configurations on validation data, and
evaluates the selected models on the held-out test set.
References
- MNIST database: http://yann.lecun.com/exdb/mnist/
- dslabs R package: https://cran.r-project.org/package=dslabs
- RANN R package: https://cran.r-project.org/package=RANN
- RSpectra R package: https://cran.r-project.org/package=RSpectra
- e1071 R package: https://cran.r-project.org/package=e1071
- randomForest R package:
  https://cran.r-project.org/package=randomForest
