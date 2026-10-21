# ==============================================================================
# 03_lle_core.R
# Genuine Locally Linear Embedding (LLE) & Out-of-Sample Manifold Projection
# ==============================================================================
# Mathematical Formulation (Roweis & Saul, 2000):
#
# LLE assumes that high-dimensional data lies on or near a smooth non-linear
# manifold. The algorithm operates in 3 distinct mathematical steps:
#
# Step 1: Characterize Local Geometry (Nearest Neighbors)
#   For each point x_i in R^D, find its k nearest neighbors N(i) using Euclidean distance.
#
# Step 2: Compute Reconstruction Weights (W)
#   Represent each point x_i as a linear combination of its neighbors:
#     min_W sum_i || x_i - sum_{j in N(i)} W_ij * x_j ||^2
#     subject to: sum_j W_ij = 1  (invariance to translation)
#   Solving for W_i involves the local covariance matrix C_i of neighbor differences:
#     C_i = (X_N - x_i)^T (X_N - x_i)
#   To prevent numerical instability when k > D or collinearity occurs, C_i is
#   regularized: C_i <- C_i + epsilon * I_k, where epsilon = reg * trace(C_i).
#   The weight vector is w_i = C_i^(-1) * 1_k / (1_k^T * C_i^(-1) * 1_k).
#
# Step 3: Compute Low-Dimensional Embedding (Y)
#   Map high-dimensional points x_i in R^D to low-dimensional points y_i in R^d (d << D)
#   while preserving the local reconstruction weights W:
#     min_Y sum_i || y_i - sum_j W_ij * y_j ||^2
#   This optimizes tr(Y^T M Y), where M = (I - W)^T (I - W) is sparse, symmetric (N x N).
#   Subject to constraints: (1/N) Y^T Y = I_d  (unit covariance) and sum_i y_i = 0 (centered).
#   The embedding Y is formed by the 2nd to (d+1)-th smallest eigenvectors of M.
#
# Out-of-Sample Projection (Saul & Roweis, 2003):
#   To project a new test sample x* into the learned embedding without data leakage:
#   1. Find k nearest neighbors of x* in the training dataset X_train.
#   2. Compute reconstruction weights w* such that x* ~ sum_{j in N(x*)} w*_j * x_j.
#   3. Map into embedding space: y* = sum_{j in N(x*)} w*_j * y_j.
# ==============================================================================

suppressPackageStartupMessages({
  library(Matrix)
  library(RSpectra)
  library(RANN)
})

compute_lle <- function(X, k = 15, d = 30, reg = 1e-3) {
  N <- nrow(X)
  D <- ncol(X)
  
  if (k >= N) stop("k must be smaller than the number of samples N.")
  if (d >= D) stop("Target dimension d must be smaller than feature dimension D.")
  
  # Step 1: k-Nearest Neighbors Search
  nn_res <- RANN::nn2(data = X, k = k + 1)
  nn_idx <- nn_res$nn.idx[, -1, drop = FALSE] # Exclude self-index (1st col)
  
  # Step 2: Compute Sparse Reconstruction Weights Matrix W
  i_vec <- integer(N * k)
  j_vec <- integer(N * k)
  w_vec <- numeric(N * k)
  
  idx_counter <- 1
  for (i in 1:N) {
    neighbors <- nn_idx[i, ]
    # Neighborhood matrix centered around x_i
    Z <- t(X[neighbors, , drop = FALSE]) - X[i, ] # D x k matrix of differences
    C <- t(Z) %*% Z                               # k x k local covariance matrix
    
    # Regularization for ill-conditioned covariance matrices
    trace_C <- sum(diag(C))
    eps <- if (trace_C > 0) reg * trace_C else reg
    C_reg <- C + eps * diag(k)
    
    # Solve C_reg * w = 1_k
    w <- solve(C_reg, rep(1, k))
    w <- w / sum(w) # Normalize so weights sum to 1
    
    range_idx <- idx_counter:(idx_counter + k - 1)
    i_vec[range_idx] <- i
    j_vec[range_idx] <- neighbors
    w_vec[range_idx] <- w
    idx_counter <- idx_counter + k
  }
  
  W <- Matrix::sparseMatrix(i = i_vec, j = j_vec, x = w_vec, dims = c(N, N))
  
  # Step 3: Construct Cost Matrix M = (I - W)^T * (I - W)
  I_mat <- Matrix::Diagonal(N)
  Iw <- I_mat - W
  M  <- Matrix::t(Iw) %*% Iw
  
  # Ensure M is formatted as a dgCMatrix for RSpectra S3 method dispatch
  M_dgc <- as(M, "dgCMatrix")
  
  # Step 4: Robust Eigen-decomposition for Bottom Eigenvectors of M
  # M is symmetric positive semi-definite. Smallest eigenvalues represent optimal embedding.
  ncv_val <- max(4 * (d + 1), 40)
  
  res_eig <- tryCatch({
    RSpectra::eigs_sym(M_dgc, k = d + 1, which = "SM", opts = list(ncv = ncv_val))
  }, error = function(e1) {
    tryCatch({
      RSpectra::eigs_sym(M_dgc, k = d + 1, sigma = -1e-6, opts = list(ncv = ncv_val))
    }, error = function(e2) {
      RSpectra::eigs_sym(M_dgc, k = d + 1, which = "SA")
    })
  })
  
  ord <- order(res_eig$values)
  sorted_evals <- res_eig$values[ord]
  sorted_evecs <- res_eig$vectors[, ord, drop = FALSE]
  
  if (ncol(sorted_evecs) < (d + 1)) {
    stop(sprintf("Eigen solver returned insufficient eigenvectors: %d available, %d required.",
                 ncol(sorted_evecs), d + 1))
  }
  
  # Omit the 1st trivial eigenvector (corresponding to eigenvalue 0, constant vector)
  Y <- sorted_evecs[, 2:(d + 1), drop = FALSE]
  
  return(list(
    Y           = Y,
    X_train     = X,
    k           = k,
    d           = d,
    reg         = reg,
    W           = W,
    M           = M,
    eigenvalues = sorted_evals[2:(d + 1)]
  ))
}

project_lle <- function(lle_obj, X_test, k = NULL, reg = 1e-3) {
  X_train <- lle_obj$X_train
  Y_train <- lle_obj$Y
  if (is.null(k)) k <- lle_obj$k
  
  N_test <- nrow(X_test)
  d <- ncol(Y_train)
  
  # Find k-NN in the training set for each test sample
  nn_res <- RANN::nn2(data = X_train, query = X_test, k = k)
  nn_idx <- nn_res$nn.idx
  
  Y_test <- matrix(0, nrow = N_test, ncol = d)
  
  for (i in 1:N_test) {
    neighbors <- nn_idx[i, ]
    Z <- t(X_train[neighbors, , drop = FALSE]) - X_test[i, ]
    C <- t(Z) %*% Z
    
    trace_C <- sum(diag(C))
    eps <- if (trace_C > 0) reg * trace_C else reg
    C_reg <- C + eps * diag(k)
    
    w <- solve(C_reg, rep(1, k))
    w <- w / sum(w)
    
    # Linear projection onto training embedding coordinates
    Y_test[i, ] <- colSums(w * Y_train[neighbors, , drop = FALSE])
  }
  
  return(Y_test)
}
