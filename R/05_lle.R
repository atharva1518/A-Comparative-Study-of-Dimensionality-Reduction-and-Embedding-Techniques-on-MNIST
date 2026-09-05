# ============================================
# 05_lle.R
# Create 2D LLE embedding
# ============================================

library(Matrix)
library(RSpectra)

# --------------------------------------------
# 1. Parameters
# --------------------------------------------

n <- nrow(X)
K <- ncol(weights)

print(paste("Number of samples:", n))
print(paste("Number of neighbours:", K))


# --------------------------------------------
# 2. Create sparse weight matrix
# --------------------------------------------

W <- sparseMatrix(
  i = rep(1:n, each = K),
  j = as.vector(t(neighbours)),
  x = as.vector(t(weights)),
  dims = c(n, n)
)

print("Dimensions of W:")
print(dim(W))


# --------------------------------------------
# 3. Create identity matrix
# --------------------------------------------

I <- Diagonal(n)


# --------------------------------------------
# 4. Create LLE matrix
# --------------------------------------------

M <- t(I - W) %*% (I - W)

print("Dimensions of M:")
print(dim(M))


# --------------------------------------------
# 5. Find eigenvalues near zero
# --------------------------------------------

eigen_result <- eigs(
  M,
  k = 3,
  sigma = 0,
  which = "LM"
)

# Sort eigenvalues
order_index <- order(eigen_result$values)

sorted_values <- eigen_result$values[order_index]

sorted_vectors <- eigen_result$vectors[, order_index]


# --------------------------------------------
# 6. Display eigenvalues
# --------------------------------------------

print("Sorted eigenvalues:")
print(sorted_values)


# --------------------------------------------
# 7. Create 2D embedding
# --------------------------------------------

# The first eigenvector is the trivial solution.
# Use the next two eigenvectors.

Y_embedding <- sorted_vectors[, 2:3]

print("Dimensions of LLE embedding:")
print(dim(Y_embedding))