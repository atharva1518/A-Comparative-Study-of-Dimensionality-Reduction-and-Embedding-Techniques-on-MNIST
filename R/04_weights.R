# ============================================
# 04_weights.R
# Calculate reconstruction weights for LLE
# ============================================

K <- 10
n <- nrow(X)

# Store weights for every data point
weights <- matrix(0, nrow = n, ncol = K)

# Small value to make calculations more stable
reg <- 1e-3

# Calculate weights for every data point
for (i in 1:n) {

    # Current data point
    Xi <- X[i, ]

    # Get indices of its K neighbours
    neighbor_indices <- neighbours[i, ]

    # Get the actual neighbour data
    Xn <- X[neighbor_indices, , drop = FALSE]

    # Difference between neighbours and current point
    Z <- sweep(Xn, 2, Xi, "-")

    # Local covariance matrix
    C <- Z %*% t(Z)

    # Vector of ones
    ones <- rep(1, K)

    # Calculate weights
    w <- solve(C + reg * diag(K), ones)

    # Make weights add up to 1
    w <- w / sum(w)

    # Store weights
    weights[i, ] <- w
}

print(dim(weights))
print(weights[1, ])
print(sum(weights[1, ]))