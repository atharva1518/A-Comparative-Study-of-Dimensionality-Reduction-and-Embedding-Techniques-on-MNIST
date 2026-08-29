# ==========================================
# 03 - Find Nearest Neighbours
# ==========================================

# Load preprocessing
'''source("R/02_preprocessing.R")

# Number of neighbours
K <- 10

print(paste("Number of neighbours:", K))
# ------------------------------------------
# Find neighbours for one sample
# ------------------------------------------

i <- 1

distances <- sqrt(
    rowSums(
        (X - X[i, ])^2
    )
)


distances[i] <- Inf

# Find K nearest neighbours
neighbours_i <- order(distances)[1:K]

print(neighbours_i)


Y[1]
Y[neighbours_i]'''

# ==========================================
# 03 - Find Nearest Neighbours
# ==========================================

# Load preprocessed MNIST data
source("R/02_preprocessing.R")


# ------------------------------------------
# 1. Set number of neighbours
# ------------------------------------------

K <- 10

print(paste("Number of neighbours:", K))


# ------------------------------------------
# 2. Number of samples
# ------------------------------------------

n <- nrow(X)

print(paste("Number of samples:", n))


# ------------------------------------------
# 3. Create neighbour matrix
# ------------------------------------------

# Each row will contain the K nearest
# neighbours for one image

neighbours <- matrix(
    0,
    nrow = n,
    ncol = K
)


# ------------------------------------------
# 4. Find K nearest neighbours
# ------------------------------------------

for (i in 1:n) {

    # Calculate Euclidean distance from
    # image i to every image

    distances <- sqrt(
        rowSums(
            (X - X[i, ])^2
        )
    )

    # Ignore the image itself
    distances[i] <- Inf

    # Get indices of K closest images
    neighbours[i, ] <- order(distances)[1:K]
}


# ------------------------------------------
# 5. Check results
# ------------------------------------------

print("Dimensions of neighbour matrix:")
print(dim(neighbours))

print("Neighbours of image 1:")
print(neighbours[1, ])

print("Neighbours of image 2:")
print(neighbours[2, ])

print("Neighbours of image 3:")
print(neighbours[3, ])