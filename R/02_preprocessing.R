# ==========================================
# 02 - MNIST Preprocessing
# ==========================================

# Load the data from Step 1
source("R/01_load_data.R")

# ------------------------------------------
# 1. Flatten images
# ------------------------------------------

x_train_flat <- array_reshape(
    x_train,
    c(dim(x_train)[1], 784)
)

print("After flattening:")
print(dim(x_train_flat))


# ------------------------------------------
# 2. Normalize pixel values
# ------------------------------------------

x_train_flat <- x_train_flat / 255

print("Pixel value range:")
print(range(x_train_flat))


# ------------------------------------------
# 3. Select a reproducible subset
# ------------------------------------------

set.seed(42)

n_samples <- 5000

sample_indices <- sample(
    1:nrow(x_train_flat),
    n_samples
)

X <- x_train_flat[sample_indices, ]
Y <- y_train[sample_indices]


# ------------------------------------------
# 4. Check final dataset
# ------------------------------------------

print("Final feature matrix:")
print(dim(X))

print("Final labels:")
print(length(Y))