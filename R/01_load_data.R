library(keras3)

mnist <- dataset_mnist()

x_train <- mnist$train$x
y_train <- mnist$train$y

x_test <- mnist$test$x
y_test <- mnist$test$y

print(dim(x_train))
print(dim(y_train))

print(dim(x_test))
print(dim(y_test))

image(
    x_train[1,,],
    col = gray.colors(256),
    axes = FALSE,
    useRaster = TRUE
)

print(y_train[1])

'''
# Flatten images
x_train_flat <- array_reshape(
    x_train,
    c(dim(x_train)[1], 784)
)

print(dim(x_train_flat))
range(x_train_flat)

# -------------------------
# Normalize
# -------------------------

x_train_flat <- x_train_flat / 255

print(range(x_train_flat))'''