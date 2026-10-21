# ==============================================================================
# 01_load_and_preprocess.R
# MNIST Data Loading, Inspection, Robust Stratified Sampling, and Preprocessing
# ==============================================================================
# Goal: Load the MNIST dataset in R, perform data checks, plot representative
# digits, and normalize pixel intensities without data leakage.
# Implements TRUE Robust Stratified Sampling across all 10 digit classes (0-9).
# ==============================================================================

suppressPackageStartupMessages({
  library(dslabs)
  library(ggplot2)
  library(gridExtra)
})

load_and_preprocess_mnist <- function(seed = 42, 
                                      n_lle_train = 4000, 
                                      n_val = 1000, 
                                      save_plot = TRUE) {
  set.seed(seed)
  cat("\n=========================================================\n")
  cat("STEP 1: LOADING & PREPROCESSING MNIST DATASET\n")
  cat("=========================================================\n")
  
  # 1. Load Dataset
  mnist <- dslabs::read_mnist()
  
  X_train_full <- mnist$train$images # 60,000 x 784
  y_train_full <- factor(mnist$train$labels)
  
  X_test_full  <- mnist$test$images  # 10,000 x 784
  y_test_full  <- factor(mnist$test$labels)
  
  cat("Dataset dimensions loaded:\n")
  cat("  - Full Train Images:", dim(X_train_full)[1], "samples x", dim(X_train_full)[2], "features\n")
  cat("  - Full Test Images :", dim(X_test_full)[1], "samples x", dim(X_test_full)[2], "features\n")
  cat("  - Image Resolution  : 28 x 28 pixels = 784 total features\n\n")
  
  # 2. Check Missing Values
  na_train <- sum(is.na(X_train_full))
  na_test  <- sum(is.na(X_test_full))
  cat("Missing value audit:\n")
  cat("  - Missing values in train:", na_train, "\n")
  cat("  - Missing values in test :", na_test, "\n\n")
  
  # 3. Robust Stratified Sampling across Digit Classes (0-9)
  # Computes exact quota per class even if totals are not divisible by 10.
  cat("Performing genuine Stratified Sampling by digit class...\n")
  split_indices <- split(seq_len(nrow(X_train_full)), y_train_full)
  classes <- levels(y_train_full)
  n_classes <- length(classes)
  
  # Calculate exact quotas per class
  base_tr_quota <- rep(floor(n_lle_train / n_classes), n_classes)
  rem_tr        <- n_lle_train %% n_classes
  if (rem_tr > 0) base_tr_quota[1:rem_tr] <- base_tr_quota[1:rem_tr] + 1
  names(base_tr_quota) <- classes
  
  base_val_quota <- rep(floor(n_val / n_classes), n_classes)
  rem_val        <- n_val %% n_classes
  if (rem_val > 0) base_val_quota[1:rem_val] <- base_val_quota[1:rem_val] + 1
  names(base_val_quota) <- classes
  
  lle_train_idx <- integer(0)
  val_idx       <- integer(0)
  
  for (cls in classes) {
    cls_idx <- split_indices[[cls]]
    cls_shuffled <- sample(cls_idx)
    
    n_tr_c  <- base_tr_quota[cls]
    n_val_c <- base_val_quota[cls]
    
    idx_tr  <- cls_shuffled[1:n_tr_c]
    idx_v   <- cls_shuffled[(n_tr_c + 1):(n_tr_c + n_val_c)]
    
    lle_train_idx <- c(lle_train_idx, idx_tr)
    val_idx       <- c(val_idx, idx_v)
  }
  
  # Shuffle final subset order
  set.seed(seed)
  lle_train_idx <- sample(lle_train_idx)
  val_idx       <- sample(val_idx)
  
  # ----------------------------------------------------------------------------
  # EXPLICIT AUDIT CHECKS
  # ----------------------------------------------------------------------------
  overlap_count <- length(intersect(lle_train_idx, val_idx))
  if (overlap_count != 0) {
    stop(paste("CRITICAL ERROR: Training and validation index overlap detected! Overlap =", overlap_count))
  }
  
  if (length(lle_train_idx) != n_lle_train) {
    stop(sprintf("CRITICAL ERROR: Expected %d LLE train samples, got %d", n_lle_train, length(lle_train_idx)))
  }
  
  if (length(val_idx) != n_val) {
    stop(sprintf("CRITICAL ERROR: Expected %d validation samples, got %d", n_val, length(val_idx)))
  }
  
  if (nrow(X_test_full) != 10000) {
    stop(sprintf("CRITICAL ERROR: Expected 10000 test samples, got %d", nrow(X_test_full)))
  }
  
  X_tr_lle <- X_train_full[lle_train_idx, ] / 255.0
  y_tr_lle <- y_train_full[lle_train_idx]
  
  X_val    <- X_train_full[val_idx, ] / 255.0
  y_val    <- y_train_full[val_idx]
  
  X_test   <- X_test_full / 255.0
  y_test   <- y_test_full
  
  cat("Data Splits for LLE Pipeline:\n")
  cat("  - LLE Fitting Subset (Train):", nrow(X_tr_lle), "samples (Stratified)\n")
  cat("  - Validation Subset         :", nrow(X_val), "samples (Stratified)\n")
  cat("  - Holdout Test Set          :", nrow(X_test), "samples (Untouched)\n")
  cat("  - Zero-Overlap Audit        : PASSED (length(intersect) = 0)\n\n")
  
  # 4. Verify Class Distributions
  cat("Class Distribution in LLE Train Subset (Stratified):\n")
  print(table(y_tr_lle))
  cat("\nClass Distribution in Validation Subset (Stratified):\n")
  print(table(y_val))
  cat("\nClass Distribution in Holdout Test Set:\n")
  print(table(y_test))
  cat("\n")
  
  # 5. Render Representative Handwritten Digits Visualization
  if (save_plot) {
    plot_list <- list()
    set.seed(seed)
    sample_indices <- sample(1:length(y_tr_lle), 25)
    
    for (idx in seq_along(sample_indices)) {
      img_matrix <- matrix(X_tr_lle[sample_indices[idx], ], nrow = 28, ncol = 28, byrow = TRUE)
      img_matrix <- t(img_matrix[28:1, ]) # Orient correctly for plotting
      df_img <- expand.grid(x = 1:28, y = 1:28)
      df_img$val <- as.vector(img_matrix)
      
      digit_label <- as.character(y_tr_lle[sample_indices[idx]])
      
      p <- ggplot(df_img, aes(x = x, y = y, fill = val)) +
        geom_tile() +
        scale_fill_gradient(low = "white", high = "black") +
        theme_void() +
        theme(legend.position = "none", 
              panel.border = element_rect(color = "#cccccc", fill = NA, linewidth = 0.5),
              plot.title = element_text(size = 9, hjust = 0.5, face = "bold")) +
        ggtitle(paste("Digit:", digit_label))
      
      plot_list[[idx]] <- p
    }
    
    grid_grob <- gridExtra::arrangeGrob(grobs = plot_list, ncol = 5, top = "Representative MNIST Handwritten Digits (28x28 = 784 Pixels)")
    ggsave(file.path(getwd(), "mnist_sample_digits.png"), plot = grid_grob, width = 8, height = 8, dpi = 120)
    cat("Saved digit visualization to 'mnist_sample_digits.png'\n")
  }
  
  return(list(
    X_tr_lle = X_tr_lle,
    y_tr_lle = y_tr_lle,
    X_val    = X_val,
    y_val    = y_val,
    X_test   = X_test,
    y_test   = y_test,
    X_train_full = X_train_full / 255.0,
    y_train_full = y_train_full
  ))
}

if (sys.nframe() == 0) {
  data_obj <- load_and_preprocess_mnist()
}
