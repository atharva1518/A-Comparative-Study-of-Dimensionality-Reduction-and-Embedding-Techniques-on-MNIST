# ==============================================================================
# 02_baseline_model.R
# Baseline Classification on Raw 784-Dimensional Features
# ==============================================================================
# Goal: Establish benchmark performance on original 784 pixel features.
# Benchmark uses the SAME 4,000-sample training subset as the LLE experiment.
# Computes Accuracy, Macro-Precision, Macro-Recall, Macro-F1, and confusion matrix.
# ==============================================================================

suppressPackageStartupMessages({
  library(class)
  library(e1071)
})

# Universal Metric Evaluation Helper
compute_metrics <- function(actual, predicted) {
  actual_fac <- factor(actual)
  pred_fac   <- factor(predicted, levels = levels(actual_fac))
  
  classes <- levels(actual_fac)
  cm <- table(Actual = actual_fac, Predicted = pred_fac)
  
  acc <- sum(actual_fac == pred_fac, na.rm = TRUE) / length(actual_fac)
  
  precision_vec <- numeric(length(classes))
  recall_vec    <- numeric(length(classes))
  f1_vec        <- numeric(length(classes))
  
  names(precision_vec) <- classes
  names(recall_vec)    <- classes
  names(f1_vec)        <- classes
  
  for (i in seq_along(classes)) {
    cls <- classes[i]
    tp <- if (cls %in% rownames(cm) && cls %in% colnames(cm)) cm[cls, cls] else 0
    fp <- if (cls %in% colnames(cm)) sum(cm[, cls]) - tp else 0
    fn <- if (cls %in% rownames(cm)) sum(cm[cls, ]) - tp else 0
    
    prec <- if ((tp + fp) > 0) tp / (tp + fp) else 0
    rec  <- if ((tp + fn) > 0) tp / (tp + fn) else 0
    f1   <- if ((prec + rec) > 0) 2 * prec * rec / (prec + rec) else 0
    
    precision_vec[i] <- prec
    recall_vec[i]    <- rec
    f1_vec[i]        <- f1
  }
  
  return(list(
    accuracy        = acc,
    precision_macro = mean(precision_vec),
    recall_macro    = mean(recall_vec),
    f1_macro        = mean(f1_vec),
    confusion_matrix= cm,
    class_precision = precision_vec,
    class_recall    = recall_vec,
    class_f1        = f1_vec
  ))
}

run_baseline_model <- function(X_train, y_train, X_test, y_test, k_nn = 5) {
  cat("\n=========================================================\n")
  cat("STEP 2: RUNNING BASELINE MODEL (RAW 784 FEATURES)\n")
  cat("=========================================================\n")
  cat("Model: k-Nearest Neighbors (k =", k_nn, ") on 784 raw pixel features\n")
  cat("Training Subset Size: ", nrow(X_train), "samples (Same subset used for LLE)\n")
  cat("Test Set Size       : ", nrow(X_test), "samples\n")
  
  t0 <- Sys.time()
  preds_knn <- class::knn(train = X_train, test = X_test, cl = y_train, k = k_nn)
  t1 <- Sys.time()
  runtime_sec <- as.numeric(t1 - t0, units = "secs")
  
  metrics <- compute_metrics(y_test, preds_knn)
  
  cat("\nBaseline (784 features, 4k train subset) Results:\n")
  cat(sprintf("  - Accuracy        : %.4f (%.2f%%)\n", metrics$accuracy, metrics$accuracy * 100))
  cat(sprintf("  - Macro Precision : %.4f\n", metrics$precision_macro))
  cat(sprintf("  - Macro Recall    : %.4f\n", metrics$recall_macro))
  cat(sprintf("  - Macro F1-Score  : %.4f\n", metrics$f1_macro))
  cat(sprintf("  - Computation Time: %.2f seconds\n\n", runtime_sec))
  
  return(list(
    model_name = paste0("Baseline 5-NN (784 features, ", nrow(X_train), " train)"),
    metrics    = metrics,
    predictions= preds_knn,
    runtime_sec= runtime_sec
  ))
}
