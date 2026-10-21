# ==============================================================================
# 06_final_evaluation.R
# Final Test Set Evaluation, Confusion Matrix Analysis, and PCA vs LLE Comparison
# ==============================================================================
# Goal: Evaluate the dynamically selected optimal LLE + Classifier model on the 10,000 test set.
# Supports 5-NN, RBF SVM, and Random Forest classifiers.
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(e1071)
  library(class)
  library(randomForest)
  library(stats)
})

run_final_evaluation <- function(data_obj, best_config, baseline_result = NULL) {
  best_k          <- best_config$LLE_k
  best_d          <- best_config$Dimensions
  best_classifier <- best_config$Classifier
  cls_family      <- best_config$Classifier_Family
  cls_params      <- best_config$Classifier_Params
  
  cat("\n=========================================================\n")
  cat("STEP 6: FINAL HOLDOUT TEST SET EVALUATION\n")
  cat("=========================================================\n")
  cat(sprintf("Selected Final LLE Configuration: k = %d, d = %d\n", best_k, best_d))
  cat(sprintf("Selected Classifier             : %s (%s)\n\n", best_classifier, cls_params))
  
  X_tr_lle <- data_obj$X_tr_lle
  y_tr_lle <- data_obj$y_tr_lle
  X_test   <- data_obj$X_test
  y_test   <- data_obj$y_test
  
  # ----------------------------------------------------------------------------
  # 1. Fit Final LLE & Project Test Set
  # ----------------------------------------------------------------------------
  cat("1. Fitting LLE on Training Subset...")
  t0_lle_fit <- Sys.time()
  lle_final  <- compute_lle(X_tr_lle, k = best_k, d = best_d)
  t1_lle_fit <- Sys.time()
  time_lle_fit <- as.numeric(t1_lle_fit - t0_lle_fit, units = "secs")
  cat(sprintf(" Done (%.2fs)\n", time_lle_fit))
  
  cat("2. Out-of-Sample Projecting 10,000 Test Set Samples into LLE Space...")
  t0_lle_proj <- Sys.time()
  Y_test_lle  <- project_lle(lle_final, X_test, k = best_k)
  t1_lle_proj <- Sys.time()
  time_lle_proj <- as.numeric(t1_lle_proj - t0_lle_proj, units = "secs")
  cat(sprintf(" Done (%.2fs)\n", time_lle_proj))
  
  # ----------------------------------------------------------------------------
  # 2. Train Selected Classifier on LLE Embedding
  # ----------------------------------------------------------------------------
  cat(sprintf("3. Training Selected Classifier (%s) on LLE %dD Embedding...", best_classifier, best_d))
  t0_cls <- Sys.time()
  
  if (cls_family == "svm" || grepl("SVM", best_classifier, ignore.case = TRUE)) {
    model_obj <- e1071::svm(x = lle_final$Y, y = y_tr_lle, kernel = "radial", cost = 10, gamma = 0.05)
    t1_cls <- Sys.time()
    time_cls_train <- as.numeric(t1_cls - t0_cls, units = "secs")
    
    t0_pred <- Sys.time()
    preds_lle_test <- predict(model_obj, Y_test_lle)
    t1_pred <- Sys.time()
    time_pred <- as.numeric(t1_pred - t0_pred, units = "secs")
    
  } else if (cls_family == "knn" || grepl("NN", best_classifier, ignore.case = TRUE)) {
    t0_pred <- Sys.time()
    preds_lle_test <- class::knn(train = lle_final$Y, test = Y_test_lle, cl = y_tr_lle, k = 5)
    t1_pred <- Sys.time()
    time_cls_train <- 0.0
    time_pred <- as.numeric(t1_pred - t0_pred, units = "secs")
    
  } else if (cls_family == "rf" || grepl("Forest", best_classifier, ignore.case = TRUE)) {
    model_obj <- randomForest::randomForest(x = lle_final$Y, y = y_tr_lle, ntree = 100)
    t1_cls <- Sys.time()
    time_cls_train <- as.numeric(t1_cls - t0_cls, units = "secs")
    
    t0_pred <- Sys.time()
    preds_lle_test <- predict(model_obj, Y_test_lle)
    t1_pred <- Sys.time()
    time_pred <- as.numeric(t1_pred - t0_pred, units = "secs")
  } else {
    stop("Unknown classifier family selected.")
  }
  
  cat(sprintf(" Done (Train: %.2fs, Predict: %.2fs)\n\n", time_cls_train, time_pred))
  
  metrics_lle <- compute_metrics(y_test, preds_lle_test)
  total_time_lle <- time_lle_fit + time_lle_proj + time_cls_train + time_pred
  
  cat("=========================================================\n")
  cat(sprintf("FINAL TEST SET PERFORMANCE (LLE + %s):\n", best_classifier))
  cat("=========================================================\n")
  cat(sprintf("  - Final Test Accuracy: %.4f (%.2f%%)\n", metrics_lle$accuracy, metrics_lle$accuracy * 100))
  cat(sprintf("  - Macro Precision    : %.4f\n", metrics_lle$precision_macro))
  cat(sprintf("  - Macro Recall       : %.4f\n", metrics_lle$recall_macro))
  cat(sprintf("  - Macro F1-Score     : %.4f\n", metrics_lle$f1_macro))
  cat(sprintf("  - Total Pipeline Time: %.2f seconds\n\n", total_time_lle))
  
  # ----------------------------------------------------------------------------
  # 3. Confusion Matrix Analysis & Visualization
  # ----------------------------------------------------------------------------
  cm <- metrics_lle$confusion_matrix
  df_cm <- as.data.frame(cm)
  colnames(df_cm) <- c("Actual", "Predicted", "Count")
  
  p_cm <- ggplot(df_cm, aes(x = Predicted, y = Actual, fill = Count)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Count), color = ifelse(df_cm$Count > max(df_cm$Count)/2, "white", "black"), size = 3.5, fontface = "bold") +
    scale_fill_gradient(low = "#e8f4f8", high = "#104e8b") +
    scale_y_discrete(limits = rev(levels(df_cm$Actual))) +
    theme_minimal() +
    labs(
      title = sprintf("MNIST Digit Confusion Matrix (LLE + %s)", best_classifier),
      subtitle = sprintf("Test Accuracy: %.2f%% | Total Samples: %d", metrics_lle$accuracy * 100, length(y_test)),
      x = "Predicted Digit Label",
      y = "Actual Digit Label"
    ) +
    theme(
      plot.title = element_text(face = "bold", size = 13),
      axis.text  = element_text(size = 10, face = "bold"),
      legend.position = "right"
    )
  
  ggsave(file.path(getwd(), "confusion_matrix.png"), plot = p_cm, width = 7, height = 6, dpi = 140)
  cat("Saved confusion matrix heatmap to 'confusion_matrix.png'\n\n")
  
  # Dynamically extract top 5 most frequently confused digit pairs (excluding zero counts)
  cm_mat <- as.matrix(cm)
  diag(cm_mat) <- 0 # Remove correct predictions
  
  nonzero_idx <- which(cm_mat > 0)
  if (length(nonzero_idx) > 0) {
    ord_idx   <- nonzero_idx[order(cm_mat[nonzero_idx], decreasing = TRUE)]
    top_k_idx <- head(ord_idx, 5)
  } else {
    top_k_idx <- integer(0)
  }
  
  confused_pairs_list <- list()
  if (length(top_k_idx) > 0) {
    cat("Top Dynamically Derived Confused Digit Pairs:\n")
    for (i in seq_along(top_k_idx)) {
      idx <- top_k_idx[i]
      r <- row(cm_mat)[idx]
      c <- col(cm_mat)[idx]
      actual_lbl <- rownames(cm_mat)[r]
      pred_lbl   <- colnames(cm_mat)[c]
      cnt        <- cm_mat[r, c]
      confused_pairs_list[[i]] <- list(actual = actual_lbl, predicted = pred_lbl, count = cnt)
      cat(sprintf("  %d. Actual %s misclassified as %s: %d cases\n", i, actual_lbl, pred_lbl, cnt))
    }
  } else {
    cat("No misclassification errors recorded (100% test accuracy).\n")
  }
  cat("\n")
  
  # ----------------------------------------------------------------------------
  # 4. Comparative Benchmark: PCA + Selected Classifier
  # ----------------------------------------------------------------------------
  cat(sprintf("Running Benchmark: PCA (%d components) + %s...\n", best_d, best_classifier))
  t0_pca <- Sys.time()
  pca_mod <- prcomp(X_tr_lle, center = TRUE, scale. = FALSE)
  Y_tr_pca <- pca_mod$x[, 1:best_d, drop = FALSE]
  
  # Center and project test set onto PCA eigenvectors
  X_test_centered <- scale(X_test, center = pca_mod$center, scale = FALSE)
  Y_test_pca <- X_test_centered %*% pca_mod$rotation[, 1:best_d, drop = FALSE]
  
  if (cls_family == "svm" || grepl("SVM", best_classifier, ignore.case = TRUE)) {
    svm_pca <- e1071::svm(x = Y_tr_pca, y = y_tr_lle, kernel = "radial", cost = 10, gamma = 0.05)
    preds_pca <- predict(svm_pca, Y_test_pca)
  } else if (cls_family == "knn" || grepl("NN", best_classifier, ignore.case = TRUE)) {
    preds_pca <- class::knn(train = Y_tr_pca, test = Y_test_pca, cl = y_tr_lle, k = 5)
  } else if (cls_family == "rf" || grepl("Forest", best_classifier, ignore.case = TRUE)) {
    rf_pca <- randomForest::randomForest(x = Y_tr_pca, y = y_tr_lle, ntree = 100)
    preds_pca <- predict(rf_pca, Y_test_pca)
  }
  
  t1_pca <- Sys.time()
  total_time_pca <- as.numeric(t1_pca - t0_pca, units = "secs")
  
  metrics_pca <- compute_metrics(y_test, preds_pca)
  
  # ----------------------------------------------------------------------------
  # 5. Build Final Comprehensive Comparison Table
  # ----------------------------------------------------------------------------
  comp_table <- data.frame(
    Pipeline      = c("A. Baseline (784 features)", 
                      paste0("B. PCA + ", best_classifier, " (", best_d, "D)"), 
                      paste0("C. LLE + ", best_classifier, " (", best_d, "D)")),
    Features_Dim  = c(784, best_d, best_d),
    LLE_k         = c("-", "-", best_k),
    Accuracy      = c(
      if (!is.null(baseline_result)) baseline_result$metrics$accuracy else NA,
      metrics_pca$accuracy,
      metrics_lle$accuracy
    ),
    Precision_Macro = c(
      if (!is.null(baseline_result)) baseline_result$metrics$precision_macro else NA,
      metrics_pca$precision_macro,
      metrics_lle$precision_macro
    ),
    Recall_Macro = c(
      if (!is.null(baseline_result)) baseline_result$metrics$recall_macro else NA,
      metrics_pca$recall_macro,
      metrics_lle$recall_macro
    ),
    F1_Macro = c(
      if (!is.null(baseline_result)) baseline_result$metrics$f1_macro else NA,
      metrics_pca$f1_macro,
      metrics_lle$f1_macro
    ),
    Runtime_Sec = c(
      if (!is.null(baseline_result)) round(baseline_result$runtime_sec, 2) else NA,
      round(total_time_pca, 2),
      round(total_time_lle, 2)
    ),
    stringsAsFactors = FALSE
  )
  
  cat("=========================================================\n")
  cat("FINAL MODEL COMPARISON TABLE:\n")
  cat("=========================================================\n")
  print(comp_table)
  cat("\n")
  
  return(list(
    metrics_lle         = metrics_lle,
    metrics_pca         = metrics_pca,
    comparison_table    = comp_table,
    confused_pairs_list = confused_pairs_list,
    cm_plot             = p_cm,
    Y_test_lle          = Y_test_lle,
    preds_lle           = preds_lle_test
  ))
}
