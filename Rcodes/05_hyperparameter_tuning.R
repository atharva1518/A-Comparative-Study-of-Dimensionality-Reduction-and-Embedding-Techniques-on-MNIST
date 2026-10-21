# ==============================================================================
# 05_hyperparameter_tuning.R
# Systematic Hyperparameter Search on Validation Set
# ==============================================================================
# Goal: Tune LLE parameters (neighborhood size k, embedding dimension d)
# and select the optimal classifier family strictly using the Validation Set.
#
# NOTE ON CLASSIFIER PARAMETERS:
# In this experiment, classifier hyperparameter values are fixed per model family:
#   - k-NN          : k_neighbors = 5
#   - RBF SVM       : cost = 10, gamma = 0.05
#   - Random Forest : ntree = 100
# LLE neighborhood size k and embedding dimension d are systematically grid-searched.
# ==============================================================================

suppressPackageStartupMessages({
  library(class)
  library(e1071)
  library(randomForest)
})

tune_lle_pipeline <- function(X_tr_lle, y_tr_lle, X_val, y_val,
                              k_values = c(10, 15, 20, 30),
                              d_values = c(10, 20, 30, 50)) {
  cat("\n=========================================================\n")
  cat("STEP 5: HYPERPARAMETER TUNING ON VALIDATION SET\n")
  cat("=========================================================\n")
  cat("Searching grid of LLE k values:", paste(k_values, collapse=", "), "\n")
  cat("Searching grid of Dimensions d:", paste(d_values, collapse=", "), "\n")
  cat("Classifiers evaluated (with fixed model family parameters):\n")
  cat("  1. 5-NN          (k = 5)\n")
  cat("  2. RBF SVM       (cost = 10, gamma = 0.05)\n")
  cat("  3. Random Forest (ntree = 100)\n\n")
  
  results_list <- list()
  row_count <- 1
  
  max_d <- max(d_values)
  
  for (k_val in k_values) {
    cat(sprintf("Fitting LLE base manifold with k = %2d, max_d = %d...", k_val, max_d))
    t0_lle <- Sys.time()
    lle_base <- compute_lle(X_tr_lle, k = k_val, d = max_d)
    Y_val_base <- project_lle(lle_base, X_val, k = k_val)
    t1_lle <- Sys.time()
    cat(sprintf(" Done (%.2fs)\n", as.numeric(t1_lle - t0_lle, units = "secs")))
    
    for (d_val in d_values) {
      Y_tr_sub  <- lle_base$Y[, 1:d_val, drop = FALSE]
      Y_val_sub <- Y_val_base[, 1:d_val, drop = FALSE]
      
      # 1. k-NN Classifier (k=5)
      pred_knn <- class::knn(train = Y_tr_sub, test = Y_val_sub, cl = y_tr_lle, k = 5)
      m_knn    <- compute_metrics(y_val, pred_knn)
      
      results_list[[row_count]] <- data.frame(
        Method            = "LLE + 5-NN",
        LLE_k             = k_val,
        Dimensions        = d_val,
        Classifier        = "5-NN",
        Classifier_Family = "knn",
        Classifier_Params = "k = 5",
        Accuracy          = m_knn$accuracy,
        Precision         = m_knn$precision_macro,
        Recall            = m_knn$recall_macro,
        F1_Score          = m_knn$f1_macro,
        stringsAsFactors  = FALSE
      )
      row_count <- row_count + 1
      
      # 2. RBF SVM Classifier (cost=10, gamma=0.05)
      svm_mod  <- e1071::svm(x = Y_tr_sub, y = y_tr_lle, kernel = "radial", cost = 10, gamma = 0.05)
      pred_svm <- predict(svm_mod, Y_val_sub)
      m_svm    <- compute_metrics(y_val, pred_svm)
      
      results_list[[row_count]] <- data.frame(
        Method            = "LLE + RBF SVM",
        LLE_k             = k_val,
        Dimensions        = d_val,
        Classifier        = "RBF SVM",
        Classifier_Family = "svm",
        Classifier_Params = "cost = 10, gamma = 0.05",
        Accuracy          = m_svm$accuracy,
        Precision         = m_svm$precision_macro,
        Recall            = m_svm$recall_macro,
        F1_Score          = m_svm$f1_macro,
        stringsAsFactors  = FALSE
      )
      row_count <- row_count + 1
      
      # 3. Random Forest Classifier (ntree=100)
      rf_mod  <- randomForest::randomForest(x = Y_tr_sub, y = y_tr_lle, ntree = 100)
      pred_rf <- predict(rf_mod, Y_val_sub)
      m_rf    <- compute_metrics(y_val, pred_rf)
      
      results_list[[row_count]] <- data.frame(
        Method            = "LLE + Random Forest",
        LLE_k             = k_val,
        Dimensions        = d_val,
        Classifier        = "Random Forest",
        Classifier_Family = "rf",
        Classifier_Params = "ntree = 100",
        Accuracy          = m_rf$accuracy,
        Precision         = m_rf$precision_macro,
        Recall            = m_rf$recall_macro,
        F1_Score          = m_rf$f1_macro,
        stringsAsFactors  = FALSE
      )
      row_count <- row_count + 1
    }
  }
  
  results_df <- do.call(rbind, results_list)
  
  # Explicit Audit Check: Ensure expected number of tuning combinations (e.g., 4x4x3 = 48)
  expected_rows <- length(k_values) * length(d_values) * 3
  if (nrow(results_df) != expected_rows) {
    stop(sprintf("CRITICAL ERROR: Expected %d tuning configurations, got %d", expected_rows, nrow(results_df)))
  }
  
  # Sort results by Accuracy descending, breaking ties by F1_Score
  results_df <- results_df[order(-results_df$Accuracy, -results_df$F1_Score), ]
  rownames(results_df) <- NULL
  
  best_config <- results_df[1, ]
  
  cat("\n=========================================================\n")
  cat("TOP 5 HYPERPARAMETER CONFIGURATIONS (VALIDATION SET):\n")
  cat("=========================================================\n")
  print(head(results_df[, c("Method", "LLE_k", "Dimensions", "Classifier", "Accuracy", "F1_Score")], 5))
  cat("\nSelected Optimal Configuration from Validation Search:\n")
  cat("  - LLE k              :", best_config$LLE_k, "\n")
  cat("  - Dimensions d       :", best_config$Dimensions, "\n")
  cat("  - Classifier Selected:", best_config$Classifier, "(Family:", best_config$Classifier_Family, ")\n")
  cat("  - Classifier Params  :", best_config$Classifier_Params, "\n")
  cat(sprintf("  - Validation Accuracy: %.4f (%.2f%%)\n\n", best_config$Accuracy, best_config$Accuracy * 100))
  
  return(list(
    tuning_table = results_df,
    best_config  = best_config
  ))
}
