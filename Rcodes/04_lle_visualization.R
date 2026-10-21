# ==============================================================================
# 04_lle_visualization.R
# 2D LLE vs PCA Manifold Visualizations
# ==============================================================================
# Goal: Compare 2D linear projection (PCA) with 2D non-linear embedding (LLE).
# Demonstrates visually how LLE unfolds non-linear manifold structures of digit classes.
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(gridExtra)
  library(stats)
})

visualize_pca_vs_lle <- function(X_train, y_train, k_lle = 15, save_plot = TRUE) {
  cat("\n=========================================================\n")
  cat("STEP 4: GENERATING 2D LLE vs PCA VISUALIZATIONS\n")
  cat("=========================================================\n")
  
  # 1. Compute 2D PCA Embedding
  cat("Computing 2D PCA embedding...\n")
  pca_res <- prcomp(X_train, center = TRUE, scale. = FALSE)
  df_pca  <- data.frame(
    PC1   = pca_res$x[, 1],
    PC2   = pca_res$x[, 2],
    Digit = factor(y_train)
  )
  
  var_explained <- summary(pca_res)$importance[2, 1:2] * 100
  
  p_pca <- ggplot(df_pca, aes(x = PC1, y = PC2, color = Digit)) +
    geom_point(alpha = 0.6, size = 1.2) +
    scale_color_brewer(palette = "Set1") +
    theme_minimal() +
    labs(
      title = "2D Linear PCA Embedding of MNIST",
      subtitle = sprintf("PC1 (%.1f%% var) vs PC2 (%.1f%% var)", var_explained[1], var_explained[2]),
      x = "Principal Component 1",
      y = "Principal Component 2"
    ) +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      legend.position = "right"
    )
  
  # 2. Compute 2D LLE Embedding
  cat("Computing 2D LLE embedding (k =", k_lle, ")...\n")
  lle_2d <- compute_lle(X_train, k = k_lle, d = 2)
  df_lle <- data.frame(
    LLE1  = lle_2d$Y[, 1],
    LLE2  = lle_2d$Y[, 2],
    Digit = factor(y_train)
  )
  
  p_lle <- ggplot(df_lle, aes(x = LLE1, y = LLE2, color = Digit)) +
    geom_point(alpha = 0.6, size = 1.2) +
    scale_color_brewer(palette = "Set1") +
    theme_minimal() +
    labs(
      title = sprintf("2D Non-Linear LLE Embedding of MNIST (k = %d)", k_lle),
      subtitle = "Locally Linear Embedding manifold structure",
      x = "LLE Dimension 1",
      y = "LLE Dimension 2"
    ) +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      legend.position = "right"
    )
  
  if (save_plot) {
    grid_comp <- gridExtra::arrangeGrob(p_pca, p_lle, ncol = 2, top = "Comparison of Linear PCA vs Non-Linear LLE Digit Separation")
    ggsave(file.path(getwd(), "pca_vs_lle_2d.png"), plot = grid_comp, width = 11, height = 5, dpi = 130)
    cat("Saved 2D visualization comparison to 'pca_vs_lle_2d.png'\n")
  }
  
  return(list(
    pca_plot = p_pca,
    lle_plot = p_lle,
    lle_2d   = lle_2d
  ))
}
