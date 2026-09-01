"""
Model Evaluation and Visualization Module for Decision Tree AQI Model.
Computes regression metrics (MAE, MSE, RMSE, R2) and renders diagnostic charts:
- Actual vs Predicted AQI scatter
- Residual error distribution
- Feature importance bar chart
- Decision tree structure diagram
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for web server environments
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import plot_tree

def compute_metrics(y_true, y_pred) -> dict:
    """
    Computes standard regression metrics.
    Explicitly distinguishes regression R^2 from classification accuracy.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': round(float(mae), 3),
        'MSE': round(float(mse), 3),
        'RMSE': round(float(rmse), 3),
        'R2_Score': round(float(r2), 4)
    }

def print_metrics_summary(baseline_metrics: dict, tuned_metrics: dict):
    """Prints a comparison table between baseline and tuned models."""
    print("\n=======================================================")
    print("               MODEL EVALUATION COMPARISON            ")
    print("=======================================================")
    print(f"{'Metric':<15} | {'Baseline Model':<18} | {'Tuned Model':<15}")
    print("-------------------------------------------------------")
    print(f"{'MAE':<15} | {baseline_metrics['MAE']:<18} | {tuned_metrics['MAE']:<15}")
    print(f"{'MSE':<15} | {baseline_metrics['MSE']:<18} | {tuned_metrics['MSE']:<15}")
    print(f"{'RMSE':<15} | {baseline_metrics['RMSE']:<18} | {tuned_metrics['RMSE']:<15}")
    print(f"{'R² Score':<15} | {baseline_metrics['R2_Score']:<18} | {tuned_metrics['R2_Score']:<15}")
    print("=======================================================")
    print("Note: R² indicates the proportion of variance explained by the model.")
    print("Regression metrics should not be confused with classification accuracy.\n")

def generate_evaluation_plots(model, X_test, y_test, feature_names: list, output_dir: str):
    """
    Generates and saves diagnostic evaluation plots into the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Configure dark theme styling consistent with dashboard UI
    plt.style.use('dark_background')
    colors = {
        'accent': '#00ff88',
        'sub': '#00bfff',
        'bg': '#071019',
        'card_bg': '#0e2233',
        'grid': '#1c3b52',
        'red': '#ff4d6d'
    }

    # 1. Actual vs Predicted AQI
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=colors['bg'])
    ax.set_facecolor(colors['card_bg'])
    ax.scatter(y_test, y_pred, color=colors['sub'], alpha=0.6, edgecolors='none', label='Test Samples')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color=colors['accent'], linestyle='--', linewidth=2, label='Ideal Fit')
    ax.set_xlabel('Actual AQI', color='white', fontsize=11)
    ax.set_ylabel('Predicted AQI', color='white', fontsize=11)
    ax.set_title('Actual vs Predicted AQI (Decision Tree)', color='white', fontsize=13, fontweight='bold')
    ax.grid(True, color=colors['grid'], linestyle=':', alpha=0.6)
    ax.legend(facecolor=colors['card_bg'], edgecolor=colors['grid'])
    plt.tight_layout()
    plot_actual_vs_pred = os.path.join(output_dir, 'actual_vs_predicted.png')
    plt.savefig(plot_actual_vs_pred, dpi=180)
    plt.close()

    # 2. Residual Distribution Plot
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=colors['bg'])
    ax.set_facecolor(colors['card_bg'])
    ax.scatter(y_pred, residuals, color=colors['red'], alpha=0.6, edgecolors='none')
    ax.axhline(0, color=colors['accent'], linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted AQI', color='white', fontsize=11)
    ax.set_ylabel('Residual (Actual - Predicted)', color='white', fontsize=11)
    ax.set_title('Prediction Error Residuals', color='white', fontsize=13, fontweight='bold')
    ax.grid(True, color=colors['grid'], linestyle=':', alpha=0.6)
    plt.tight_layout()
    plot_residuals = os.path.join(output_dir, 'residuals.png')
    plt.savefig(plot_residuals, dpi=180)
    plt.close()

    # 3. Feature Importance Bar Chart
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=colors['bg'])
    ax.set_facecolor(colors['card_bg'])
    y_pos = np.arange(len(sorted_idx))
    bars = ax.barh(y_pos, importances[sorted_idx], color=colors['accent'], edgecolor='none', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], color='white', fontsize=11)
    ax.set_xlabel('Relative Importance (Gini/Variance Reduction)', color='white', fontsize=11)
    ax.set_title('Feature Importances for AQI Prediction', color='white', fontsize=13, fontweight='bold')
    ax.grid(True, color=colors['grid'], linestyle=':', alpha=0.6, axis='x')

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                ha='left', va='center', color='white', fontsize=9)

    plt.tight_layout()
    plot_importance = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(plot_importance, dpi=180)
    plt.close()

    # 4. Decision Tree Visualization (pruned visualization to depth 3 for readability)
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=colors['bg'])
    plot_tree(
        model,
        max_depth=3,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=8
    )
    ax.set_title('Decision Tree Structure (Top 3 Levels)', color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_tree_path = os.path.join(output_dir, 'decision_tree.png')
    plt.savefig(plot_tree_path, dpi=180)
    plt.close()

    print(f"[+] Diagnostic evaluation plots saved to {output_dir}")
