"""
Evaluation and Diagnostic Visualization Module for Multi-Model AQI Prediction.
Computes regression metrics (MAE, MSE, RMSE, R2) and generates:
- Actual vs Predicted AQI scatter plots for each model
- Residual error distribution plots for each model
- Comparative Feature Importance plot across tree-based ensembles
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for web server environments
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_regression_metrics(y_true, y_pred) -> dict:
    """Computes standard regression metrics (MAE, MSE, RMSE, R2)."""
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

def generate_model_diagnostic_plots(model_key: str, model_name: str, model, X_test, y_test, output_dir: str):
    """
    Generates Actual vs Predicted AQI scatter plot and Residuals distribution plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

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
    fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor=colors['bg'])
    ax.set_facecolor(colors['card_bg'])
    ax.scatter(y_test, y_pred, color=colors['sub'], alpha=0.55, edgecolors='none', label='Test Samples')
    min_val = min(float(y_test.min()), float(y_pred.min()))
    max_val = max(float(y_test.max()), float(y_pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val], color=colors['accent'], linestyle='--', linewidth=2, label='Ideal Fit Line')
    ax.set_xlabel('Actual AQI', color='white', fontsize=10)
    ax.set_ylabel('Predicted AQI', color='white', fontsize=10)
    ax.set_title(f'{model_name}: Actual vs Predicted AQI', color='white', fontsize=11, fontweight='bold')
    ax.grid(True, color=colors['grid'], linestyle=':', alpha=0.6)
    ax.legend(facecolor=colors['card_bg'], edgecolor=colors['grid'], fontsize=9)
    plt.tight_layout()
    actual_pred_path = os.path.join(output_dir, f'{model_key}_actual_pred.png')
    plt.savefig(actual_pred_path, dpi=160)
    plt.close()

    # 2. Residual Distribution Plot
    fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor=colors['bg'])
    ax.set_facecolor(colors['card_bg'])
    ax.scatter(y_pred, residuals, color=colors['red'], alpha=0.55, edgecolors='none')
    ax.axhline(0, color=colors['accent'], linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted AQI', color='white', fontsize=10)
    ax.set_ylabel('Residual (Actual - Predicted)', color='white', fontsize=10)
    ax.set_title(f'{model_name}: Residual Error Analysis', color='white', fontsize=11, fontweight='bold')
    ax.grid(True, color=colors['grid'], linestyle=':', alpha=0.6)
    plt.tight_layout()
    residuals_path = os.path.join(output_dir, f'{model_key}_residuals.png')
    plt.savefig(residuals_path, dpi=160)
    plt.close()

def generate_feature_importance_comparison(tree_models: dict, feature_names: list, output_dir: str):
    """
    Generates a grouped horizontal bar chart comparing feature importances
    across tree-based models (Random Forest, Gradient Boosting, Extra Trees, Decision Tree).
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('dark_background')
    colors = ['#00ff88', '#00bfff', '#ff9900', '#a000ff']
    bg_color = '#071019'
    card_bg = '#0e2233'
    grid_color = '#1c3b52'

    fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor=bg_color)
    ax.set_facecolor(card_bg)

    n_features = len(feature_names)
    y_pos = np.arange(n_features)
    n_models = len(tree_models)
    bar_height = 0.8 / n_models

    for idx, (m_name, m_obj) in enumerate(tree_models.items()):
        # Handle Pipeline vs raw estimator
        estimator = m_obj.named_steps[list(m_obj.named_steps.keys())[-1]] if hasattr(m_obj, 'named_steps') else m_obj
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
            offset = (idx - n_models / 2 + 0.5) * bar_height
            ax.barh(y_pos + offset, importances, height=bar_height,
                    label=m_name, color=colors[idx % len(colors)], alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, color='white', fontsize=10)
    ax.set_xlabel('Relative Importance (Gini / Variance Reduction)', color='white', fontsize=10)
    ax.set_title('Feature Importance Comparison Across Ensemble Models', color='white', fontsize=12, fontweight='bold')
    ax.grid(True, color=grid_color, linestyle=':', alpha=0.6, axis='x')
    ax.legend(facecolor=card_bg, edgecolor=grid_color, fontsize=9)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'feature_importances_comparison.png')
    plt.savefig(plot_path, dpi=160)
    plt.close()
