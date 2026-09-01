"""
End-to-End Model Training and Hyperparameter Tuning Script.
Trains Decision Tree Regressor for AQI prediction, tunes hyperparameters via GridSearchCV,
evaluates performance, generates diagnostic plots, and exports the model & metadata.
"""

import os
import sys
import json
import pickle
import datetime

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

from ml.preprocessing import clean_dataset, impute_missing_values, prepare_features, split_data
from ml.evaluate import compute_metrics, print_metrics_summary, generate_evaluation_plots

def run_training_pipeline(
    data_path: str = None,
    models_dir: str = None,
    static_plots_dir: str = None
):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_path is None:
        data_path = os.path.join(base_dir, 'data', 'air_quality_data.csv')
    if models_dir is None:
        models_dir = os.path.join(base_dir, 'models')
    if static_plots_dir is None:
        static_plots_dir = os.path.join(base_dir, 'static', 'models')

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(static_plots_dir, exist_ok=True)

    print("\n=======================================================")
    print("      SMART AQI - DECISION TREE TRAINING PIPELINE     ")
    print("=======================================================")

    # 1. Load raw dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"[1/8] Loading dataset from {data_path}...")
    df_raw = pd.read_csv(data_path)
    print(f"      Initial shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")

    # 2. Clean & Preprocess Data
    print("\n[2/8] Preprocessing and cleaning dataset...")
    df_clean = clean_dataset(df_raw)
    df_clean = impute_missing_values(df_clean)

    # 3. Select Features & Target
    print("\n[3/8] Selecting features and target variable...")
    X, y, feature_names = prepare_features(df_clean)
    print(f"      Selected Features ({len(feature_names)}): {feature_names}")
    print(f"      Target: AQI (enforced excluded from feature matrix)")

    # 4. Train / Test Split (80/20)
    print("\n[4/8] Performing Train/Test split...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # 5. Train Baseline Decision Tree
    print("\n[5/8] Training Baseline DecisionTreeRegressor...")
    baseline_model = DecisionTreeRegressor(random_state=42)
    baseline_model.fit(X_train, y_train)
    baseline_preds = baseline_model.predict(X_test)
    baseline_metrics = compute_metrics(y_test, baseline_preds)

    # 6. Hyperparameter Optimization via GridSearchCV
    print("\n[6/8] Running Hyperparameter Tuning via GridSearchCV...")
    param_grid = {
        'max_depth': [4, 6, 8, 10, 15],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['squared_error', 'friedman_mse']
    }

    grid_search = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print(f"      Optimal Hyperparameters found:")
    for k, v in best_params.items():
        print(f"        • {k}: {v}")

    # 7. Evaluate Tuned Model & Print Comparison
    print("\n[7/8] Evaluating tuned model...")
    tuned_preds = best_model.predict(X_test)
    tuned_metrics = compute_metrics(y_test, tuned_preds)
    print_metrics_summary(baseline_metrics, tuned_metrics)

    # Display Feature Importances
    importances = best_model.feature_importances_
    feat_importance_dict = {
        name: round(float(imp), 4)
        for name, imp in zip(feature_names, importances)
    }
    print("Feature Importances:")
    for name, val in sorted(feat_importance_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {name:<14} -> {val:.4f}")

    # Generate Evaluation Charts (Saved to both models/plots and static/models for UI)
    print("\nGenerating diagnostic evaluation plots...")
    eval_plots_dir = os.path.join(models_dir, 'evaluation_plots')
    generate_evaluation_plots(best_model, X_test, y_test, feature_names, eval_plots_dir)
    generate_evaluation_plots(best_model, X_test, y_test, feature_names, static_plots_dir)

    # 8. Save Model & Metadata
    print("\n[8/8] Saving artifacts and metadata...")
    model_export_path = os.path.join(models_dir, 'decision_tree_aqi.pkl')
    with open(model_export_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"      [OK] Model saved to {model_export_path}")

    # Also save to root model.pkl and models.pkl for backward compatibility
    root_model_path = os.path.join(base_dir, 'model.pkl')
    root_models_path = os.path.join(base_dir, 'models.pkl')
    with open(root_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    with open(root_models_path, 'wb') as f:
        pickle.dump({
            "decision_tree": best_model,
            "tuned_model": best_model
        }, f)
    print(f"      [OK] Compatibility models updated at {root_model_path} and {root_models_path}")

    # Metadata JSON
    metadata = {
        "model_name": "DecisionTreeRegressor",
        "algorithm": "Decision Tree Regression",
        "version": "2.0.0",
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": feature_names,
        "n_features": len(feature_names),
        "target": "AQI",
        "best_hyperparameters": best_params,
        "evaluation_metrics": tuned_metrics,
        "baseline_metrics": baseline_metrics,
        "feature_importances": feat_importance_dict,
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }

    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"      [OK] Metadata exported to {metadata_path}")

    print("\nPipeline execution complete! Model is ready for production inference.\n")
    return best_model, metadata

if __name__ == '__main__':
    run_training_pipeline()
